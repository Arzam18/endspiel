//! Shared lock-free transposition table for Lazy SMP.
//!
//! Each entry is 16 bytes: two `AtomicU64` fields. XOR-based verification
//! detects torn reads without any locking. `Ordering::Relaxed` is used
//! throughout — the TT is a best-effort cache and occasional corruption
//! from races is acceptable.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use chess_common::Move;

// ---------------------------------------------------------------------------
// TT entry flag (2 bits)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TTFlag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
}

impl TTFlag {
    fn from_u8(v: u8) -> Self {
        match v & 3 {
            0 => Self::Exact,
            1 => Self::LowerBound,
            _ => Self::UpperBound,
        }
    }
}

// ---------------------------------------------------------------------------
// Unpacked TT entry (for caller convenience)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct TTData {
    pub depth: u8,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Move,
    /// Static eval of the position before the learned correction, if stored.
    pub eval: Option<i32>,
}

// ---------------------------------------------------------------------------
// Packed data layout (64 bits):
//   bits  0..15 : best_move (u16)
//   bits 16..31 : score (i16, clamped)
//   bits 32..39 : depth (u8)
//   bits 40..41 : flag (2 bits)
//   bits 42..47 : generation (6 bits)
//   bits 48..63 : static eval (i16, `EVAL_NONE` = not stored)
// ---------------------------------------------------------------------------

/// Generations wrap at 64. Only equality is compared, so after a wrap an entry
/// from 64 searches ago merely looks current to the replacement rule.
const GENERATION_MASK: u8 = 0x3F;
const EVAL_NONE: i16 = i16::MIN;

fn pack_data(
    depth: u8,
    score: i32,
    flag: TTFlag,
    best_move: Move,
    generation: u8,
    eval: Option<i32>,
) -> u64 {
    let m = best_move.0 as u64;
    let s = (score.clamp(-32000, 32000) as i16) as u16 as u64;
    let d = depth as u64;
    let f = flag as u64;
    let g = (generation & GENERATION_MASK) as u64;
    // an eval outside i16 is not cached rather than clamped
    let e = eval
        .and_then(|eval| i16::try_from(eval).ok())
        .filter(|&eval| eval != EVAL_NONE)
        .unwrap_or(EVAL_NONE) as u16 as u64;
    m | (s << 16) | (d << 32) | (f << 40) | (g << 42) | (e << 48)
}

fn unpack_data(data: u64) -> (TTData, u8) {
    let best_move = Move((data & 0xFFFF) as u16);
    let score = ((data >> 16) & 0xFFFF) as u16 as i16 as i32;
    let depth = ((data >> 32) & 0xFF) as u8;
    let flag = TTFlag::from_u8(((data >> 40) & 3) as u8);
    let generation = ((data >> 42) as u8) & GENERATION_MASK;
    let eval = (data >> 48) as u16 as i16;
    (
        TTData {
            depth,
            score,
            flag,
            best_move,
            eval: (eval != EVAL_NONE).then_some(i32::from(eval)),
        },
        generation,
    )
}

// ---------------------------------------------------------------------------
// Atomic TT entry (16 bytes, cache-line friendly)
// ---------------------------------------------------------------------------

#[repr(C, align(16))]
struct AtomicTTEntry {
    /// Stores `hash_key XOR data` for torn-read detection.
    key: AtomicU64,
    /// Packed entry data.
    data: AtomicU64,
}

impl AtomicTTEntry {
    fn new() -> Self {
        Self {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared transposition table
// ---------------------------------------------------------------------------

pub struct SharedTT {
    entries: Box<[AtomicTTEntry]>,
    mask: usize,
    generation: AtomicU8,
}

impl SharedTT {
    /// Create a new transposition table with the given size in MB.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = 16; // two u64
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        let num_entries = num_entries.next_power_of_two() / 2;
        let num_entries = num_entries.max(1024);

        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            entries.push(AtomicTTEntry::new());
        }

        Self {
            entries: entries.into_boxed_slice(),
            mask: num_entries - 1,
            generation: AtomicU8::new(0),
        }
    }

    /// Probe the TT for the given hash. Returns `None` on miss or corrupted read.
    pub fn probe(&self, hash: u64) -> Option<TTData> {
        let idx = hash as usize & self.mask;
        let entry = &self.entries[idx];

        let data = entry.data.load(Ordering::Relaxed);
        let key = entry.key.load(Ordering::Relaxed);

        // XOR verification: the stored key should equal hash ^ data.
        // If a concurrent write tore either field, this will (very likely) fail.
        if key ^ data != hash {
            return None;
        }

        let (tt_data, _generation) = unpack_data(data);
        Some(tt_data)
    }

    /// Store an entry. Uses a generation + depth replacement scheme:
    /// - Always replace if same hash (refinement)
    /// - Replace if current generation is newer than stored
    /// - Replace if depth is >= stored depth within same generation
    ///
    /// `eval` is the node's static eval before the correction, if it has one.
    pub fn store(
        &self,
        hash: u64,
        depth: u8,
        score: i32,
        flag: TTFlag,
        best_move: Move,
        eval: Option<i32>,
    ) {
        let idx = hash as usize & self.mask;
        let entry = &self.entries[idx];
        let generation = self.generation.load(Ordering::Relaxed) & GENERATION_MASK;

        let data = pack_data(depth, score, flag, best_move, generation, eval);

        // Check if we should replace
        let old_data = entry.data.load(Ordering::Relaxed);
        let old_key = entry.key.load(Ordering::Relaxed);

        if old_data != 0 {
            let old_hash = old_key ^ old_data;
            let (_old_entry, old_generation) = unpack_data(old_data);

            // Keep deeper entries from the current generation
            if old_hash != hash && old_generation == generation && _old_entry.depth >= depth + 2 {
                return;
            }
        }

        entry.data.store(data, Ordering::Relaxed);
        entry.key.store(hash ^ data, Ordering::Relaxed);
    }

    /// Increment the generation counter. Call once per search.
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Approximate fill ratio in permille (0..1000).
    pub fn hashfull(&self) -> u16 {
        let sample = 1000.min(self.entries.len());
        let mut used = 0u32;
        let generation = self.generation.load(Ordering::Relaxed) & GENERATION_MASK;

        for i in 0..sample {
            let data = self.entries[i].data.load(Ordering::Relaxed);
            if data != 0 {
                let (_entry, entry_generation) = unpack_data(data);
                if entry_generation == generation {
                    used += 1;
                }
            }
        }

        ((used as u64 * 1000) / sample as u64) as u16
    }

    /// Clear all entries.
    pub fn clear(&self) {
        for entry in self.entries.iter() {
            entry.key.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Resize the table. Returns a new SharedTT with the given size.
    /// The old table is dropped.
    pub fn resize(size_mb: usize) -> Self {
        Self::new(size_mb)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let m = Move(1234);
        let data = pack_data(12, -500, TTFlag::LowerBound, m, 7, Some(-321));
        let (entry, generation) = unpack_data(data);
        assert_eq!(entry.depth, 12);
        assert_eq!(entry.score, -500);
        assert_eq!(entry.flag, TTFlag::LowerBound);
        assert_eq!(entry.best_move, m);
        assert_eq!(generation, 7);
        assert_eq!(entry.eval, Some(-321));
    }

    #[test]
    fn test_pack_unpack_positive_score() {
        let m = Move(4321);
        let data = pack_data(20, 31000, TTFlag::Exact, m, 63, Some(i32::from(i16::MAX)));
        let (entry, generation) = unpack_data(data);
        assert_eq!(entry.depth, 20);
        assert_eq!(entry.score, 31000);
        assert_eq!(entry.flag, TTFlag::Exact);
        assert_eq!(entry.best_move, m);
        assert_eq!(generation, 63);
        assert_eq!(entry.eval, Some(i32::from(i16::MAX)));
    }

    #[test]
    fn test_eval_outside_i16_is_not_stored() {
        for eval in [None, Some(40_000), Some(-40_000), Some(i32::from(i16::MIN))] {
            let data = pack_data(5, 0, TTFlag::Exact, Move(1), 64 + 3, eval);
            let (entry, generation) = unpack_data(data);
            assert_eq!(entry.eval, None, "{eval:?}");
            assert_eq!(generation, 3);
            assert_eq!(entry.depth, 5);
        }
    }

    #[test]
    fn test_store_and_probe() {
        let tt = SharedTT::new(1);
        let hash = 0xDEAD_BEEF_1234_5678u64;
        let m = Move(100);

        tt.store(hash, 8, 150, TTFlag::Exact, m, Some(42));
        let result = tt.probe(hash);
        assert!(result.is_some());
        let entry = result.unwrap();
        assert_eq!(entry.depth, 8);
        assert_eq!(entry.score, 150);
        assert_eq!(entry.flag, TTFlag::Exact);
        assert_eq!(entry.best_move, m);
        assert_eq!(entry.eval, Some(42));
    }

    #[test]
    fn test_probe_miss() {
        let tt = SharedTT::new(1);
        assert!(tt.probe(12345).is_none());
    }

    #[test]
    fn test_hashfull() {
        let tt = SharedTT::new(1);
        assert_eq!(tt.hashfull(), 0);

        // Store some entries
        for i in 0..500u64 {
            tt.store(i * 7919, 5, 0, TTFlag::Exact, Move::NULL, None);
        }
        let hf = tt.hashfull();
        assert!(hf > 0, "hashfull should be > 0 after storing entries");
    }

    #[test]
    fn test_generation_replacement() {
        let tt = SharedTT::new(1);
        let hash = 0x1234_5678_ABCD_EF01u64;

        // Store at generation 0
        tt.store(hash, 10, 100, TTFlag::Exact, Move(200), None);

        // New search bumps generation
        tt.new_search();

        // Store at generation 1 — should replace even at lower depth
        tt.store(hash, 3, 50, TTFlag::UpperBound, Move(300), None);

        let entry = tt.probe(hash).unwrap();
        assert_eq!(entry.depth, 3);
        assert_eq!(entry.score, 50);
        assert_eq!(entry.best_move, Move(300));
    }

    #[test]
    fn test_clear() {
        let tt = SharedTT::new(1);
        tt.store(42, 5, 100, TTFlag::Exact, Move(10), None);
        assert!(tt.probe(42).is_some());

        tt.clear();
        assert!(tt.probe(42).is_none());
    }
}
