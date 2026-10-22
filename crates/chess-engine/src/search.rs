use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use chess_common::moves::MoveFlag;
use chess_common::{Board, Color, Move, PieceKind, Score, Square};

use chess_nnue::{
    AccumulatorStack, MoveUpdate, NnueNetwork, PieceDelta, RefreshCache, nnue_evaluate,
};

use crate::eval::evaluate;
use crate::polyglot::OpeningBook;
use crate::see;
use crate::syzygy::{self, SyzygyTB};
use crate::tt::{SharedTT, TTFlag};
use crate::{InfoCallback, SearchInfo, SearchParams, SearchResult, TuneParams};

// ---------------------------------------------------------------------------
// LMR reduction table (built at search start from TuneParams)
// ---------------------------------------------------------------------------

fn build_lmr_table(base_x100: i32, div_x100: i32) -> [[u8; 64]; 64] {
    let base = base_x100 as f64 / 100.0;
    let div = div_x100 as f64 / 100.0;
    let mut table = [[0u8; 64]; 64];
    #[allow(clippy::needless_range_loop)]
    for depth in 1..64 {
        for moves in 1..64 {
            table[depth][moves] = (base + (depth as f64).ln() * (moves as f64).ln() / div) as u8;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Killer moves, history & search state
// ---------------------------------------------------------------------------

const MAX_PLY: usize = 128;
/// The TT move gets +2 plies when every alternative fails this far below the singular beta.
const DOUBLE_EXT_MARGIN: i32 = 25;
/// Cap on double extensions along one root-to-leaf path, so forcing lines cannot explode.
const MAX_DOUBLE_EXTS: u8 = 8;
const MAX_KILLERS: usize = 2;
const MATE_THRESHOLD: i32 = 29_000;

/// A search line cannot exceed the ply limit. Keep recursive scratch inline;
/// only completed root lines need owned, heap-backed output buffers.
struct PvLine {
    moves: [Move; MAX_PLY],
    len: usize,
}

impl PvLine {
    fn new() -> Self {
        Self {
            moves: [Move::NULL; MAX_PLY],
            len: 0,
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn push(&mut self, m: Move) {
        self.moves[self.len] = m;
        self.len += 1;
    }

    fn extend_from_slice(&mut self, moves: &[Move]) {
        let end = self.len + moves.len();
        self.moves[self.len..end].copy_from_slice(moves);
        self.len = end;
    }
}

impl std::ops::Deref for PvLine {
    type Target = [Move];
    fn deref(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

/// Iterations worth running once the root is confined to the tablebase's
/// win-preserving moves. Syzygy has already chosen the move — the DTZ order is
/// sticky and cp scores cannot displace it — so the search is only still here to
/// promote a proven shorter mate and to demote a move whose child repeats (the
/// 45geMoUz fix). Both act within a few plies; beyond that the search re-derives
/// a decision it is not allowed to change, at full price (KQvKR: 6.4s of a 60s
/// clock). Deep enough to keep those two mechanisms, cheap enough to be free.
const TB_RESTRICTED_MAX_DEPTH: u8 = 12;

/// Adjust a mate score from root-relative to position-relative before TT store.
/// Non-mate scores pass through unchanged.
#[inline]
fn score_to_tt(score: i32, ply: u8) -> i32 {
    if score > MATE_THRESHOLD {
        score + ply as i32
    } else if score < -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

/// Adjust a mate score from position-relative to root-relative after TT probe.
/// Non-mate scores pass through unchanged.
#[inline]
fn score_from_tt(score: i32, ply: u8) -> i32 {
    if score > MATE_THRESHOLD {
        score - ply as i32
    } else if score < -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

/// Continuation history: indexed by
/// [cur_color][piece_kind][to_sq][cur_piece_kind][cur_to_sq]. The earlier
/// move's colour follows from the lookback distance, so the mover's colour is
/// enough to keep White's and Black's contexts apart.
type ContHistory = Box<[[[[[i32; 64]; 6]; 64]; 6]; 2]>;

fn new_cont_history() -> ContHistory {
    Box::new([[[[[0i32; 64]; 6]; 64]; 6]; 2])
}

/// Quiet-move history: indexed by [color][from_sq][to_sq].
type QuietHistory = [[[i32; 64]; 64]; 2];

/// Per-ply move context for continuation history lookback.
#[derive(Clone, Copy)]
struct PlyContext {
    piece_kind: usize, // PieceKind index of the moved piece
    to_sq: usize,      // destination square index
}

const NULL_PLY_CONTEXT: PlyContext = PlyContext {
    piece_kind: 0,
    to_sq: 0,
};

/// Capture history: indexed by [moving_piece][to_sq][captured_piece].
type CaptureHistory = Box<[[[i32; 6]; 64]; 6]>;

fn new_capture_history() -> CaptureHistory {
    Box::new([[[0i32; 6]; 64]; 6])
}

/// Correction history: learns the systematic error of the static eval, keyed
/// by various facets of the position (pawn structure, non-pawn placement,
/// minor/major placement, and the previous move). Each facet has its own
/// table indexed by [stm][key]; per-facet values are stored in fixed-point
/// with scale `CORRHIST_GRAIN` and clamped to `CORRHIST_LIMIT` (±32cp). The
/// applied correction sums the facets and is capped at `CORR_TOTAL_CAP`.
const CORRHIST_SIZE: usize = 16384; // power of two
const CORR_MASK: usize = CORRHIST_SIZE - 1;
const CORRHIST_GRAIN: i32 = 256; // fixed-point scale
const CORRHIST_LIMIT: i32 = CORRHIST_GRAIN * 32; // per-facet cap: ±32cp
const CORR_TOTAL_CAP: i32 = 64; // cap on the summed correction (cp)

/// A hash-keyed correction table, indexed by [stm][key & CORR_MASK].
type CorrTable = Box<[[i32; CORRHIST_SIZE]; 2]>;
fn new_corr_table() -> CorrTable {
    Box::new([[0i32; CORRHIST_SIZE]; 2])
}

/// Continuation correction table, keyed by [stm][prev_piece_kind][prev_to].
type ContCorrTable = Box<[[[i32; 64]; 6]; 2]>;
fn new_cont_corr_table() -> ContCorrTable {
    Box::new([[[0i32; 64]; 6]; 2])
}

/// Game-level learning tables that persist across moves (cleared on
/// `ucinewgame`). They are borrowed by the per-search [`SearchState`] for the
/// duration of a search so history and corrections accumulate over the whole
/// game instead of resetting every move. Ply-local scratch (killers,
/// `ply_context`, `static_evals`, accumulators) stays per-search and is *not*
/// carried here.
pub struct PersistentHistory {
    history: QuietHistory,
    capture_history: CaptureHistory,
    counter_moves: [[Move; 64]; 64],
    cont_history: ContHistory,
    pawn_corrhist: CorrTable,
    white_corrhist: CorrTable,
    black_corrhist: CorrTable,
    minor_corrhist: CorrTable,
    major_corrhist: CorrTable,
    cont_corrhist: ContCorrTable,
}

impl PersistentHistory {
    pub fn new() -> Self {
        Self {
            history: [[[0; 64]; 64]; 2],
            capture_history: new_capture_history(),
            counter_moves: [[Move::NULL; 64]; 64],
            cont_history: new_cont_history(),
            pawn_corrhist: new_corr_table(),
            white_corrhist: new_corr_table(),
            black_corrhist: new_corr_table(),
            minor_corrhist: new_corr_table(),
            major_corrhist: new_corr_table(),
            cont_corrhist: new_cont_corr_table(),
        }
    }
}

impl Default for PersistentHistory {
    fn default() -> Self {
        Self::new()
    }
}

struct SearchState<'a> {
    learning: &'a mut PersistentHistory,
    tt: Arc<SharedTT>,
    killers: [[Move; MAX_KILLERS]; MAX_PLY],
    ply_context: [PlyContext; MAX_PLY],
    static_evals: [i32; MAX_PLY],
    /// Double singular extensions taken on the path from the root to each ply.
    double_exts: [u8; MAX_PLY + 1],
    accumulators: AccumulatorStack,
    /// Refresh cache for king moves; built for `net`.
    refresh_cache: RefreshCache,
    use_nnue: bool,
    net: Arc<NnueNetwork>,
    lmr_table: [[u8; 64]; 64],
    tune: TuneParams,
    nodes: u64,
    seldepth: u8,
    start_time: Instant,
    /// Hard ceiling: the search is aborted here, whatever the position.
    time_limit_ms: Option<u64>,
    /// What a typical position is worth; the soft limit is a multiple of this.
    soft_target_ms: Option<u64>,
    use_soft_limit: bool,
    inc_ms: u64,
    time_remaining_ms: u64,
    max_nodes: Option<u64>,
    game_ply: usize,
    contempt: i32,
    singular_ext_mode: u8,
    stop: bool,
    /// Syzygy tablebase handle for WDL probing (None if disabled or not loaded).
    syzygy_tb: Option<SyzygyTB>,
    /// `go ponder` flag shared with the UCI handler. While it reads true every
    /// time-based stop is suspended; `ponderhit`/`stop` clear it and normal
    /// time management takes over. Pondered work still counts toward the move's
    /// budget, but a successful hit gets a small amount of fresh thinking on
    /// our own clock before time-based stops become eligible.
    ponder: Option<Arc<AtomicBool>>,
    ponder_was_active: bool,
    ponder_hit_at: Option<Instant>,
    ponder_fresh_ms: u64,
}

impl<'a> SearchState<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        time_limit_ms: Option<u64>,
        soft_target_ms: Option<u64>,
        use_soft_limit: bool,
        inc_ms: u64,
        time_remaining_ms: u64,
        max_nodes: Option<u64>,
        game_ply: usize,
        contempt: i32,
        singular_ext_mode: u8,
        tt: &Arc<SharedTT>,
        use_nnue: bool,
        net: Arc<NnueNetwork>,
        syzygy_tb: Option<SyzygyTB>,
        tune: TuneParams,
        learning: &'a mut PersistentHistory,
    ) -> Self {
        let lmr_table = build_lmr_table(tune.lmr_base, tune.lmr_div);

        Self {
            learning,
            tt: Arc::clone(tt),
            killers: [[Move::NULL; MAX_KILLERS]; MAX_PLY],
            ply_context: [NULL_PLY_CONTEXT; MAX_PLY],
            static_evals: [0; MAX_PLY],
            double_exts: [0; MAX_PLY + 1],
            accumulators: AccumulatorStack::new(MAX_PLY),
            refresh_cache: RefreshCache::new(&net),
            use_nnue,
            net,
            lmr_table,
            tune,
            nodes: 0,
            seldepth: 0,
            start_time: Instant::now(),
            time_limit_ms,
            soft_target_ms,
            use_soft_limit,
            inc_ms,
            time_remaining_ms,
            max_nodes,
            game_ply,
            contempt,
            singular_ext_mode,
            stop: false,
            syzygy_tb,
            ponder: None,
            ponder_was_active: false,
            ponder_hit_at: None,
            ponder_fresh_ms: ponder_fresh_time_ms(soft_target_ms, time_remaining_ms, game_ply),
        }
    }

    fn set_ponder(&mut self, ponder: Option<Arc<AtomicBool>>) {
        self.ponder_was_active = ponder
            .as_ref()
            .is_some_and(|pending| pending.load(Ordering::Relaxed));
        self.ponder = ponder;
    }

    fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    fn pondering(&self) -> bool {
        self.ponder
            .as_ref()
            .is_some_and(|p| p.load(Ordering::Relaxed))
    }

    /// True while a successful ponder hit is owed a small amount of fresh
    /// thinking. Detecting the shared flag transition here keeps the UCI and
    /// search threads decoupled and timestamps the hit, not the ponder start.
    fn ponder_fresh_think_pending(&mut self) -> bool {
        if !self.ponder_was_active || self.ponder_fresh_ms == 0 {
            return false;
        }
        if self.pondering() {
            return true;
        }
        let hit_at = *self.ponder_hit_at.get_or_insert_with(Instant::now);
        (hit_at.elapsed().as_millis() as u64) < self.ponder_fresh_ms
    }

    fn should_stop(&mut self, stop_flag: &AtomicBool) -> bool {
        if self.stop || stop_flag.load(Ordering::Relaxed) {
            return true;
        }
        // Poll the ponder flag on every periodic stop check so the fresh-time
        // window starts close to `ponderhit`, not only after the old hard limit.
        let pondering = self.pondering();
        let ponder_fresh_pending = self.ponder_fresh_think_pending();
        if let Some(limit) = self.time_limit_ms
            && self.elapsed_ms() >= limit
            && !pondering
            && !ponder_fresh_pending
        {
            return true;
        }
        if let Some(max) = self.max_nodes
            && self.nodes >= max
        {
            return true;
        }
        false
    }

    fn store_killer(&mut self, ply: usize, m: Move) {
        if ply < MAX_PLY && self.killers[ply][0] != m {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = m;
        }
    }

    fn update_history(&mut self, m: Move, depth: u8, board: &Board, ply: u8) {
        let bonus = depth as i32 * depth as i32;
        let from = m.from_sq().index();
        let to = m.to_sq().index();

        // Regular from-to history (gravity formula)
        let max_val = 16384;
        let us = board.side_to_move.index();
        let entry = &mut self.learning.history[us][from][to];
        *entry += bonus - *entry * bonus.abs() / max_val;
        *entry = (*entry).clamp(-max_val, max_val);

        // Continuation history (1-ply and 2-ply lookback)
        if let Some(piece) = board.piece_at(m.from_sq()) {
            let cur_kind = piece.kind.index();
            let cur_to = to;
            // 1-ply lookback
            if ply >= 1 && (ply as usize - 1) < MAX_PLY {
                let ctx = self.ply_context[ply as usize - 1];
                let entry = &mut self.learning.cont_history[us][ctx.piece_kind][ctx.to_sq]
                    [cur_kind][cur_to];
                *entry += bonus - *entry * bonus.abs() / max_val;
                *entry = (*entry).clamp(-max_val, max_val);
            }
            // 2-ply lookback
            if ply >= 2 && (ply as usize - 2) < MAX_PLY {
                let ctx = self.ply_context[ply as usize - 2];
                let entry = &mut self.learning.cont_history[us][ctx.piece_kind][ctx.to_sq]
                    [cur_kind][cur_to];
                *entry += bonus - *entry * bonus.abs() / max_val;
                *entry = (*entry).clamp(-max_val, max_val);
            }
        }
    }

    fn update_history_malus(&mut self, m: Move, depth: u8, board: &Board, ply: u8) {
        let malus = depth as i32 * depth as i32;
        let from = m.from_sq().index();
        let to = m.to_sq().index();

        // Regular history malus (gravity formula)
        let max_val = 16384;
        let us = board.side_to_move.index();
        let entry = &mut self.learning.history[us][from][to];
        *entry += -malus - *entry * malus.abs() / max_val;
        *entry = (*entry).clamp(-max_val, max_val);

        // Continuation history malus (1-ply and 2-ply lookback)
        if let Some(piece) = board.piece_at(m.from_sq()) {
            let cur_kind = piece.kind.index();
            let cur_to = to;
            // 1-ply lookback
            if ply >= 1 && (ply as usize - 1) < MAX_PLY {
                let ctx = self.ply_context[ply as usize - 1];
                let entry = &mut self.learning.cont_history[us][ctx.piece_kind][ctx.to_sq]
                    [cur_kind][cur_to];
                *entry += -malus - *entry * malus.abs() / max_val;
                *entry = (*entry).clamp(-max_val, max_val);
            }
            // 2-ply lookback
            if ply >= 2 && (ply as usize - 2) < MAX_PLY {
                let ctx = self.ply_context[ply as usize - 2];
                let entry = &mut self.learning.cont_history[us][ctx.piece_kind][ctx.to_sq]
                    [cur_kind][cur_to];
                *entry += -malus - *entry * malus.abs() / max_val;
                *entry = (*entry).clamp(-max_val, max_val);
            }
        }
    }

    /// Learned static-eval correction for the current position, in centipawns
    /// (side-to-move perspective). Sums all corrhist facets, caps the total,
    /// then scales by the tunable `corrhist_mult` (×100).
    fn correction(&self, board: &Board, ply: u8) -> i32 {
        let stm = if board.side_to_move == Color::White {
            0
        } else {
            1
        };
        let k = crate::eval::corr_keys(board);
        let mut sum = self.learning.pawn_corrhist[stm][k.pawn as usize & CORR_MASK]
            + self.learning.white_corrhist[stm][k.white as usize & CORR_MASK]
            + self.learning.black_corrhist[stm][k.black as usize & CORR_MASK]
            + self.learning.minor_corrhist[stm][k.minor as usize & CORR_MASK]
            + self.learning.major_corrhist[stm][k.major as usize & CORR_MASK];
        if ply >= 1 && (ply as usize - 1) < MAX_PLY {
            let ctx = self.ply_context[ply as usize - 1];
            sum += self.learning.cont_corrhist[stm][ctx.piece_kind][ctx.to_sq];
        }
        let cp = (sum / CORRHIST_GRAIN).clamp(-CORR_TOTAL_CAP, CORR_TOTAL_CAP);
        cp * self.tune.corrhist_mult / 100
    }

    /// Update every corrhist facet from the residual `diff = search_score -
    /// static_eval`. Deeper searches are trusted more (larger weight).
    fn update_corrhist(&mut self, board: &Board, depth: u8, diff: i32, ply: u8) {
        let stm = if board.side_to_move == Color::White {
            0
        } else {
            1
        };
        let k = crate::eval::corr_keys(board);
        let scaled = (diff * CORRHIST_GRAIN).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
        let weight = (depth as i32 + 1).min(16);
        let blend = |e: &mut i32| {
            *e = (*e * (256 - weight) + scaled * weight) / 256;
            *e = (*e).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
        };
        blend(&mut self.learning.pawn_corrhist[stm][k.pawn as usize & CORR_MASK]);
        blend(&mut self.learning.white_corrhist[stm][k.white as usize & CORR_MASK]);
        blend(&mut self.learning.black_corrhist[stm][k.black as usize & CORR_MASK]);
        blend(&mut self.learning.minor_corrhist[stm][k.minor as usize & CORR_MASK]);
        blend(&mut self.learning.major_corrhist[stm][k.major as usize & CORR_MASK]);
        if ply >= 1 && (ply as usize - 1) < MAX_PLY {
            let ctx = self.ply_context[ply as usize - 1];
            blend(&mut self.learning.cont_corrhist[stm][ctx.piece_kind][ctx.to_sq]);
        }
    }

    fn store_counter_move(&mut self, prev_move: Move, counter: Move) {
        if !prev_move.is_null() {
            self.learning.counter_moves[prev_move.from_sq().index()][prev_move.to_sq().index()] =
                counter;
        }
    }

    fn get_counter_move(&self, prev_move: Move) -> Move {
        if prev_move.is_null() {
            Move::NULL
        } else {
            self.learning.counter_moves[prev_move.from_sq().index()][prev_move.to_sq().index()]
        }
    }

    fn get_cont_history_bonus(&self, m: Move, board: &Board, ply: u8) -> i32 {
        let piece = match board.piece_at(m.from_sq()) {
            Some(p) => p,
            None => return 0,
        };
        let us = piece.color.index();
        let cur_kind = piece.kind.index();
        let cur_to = m.to_sq().index();

        let mut bonus = 0i32;
        // 1-ply lookback
        if ply >= 1 && (ply as usize - 1) < MAX_PLY {
            let ctx = self.ply_context[ply as usize - 1];
            bonus += self.learning.cont_history[us][ctx.piece_kind][ctx.to_sq][cur_kind][cur_to];
        }
        // 2-ply lookback (follow-up history)
        if ply >= 2 && (ply as usize - 2) < MAX_PLY {
            let ctx = self.ply_context[ply as usize - 2];
            bonus +=
                self.learning.cont_history[us][ctx.piece_kind][ctx.to_sq][cur_kind][cur_to] / 2;
        }
        bonus
    }

    fn get_capture_history(&self, m: Move, board: &Board) -> i32 {
        let piece = match board.piece_at(m.from_sq()) {
            Some(p) => p,
            None => return 0,
        };
        let captured = match board.piece_at(m.to_sq()) {
            Some(p) => p,
            None => return 0,
        };
        self.learning.capture_history[piece.kind.index()][m.to_sq().index()][captured.kind.index()]
    }

    fn update_capture_history(&mut self, m: Move, depth: u8, board: &Board) {
        let piece = match board.piece_at(m.from_sq()) {
            Some(p) => p,
            None => return,
        };
        let captured = match board.piece_at(m.to_sq()) {
            Some(p) => p,
            None => return,
        };
        let bonus = depth as i32 * depth as i32;
        let max_val = 16384;
        let entry = &mut self.learning.capture_history[piece.kind.index()][m.to_sq().index()]
            [captured.kind.index()];
        *entry += bonus - *entry * bonus.abs() / max_val;
        *entry = (*entry).clamp(-max_val, max_val);
    }

    fn update_capture_history_malus(&mut self, m: Move, depth: u8, board: &Board) {
        let piece = match board.piece_at(m.from_sq()) {
            Some(p) => p,
            None => return,
        };
        let captured = match board.piece_at(m.to_sq()) {
            Some(p) => p,
            None => return,
        };
        let malus = depth as i32 * depth as i32;
        let max_val = 16384;
        let entry = &mut self.learning.capture_history[piece.kind.index()][m.to_sq().index()]
            [captured.kind.index()];
        *entry += -malus - *entry * malus.abs() / max_val;
        *entry = (*entry).clamp(-max_val, max_val);
    }

    fn store_ply_context(&mut self, ply: u8, m: Move, board: &Board) {
        if (ply as usize) < MAX_PLY
            && let Some(piece) = board.piece_at(m.from_sq())
        {
            self.ply_context[ply as usize] = PlyContext {
                piece_kind: piece.kind.index(),
                to_sq: m.to_sq().index(),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Time management
// ---------------------------------------------------------------------------

/// Per-move time cap (ms): never spend more than the clock can sustain.
///
/// The piece-count `moves_left` estimate in `compute_time_limit` spends *faster*
/// as pieces come off, so the raw target is roughly a tenth of the remaining
/// clock every move. That decays the clock geometrically toward `~10 * inc`,
/// however long the game actually runs: a 180+0 game flagged ~move 90 while
/// mating, and a 1800+5 game (60rqL48Y) reached move 57 with 1:00 left, having
/// burned 29 of its 30 minutes and played the endgame on the increment.
///
/// Bound the slice to what is refundable — the increment, which is free — plus
/// `time/N` of the bank (large `N`), so the clock decays slowly enough to
/// survive a long endgame conversion. With `inc == 0` this reduces to the
/// original sudden-death `time/N` cap.
///
/// The reserve is *not* free. `time/N` is a sudden-death constant — sized so the
/// clock survives ~90 moves with no refund — and as a general per-move budget it
/// forbids ever investing more than ~3% of the clock in a critical position. Below
/// a long clock that trade loses badly: an increment control refunds every move,
/// so the drain is slow and the game ends long before the reserve pays for itself.
///
/// Applying it everywhere was a mistake, caught in live play (2026-07-11): capped
/// at every clock, the bot finished a 180+5 game having used **1 second of its
/// 180-second bank** — peak spend pinned to the cap on every move, the rest of its
/// thinking donated to ponder hits. Blitz went 22s/move → 10s, rapid 55s → 21s.
/// A 144-game `180+2` A/B had called that "parity" (+4.8 ±27.6): the interval was
/// far too wide to resolve it, and fastchess does not ponder, so the gate was blind
/// to the interaction that makes it bite. Bullet said so outright: −28.3 ±22.1 Elo
/// over 320 games at `10+0.1`.
///
/// So the reserve is fully charged above ten minutes and removed below five,
/// with a linear transition in between. This avoids a spending discontinuity
/// as a long game drains through the old five-minute boundary. The reserve is
/// also phase-aware: early play protects a long future, while late endgames can
/// deliberately convert more of a healthy bank into search depth.
///
/// Sudden death keeps a reserve at every clock: there a flag is fatal, not
/// merely expensive.
///
/// Above a 5-minute clock the bank slice is `time/(2N/3)` (20/17/14/12 by phase),
/// with or without an increment. Sudden death keeps that slice down to 400 s and
/// only then blends to the flag-safe `time/34` at 300 s, so 3+0 and 5+0 never see
/// it; blending from ten minutes changed nothing in a 10+0 game, which leaves
/// ten minutes after a few moves (187 s and 248 s of 600 unused in QQHtFZaq and
/// 5bjcnIB6, 2026-09-29).
/// With an increment, at `time/N` rapid under-spent badly: after the search's own stability stop
/// and next-iteration prediction the bot averaged ~`time/43` per move — 16.6s
/// against the opponent's 32.6s over the first 18 moves of a 900+3 game
/// (ELg7ivU3), and 9:16 left at move 27 of evhAQeHC (2026-09-27).
fn long_game_time_cap(target_ms: u64, inc_ms: u64, time_ms: u64, game_ply: u64) -> u64 {
    let phase_divisor = match game_ply {
        0..40 => 30u64,
        40..80 => 26,
        80..120 => 22,
        _ => 18,
    };

    if inc_ms == 0 {
        // Keep the long-clock slice down to 400 s, then blend to the flag-safe
        // sudden-death value by 5 min (see fn docs).
        let long_divisor = phase_divisor * 2 / 3;
        let divisor = if time_ms >= 400_000 {
            long_divisor
        } else if time_ms <= 300_000 {
            34
        } else {
            let classical_weight = time_ms - 300_000;
            let reserve_weight = 400_000 - time_ms;
            (long_divisor * classical_weight + 34 * reserve_weight) / 100_000
        };
        return target_ms.min(time_ms / divisor);
    }

    if time_ms <= 300_000 {
        return target_ms;
    }

    // The increment refunds every move, so the bank may drain faster than in
    // sudden death (see fn docs).
    let sustainable = inc_ms.saturating_add(time_ms / (phase_divisor * 2 / 3));
    if time_ms >= 600_000 {
        return target_ms.min(sustainable);
    }

    // Between five and ten minutes, blend continuously from the capped
    // classical allocation to the tuned short-control allocation.
    let capped_weight = time_ms - 300_000;
    let uncapped_weight = 600_000 - time_ms;
    let blended = sustainable
        .saturating_mul(capped_weight)
        .saturating_add(target_ms.saturating_mul(uncapped_weight))
        / 300_000;
    target_ms.min(blended)
}

/// Minimum wall-clock thinking after a successful ponder hit. A deep ponder is
/// valuable, but immediately playing every hit leaves most of a classical bank
/// unused and gives the actual reply no verification on our clock.
fn ponder_fresh_time_ms(
    soft_target_ms: Option<u64>,
    time_remaining_ms: u64,
    game_ply: usize,
) -> u64 {
    let Some(soft_target_ms) = soft_target_ms else {
        return 0;
    };
    if time_remaining_ms <= 300_000 {
        return 0;
    }
    let divisor = match game_ply {
        0..40 => 4,
        40..80 => 3,
        _ => 2,
    };
    (soft_target_ms / divisor).clamp(1_000, 12_000)
}

/// Opening ramp for long clocks, in permille of the soft target: 40% at ply 0,
/// rising linearly to 100% by ply 20.
///
/// The `time/20` bank slice (see `long_game_time_cap`) applies from move 1, so the
/// Pi spent a median ~20 s per move on plies 3-8 at 10+x after it (vs ~10 s
/// before) — on known openings where `OpeningVariety` then picks within 15 cp
/// anyway — and entered the middlegame ~70 s poorer (2026-09-30). The saved time
/// stays in the bank. Clocks under 400 s (blitz, 5+x) and Chess960 keep the full
/// target; repeating controls are budgeted separately (`movestogo`).
fn opening_ramp_permille(game_ply: u64, time_ms: u64, chess960: bool) -> u64 {
    const RAMP_PLIES: u64 = 20;
    if chess960 || time_ms < 400_000 || game_ply >= RAMP_PLIES {
        return 1000;
    }
    400 + 600 * game_ply / RAMP_PLIES
}

/// Early-opening soft-limit discount in permille points (negative = stop sooner).
/// Chess960 returns 0: there is no book, and the first moves need the long think.
fn opening_time_adjust(game_ply: usize, time_remaining_ms: u64, chess960: bool) -> i64 {
    if chess960 {
        return 0;
    }
    if game_ply < 6 {
        if time_remaining_ms <= 300_000 {
            -240
        } else {
            -160
        }
    } else if game_ply < 12 {
        if time_remaining_ms <= 300_000 {
            -140
        } else {
            -90
        }
    } else if game_ply < 20 {
        if time_remaining_ms <= 300_000 {
            -60
        } else {
            -30
        }
    } else {
        0
    }
}

/// Retain only recent score volatility for time management. A large shallow
/// swing must not suppress the clearly-decided-position discount forever once
/// several deeper iterations have converged.
#[inline]
fn update_recent_score_swing(recent: i32, current: i32) -> i32 {
    (recent / 2).max(current.saturating_abs())
}

/// Track a short-lived count of recent best-move changes. Stable completed
/// iterations pay down old instability instead of carrying it for the entire
/// search.
#[inline]
fn update_recent_pv_changes(recent: u32, changed: bool) -> u32 {
    if changed {
        recent.saturating_add(1).min(5)
    } else {
        recent.saturating_sub(1)
    }
}

/// How much longer than the soft target a difficult position may run. The soft
/// limit itself can reach 1.8x (see `soft_frac`), so this only has to leave room
/// for that plus the fail-low extension; it is not a spending target.
const HARD_LIMIT_SOFT_MULT: u64 = 3;

/// Returns (soft_target_ms, hard_limit_ms, use_soft_limit, inc_ms, time_remaining_ms).
///
/// Two budgets, not one. `soft_target` is what a *typical* position is worth;
/// `hard_limit` is the ceiling the search may never cross. The iterative-deepening
/// loop scores how hard the position actually is — PV instability, a dropping score,
/// aspiration failures, how concentrated the nodes are on the best move — and stops
/// anywhere between ~0.35x and ~1.8x of `soft_target`, bounded by `hard_limit`.
///
/// They used to be the same number, which quietly disabled half the machinery: the
/// difficulty *bonuses* could never fire, because the hard stop killed the search at
/// 1.0x before any of them could buy a millisecond. The engine could play an obvious
/// position fast but was structurally unable to think longer about a hard one.
///
/// `use_soft_limit` is true for clock-based time controls (wtime/btime) where we must
/// save time for future moves, false for fixed movetime.
///
/// Supports typical tournament time controls:
///   - FIDE/DSB Classical: 40 moves / 90 min + 30s inc, then 15-30 min + 30s
///   - FIDE/DSB Rapid:     15 min + 10s inc
///   - FIDE/DSB Blitz:     3-5 min + 2-3s inc
///   - Bullet:             1 min + 0-1s inc
fn compute_time_limit(
    params: &SearchParams,
    board: &Board,
) -> (Option<u64>, Option<u64>, bool, u64, u64) {
    if params.infinite {
        return (None, None, false, 0, 0);
    }

    let overhead = params.move_overhead_ms;

    if let Some(mt) = params.move_time_ms {
        // Fixed movetime: the caller dictated the budget, so soft == hard.
        let mt = mt.saturating_sub(overhead);
        return (Some(mt), Some(mt), false, 0, 0);
    }

    let side = board.side_to_move;
    let (time_ms, inc_ms) = match side {
        Color::White => (params.white_time_ms, params.white_inc_ms),
        Color::Black => (params.black_time_ms, params.black_inc_ms),
    };

    if let Some(time) = time_ms {
        let inc = inc_ms.unwrap_or(0);
        let game_ply = board.position_history.len() as u64;

        // Sudden death: estimate the moves remaining in the game. A repeating
        // control (`movestogo`) overrides the whole allocation below.
        let mut moves_left: u64 = {
            // Game phase-aware estimate based on piece count
            let total_pieces = (board.occupancy[0] | board.occupancy[1]).count();
            let base_moves = match total_pieces {
                28.. => 32,    // opening
                22..=27 => 26, // middlegame
                16..=21 => 20, // late middlegame
                _ => 16,       // endgame
            };
            // In fast time controls, use time more aggressively since
            // games tend to end sooner and deep calculation matters more
            // per move. Scale down moves_left for blitz/bullet.
            if time <= 15_000 {
                // Ultra-blitz / bullet: still reduce moves_left, but keep a
                // safer reserve to avoid practical flags.
                (base_moves * 65 / 100).max(1)
            } else if time < 120_000 {
                // Bullet (< 2 min): moderately aggressive
                (base_moves * 3 / 4).max(1)
            } else if time < 300_000 {
                // Blitz (< 5 min): moderately aggressive
                (base_moves * 3 / 4).max(1)
            } else {
                base_moves
            }
        };

        // In increment games, we can safely plan for fewer "must-save" moves,
        // so spend more time per move in the middlegame.
        let moves_left_scale_permille = if inc >= 5_000 {
            650
        } else if inc >= 2_000 {
            740
        } else if inc >= 1_000 {
            840
        } else if inc > 0 {
            930
        } else {
            1000
        };
        moves_left = (moves_left * moves_left_scale_permille / 1000).max(1);

        // Base time per move
        let base = time / moves_left;

        // Add an increment share. Keep it conservative in severe time trouble,
        // but in normal increment controls we can spend close to the increment.
        let inc_share_permille = if time <= 10_000 {
            500
        } else if time <= 30_000 {
            650
        } else if time <= 120_000 {
            800
        } else {
            900
        };
        let mut inc_term = inc * inc_share_permille / 1000;

        // If increment is large relative to the base slice, safely use a bit
        // more of it (up to +20%). This helps typical increment games avoid
        // under-spending while preserving low-time safety.
        if base > 0 && inc > 0 {
            let inc_ratio_permille = (inc.saturating_mul(1000) / base).min(1000);
            inc_term += inc * inc_ratio_permille / 5000;
        }
        // Use a stronger share of excess bank and steer toward a phase-aware
        // target reserve. This reduces ending the game with large unused time.
        let target_bank_ms = if inc > 0 {
            if game_ply < 20 {
                8_000u64.saturating_add(inc.saturating_mul(16))
            } else if game_ply < 40 {
                6_000u64.saturating_add(inc.saturating_mul(12))
            } else if game_ply < 60 {
                5_000u64.saturating_add(inc.saturating_mul(9))
            } else {
                4_000u64.saturating_add(inc.saturating_mul(7))
            }
        } else if game_ply < 40 {
            5_000
        } else {
            3_000
        };
        let usable_bank = time.saturating_sub(target_bank_ms);
        let burn_horizon: u64 = if game_ply < 20 {
            18
        } else if game_ply < 40 {
            14
        } else {
            10
        };
        let bank_term_raw = usable_bank / burn_horizon.saturating_mul(3).max(1);
        // Opening throttle: avoid spending too much bank time in the first
        // few moves when there is no opening book. Chess960 skips this — there
        // is no book, and those first moves need the long think.
        let opening_bank_scale_permille = if params.chess960 {
            1000u64
        } else if game_ply < 6 {
            if time <= 15_000 { 600u64 } else { 350u64 }
        } else if game_ply < 12 {
            if time <= 15_000 {
                700u64
            } else if time <= 300_000 {
                550u64
            } else {
                600u64
            }
        } else if game_ply < 20 {
            if time <= 300_000 { 800u64 } else { 900u64 }
        } else {
            1000u64
        };
        let bank_term_raw = bank_term_raw * opening_bank_scale_permille / 1000;
        let bank_term_cap = base.saturating_mul(3) / 2 + inc;
        let bank_term = bank_term_raw.min(bank_term_cap);

        let target = base + inc_term + bank_term;

        // Apply slow mover scaling factor
        let target = target * params.slow_mover / 100;

        // Long-game safety: keep enough clock in reserve for a long game (incl.
        // long endgame conversions) on every clock control. See fn docs.
        let target = long_game_time_cap(target, inc, time, game_ply);

        // Spend the long-clock slice on the middlegame, not on known openings.
        let target = target * opening_ramp_permille(game_ply, time, params.chess960) / 1000;

        // Repeating control ("40 moves in 15 min"): the clock is refilled after
        // `movestogo` moves, so the sudden-death reserves above (bank term, long-game
        // cap) do not apply — they left 27-39% of every 40/15 period unused. Spread
        // the whole clock over the moves left with one move in reserve; the search
        // stops at ~0.65-1.0x of the soft target, and `max`/`hard` below still bound
        // every single move.
        let (target, base) = match params.moves_to_go {
            Some(mtg) => {
                let per_move = time / ((mtg as u64).clamp(1, 50) + 1);
                (
                    (per_move + inc * 9 / 10) * params.slow_mover / 100,
                    per_move,
                )
            }
            None => (target, base),
        };

        // Hard maximum with stronger low-time safety to reduce flags.
        let mut max = if time <= 30_000 {
            time / 8 + inc * 2
        } else if time < 120_000 {
            time / 6 + inc
        } else {
            time / 4 + inc / 2
        };

        // In healthy increment controls, allow a little more burst usage.
        if base > 0 && inc.saturating_mul(2) >= base {
            max = max.saturating_add(inc / 2);
        }

        // Additional early-opening cap to prevent very long thinks in the
        // first moves without a book. Chess960 skips this (see bank scale).
        if !params.chess960 {
            if game_ply < 6 {
                let frac = if time <= 300_000 { 16u64 } else { 30u64 };
                max = max.min(time / frac + inc.saturating_mul(2));
            } else if game_ply < 12 {
                let frac = if time <= 300_000 { 13u64 } else { 22u64 };
                max = max.min(time / frac + inc.saturating_mul(2));
            }
        }

        // Minimum think time: always use at least the increment when
        // there's meaningful clock time remaining. This guarantees the
        // engine never plays faster than its increment allows.
        let min = if time > 1000 {
            (time / 60).max(inc.saturating_sub(overhead)).max(50)
        } else {
            // Ultra-low time (bullet flag): use increment if available,
            // otherwise a tiny amount to avoid instant moves.
            if inc > overhead {
                inc.saturating_sub(overhead)
            } else {
                10
            }
        };

        let soft = target.min(max).max(min);

        // Hard ceiling: the most a genuinely difficult position may take. Bounded
        // by `max` (the flag-safety limit) and by a multiple of the soft target, so
        // a single position can never eat an unbounded slice of the clock.
        //
        // Leave headroom below `max` for the fail-low extension further down, which
        // may overrun the hard limit by 1/5 — 5/6 * 6/5 == 1, so even a search that
        // takes the extension stays inside `max`.
        //
        // Then bound it by the clock itself. `max` is NOT a safe spend: its low-time
        // branch is `time/8 + inc*2`, and with a fat increment that `inc*2` alone can
        // exceed the clock (5s inc, 6s left => `max` ~10.7s). That never mattered
        // while `max` was only a ceiling on an allocation that stayed far below it —
        // but a hard limit is a number the search will actually spend, and spending
        // it flags. Measured: 3 losses on time in 21 blitz games before this bound.
        let clock_cap = time.saturating_sub(overhead).saturating_mul(4) / 5;
        let hard = soft
            .saturating_mul(HARD_LIMIT_SOFT_MULT)
            .min(max.saturating_mul(5) / 6)
            .min(clock_cap)
            .max(soft.min(clock_cap));

        (
            Some(soft.saturating_sub(overhead)),
            Some(hard.saturating_sub(overhead)),
            true,
            inc,
            time,
        )
    } else {
        (None, None, false, 0, 0)
    }
}

/// Soft move-time budget (ms) the engine would allocate for `params` at `board`,
/// or `None` for an infinite/untimed search. Used by the UCI layer to convert
/// a pondering (infinite) search into a timed one when `ponderhit` arrives.
///
/// This is the *soft target*, not the hard ceiling: a pondered move gets the budget
/// a typical position is worth. The difficulty-aware widening only applies to
/// searches that run under the soft limit, which a ponder search does not.
pub fn allocated_move_time_ms(params: &SearchParams, board: &Board) -> Option<u64> {
    compute_time_limit(params, board).0
}

// ---------------------------------------------------------------------------
// Move ordering helpers
// ---------------------------------------------------------------------------

/// Pick the highest-scored move from `start..len`, swap it to position `start`.
#[inline]
fn pick_next_move(moves: &mut [Move], scores: &mut [i32], start: usize) {
    let mut best_idx = start;
    let mut best_score = scores[start];
    for (i, &score) in scores.iter().enumerate().take(moves.len()).skip(start + 1) {
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    if best_idx != start {
        moves.swap(start, best_idx);
        scores.swap(start, best_idx);
    }
}

// ---------------------------------------------------------------------------
// Staged move picker — generates moves lazily in priority order
// ---------------------------------------------------------------------------
//
// Stages (main search):
//   1. TT move
//   2. Good captures  (SEE ≥ 0, sorted by MVV-LVA)
//   3. Quiet moves    (killers / counter / history-sorted)
//   4. Bad captures   (SEE < 0)
//
// This avoids generating quiet moves entirely when a beta cutoff happens
// on a capture, and eliminates the board-clone legality check from movegen.
// Legality is verified inline in the search after make_move.

const MAX_PICKER_MOVES: usize = 256;
const MAX_BAD_CAPTURES: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum MpStage {
    TtMove,
    InitCaptures,
    GoodCaptures,
    InitQuiets,
    Quiets,
    BadCaptures,
    Done,
}

struct MovePicker {
    stage: MpStage,
    tt_move: Move,

    moves: [Move; MAX_PICKER_MOVES],
    scores: [i32; MAX_PICKER_MOVES],
    count: usize,
    idx: usize,

    bad_captures: [Move; MAX_BAD_CAPTURES],
    num_bad: usize,
    bad_idx: usize,
}

impl MovePicker {
    fn new(tt_move: Move) -> Self {
        Self {
            stage: if !tt_move.is_null() {
                MpStage::TtMove
            } else {
                MpStage::InitCaptures
            },
            tt_move,
            moves: [Move::NULL; MAX_PICKER_MOVES],
            scores: [0; MAX_PICKER_MOVES],
            count: 0,
            idx: 0,
            bad_captures: [Move::NULL; MAX_BAD_CAPTURES],
            num_bad: 0,
            bad_idx: 0,
        }
    }

    /// Return the next pseudo-legal move in priority order.
    /// The caller must check legality after `make_move`.
    fn next(
        &mut self,
        board: &Board,
        state: &SearchState,
        ply: u8,
        killers: &[Move; MAX_KILLERS],
        counter_move: Move,
    ) -> Option<Move> {
        loop {
            match self.stage {
                // ── 1. Hash move ────────────────────────────────────
                MpStage::TtMove => {
                    self.stage = MpStage::InitCaptures;
                    if chess_core::is_pseudo_legal(board, self.tt_move) {
                        return Some(self.tt_move);
                    }
                }
                // ── 2a. Generate captures ───────────────────────────
                MpStage::InitCaptures => {
                    let captures = chess_core::generate_pseudo_legal_captures(board);
                    self.count = 0;
                    for &m in captures.iter() {
                        if m == self.tt_move {
                            continue;
                        }
                        let mvv = mvv_lva_score(board, m);
                        let cap_hist = state.get_capture_history(m, board);
                        let score = mvv * 16 + cap_hist / 64;
                        self.moves[self.count] = m;
                        self.scores[self.count] = score;
                        self.count += 1;
                    }
                    self.idx = 0;
                    self.stage = MpStage::GoodCaptures;
                }
                // ── 2b. Yield good captures (SEE ≥ 0) ──────────────
                MpStage::GoodCaptures => {
                    while self.idx < self.count {
                        pick_next_move(
                            &mut self.moves[..self.count],
                            &mut self.scores[..self.count],
                            self.idx,
                        );
                        let m = self.moves[self.idx];
                        self.idx += 1;
                        if see::see_ge(board, m, 0) {
                            return Some(m);
                        } else if self.num_bad < MAX_BAD_CAPTURES {
                            self.bad_captures[self.num_bad] = m;
                            self.num_bad += 1;
                        }
                    }
                    self.stage = MpStage::InitQuiets;
                }
                // ── 3a. Generate quiet moves ────────────────────────
                MpStage::InitQuiets => {
                    let quiets = chess_core::generate_pseudo_legal_quiets(board);
                    self.count = 0;
                    for &m in quiets.iter() {
                        if m == self.tt_move {
                            continue;
                        }
                        let hist = state.learning.history[board.side_to_move.index()]
                            [m.from_sq().index()][m.to_sq().index()];
                        let cont = state.get_cont_history_bonus(m, board, ply);
                        let score = if m == killers[0] {
                            900_000
                        } else if m == killers[1] {
                            800_000
                        } else if m == counter_move && !counter_move.is_null() {
                            700_000
                        } else {
                            hist + cont / 2
                        };
                        self.moves[self.count] = m;
                        self.scores[self.count] = score;
                        self.count += 1;
                    }
                    self.idx = 0;
                    self.stage = MpStage::Quiets;
                }
                // ── 3b. Yield quiet moves sorted by score ───────────
                MpStage::Quiets => {
                    if self.idx < self.count {
                        pick_next_move(
                            &mut self.moves[..self.count],
                            &mut self.scores[..self.count],
                            self.idx,
                        );
                        let m = self.moves[self.idx];
                        self.idx += 1;
                        return Some(m);
                    }
                    self.stage = MpStage::BadCaptures;
                    self.bad_idx = 0;
                }
                // ── 4. Bad captures (SEE < 0) ───────────────────────
                MpStage::BadCaptures => {
                    if self.bad_idx < self.num_bad {
                        let m = self.bad_captures[self.bad_idx];
                        self.bad_idx += 1;
                        return Some(m);
                    }
                    self.stage = MpStage::Done;
                }
                MpStage::Done => return None,
            }
        }
    }
}

/// MVV-LVA score for capture ordering (higher = better).
#[inline]
fn mvv_lva_score(board: &Board, m: Move) -> i32 {
    let flag = m.flag();
    let victim = if flag == MoveFlag::EnPassant {
        PieceKind::Pawn.value()
    } else {
        board
            .piece_at(m.to_sq())
            .map(|p| p.kind.value())
            .unwrap_or(0)
    };
    let attacker = board
        .piece_at(m.from_sq())
        .map(|p| p.kind.value())
        .unwrap_or(0);
    let promo = if flag.is_promotion() { 800 } else { 0 };
    victim * 10 - attacker + promo
}

/// After `board.make_move(m)`, returns true when the side that just moved
/// left its own king in check (i.e. the move was illegal).
#[inline]
fn move_is_illegal(board: &Board, mover: Color) -> bool {
    let king_sq = board.king_square(mover);
    chess_core::attacks::is_square_attacked(board, king_sq, board.side_to_move)
}

// ---------------------------------------------------------------------------
// Opening book
// ---------------------------------------------------------------------------

fn probe_opening_book(board: &Board, book: &OpeningBook) -> Option<Move> {
    book.pick_move(board)
}

// ---------------------------------------------------------------------------
// Iterative deepening
// ---------------------------------------------------------------------------

/// Depth offsets for Lazy SMP helper threads (Stockfish-style).
/// Thread 0 gets no offset. Helpers alternate between +/-1, +/-2, etc.
/// This creates depth diversity so different threads explore different depths.
const DEPTH_OFFSETS: [i8; 16] = [0, 1, -1, 2, -2, 1, -1, 3, -3, 2, -2, 1, -1, 4, -4, 3];

fn depth_offset(thread_id: usize) -> i8 {
    if thread_id == 0 {
        0
    } else {
        DEPTH_OFFSETS[thread_id % DEPTH_OFFSETS.len()]
    }
}

#[allow(clippy::too_many_arguments)]
pub fn iterative_deepening(
    board: &Board,
    params: &SearchParams,
    stop: &Arc<AtomicBool>,
    tt: &Arc<SharedTT>,
    info_callback: Option<InfoCallback>,
    thread_id: usize,
    net: &Arc<NnueNetwork>,
    node_counter: Option<&AtomicU64>,
    syzygy_tb: Option<SyzygyTB>,
    root_tb_ranking: Option<syzygy::RootTbRanking>,
    external_book: Option<Arc<OpeningBook>>,
    persistent: Option<&mut PersistentHistory>,
) -> SearchResult {
    // Try opening book first (only in the opening — limit to 30 half-moves
    // to avoid polyglot hash collisions returning garbage moves in endgames)
    let game_ply = board.position_history.len();
    if let Some(book) = external_book.as_deref()
        && params.search_moves.is_empty()
        && game_ply <= 30
        && let Some(book_move) = probe_opening_book(board, book)
    {
        // Report a stable static eval of the current position (before the
        // book move).  Evaluating *after* the move can be misleading for
        // tactical captures.  For book UI reporting, clamp aggressively so
        // GUIs never display absurd outlier values.
        let raw_book_eval = evaluate(board)
            .0
            .clamp(-MATE_THRESHOLD + 1, MATE_THRESHOLD - 1);
        let stm_eval = match board.side_to_move {
            Color::White => raw_book_eval,
            Color::Black => raw_book_eval.saturating_neg(),
        };
        let score = Score(stm_eval.clamp(-40, 40));
        if let Some(ref cb) = info_callback {
            cb(&SearchInfo {
                depth: 1,
                seldepth: 1,
                score,
                nodes: 1,
                time_ms: 0,
                pv: vec![book_move],
                hashfull: 0,
                multipv_line: 1,
            });
        }
        return SearchResult {
            best_move: book_move,
            score,
            depth: 1,
            nodes: 1,
            pv: vec![book_move],
        };
    }

    let (soft_target, hard_limit, use_soft_limit, inc, time_remaining) =
        compute_time_limit(params, board);
    let mut local_history;
    let learning = match persistent {
        Some(history) => history,
        None => {
            local_history = PersistentHistory::new();
            &mut local_history
        }
    };
    let mut state = SearchState::new(
        hard_limit,
        soft_target,
        use_soft_limit,
        inc,
        time_remaining,
        params.max_nodes,
        game_ply,
        if root_tb_ranking
            .as_ref()
            .is_some_and(syzygy::RootTbRanking::is_theoretical_draw)
        {
            0
        } else {
            params.contempt
        },
        params.singular_ext_mode,
        tt,
        params.use_nnue,
        Arc::clone(net),
        syzygy_tb,
        params.tune.clone(),
        learning,
    );
    state.set_ponder(params.ponder.clone());

    // Initialize root accumulator for NNUE
    if state.use_nnue {
        state.accumulators.refresh(0, board, &state.net);
    }
    let offset = depth_offset(thread_id);

    // Book-exit time boost: allocate extra time for the first few moves
    // of original play after exiting the opening book.
    let book_exit_bonus: u64 = if external_book.is_some() && (8..=33).contains(&game_ply) {
        if game_ply <= 20 {
            200
        } else {
            200 * (33 - game_ply) as u64 / 13
        }
    } else {
        0
    };

    let mut best_move = Move::NULL;
    let mut best_score = Score(0);
    let mut best_pv = Vec::new();
    let mut best_depth = 0u8;

    let mut root_moves: Vec<Move> = chess_core::generate_legal_moves(board)
        .iter()
        .copied()
        .collect();
    // `go searchmoves`: keep only the listed moves; if none of them is legal,
    // search everything rather than return no move.
    if root_moves.iter().any(|m| params.search_moves.contains(m)) {
        root_moves.retain(|m| params.search_moves.contains(m));
    }
    if root_moves.is_empty() {
        let score = if chess_core::is_in_check(board) {
            Score::NEG_MATE
        } else {
            Score::DRAW
        };
        return SearchResult {
            best_move: Move::NULL,
            score,
            depth: 0,
            nodes: 0,
            pv: Vec::new(),
        };
    }

    if root_moves.len() == 1 && params.search_moves.is_empty() && !params.infinite {
        // Only one legal move — do a quick search after making it to get
        // an accurate score (important for mate detection and analysis).
        // Not for `searchmoves` of one move: that asks for a full analysis.
        let only_move = root_moves[0];
        let mut board_copy = board.clone();
        let prev_castling = board_copy.castling;
        let prev_ep = board_copy.en_passant;
        let prev_halfmove = board_copy.halfmove_clock;
        let captured = board_copy.make_move(only_move);
        // Quick fixed-depth search to evaluate the resulting position
        let quick_depth = 10u8;
        let mut quick_history = PersistentHistory::new();
        let mut quick_state = SearchState::new(
            Some(2000),
            Some(2000),
            false,
            0,
            0,
            None,
            game_ply,
            params.contempt,
            params.singular_ext_mode,
            tt,
            params.use_nnue,
            Arc::clone(net),
            state.syzygy_tb.clone(),
            params.tune.clone(),
            &mut quick_history,
        );
        if quick_state.use_nnue {
            quick_state.accumulators.refresh(0, board, &quick_state.net);
            update_accumulator_for_move(
                &mut quick_state,
                &board_copy,
                only_move,
                captured,
                prev_castling,
                prev_ep,
                0,
                1,
            );
        }
        let mut quick_pv = PvLine::new();
        let child_score = -alpha_beta(
            &mut board_copy,
            quick_depth,
            1,
            Score::NEG_INF.0,
            Score::INF.0,
            &mut quick_pv,
            &mut quick_state,
            stop,
            only_move,
            Move::NULL,
        );
        board_copy.unmake_move(only_move, captured, prev_castling, prev_ep, prev_halfmove);
        let score = Score(child_score);
        let mut pv = vec![only_move];
        pv.extend_from_slice(&quick_pv);
        let clean_pv = sanitize_pv(board, &pv);
        if let Some(ref cb) = info_callback {
            cb(&SearchInfo {
                depth: quick_depth,
                seldepth: quick_state.seldepth,
                score,
                nodes: quick_state.nodes,
                time_ms: quick_state.elapsed_ms(),
                pv: clean_pv.clone(),
                hashfull: tt.hashfull(),
                multipv_line: 1,
            });
        }
        return SearchResult {
            best_move: only_move,
            score,
            depth: quick_depth,
            nodes: quick_state.nodes,
            pv: clean_pv,
        };
    }

    let max_depth = params.max_depth.min(MAX_PLY as u8);
    let mut prev_score = 0i32;

    // Root move ordering: track scores from previous iteration
    let mut root_move_scores: Vec<(Move, i32)> = root_moves.iter().map(|&m| (m, 0i32)).collect();

    // Pre-order root moves using TT so the 0-nodes fallback (stop before any
    // evaluation) picks the TT move rather than an arbitrary first generated move.
    {
        let tt_best = tt
            .probe(board.hash)
            .map(|e| e.best_move)
            .unwrap_or(Move::NULL);
        if !tt_best.is_null()
            && let Some(pos) = root_move_scores.iter().position(|(m, _)| *m == tt_best)
            && pos > 0
        {
            root_move_scores.swap(0, pos);
        }
    }

    // Tablebase root restriction. In a TB-won position, confine the search to
    // the win-preserving moves, ordered by the tablebase ranking (fastest DTZ
    // conversion first). In a TB-drawn position, confine it to the
    // draw-preserving moves so a losing king step cannot sneak in (lichess
    // RBmggG2n). This is the one thing the search cannot do on its own:
    // interior WDL probes fire only at halfmove_clock == 0, so inside a
    // conversion the search sees no tablebase gradient at all and would drift
    // toward the 50-move or repetition draw. Win restrictions keep DTZ order
    // sticky end to end; draw restrictions only filter the list and let
    // search pick among the holders. Single-PV play only; multi-PV keeps the
    // full list.
    let mut tb_win_restricted = false;
    if params.multi_pv.max(1) == 1
        && let Some(ranking) = root_tb_ranking.as_ref()
        && !ranking.restriction_moves().is_empty()
    {
        // Syzygy is blind to game history: a statically win-preserving move
        // can still complete a claimable threefold on the board (lichess
        // 45geMoUz: 82.Kg4 in TB-won K+N+2P vs K+B reached the position a
        // third time). Grade each winning move by how often its child
        // position already occurred in the game: twice-seen children are
        // dropped outright, and fresh children keep their DTZ order ahead of
        // once-seen ones, so the repetition budget can never be spent faster
        // than conversion progress is made.
        let prior_occurrences = |m: Move| -> usize {
            let mut child = board.clone();
            child.make_move(m);
            let target = child.hash;
            child
                .position_history
                .iter()
                .filter(|&&h| h == target)
                .count()
        };
        let mut restricted: Vec<(Move, i32)> = Vec::new();
        let mut seen_once: Vec<(Move, i32)> = Vec::new();
        for &m in ranking.restriction_moves() {
            if !root_move_scores.iter().any(|(rm, _)| *rm == m) {
                continue;
            }
            match prior_occurrences(m) {
                0 => restricted.push((m, 0i32)),
                1 => seen_once.push((m, 0i32)),
                _ => {}
            }
        }
        restricted.extend(seen_once);
        if !restricted.is_empty() {
            root_move_scores = restricted;
            tb_win_restricted = !ranking.winning_moves.is_empty();
        }
    }

    // Set ENDSPIEL_TM_DEBUG=1 to dump the time manager's per-iteration reasoning
    // (which difficulty terms fired, and the soft/hard budgets). Probing a position
    // by bare FEN gives `game_ply == 0`, which silently trips the early-opening cap
    // and shrinks `max` — always probe with the real move history, or the numbers lie.
    let tm_debug = std::env::var("ENDSPIEL_TM_DEBUG").is_ok();

    // Recent PV instability: old shallow best-move changes decay after stable depths.
    let mut recent_pv_changes = 0u32;
    // Best move stability: consecutive iterations with same best move
    let mut best_move_stability = 0u32;
    // Recent score volatility: the peak decays as deeper iterations converge.
    let mut recent_score_swing = 0i32;
    // Track last iteration duration for next-iteration prediction
    let mut last_iter_ms: u64;

    for base_depth in 1..=max_depth {
        // Apply depth offset for helper threads (Lazy SMP depth diversity)
        let depth = (base_depth as i16 + offset as i16).clamp(1, max_depth as i16) as u8;
        if depth == 0 {
            continue;
        }

        // Sort root moves by previous iteration score (best first).
        // In TB-restricted mode every listed move is a tablebase win, so cp
        // differences between them are search noise that must not displace
        // the DTZ/repetition-freshness order (a shuffle move that "scores"
        // 20cp above the converting move burns the repetition budget —
        // lichess 45geMoUz). Sort into buckets only: proven mates first
        // (fine-grained, shorter mate wins), then still-winning scores in
        // their tablebase order (stable), then draw-or-worse (a repetition
        // or loss the search actually found) last.
        if base_depth > 1 {
            if tb_win_restricted {
                root_move_scores.sort_by_key(|&(_, s)| {
                    let bucket = if Score(s).is_mate() && s > 0 {
                        2
                    } else if s > 0 {
                        1
                    } else {
                        0
                    };
                    std::cmp::Reverse((bucket, if bucket == 2 { s } else { 0 }))
                });
            } else {
                root_move_scores.sort_by_key(|b| std::cmp::Reverse(b.1));
            }
        }
        // If the best move has been highly stable, ensure it is searched first
        // at root regardless of TT or score ordering. In PVS, the first move
        // gets a full window while others get null windows — this ensures a
        // long-stable move isn't displaced by SMP TT noise. Not in
        // TB-restricted mode: there the master order IS the choice and must
        // stay in tablebase order.
        if !tb_win_restricted && best_move_stability >= 4 && !best_move.is_null() {
            for i in 1..root_move_scores.len() {
                if root_move_scores[i].0 == best_move {
                    root_move_scores.swap(0, i);
                    break;
                }
            }
        }

        let iter_start_ms = state.elapsed_ms();
        let mut best_node_fraction = 1.0f32;
        let multi_pv_count = params.multi_pv.max(1).min(root_move_scores.len());
        // Per-line results: (score, sanitized_pv, seldepth)
        let mut line_results: Vec<(i32, Vec<Move>, u8)> = Vec::with_capacity(multi_pv_count);
        let mut excluded_moves: Vec<Move> = Vec::with_capacity(multi_pv_count);
        let mut aspiration_failures = 0u32;
        let mut depth_completed = true;

        for line_idx in 0..multi_pv_count {
            // Build working_scores: excluded moves are pushed to the back with i32::MIN
            let mut working_scores: Vec<(Move, i32)> = root_move_scores
                .iter()
                .map(|&(m, s)| {
                    if excluded_moves.contains(&m) {
                        (m, i32::MIN)
                    } else {
                        (m, s)
                    }
                })
                .collect();
            // In TB-restricted mode the master list is already in the bucketed
            // tablebase order; a cp re-sort here would hand the full-window
            // slot to whichever shuffle move scored highest last iteration and
            // defeat the whole restriction (lichess xp85KNLq: 28 moves of
            // queen checks in a TB-won KQvKR, halfmove clock at 95).
            if !tb_win_restricted {
                working_scores.sort_by_key(|b| std::cmp::Reverse(b.1));
            }

            let mut pv = Vec::new();
            let score;

            if line_idx == 0 {
                // Aspiration windows: narrow window with progressive widening
                let (mut alpha, mut beta) = if depth >= 4 {
                    (prev_score - 30, prev_score + 30)
                } else {
                    (Score::NEG_INF.0, Score::INF.0)
                };
                let mut delta = 60i32;
                let mut s;
                loop {
                    pv.clear();
                    s = alpha_beta_root(
                        &mut board.clone(),
                        depth,
                        alpha,
                        beta,
                        &mut pv,
                        &mut state,
                        stop,
                        &mut working_scores,
                        &mut best_node_fraction,
                    );

                    if state.should_stop(stop) && depth > 1 {
                        // Don't abort during aspiration fail-low if we have time safety margin
                        let in_fail_low = s <= alpha;
                        let can_extend = state.time_limit_ms.is_some_and(|limit| {
                            state.elapsed_ms() < limit * 6 / 5
                                && state.time_remaining_ms >= 10_000
                                && state.time_remaining_ms > state.inc_ms * 8
                        });
                        if !(in_fail_low && can_extend) {
                            break;
                        }
                    }

                    if s <= alpha {
                        alpha = (s - delta).max(Score::NEG_INF.0);
                        delta += delta / 2;
                        aspiration_failures += 1;
                        continue;
                    }
                    if s >= beta {
                        beta = s + delta;
                        delta += delta / 2;
                        aspiration_failures += 1;
                        continue;
                    }
                    break;
                }
                score = s;
                // Propagate line 0 score ordering back to master root_move_scores
                for (m, ws) in &working_scores {
                    if let Some(entry) = root_move_scores.iter_mut().find(|(rm, _)| rm == m) {
                        entry.1 = *ws;
                    }
                }
            } else {
                // Full window for lines 2+
                score = alpha_beta_root(
                    &mut board.clone(),
                    depth,
                    Score::NEG_INF.0,
                    Score::INF.0,
                    &mut pv,
                    &mut state,
                    stop,
                    &mut working_scores,
                    &mut best_node_fraction,
                );
            }

            if state.should_stop(stop) && depth > 1 {
                depth_completed = false;
                break;
            }

            let line_best = if !pv.is_empty() {
                pv[0]
            } else {
                working_scores[0].0
            };
            excluded_moves.push(line_best);
            line_results.push((score, sanitize_pv(board, &pv), state.seldepth));
        }

        last_iter_ms = state.elapsed_ms().saturating_sub(iter_start_ms);

        if !depth_completed && depth > 1 {
            break;
        }

        if line_results.is_empty() {
            break;
        }

        // The searched root line stands as-is. Tablebase guidance is applied by
        // restricting/ordering the root moves up front (see rank_root_moves and
        // the root-restriction block above), never by overwriting the searched
        // PV with a DTZ "mate in N" line — that line assumes optimal defence and
        // breaks into a repetition the moment the defender deviates (lichess
        // R2KlN7mg perpetual trap, see the regression test).
        //
        // ...with one exception: in TB-restricted mode the searched argmax must
        // not pick the move either. Every listed move preserves the win, so cp
        // differences between them are noise, and inside a conversion with no
        // zeroing move in horizon (pawnless KQvKR) the subtree never sees a
        // tablebase score — the argmax is then a random shuffle check (lichess
        // xp85KNLq drifted to halfmove clock 95 that way). Re-derive the choice
        // from the freshly propagated root scores: shortest proven mate first,
        // else the first move in tablebase order that still scores as winning.
        // If everything scores draw-or-worse the searched line stands — the
        // bucketed sort is mid-demotion and the next iteration re-sorts.
        if tb_win_restricted {
            let mut mate_pick: Option<(Move, i32)> = None;
            let mut win_pick: Option<(Move, i32)> = None;
            for &(m, s) in root_move_scores.iter() {
                if Score(s).is_mate() && s > 0 {
                    if mate_pick.is_none_or(|(_, best)| s > best) {
                        mate_pick = Some((m, s));
                    }
                } else if s > 0 && win_pick.is_none() {
                    win_pick = Some((m, s));
                }
            }
            if let Some((m, s)) = mate_pick.or(win_pick)
                && line_results[0].1.first() != Some(&m)
            {
                let seldepth = line_results[0].2;
                line_results[0] = (s, sanitize_pv(board, &[m]), seldepth);
            }
        }

        let score = line_results[0].0;
        let score_drop = if base_depth > 1 {
            (prev_score - score).max(0)
        } else {
            0
        };
        if base_depth > 1 {
            recent_score_swing = update_recent_score_swing(recent_score_swing, prev_score - score);
        }
        prev_score = score;
        let line0_pv = &line_results[0].1;
        let new_best = if !line0_pv.is_empty() {
            line0_pv[0]
        } else {
            let tt_best = tt
                .probe(board.hash)
                .map(|e| e.best_move)
                .unwrap_or(Move::NULL);
            if !tt_best.is_null() && root_move_scores.iter().any(|(m, _)| *m == tt_best) {
                tt_best
            } else {
                root_move_scores[0].0
            }
        };
        if base_depth > 1 && new_best != best_move {
            recent_pv_changes = update_recent_pv_changes(recent_pv_changes, true);
            best_move_stability = 0;
        } else if base_depth > 1 {
            recent_pv_changes = update_recent_pv_changes(recent_pv_changes, false);
            best_move_stability += 1;
        }
        best_move = new_best;
        best_score = Score(score);
        best_pv = line_results[0].1.clone();
        best_depth = depth;

        if let Some(ref cb) = info_callback {
            let elapsed = state.elapsed_ms();
            for (line_idx, (line_score, line_pv, line_seldepth)) in line_results.iter().enumerate()
            {
                cb(&SearchInfo {
                    depth,
                    seldepth: *line_seldepth,
                    score: Score(*line_score),
                    nodes: state.nodes,
                    time_ms: elapsed,
                    pv: line_pv.clone(),
                    hashfull: tt.hashfull(),
                    multipv_line: line_idx + 1,
                });
            }
        }

        if let Some(counter) = node_counter {
            counter.store(state.nodes, Ordering::Relaxed);
        }

        log::info!(
            "depth {} score {} nodes {} time {}ms pv {}",
            depth,
            best_score,
            state.nodes,
            state.elapsed_ms(),
            best_pv
                .iter()
                .map(|m| m.to_uci())
                .collect::<Vec<_>>()
                .join(" ")
        );

        // When we find a forced mate for us, keep searching for shorter mates
        // but stop if the mate is very short (within 3 moves) since
        // further improvement is unlikely, or if we've spent > 20% of time.
        // Only applies to winning mates (best_score > 0): when we are being
        // mated we must keep searching to find the best defence.
        if best_score.is_mate() && best_score.0 > 0 {
            let mate_distance = (Score::MATE.0 - best_score.0) as u32;
            if mate_distance <= 6 {
                // mate in 3 or fewer full moves
                break;
            }
            if let Some(limit) = state.soft_target_ms
                && state.elapsed_ms() > limit / 5
            {
                break;
            }
        }

        // Root confined to tablebase wins: stop once the two things the search
        // can still contribute have had their chance. `best_score > 0` is the
        // guard — a top move scored draw-or-worse means the search found a
        // repetition and the bucketed sort is mid-demotion, so keep going until
        // a move that actually preserves the win leads.
        if tb_win_restricted && base_depth >= TB_RESTRICTED_MAX_DEPTH && best_score.0 > 0 {
            break;
        }

        if state.should_stop(stop) {
            break;
        }

        // Everything below scales the *soft target*, not the hard ceiling: this is
        // the block that decides how hard the position is and therefore what this
        // move is worth. `should_stop` above enforces the ceiling. Skipped in
        // full while pondering: the opponent has not moved, so no budget is
        // running — the same block takes over the moment `ponderhit` clears
        // the flag, with the pondered time already on the elapsed clock.
        if state.use_soft_limit
            && !state.pondering()
            && let Some(limit) = state.soft_target_ms
        {
            let ponder_fresh_pending = state.ponder_fresh_think_pending();
            // Dynamic soft time limit based on multiple factors:
            // Base: increase in increment controls to avoid under-spending.
            let inc_ratio_permille = state
                .inc_ms
                .saturating_mul(1000)
                .checked_div(limit)
                .unwrap_or(0)
                .min(2000);
            let base_frac = if inc_ratio_permille >= 600 {
                860u64
            } else if inc_ratio_permille >= 350 {
                800u64
            } else if inc_ratio_permille >= 200 {
                740u64
            } else {
                650u64
            };

            // PV instability: +8% per change, up to +40%
            let instability_bonus = (recent_pv_changes as u64).min(5) * 80;

            // Score drop: +5% per 25cp drop, up to +40%
            let drop_bonus = ((score_drop as u64) / 25).min(8) * 50;

            // Score volatility: +6% per 30cp swing, up to +30%
            let volatility_bonus = ((recent_score_swing as u64) / 30).min(5) * 60;

            // Aspiration failures: +8% per failure, up to +32%
            let aspiration_bonus = (aspiration_failures as u64).min(4) * 80;

            // Root move complexity: +1% per move above 20, up to +20%
            let root_moves_count = root_move_scores.len() as u64;
            let complexity_bonus = root_moves_count.saturating_sub(20).min(20) * 10;

            // Increment safety net: when increment is significant relative
            // to allocated time, we can afford to spend more since it
            // replenishes. +0-15% based on inc/allocated ratio.
            let inc_bonus = state
                .inc_ms
                .saturating_mul(150)
                .checked_div(limit)
                .unwrap_or(0)
                .min(150);

            // Clock-surplus bonus: if we have a healthy bank beyond a
            // phase-aware reserve, bias more strongly toward spending now.
            let desired_bank = if state.inc_ms > 0 {
                if game_ply < 20 {
                    8_000u64.saturating_add(state.inc_ms.saturating_mul(16))
                } else if game_ply < 40 {
                    6_000u64.saturating_add(state.inc_ms.saturating_mul(12))
                } else if game_ply < 60 {
                    5_000u64.saturating_add(state.inc_ms.saturating_mul(9))
                } else {
                    4_000u64.saturating_add(state.inc_ms.saturating_mul(7))
                }
            } else if game_ply < 40 {
                5_000
            } else {
                3_000
            };
            let surplus = state.time_remaining_ms.saturating_sub(desired_bank);
            let surplus_bonus = if limit > 0 {
                (surplus.saturating_mul(320) / (limit.saturating_mul(8).max(1))).min(320)
            } else {
                0
            };

            // Blitz safety: in low time, bias toward spending less of the
            // allocated budget to reduce practical timeout risk.
            let blitz_adjust: i64 = if state.time_remaining_ms <= 15_000 {
                -180 // -18%
            } else if state.time_remaining_ms <= 30_000 {
                -80 // -8%
            } else {
                0
            };

            // Best move stability: -3% per stable iteration, up to -15%
            let stability_discount = (best_move_stability as u64).min(5) * 30;

            // Best-move node fraction: if >80% nodes go to best move,
            // position is decided → stop sooner. If <50%, unclear → longer.
            let node_frac_adj: i64 = if best_node_fraction > 0.80 {
                -100 // stop 10% sooner
            } else if best_node_fraction < 0.50 {
                200 // search 20% longer
            } else {
                0
            };

            // Score-aware adjustment:
            // - Spend less in clearly decided and stable positions (won/lost)
            // - Spend more in unclear or volatile positions.
            let eval_abs = best_score.0.saturating_abs() as u64;
            let eval_adjust: i64 = if best_score.is_mate() {
                // Once mate is visible, prioritize conversion speed.
                -220
            } else if eval_abs >= 500 && best_move_stability >= 2 && recent_score_swing <= 40 {
                -220
            } else if eval_abs >= 300 && best_move_stability >= 2 && recent_score_swing <= 60 {
                -140
            } else if eval_abs <= 50 && recent_score_swing >= 35 {
                180
            } else if eval_abs <= 90 && recent_score_swing >= 25 {
                100
            } else {
                0
            };

            // Early opening brake: keep first moves snappier and save heavy
            // spending for richer middlegame positions. Chess960 skips this
            // (see compute_time_limit).
            let opening_adjust: i64 =
                opening_time_adjust(game_ply, state.time_remaining_ms, params.chess960);

            let soft_frac = (base_frac
                + instability_bonus
                + drop_bonus
                + volatility_bonus
                + aspiration_bonus
                + complexity_bonus
                + inc_bonus
                + surplus_bonus
                + book_exit_bonus)
                .saturating_sub(stability_discount);
            let soft_frac =
                (soft_frac as i64 + node_frac_adj + eval_adjust + opening_adjust + blitz_adjust)
                    .max(0) as u64;
            // Range: 35% of the soft target for an obvious position, up to the hard
            // ceiling for a genuinely difficult one. The old 180% ceiling was set
            // when soft == hard, so everything above 100% was unreachable and it was
            // never a real knob; the hard limit (and `should_stop`) is the true bound
            // now, so let the difficulty bonuses run all the way up to it.
            let soft_frac = soft_frac.clamp(350, HARD_LIMIT_SOFT_MULT * 1000);
            let soft_limit = limit * soft_frac / 1000;
            let elapsed = state.elapsed_ms();
            if tm_debug {
                println!(
                    "info string tm d={depth} frac={soft_frac} soft={limit} soft_lim={soft_limit} \
                     hard={:?} elapsed={elapsed} | inst={instability_bonus} drop={drop_bonus} \
                     vol={volatility_bonus} asp={aspiration_bonus} cplx={complexity_bonus} \
                     surp={surplus_bonus} stab=-{stability_discount} nodefrac={node_frac_adj} \
                     eval={eval_adjust} pvch={recent_pv_changes} swing={recent_score_swing} \
                     bnf={best_node_fraction:.2}",
                    state.time_limit_ms
                );
            }
            if elapsed > soft_limit && !ponder_fresh_pending {
                break;
            }
            // Predict whether the next iteration can complete in time.
            // Each deeper iteration typically takes 2-3x the previous one.
            // Estimate next iteration cost as ~3x the last one (conservative
            // branching factor). Stop if it would exceed the soft limit.
            // In unstable/tactical positions, the next iteration growth is
            // often closer to ~2x than ~3x; using 3x can stop too early.
            // Keep the conservative 3x in quiet/stable nodes.
            let unstable =
                recent_pv_changes > 0 || recent_score_swing >= 35 || best_node_fraction < 0.55;
            let growth = if unstable { 2 } else { 3 };
            let predicted_next = last_iter_ms * growth;
            let prediction_margin_permille = if surplus > limit {
                980u64
            } else if inc_ratio_permille >= 350 {
                950u64
            } else {
                900u64
            };
            if depth >= 6
                && elapsed + predicted_next
                    > soft_limit.saturating_mul(prediction_margin_permille) / 1000
                && !ponder_fresh_pending
            {
                break;
            }
        }
    }

    if best_move.is_null() {
        best_move = root_move_scores[0].0;
    }

    // Fallback: if the search produced no PV at all (e.g. stopped before any
    // root move was scored), fall back to the tablebase's best winning move with
    // a TB-win-band score — never a fabricated mate line.
    if best_pv.is_empty()
        && let Some(ranking) = root_tb_ranking.as_ref()
        && let Some(&first) = ranking.restriction_moves().first()
    {
        best_move = first;
        best_score = ranking.score;
        best_pv = vec![first];
        best_depth = best_depth.max(1);
    }

    SearchResult {
        best_move,
        score: best_score,
        depth: best_depth,
        nodes: state.nodes,
        pv: sanitize_pv(board, &best_pv),
    }
}

// ---------------------------------------------------------------------------
// Root-specific alpha-beta (uses pre-ordered root move list)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn alpha_beta_root(
    board: &mut Board,
    depth: u8,
    mut alpha: i32,
    beta: i32,
    pv: &mut Vec<Move>,
    state: &mut SearchState,
    stop: &AtomicBool,
    root_move_scores: &mut [(Move, i32)],
    best_node_fraction: &mut f32,
) -> i32 {
    pv.clear();

    if state.nodes & 1023 == 0 && state.should_stop(stop) {
        state.stop = true;
        return 0;
    }

    state.nodes += 1;
    let in_check = chess_core::is_in_check(board);
    let effective_depth = if in_check { depth + 1 } else { depth };

    // TT probe (for move ordering at root, not cutoff)
    let tt_entry = state.tt.probe(board.hash);
    let tt_move = tt_entry.map(|e| e.best_move).unwrap_or(Move::NULL);

    // Ensure TT move is first in root moves if present, but don't promote excluded moves (score == i32::MIN)
    if !tt_move.is_null() {
        for i in 0..root_move_scores.len() {
            if root_move_scores[i].0 == tt_move && i > 0 && root_move_scores[i].1 != i32::MIN {
                root_move_scores.swap(0, i);
                break;
            }
        }
    }

    let mut best_score = Score::NEG_INF.0;
    let mut best_move = root_move_scores[0].0;
    let mut child_pv = PvLine::new();
    let mut tmp_pv = PvLine::new();
    let mut tt_store_flag = TTFlag::UpperBound;
    let total_nodes_start = state.nodes;
    let mut best_move_nodes: u64 = 0;
    for (moves_searched, (m, move_score)) in root_move_scores.iter_mut().enumerate() {
        if *move_score == i32::MIN {
            continue; // excluded by multi-PV logic
        }
        let m = *m;
        let move_nodes_start = state.nodes;

        let prev_castling = board.castling;
        let prev_ep = board.en_passant;
        let prev_halfmove = board.halfmove_clock;

        // Store ply context BEFORE make_move (piece is still on from_sq)
        state.store_ply_context(0, m, board);

        let captured = board.make_move(m);

        // Incremental accumulator update for root (ply 0 → ply 1)
        if state.use_nnue {
            update_accumulator_for_move(state, board, m, captured, prev_castling, prev_ep, 0, 1);
        }

        let gives_check = chess_core::is_in_check(board);

        let search_depth = effective_depth - 1;
        let moves_searched = moves_searched as u32;

        let score = if moves_searched == 0 {
            -alpha_beta(
                board,
                search_depth,
                1,
                -beta,
                -alpha,
                &mut child_pv,
                state,
                stop,
                m,
                Move::NULL,
            )
        } else {
            let mut do_full_search = true;
            let mut reduced_score = 0;

            // LMR at root for late moves
            if moves_searched >= 3
                && depth >= 3
                && !in_check
                && !gives_check
                && !m.is_capture()
                && !m.is_promotion()
            {
                let d = (depth as usize).min(63);
                let ms = (moves_searched as usize).min(63);
                let reduction =
                    (state.lmr_table[d][ms] as i8).clamp(0, (search_depth as i8 - 1).max(0));

                let reduced_depth = search_depth.saturating_sub(reduction as u8);
                tmp_pv.clear();
                reduced_score = -alpha_beta(
                    board,
                    reduced_depth,
                    1,
                    -alpha - 1,
                    -alpha,
                    &mut tmp_pv,
                    state,
                    stop,
                    m,
                    Move::NULL,
                );

                if reduced_score <= alpha {
                    do_full_search = false;
                    std::mem::swap(&mut child_pv, &mut tmp_pv);
                }
            }

            if do_full_search {
                tmp_pv.clear();
                let nw_score = -alpha_beta(
                    board,
                    search_depth,
                    1,
                    -alpha - 1,
                    -alpha,
                    &mut tmp_pv,
                    state,
                    stop,
                    m,
                    Move::NULL,
                );

                if nw_score > alpha && nw_score < beta {
                    -alpha_beta(
                        board,
                        search_depth,
                        1,
                        -beta,
                        -alpha,
                        &mut child_pv,
                        state,
                        stop,
                        m,
                        Move::NULL,
                    )
                } else {
                    std::mem::swap(&mut child_pv, &mut tmp_pv);
                    nw_score
                }
            } else {
                reduced_score
            }
        };

        board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

        // Store score for root move ordering in next iteration
        *move_score = score;

        if state.stop {
            return 0;
        }

        let move_nodes = state.nodes - move_nodes_start;

        if score > best_score {
            best_score = score;
            best_move = m;
            best_move_nodes = move_nodes;

            if score > alpha {
                alpha = score;
                tt_store_flag = TTFlag::Exact;

                pv.clear();
                pv.push(m);
                pv.extend_from_slice(&child_pv);

                if score >= beta {
                    tt_store_flag = TTFlag::LowerBound;
                    break;
                }
            }
        }
    }

    // Compute best-move node fraction for time management
    let total_nodes = state.nodes - total_nodes_start;
    if total_nodes > 0 {
        *best_node_fraction = best_move_nodes as f32 / total_nodes as f32;
    }

    state.tt.store(
        board.hash,
        effective_depth,
        score_to_tt(best_score, 0),
        tt_store_flag,
        best_move,
        None,
    );

    best_score
}

// ---------------------------------------------------------------------------
// Alpha-beta search
// ---------------------------------------------------------------------------

/// Draw score from the side to move at `ply`. Contempt is relative to the
/// root side: every draw costs the side we search for `contempt`, whichever
/// side completes the repetition. (Scoring -contempt for the side to move
/// instead makes a draw worth +contempt to the root whenever the opponent is
/// to move at the drawn node, so a line where the opponent can only repeat
/// looked better than a quiet hold — Mythos–Endspiel 2026-09-18, 47...Rh1.)
fn draw_score(state: &SearchState, ply: u8) -> i32 {
    if ply.is_multiple_of(2) {
        -state.contempt
    } else {
        state.contempt
    }
}

#[allow(clippy::too_many_arguments)]
fn alpha_beta(
    board: &mut Board,
    depth: u8,
    ply: u8,
    alpha: i32,
    beta: i32,
    pv: &mut PvLine,
    state: &mut SearchState,
    stop: &AtomicBool,
    prev_move: Move,
    excluded_move: Move,
) -> i32 {
    pv.clear();

    if state.nodes & 1023 == 0 && state.should_stop(stop) {
        state.stop = true;
        return 0;
    }

    // Ply limit guard: prevent stack overflow and out-of-bounds accumulator access.
    // With non-zero contempt the engine avoids draws, potentially driving ply past
    // MAX_PLY (and eventually wrapping the u8 ply counter).  Cap it here.
    if ply as usize >= MAX_PLY {
        let eval = evaluate(board).0;
        return match board.side_to_move {
            Color::White => eval,
            Color::Black => -eval,
        };
    }

    // Draw detection
    if ply > 0 {
        if board.halfmove_clock >= 100 {
            return draw_score(state, ply);
        }
        // Twofold in the current search path: standard practice since the
        // opponent can always force a third repetition.
        if board.is_twofold_in_search(state.game_ply) {
            return draw_score(state, ply);
        }
        // Threefold across the full history (catches game-history positions).
        // A SINGLE game-history occurrence is deliberately NOT a draw here
        // (standard semantics): scoring first revisits as terminal draws lets
        // a defender poison every winning line by steering toward once-seen
        // positions — in lichess 45geMoUz it flattened all of White's TB-won
        // continuations to ~0 and the engine walked into an actual threefold.
        // Shuffle pressure at the root comes from the TB restriction's child-
        // repetition grading instead.
        if board.is_repetition() {
            return draw_score(state, ply);
        }
    }

    // -------------------------------------------------------------------
    // Mate distance pruning (MDP)
    // -------------------------------------------------------------------
    // Tighten alpha/beta based on the shortest possible mate from this
    // node. If we already know a mate-in-N, any subtree that can't
    // beat it is pruned immediately, dramatically speeding up mate
    // searches.
    let (mut alpha, mut beta) = (alpha, beta);
    if ply > 0 {
        let mating_value = Score::MATE.0 - ply as i32;
        if mating_value < beta {
            beta = mating_value;
            if alpha >= beta {
                return alpha;
            }
        }
        let mated_value = -Score::MATE.0 + ply as i32;
        if mated_value > alpha {
            alpha = mated_value;
            if alpha >= beta {
                return beta;
            }
        }
    }

    state.nodes += 1;
    let in_check = chess_core::is_in_check(board);

    // Leaf node: only enter qsearch if NOT in check.
    // Positions in check must get at least 1 ply of full search to avoid
    // missing checkmates and quiet evasion moves.
    if depth == 0 && !in_check {
        return quiescence(board, ply, alpha, beta, pv, state, stop, true);
    }

    let is_root = ply == 0;
    let is_pv = beta - alpha > 1;

    // Check extension: ensure at least depth 1 when in check
    let effective_depth = if in_check { depth.max(1) } else { depth };

    // Syzygy WDL probe: probe tablebases for positions with no castling rights
    // and a piece count within the loaded tablebase range.  Returns an exact
    // game-theoretic score that is used to bound (or immediately return from)
    // this node.  We skip probing during singular-extension searches to avoid
    // interfering with the null-window test.
    //
    // Probe only at halfmove_clock == 0 (the position right after a capture or
    // pawn move). WDL is blind to game history: at hmc > 0 a "won" position can
    // still be drawn on the board via threefold repetition or the 50-move rule,
    // and a cutoff here adjudicates the line before the repetition checks in
    // the subtree can see it (lichess 45geMoUz: 81.Kh5 Be8+ was cut off as
    // "Black is lost" two plies above the draw Black was actually forcing).
    // After a zeroing move no earlier position can recur and the 50-move count
    // restarts, so the cutoff is sound — the standard Stockfish guard.
    if let Some(ref tb) = state.syzygy_tb {
        let piece_count = board.all_occupancy().count();
        if piece_count <= tb.max_pieces()
            && board.castling.0 == 0
            && board.halfmove_clock == 0
            && excluded_move.is_null()
            && let Some(wdl) = syzygy::probe_wdl(tb, board)
        {
            use pyrrhic_rs::WdlProbeResult;
            let tb_score = syzygy::wdl_to_score(wdl);
            let flag = match wdl {
                WdlProbeResult::Win | WdlProbeResult::CursedWin => TTFlag::LowerBound,
                WdlProbeResult::Loss | WdlProbeResult::BlessedLoss => TTFlag::UpperBound,
                WdlProbeResult::Draw => TTFlag::Exact,
            };
            state.tt.store(
                board.hash,
                effective_depth,
                score_to_tt(tb_score, ply),
                flag,
                Move::NULL,
                None,
            );
            match flag {
                TTFlag::LowerBound => {
                    if tb_score >= beta {
                        return tb_score;
                    }
                    alpha = alpha.max(tb_score);
                }
                TTFlag::UpperBound => {
                    if tb_score <= alpha {
                        return tb_score;
                    }
                    beta = beta.min(tb_score);
                }
                TTFlag::Exact => return tb_score,
            }
        }
    }

    // TT probe
    let tt_entry = state.tt.probe(board.hash);
    let (tt_move, tt_score, tt_depth, tt_flag) = if let Some(entry) = tt_entry {
        let adj_score = score_from_tt(entry.score, ply);
        if !is_pv && !is_root && entry.depth >= effective_depth && excluded_move.is_null() {
            match entry.flag {
                TTFlag::Exact => return adj_score,
                TTFlag::LowerBound => {
                    if adj_score >= beta {
                        return adj_score;
                    }
                }
                TTFlag::UpperBound => {
                    if adj_score <= alpha {
                        return adj_score;
                    }
                }
            }
        }
        (entry.best_move, adj_score, entry.depth, entry.flag)
    } else {
        (Move::NULL, 0, 0, TTFlag::UpperBound)
    };

    // In check: static eval is unreliable and all eval-based pruning is
    // skipped. Use a sentinel so `improving` is true after escaping check
    // (erring on the side of searching more — the safe direction).
    let (unadjusted_eval, static_eval) = if in_check {
        (None, i32::MIN / 2)
    } else {
        let (unadjusted, raw) =
            evaluate_for_side(board, state, ply, tt_entry.and_then(|entry| entry.eval));
        // Refine with TT bound: the search result at this position is
        // often a more accurate estimate than the static eval.
        let refined = if tt_entry.is_some() && tt_score.abs() < MATE_THRESHOLD {
            match tt_flag {
                TTFlag::Exact => tt_score,
                TTFlag::LowerBound if tt_score > raw => tt_score,
                TTFlag::UpperBound if tt_score < raw => tt_score,
                _ => raw,
            }
        } else {
            raw
        };
        (unadjusted, refined)
    };

    if (ply as usize) < MAX_PLY {
        state.static_evals[ply as usize] = static_eval;
    }

    let improving = !in_check
        && ply >= 2
        && (ply as usize) < MAX_PLY
        && static_eval > state.static_evals[(ply as usize) - 2];

    // -----------------------------------------------------------------------
    // Pre-move pruning
    // -----------------------------------------------------------------------
    // Skip all pre-move pruning when the search window contains mate
    // scores. These heuristics return centipawn values and would discard
    // the forced-mate proof we are trying to build.
    let searching_for_mate = !(-MATE_THRESHOLD..=MATE_THRESHOLD).contains(&alpha)
        || !(-MATE_THRESHOLD..=MATE_THRESHOLD).contains(&beta);

    if !is_pv && !in_check && !is_root && excluded_move.is_null() && !searching_for_mate {
        // Reverse futility pruning
        if depth <= 6 {
            let margin = (if improving {
                state.tune.rfp_margin_imp
            } else {
                state.tune.rfp_margin_noimp
            }) * depth as i32;
            if static_eval - margin >= beta {
                return static_eval - margin;
            }
        }

        // Null move pruning
        if depth >= 3 && static_eval >= beta && has_non_pawn_material(board, board.side_to_move) {
            let null_r = 3 + depth as u32 / 4 + ((static_eval - beta) as u32 / 200).min(3);
            let null_depth = depth.saturating_sub(1 + null_r as u8);

            let prev_ep = board.en_passant;
            let prev_hash = board.hash;
            // Incremental hash update: XOR out old ep file (if any) + flip side.
            // Must happen before en_passant is cleared.
            board.update_hash_for_null_move();
            board.side_to_move = board.side_to_move.opposite();
            board.en_passant = None;

            if state.use_nnue && ply as usize + 1 < MAX_PLY {
                update_accumulator_for_null_move(state, board, prev_ep, ply as usize);
            }

            let mut null_pv = PvLine::new();
            state.double_exts[ply as usize + 1] = state.double_exts[ply as usize];
            let null_score = -alpha_beta(
                board,
                null_depth,
                ply + 1,
                -beta,
                -beta + 1,
                &mut null_pv,
                state,
                stop,
                Move::NULL,
                Move::NULL,
            );

            board.side_to_move = board.side_to_move.opposite();
            board.en_passant = prev_ep;
            board.hash = prev_hash;

            if null_score >= beta {
                return beta;
            }
        }

        // Probcut: if a shallow search on captures beats beta by a margin,
        // the full-depth search is very likely to beat beta too.
        if depth >= 5 && !searching_for_mate {
            let probcut_beta = beta + 200;
            if static_eval >= probcut_beta - 100 {
                let probcut_depth = depth - 4;
                let pc_us = board.side_to_move;
                let captures = chess_core::generate_pseudo_legal_captures(board);

                for &m in captures.iter() {
                    if !see::see_ge(board, m, probcut_beta - static_eval) {
                        continue;
                    }

                    let prev_castling = board.castling;
                    let prev_ep = board.en_passant;
                    let prev_halfmove = board.halfmove_clock;

                    state.store_ply_context(ply, m, board);
                    let captured = board.make_move(m);

                    if move_is_illegal(board, pc_us) {
                        board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                        continue;
                    }

                    // Same ply guard as every other accumulator push: ply 127 would
                    // write past the 128-deep stack.
                    if state.use_nnue && (ply as usize) + 1 < MAX_PLY {
                        update_accumulator_for_move(
                            state,
                            board,
                            m,
                            captured,
                            prev_castling,
                            prev_ep,
                            ply as usize,
                            ply as usize + 1,
                        );
                    }

                    state.nodes += 1;
                    let mut pc_pv = PvLine::new();
                    let score = -quiescence(
                        board,
                        ply + 1,
                        -probcut_beta,
                        -probcut_beta + 1,
                        &mut pc_pv,
                        state,
                        stop,
                        true,
                    );

                    let score = if score >= probcut_beta {
                        state.double_exts[ply as usize + 1] = state.double_exts[ply as usize];
                        -alpha_beta(
                            board,
                            probcut_depth,
                            ply + 1,
                            -probcut_beta,
                            -probcut_beta + 1,
                            &mut pc_pv,
                            state,
                            stop,
                            m,
                            Move::NULL,
                        )
                    } else {
                        score
                    };

                    board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

                    if score >= probcut_beta {
                        state.tt.store(
                            board.hash,
                            depth,
                            score_to_tt(score, ply),
                            TTFlag::LowerBound,
                            m,
                            unadjusted_eval,
                        );
                        return score;
                    }
                }
            }
        }

        // Razoring
        if depth <= 3 {
            let razor_margin = 300 + 200 * depth as i32;
            if static_eval + razor_margin <= alpha {
                let qscore = quiescence(board, ply, alpha, beta, pv, state, stop, true);
                if qscore <= alpha {
                    return qscore;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // IIR: Internal Iterative Reduction
    // -----------------------------------------------------------------------
    // When no TT move exists, reduce depth by 1 instead of doing a full
    // recursive IID search. Much cheaper and nearly as effective.
    let iir_reduction = if tt_move.is_null() && depth >= 3 {
        if !is_pv && depth >= 8 { 2i8 } else { 1 }
    } else {
        0
    };

    // -----------------------------------------------------------------------
    // Staged move generation via MovePicker
    // -----------------------------------------------------------------------
    // Moves are generated lazily: TT move → captures → quiets → bad captures.
    // Legality is checked inline after make_move (no board cloning).

    let killers = if (ply as usize) < MAX_PLY {
        state.killers[ply as usize]
    } else {
        [Move::NULL; MAX_KILLERS]
    };
    let counter_move = state.get_counter_move(prev_move);
    let us = board.side_to_move;

    let mut picker = MovePicker::new(tt_move);

    let mut best_score = Score::NEG_INF.0;
    let mut best_move = Move::NULL;
    let mut child_pv = PvLine::new();
    let mut tt_store_flag = TTFlag::UpperBound;
    let mut moves_searched = 0u32;
    // Fixed-size stack arrays avoid a heap allocation per node.
    // 128 slots is far more than any real search node will fill before a cutoff.
    let mut quiet_moves_searched = [Move::NULL; 128];
    let mut quiet_count: usize = 0;
    let mut captures_searched = [Move::NULL; 128];
    let mut capture_count: usize = 0;
    // Single reusable buffer for LMR/null-window PV — cleared before each use.
    let mut tmp_pv = PvLine::new();

    while let Some(m) = picker.next(board, state, ply, &killers, counter_move) {
        if m == excluded_move && !excluded_move.is_null() {
            continue;
        }

        // -------------------------------------------------------------------
        // Singular extension (BEFORE making the move)
        // -------------------------------------------------------------------
        let mut extension: i8 = 0;
        let singular_mode = state.singular_ext_mode;
        let singular_candidate = !is_root
            && !in_check
            && moves_searched == 0
            && m == tt_move
            && !tt_move.is_null()
            && excluded_move.is_null()
            && tt_score.abs() < MATE_THRESHOLD;
        let conservative_gate = singular_mode == 1
            && !is_pv
            && depth >= 8
            && tt_depth >= depth - 2
            && tt_flag == TTFlag::LowerBound
            && tt_score >= alpha + 50;
        let aggressive_gate = singular_mode >= 2
            && depth >= 7
            && tt_depth >= depth - 3
            && tt_flag != TTFlag::UpperBound
            && tt_score >= alpha + 30;
        if singular_mode > 0 && singular_candidate && (conservative_gate || aggressive_gate) {
            let se_beta = tt_score - 2 * depth as i32;
            let se_depth = (depth - 1) / 2;

            let mut se_pv = PvLine::new();
            let se_score = alpha_beta(
                board,
                se_depth,
                ply,
                se_beta - 1,
                se_beta,
                &mut se_pv,
                state,
                stop,
                prev_move,
                m,
            );

            if se_score < se_beta {
                extension = if !is_pv
                    && se_score < se_beta - DOUBLE_EXT_MARGIN
                    && state.double_exts[ply as usize] < MAX_DOUBLE_EXTS
                {
                    2
                } else {
                    1
                };
            } else if !is_pv && se_beta >= beta {
                // Multi-cut: even without the TT move another move beats beta,
                // so this node fails high on more than one move.
                return se_beta;
            }
        }

        // -------------------------------------------------------------------
        // Passed pawn push extension: pawn to 7th rank or promotion
        // These moves create immediate promotion threats that drastically
        // change the position.  Extending ensures the search sees the
        // consequences (promotion, new queen activity, mating attacks).
        // -------------------------------------------------------------------
        if extension == 0 {
            if let Some(piece) = board.piece_at(m.from_sq())
                && piece.kind == PieceKind::Pawn
            {
                let to_rank = m.to_sq().rank();
                let is_7th = match piece.color {
                    Color::White => to_rank == 6,
                    Color::Black => to_rank == 1,
                };
                if is_7th {
                    extension = 1;
                }
            }
            // Also extend promotion moves (queen promotions are critical)
            if m.is_promotion() {
                extension = 1;
            }
        }

        let prev_castling = board.castling;
        let prev_ep = board.en_passant;
        let prev_halfmove = board.halfmove_clock;

        // -------------------------------------------------------------------
        // Pre-make pruning (SEE on PRE-move board)
        // -------------------------------------------------------------------
        if moves_searched > 0 && !is_pv && !in_check {
            // SEE pruning for bad captures (extended to depth 7)
            // Must be done BEFORE make_move — SEE reads the pre-move board.
            if depth <= 7 && m.is_capture() {
                let see_threshold = -30 * depth as i32;
                if !see::see_ge(board, m, see_threshold) {
                    continue;
                }
            }

            // SEE pruning for quiet moves: prune moves that lose material
            // (e.g., moving a piece to a square attacked by a pawn)
            if depth <= 5
                && !m.is_capture()
                && !m.is_promotion()
                && !see::see_ge(
                    board,
                    m,
                    -state.tune.see_quiet_margin * depth as i32
                        - state.learning.history[us.index()][m.from_sq().index()]
                            [m.to_sq().index()]
                        .max(0)
                            / 64,
                )
            {
                continue;
            }
        }

        // Store ply context BEFORE make_move (piece is still on from_sq)
        state.store_ply_context(ply, m, board);
        // Both histories are keyed on the mover, so read them while the piece is
        // still on from_sq and before the side to move flips; pruning and LMR
        // below run after make_move.
        let cont = state.get_cont_history_bonus(m, board, ply);
        let hist = state.learning.history[us.index()][m.from_sq().index()][m.to_sq().index()];

        let captured = board.make_move(m);

        // ── Inline legality check ──────────────────────────────────────
        if move_is_illegal(board, us) {
            board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
            continue;
        }

        let gives_check = chess_core::is_in_check(board);

        // -------------------------------------------------------------------
        // Move pruning (non-first moves, non-PV)
        // -------------------------------------------------------------------
        if moves_searched > 0 && !is_pv && !in_check {
            // Futility pruning for quiet moves (extended to depth 5)
            if !gives_check
                && depth <= 5
                && !m.is_capture()
                && !m.is_promotion()
                && !searching_for_mate
            {
                let futility_margin = (if improving {
                    state.tune.fut_margin_imp
                } else {
                    state.tune.fut_margin_noimp
                }) * depth as i32
                    + 50
                    + (hist + cont / 2) / 64;
                if static_eval + futility_margin <= alpha {
                    board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                    continue;
                }
            }

            // Late move pruning (extended to depth 6)
            if !gives_check
                && depth <= 6
                && !m.is_capture()
                && !m.is_promotion()
                && !searching_for_mate
            {
                let lmp_threshold = if improving {
                    3 + depth as usize * depth as usize
                } else {
                    (3 + depth as usize * depth as usize) * 2 / 3
                };
                if moves_searched as usize >= lmp_threshold {
                    board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                    continue;
                }
            }

            // History pruning: prune quiet moves with consistently terrible history
            if !gives_check
                && depth <= 4
                && !m.is_capture()
                && !m.is_promotion()
                && !searching_for_mate
                && hist + cont / 2 < -3000 * depth as i32
            {
                board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                continue;
            }
        }

        // Incremental accumulator update — deferred until after pruning so
        // moves that get pruned away never pay this cost.
        if state.use_nnue && ply as usize + 1 < MAX_PLY {
            update_accumulator_for_move(
                state,
                board,
                m,
                captured,
                prev_castling,
                prev_ep,
                ply as usize,
                ply as usize + 1,
            );
        }

        // -------------------------------------------------------------------
        // Search this move
        // -------------------------------------------------------------------
        let search_depth = (effective_depth as i8 - 1 + extension - iir_reduction).max(0) as u8;
        state.double_exts[ply as usize + 1] =
            state.double_exts[ply as usize] + (extension == 2) as u8;

        let score = if moves_searched == 0 {
            // First move: full window
            -alpha_beta(
                board,
                search_depth,
                ply + 1,
                -beta,
                -alpha,
                &mut child_pv,
                state,
                stop,
                m,
                Move::NULL,
            )
        } else {
            let mut do_full_search = true;
            let mut reduced_score = 0;

            // Late move reductions (logarithmic table)
            if moves_searched >= 2
                && depth >= 3
                && !in_check
                && !gives_check
                && !m.is_capture()
                && !m.is_promotion()
            {
                let d = (depth as usize).min(63);
                let ms = (moves_searched as usize).min(63);
                let mut reduction = state.lmr_table[d][ms] as i8;

                // Reduce less at PV nodes
                if is_pv {
                    reduction -= 1;
                }
                // Reduce more at non-PV (likely cut) nodes
                if !is_pv {
                    reduction += 1;
                }
                // Reduce less for killers/counter-moves
                if m == killers[0] || m == killers[1] || m == counter_move {
                    reduction -= 1;
                }
                // Reduce more when not improving
                if !improving {
                    reduction += 1;
                }
                // Continuous history-based reduction with continuation history:
                // good history → less reduction, bad history → more reduction.
                reduction -= ((hist + cont / 2) / state.tune.hist_lmr_div) as i8;
                // Extra reduction for very negative history
                if hist < -4000 {
                    reduction += 1;
                }
                // Reduce more when eval is below alpha (position looks bad)
                if !is_pv && static_eval + 150 < alpha {
                    reduction += 1;
                }

                reduction = reduction.clamp(0, (search_depth as i8 - 1).max(0));

                let reduced_depth = search_depth.saturating_sub(reduction as u8);
                tmp_pv.clear();
                reduced_score = -alpha_beta(
                    board,
                    reduced_depth,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    &mut tmp_pv,
                    state,
                    stop,
                    m,
                    Move::NULL,
                );

                if reduced_score <= alpha {
                    do_full_search = false;
                    std::mem::swap(&mut child_pv, &mut tmp_pv);
                }
            }

            if do_full_search {
                // Null window search
                tmp_pv.clear();
                let nw_score = -alpha_beta(
                    board,
                    search_depth,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    &mut tmp_pv,
                    state,
                    stop,
                    m,
                    Move::NULL,
                );

                if nw_score > alpha && nw_score < beta {
                    -alpha_beta(
                        board,
                        search_depth,
                        ply + 1,
                        -beta,
                        -alpha,
                        &mut child_pv,
                        state,
                        stop,
                        m,
                        Move::NULL,
                    )
                } else {
                    std::mem::swap(&mut child_pv, &mut tmp_pv);
                    nw_score
                }
            } else {
                reduced_score
            }
        };

        board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

        if !m.is_capture() && !m.is_promotion() && quiet_count < 128 {
            quiet_moves_searched[quiet_count] = m;
            quiet_count += 1;
        } else if m.is_capture() && capture_count < 128 {
            captures_searched[capture_count] = m;
            capture_count += 1;
        }
        moves_searched += 1;

        if state.stop {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = m;

            if score > alpha {
                alpha = score;
                tt_store_flag = TTFlag::Exact;

                if score < beta {
                    pv.clear();
                    pv.push(m);
                    pv.extend_from_slice(&child_pv);
                }

                if score >= beta {
                    tt_store_flag = TTFlag::LowerBound;

                    if !m.is_capture() {
                        state.store_killer(ply as usize, m);
                        state.update_history(m, depth, board, ply);
                        state.store_counter_move(prev_move, m);

                        // History malus: penalize all previously searched quiet moves
                        for &qm in &quiet_moves_searched[..quiet_count] {
                            if qm != m {
                                state.update_history_malus(qm, depth, board, ply);
                            }
                        }
                    } else {
                        // Capture history: bonus for the cutoff capture
                        state.update_capture_history(m, depth, board);
                        // Malus for previously searched captures
                        for &cm in &captures_searched[..capture_count] {
                            if cm != m {
                                state.update_capture_history_malus(cm, depth, board);
                            }
                        }
                    }

                    break;
                }
            }
        }
    }

    // No legal move found → checkmate or stalemate.
    // When inside a singular-extension search, the TT move is excluded
    // and may be the only legal move, so return alpha instead.
    if moves_searched == 0 {
        if !excluded_move.is_null() {
            return alpha;
        }
        return if in_check {
            -Score::MATE.0 + ply as i32
        } else {
            draw_score(state, ply)
        };
    }

    if excluded_move.is_null() {
        // Update pawn corrhist from the residual between the search result and
        // the static eval — but only from trustworthy nodes: not in check, a
        // quiet best move, non-mate score, and a bound that agrees with the sign
        // of the residual.
        if !in_check
            && !best_move.is_null()
            && !best_move.is_capture()
            && best_score < MATE_THRESHOLD
            && best_score > -MATE_THRESHOLD
            && match tt_store_flag {
                TTFlag::Exact => true,
                TTFlag::LowerBound => best_score >= static_eval,
                TTFlag::UpperBound => best_score <= static_eval,
            }
        {
            state.update_corrhist(board, depth, best_score - static_eval, ply);
        }
        state.tt.store(
            board.hash,
            effective_depth,
            score_to_tt(best_score, ply),
            tt_store_flag,
            best_move,
            unadjusted_eval,
        );
    }

    best_score
}

// ---------------------------------------------------------------------------
// Quiescence search
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn quiescence(
    board: &mut Board,
    ply: u8,
    mut alpha: i32,
    beta: i32,
    pv: &mut PvLine,
    state: &mut SearchState,
    stop: &AtomicBool,
    generate_checks: bool,
) -> i32 {
    pv.clear();

    if state.nodes & 1023 == 0 && state.should_stop(stop) {
        state.stop = true;
        return 0;
    }

    // Ply limit guard (mirrors the alpha_beta guard).
    if ply as usize >= MAX_PLY {
        let eval = evaluate(board).0;
        let stand_pat = match board.side_to_move {
            Color::White => eval,
            Color::Black => -eval,
        };
        return stand_pat.clamp(alpha, beta);
    }

    // Draw detection (mirrors alpha_beta, needed when depth=0 during a
    // repetition cycle and quiescence is entered without going back through
    // alpha_beta's draw check).
    if ply > 0 {
        if board.halfmove_clock >= 100 {
            return draw_score(state, ply);
        }
        if board.is_twofold_in_search(state.game_ply) {
            return draw_score(state, ply);
        }
        if board.is_repetition() {
            return draw_score(state, ply);
        }
    }

    state.nodes += 1;

    if ply > state.seldepth {
        state.seldepth = ply;
    }

    // Cache bounds describe the incoming window, before stand-pat or a
    // searched move raises alpha.
    let original_alpha = alpha;
    let in_check = chess_core::is_in_check(board);

    // As in alpha_beta, PV nodes must establish their value on this path;
    // cached bounds only cut off null-window searches.
    let tt_entry = state.tt.probe(board.hash);
    if let Some(entry) = tt_entry
        && beta - alpha == 1
        && !in_check
    {
        let adj_score = score_from_tt(entry.score, ply);
        match entry.flag {
            TTFlag::Exact => return adj_score,
            TTFlag::LowerBound => {
                if adj_score >= beta {
                    return adj_score;
                }
            }
            TTFlag::UpperBound => {
                if adj_score <= alpha {
                    return adj_score;
                }
            }
        }
    }
    let tt_move = tt_entry.map(|e| e.best_move).unwrap_or(Move::NULL);

    // When in check: search ALL moves (not just captures), skip stand-pat.
    // Uses pseudo-legal generation with inline legality instead of cloning.
    if in_check {
        let pseudo = chess_core::generate_pseudo_legal_moves(board);
        let us = board.side_to_move;

        let mut best_score = Score::NEG_INF.0;
        let mut best_move = Move::NULL;
        let mut child_pv = PvLine::new();
        let mut any_legal = false;

        for &m in pseudo.iter() {
            let prev_castling = board.castling;
            let prev_ep = board.en_passant;
            let prev_halfmove = board.halfmove_clock;
            let captured = board.make_move(m);

            // Inline legality check
            if move_is_illegal(board, us) {
                board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                continue;
            }
            any_legal = true;

            if state.use_nnue && ply as usize + 1 < MAX_PLY {
                update_accumulator_for_move(
                    state,
                    board,
                    m,
                    captured,
                    prev_castling,
                    prev_ep,
                    ply as usize,
                    ply as usize + 1,
                );
            }

            let score = -quiescence(
                board,
                ply + 1,
                -beta,
                -alpha,
                &mut child_pv,
                state,
                stop,
                false,
            );

            board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

            if state.stop {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;

                if score > alpha {
                    alpha = score;
                    if score < beta {
                        pv.clear();
                        pv.push(m);
                        pv.extend_from_slice(&child_pv);
                    }

                    if score >= beta {
                        state.tt.store(
                            board.hash,
                            0,
                            score_to_tt(best_score, ply),
                            TTFlag::LowerBound,
                            best_move,
                            None,
                        );
                        return beta;
                    }
                }
            }
        }

        if !any_legal {
            // Checkmate
            return -Score::MATE.0 + ply as i32;
        }

        let flag = if best_score > original_alpha {
            TTFlag::Exact
        } else {
            TTFlag::UpperBound
        };
        state.tt.store(
            board.hash,
            0,
            score_to_tt(best_score, ply),
            flag,
            best_move,
            None,
        );
        return alpha;
    }

    // Not in check: normal qsearch with stand-pat
    let (unadjusted_eval, stand_pat) =
        evaluate_for_side(board, state, ply, tt_entry.and_then(|entry| entry.eval));

    if stand_pat >= beta {
        state.tt.store(
            board.hash,
            0,
            score_to_tt(stand_pat, ply),
            TTFlag::LowerBound,
            Move::NULL,
            unadjusted_eval,
        );
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Delta pruning
    let delta = PieceKind::Queen.value() + 200;
    if stand_pat + delta < alpha {
        return alpha;
    }

    // Generate only captures+promotions (no quiet moves at all)
    let captures_list = chess_core::generate_pseudo_legal_captures(board);
    let us = board.side_to_move;

    // An empty capture list must still reach the quiet-check search below.
    // Score and order captures via selection sort
    let mut cap_moves = [Move::NULL; 128];
    let mut cap_scores = [0i32; 128];
    let mut cap_count = 0usize;
    for &m in captures_list.iter() {
        let score = if m == tt_move && !tt_move.is_null() {
            10_000_000
        } else {
            capture_value(board, m)
        };
        cap_moves[cap_count] = m;
        cap_scores[cap_count] = score;
        cap_count += 1;
        if cap_count >= 128 {
            break;
        }
    }

    let mut best_score = stand_pat;
    let mut best_move = Move::NULL;
    let mut child_pv = PvLine::new();

    let mut idx = 0;
    while idx < cap_count {
        pick_next_move(
            &mut cap_moves[..cap_count],
            &mut cap_scores[..cap_count],
            idx,
        );
        let m = cap_moves[idx];
        idx += 1;

        if !see::see_ge(board, m, 0) {
            continue;
        }

        let prev_castling = board.castling;
        let prev_ep = board.en_passant;
        let prev_halfmove = board.halfmove_clock;
        let captured = board.make_move(m);

        // Inline legality check
        if move_is_illegal(board, us) {
            board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
            continue;
        }

        if state.use_nnue && ply as usize + 1 < MAX_PLY {
            update_accumulator_for_move(
                state,
                board,
                m,
                captured,
                prev_castling,
                prev_ep,
                ply as usize,
                ply as usize + 1,
            );
        }

        let score = -quiescence(
            board,
            ply + 1,
            -beta,
            -alpha,
            &mut child_pv,
            state,
            stop,
            false,
        );

        board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

        if state.stop {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = m;
        }

        if score >= beta {
            state.tt.store(
                board.hash,
                0,
                score_to_tt(best_score, ply),
                TTFlag::LowerBound,
                best_move,
                unadjusted_eval,
            );
            return beta;
        }
        if score > alpha {
            alpha = score;
            if score < beta {
                pv.clear();
                pv.push(m);
                pv.extend_from_slice(&child_pv);
            }
        }
    }

    // Quiet moves that give check — only at the first qsearch ply to control
    // branching factor. SEE >= 0 guards against searching moves that hang the piece.
    if generate_checks {
        let quiets = chess_core::generate_pseudo_legal_quiets(board);
        for &m in quiets.iter() {
            if !see::see_ge(board, m, 0) {
                continue;
            }

            let prev_castling = board.castling;
            let prev_ep = board.en_passant;
            let prev_halfmove = board.halfmove_clock;
            let captured = board.make_move(m);

            if move_is_illegal(board, us) {
                board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                continue;
            }

            if !chess_core::is_in_check(board) {
                board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                continue;
            }

            if state.use_nnue && ply as usize + 1 < MAX_PLY {
                update_accumulator_for_move(
                    state,
                    board,
                    m,
                    captured,
                    prev_castling,
                    prev_ep,
                    ply as usize,
                    ply as usize + 1,
                );
            }

            let score = -quiescence(
                board,
                ply + 1,
                -beta,
                -alpha,
                &mut child_pv,
                state,
                stop,
                false,
            );

            board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

            if state.stop {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
            }

            if score >= beta {
                state.tt.store(
                    board.hash,
                    0,
                    score_to_tt(best_score, ply),
                    TTFlag::LowerBound,
                    best_move,
                    unadjusted_eval,
                );
                return beta;
            }
            if score > alpha {
                alpha = score;
                if score < beta {
                    pv.clear();
                    pv.push(m);
                    pv.extend_from_slice(&child_pv);
                }
            }
        }
    }

    if !best_move.is_null() {
        let flag = if best_score > original_alpha {
            TTFlag::Exact
        } else {
            TTFlag::UpperBound
        };
        state.tt.store(
            board.hash,
            0,
            score_to_tt(best_score, ply),
            flag,
            best_move,
            unadjusted_eval,
        );
    }

    alpha
}

fn capture_value(board: &Board, m: Move) -> i32 {
    let flag = m.flag();
    let victim = if flag == MoveFlag::EnPassant {
        PieceKind::Pawn.value()
    } else {
        board
            .piece_at(m.to_sq())
            .map(|p| p.kind.value())
            .unwrap_or(0)
    };
    let attacker = board
        .piece_at(m.from_sq())
        .map(|p| p.kind.value())
        .unwrap_or(0);
    victim * 10 - attacker
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Static eval from the side to move's view, as `(unadjusted, eval)`:
/// `unadjusted` is the net's (or HCE's) score before the learned correction,
/// for the TT to cache, and `cached` is such a score from the TT, which saves
/// the evaluation. `unadjusted` is `None` for the HCE fallback past `MAX_PLY`,
/// which must not stand in for the net's score.
fn evaluate_for_side(
    board: &Board,
    state: &mut SearchState,
    ply: u8,
    cached: Option<i32>,
) -> (Option<i32>, i32) {
    // Hand-crafted endings (K+B+N vs K, K+Q vs K+R, dead draws) replace the net:
    // it has no gradient there without tablebases (see endgame.rs).
    if let Some(eval) = crate::endgame::evaluate(board) {
        return (Some(eval), eval);
    }
    let use_net = state.use_nnue && (ply as usize) < MAX_PLY;
    let eval = if let Some(eval) = cached.filter(|_| use_net || !state.use_nnue) {
        eval
    } else if use_net {
        // The moves since the last evaluated ancestor are applied only now; the
        // board is at position P_ply, so a king-move refresh is valid here.
        let acc = state
            .accumulators
            .get(ply as usize, board, &state.net, &mut state.refresh_cache);
        // nnue_evaluate returns score from side-to-move perspective. No HCE
        // endgame scaling on top: trained on WDL-blended targets, the net already
        // prices drawish endings, and a second hand-tuned discount double-counts.
        let piece_count = (board.occupancy[0].0 | board.occupancy[1].0).count_ones();
        nnue_evaluate(acc, board.side_to_move, &state.net, piece_count)
    } else {
        let e = evaluate(board).0;
        match board.side_to_move {
            Color::White => e,
            Color::Black => -e,
        }
    };
    let unadjusted = (use_net || !state.use_nnue).then_some(eval);
    // Nudge the static eval by the learned correction for this position.
    (unadjusted, eval + state.correction(board, ply))
}

/// Record a move from `src_ply` to `dst_ply` on the accumulator stack; the
/// accumulator itself is updated when `dst_ply` (or a descendant) is evaluated.
/// Must be called AFTER `make_move` (board reflects the new position).
/// The `captured` result from `make_move` is used to detect captures.
#[allow(clippy::too_many_arguments)]
fn update_accumulator_for_move(
    state: &mut SearchState,
    board: &Board,
    m: Move,
    captured: Option<chess_common::Piece>,
    previous_castling: chess_common::CastlingRights,
    previous_en_passant: Option<Square>,
    src_ply: usize,
    dst_ply: usize,
) {
    let flag = m.flag();
    let from = m.from_sq();
    let to = m.to_sq();
    // The moving piece (side that already moved; board.side_to_move is now the opponent)
    let us = board.side_to_move.opposite();

    // Determine the piece kind (needed to detect king moves before branching)
    let moving_kind = if flag.is_promotion() {
        PieceKind::Pawn
    } else if flag == MoveFlag::KingsideCastle || flag == MoveFlag::QueensideCastle {
        PieceKind::King
    } else {
        board
            .piece_at(to)
            .map(|p| p.kind)
            .unwrap_or(PieceKind::Pawn)
    };

    // Pieces the move removes and adds. Castling also moves a rook: the
    // opponent perspective stays clean across a king move and needs it.
    let mut pieces = PieceDelta::new();
    pieces.remove(us, moving_kind, from);
    if let Some(cap) = captured {
        // En passant: the captured pawn is on the file of `to`, rank of `from`
        let cap_sq = if flag == MoveFlag::EnPassant {
            Square::new(to.file(), from.rank())
        } else {
            to
        };
        pieces.remove(cap.color, cap.kind, cap_sq);
    }
    pieces.add(us, flag.promotion_piece().unwrap_or(moving_kind), to);
    let rook_move = match (flag, us) {
        (MoveFlag::KingsideCastle, Color::White) => Some((Square::H1, Square::F1)),
        (MoveFlag::KingsideCastle, Color::Black) => Some((Square::H8, Square::F8)),
        (MoveFlag::QueensideCastle, Color::White) => Some((Square::A1, Square::D1)),
        (MoveFlag::QueensideCastle, Color::Black) => Some((Square::A8, Square::D8)),
        _ => None,
    };
    if let Some((rook_from, rook_to)) = rook_move {
        pieces.remove(us, PieceKind::Rook, rook_from);
        pieces.add(us, PieceKind::Rook, rook_to);
    }

    debug_assert_eq!(src_ply + 1, dst_ply);
    state.accumulators.push(
        dst_ply,
        (moving_kind == PieceKind::King).then_some(us),
        MoveUpdate {
            pieces,
            white_king: board.king_square(Color::White),
            black_king: board.king_square(Color::Black),
            old_castling: previous_castling,
            old_en_passant: previous_en_passant,
            new_castling: board.castling,
            new_en_passant: board.en_passant,
        },
    );
}

/// A null move changes no pieces, but it clears the en-passant square. Must be
/// called AFTER the null move is made (`board.en_passant` already cleared).
fn update_accumulator_for_null_move(
    state: &mut SearchState,
    board: &Board,
    previous_en_passant: Option<Square>,
    ply: usize,
) {
    state.accumulators.push(
        ply + 1,
        None,
        MoveUpdate {
            pieces: PieceDelta::new(),
            white_king: board.king_square(Color::White),
            black_king: board.king_square(Color::Black),
            old_castling: board.castling,
            old_en_passant: previous_en_passant,
            new_castling: board.castling,
            new_en_passant: board.en_passant,
        },
    );
}

/// Truncate a PV at the first move that is not legal in the given position.
/// Stale TT entries can cause the deeper moves in a PV to be illegal.
fn sanitize_pv(board: &Board, pv: &[Move]) -> Vec<Move> {
    let mut b = board.clone();
    let mut result = Vec::with_capacity(pv.len());
    for &m in pv {
        let legal = chess_core::generate_legal_moves(&b);
        if !legal.as_slice().contains(&m) {
            break;
        }
        b.make_move(m);
        // Stop at a forced draw: the game ends here regardless of what the
        // rest of the PV claims. Drop the draw-creating move as well — GUIs
        // (fastchess) warn when the PV includes the move that triggers the
        // threefold or 50-move rule, because nothing after it can legally
        // be played.
        if b.halfmove_clock >= 100 || b.is_repetition() {
            break;
        }
        result.push(m);
    }
    result
}

fn has_non_pawn_material(board: &Board, color: Color) -> bool {
    let ci = color.index();
    !board.pieces[ci][PieceKind::Knight.index()].is_empty()
        || !board.pieces[ci][PieceKind::Bishop.index()].is_empty()
        || !board.pieces[ci][PieceKind::Rook.index()].is_empty()
        || !board.pieces[ci][PieceKind::Queen.index()].is_empty()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syzygy::SyzygyTB;
    use crate::tt::SharedTT;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    fn with_qsearch_state(f: impl FnOnce(&mut SearchState<'_>) + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let tt = Arc::new(SharedTT::new(1));
                let mut learning = PersistentHistory::new();
                let mut state = SearchState::new(
                    None,
                    None,
                    false,
                    0,
                    0,
                    None,
                    0,
                    0,
                    0,
                    &tt,
                    false,
                    NnueNetwork::embedded(),
                    None,
                    TuneParams::default(),
                    &mut learning,
                );
                f(&mut state);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    // Exercise cache semantics independently of the embedded network and search
    // heuristics. A fail-low result is only an upper bound, even if a capture
    // improved on stand-pat; a PV search must establish its own value.
    #[test]
    fn quiescence_fail_low_capture_stores_upper_bound() {
        with_qsearch_state(|state| {
            let mut board = Board::from_fen("7k/8/8/4p3/3P4/8/8/K7 w - - 0 1").unwrap();
            let alpha = evaluate_for_side(&board, state, 0, None).1 + 500;
            let mut pv = PvLine::new();
            let stop = AtomicBool::new(false);
            let score = quiescence(
                &mut board,
                0,
                alpha,
                alpha + 1,
                &mut pv,
                state,
                &stop,
                false,
            );
            assert_eq!(score, alpha);
            let entry = state.tt.probe(board.hash).expect("capture must be cached");
            assert_eq!(entry.flag, TTFlag::UpperBound);
        });
    }

    #[test]
    fn quiescence_in_window_evasion_stores_exact() {
        with_qsearch_state(|state| {
            let mut board = Board::from_fen("7k/8/8/8/8/8/r7/K7 w - - 0 1").unwrap();
            let mut pv = PvLine::new();
            let stop = AtomicBool::new(false);
            let score = quiescence(&mut board, 0, -10000, 10000, &mut pv, state, &stop, false);
            let entry = state.tt.probe(board.hash).expect("evasion must be cached");
            assert_eq!(entry.flag, TTFlag::Exact);
            assert_eq!(entry.score, score);
        });
    }

    #[test]
    fn quiescence_pv_search_does_not_cut_off_on_cached_score() {
        with_qsearch_state(|state| {
            let mut board = Board::from_fen("7k/8/8/4p3/3P4/8/8/K7 w - - 0 1").unwrap();
            let mut pv = PvLine::new();
            let stop = AtomicBool::new(false);
            state
                .tt
                .store(board.hash, 10, 7777, TTFlag::Exact, Move::NULL, None);
            assert_eq!(
                quiescence(&mut board, 0, 0, 1, &mut pv, state, &stop, false),
                7777
            );
            assert_ne!(
                quiescence(&mut board, 0, -10000, 10000, &mut pv, state, &stop, false),
                7777
            );
        });
    }

    #[test]
    fn quiescence_searches_quiet_checks_without_captures() {
        with_qsearch_state(|state| {
            let mut board = Board::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
            assert!(chess_core::generate_pseudo_legal_captures(&board).is_empty());
            let mut pv = PvLine::new();
            let stop = AtomicBool::new(false);
            let score = quiescence(&mut board, 0, -32000, 32000, &mut pv, state, &stop, false);
            assert!(score < MATE_THRESHOLD);
            let score = quiescence(&mut board, 0, -32000, 32000, &mut pv, state, &stop, true);
            assert_eq!(score, Score::MATE.0 - 1);
            assert!(!pv.is_empty());
            board.make_move(pv[0]);
            assert!(chess_core::is_in_check(&board));
            assert!(chess_core::generate_legal_moves(&board).is_empty());
        });
    }

    /// Mythos–Endspiel, 2026-09-18, after 47.Ke2. With singular extension
    /// off (forced below), 25M nodes still returns 47...Rh1; Rd8 comes at ~30.3M (10M before
    /// the null-move en-passant accumulator fix, 15M before the u8 L1 input).
    #[test]
    #[cfg(not(debug_assertions))]
    fn mythos_endgame_finds_rd8_with_bounded_nodes() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let board =
                    Board::from_fen("6k1/4q2p/2p1P1p1/1pP1Q3/2B2P2/6P1/4K2P/3r4 b - - 4 47")
                        .unwrap();
                let net = NnueNetwork::embedded();
                let stop = Arc::new(AtomicBool::new(false));
                let tt = Arc::new(SharedTT::new(64));
                let params = SearchParams {
                    max_depth: 32,
                    max_nodes: Some(40_000_000),
                    use_nnue: true,
                    contempt: 8,
                    singular_ext_mode: 0,
                    ..Default::default()
                };
                let result = iterative_deepening(
                    &board, &params, &stop, &tt, None, 0, &net, None, None, None, None, None,
                );
                assert_eq!(result.best_move.to_uci(), "d1d8");
                // Stop checks are amortized in blocks of 1024 nodes.
                assert!(result.nodes <= 40_001_024, "{} nodes", result.nodes);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn search_moves_restrict_the_root() {
        // A full search needs more than Windows' 1 MB default test stack.
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let board = Board::starting_position();
                let net = NnueNetwork::embedded();
                let search = |moves: &[&str]| {
                    let params = SearchParams {
                        max_depth: 6,
                        search_moves: moves.iter().map(|m| Move::from_uci(m).unwrap()).collect(),
                        ..Default::default()
                    };
                    let stop = Arc::new(AtomicBool::new(false));
                    let tt = Arc::new(SharedTT::new(16));
                    iterative_deepening(
                        &board, &params, &stop, &tt, None, 0, &net, None, None, None, None, None,
                    )
                };
                // one move is searched fully, not through the single-legal-move shortcut
                let only = search(&["a2a3"]);
                assert_eq!(only.best_move.to_uci(), "a2a3");
                assert_eq!(only.depth, 6);
                let pair = search(&["a2a3", "h2h3"]);
                assert!(["a2a3", "h2h3"].contains(&pair.best_move.to_uci().as_str()));
                // no legal move listed: search everything instead of returning none
                assert!(!search(&["e2e5"]).best_move.is_null());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    fn nnue_test_state<'a>(
        net: &Arc<NnueNetwork>,
        tt: &Arc<SharedTT>,
        history: &'a mut PersistentHistory,
    ) -> SearchState<'a> {
        SearchState::new(
            None,
            None,
            false,
            0,
            0,
            None,
            0,
            0,
            0,
            tt,
            true,
            Arc::clone(net),
            None,
            TuneParams::default(),
            history,
        )
    }

    /// The stack's accumulator at `ply` equals a full refresh of `board`.
    fn assert_stack_matches_refresh(state: &mut SearchState, board: &Board, ply: usize) {
        let net = Arc::clone(&state.net);
        let acc = state
            .accumulators
            .get(ply, board, &net, &mut state.refresh_cache);
        let mut expected = chess_nnue::Accumulator::new();
        expected.refresh(board, &net);
        assert_eq!(acc.white, expected.white, "{}", board.to_fen());
        assert_eq!(acc.black, expected.black, "{}", board.to_fen());
    }

    fn make_uci_move(state: &mut SearchState, board: &mut Board, uci: &str, ply: usize) {
        let legal = chess_core::generate_legal_moves(board);
        let m = legal
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("move {uci} is not legal in {}", board.to_fen()));
        let prev_castling = board.castling;
        let prev_ep = board.en_passant;
        let captured = board.make_move(m);
        update_accumulator_for_move(
            state,
            board,
            m,
            captured,
            prev_castling,
            prev_ep,
            ply,
            ply + 1,
        );
    }

    #[test]
    fn null_move_drops_the_en_passant_state_feature() {
        // After 1. e4 the en-passant square is e3; the null move clears it.
        let mut board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
                .expect("valid test FEN");
        let net = NnueNetwork::embedded();
        let tt = Arc::new(SharedTT::new(1));
        let mut history = PersistentHistory::new();
        let mut state = nnue_test_state(&net, &tt, &mut history);
        state.accumulators.refresh(0, &board, &net);

        let prev_ep = board.en_passant;
        board.side_to_move = board.side_to_move.opposite();
        board.en_passant = None;
        update_accumulator_for_null_move(&mut state, &board, prev_ep, 0);
        assert_stack_matches_refresh(&mut state, &board, 1);
    }

    #[test]
    fn king_moves_and_castling_match_refresh() {
        for (fen, uci) in [
            ("4k3/8/8/8/8/8/3R4/4K3 w - - 0 1", "e1f1"),
            ("4k3/3r4/8/8/8/8/8/4K3 b - - 0 1", "e8f8"),
            ("4k3/8/8/8/8/8/5p2/4K3 w - - 0 1", "e1f2"),
            ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1"),
            ("r3k3/8/8/8/8/8/8/4K3 b q - 0 1", "e8c8"),
        ] {
            let mut board = Board::from_fen(fen).expect("valid test FEN");
            let net = NnueNetwork::embedded();
            let tt = Arc::new(SharedTT::new(1));
            let mut history = PersistentHistory::new();
            let mut state = nnue_test_state(&net, &tt, &mut history);
            state.accumulators.refresh(0, &board, &net);
            make_uci_move(&mut state, &mut board, uci, 0);
            assert_stack_matches_refresh(&mut state, &board, 1);
        }
    }

    #[test]
    fn lazy_accumulators_match_refresh_across_unevaluated_plies() {
        // Castling, a capture, en passant, promotion, king moves and a null
        // move, with plies evaluated only now and then.
        let line = [
            "e2e4", "d7d5", "e4d5", "e7e5", "d5e6", "f8c5", "g1f3", "g8f6", "f1c4", "e8g8", "e1g1",
            "b8c6", "e6f7", "g8h8", "d2d3", "f8e8", "f7e8q",
        ];
        for evaluate_every in [1, 2, 3, 5, 100] {
            let mut board = Board::starting_position();
            let net = NnueNetwork::embedded();
            let tt = Arc::new(SharedTT::new(1));
            let mut history = PersistentHistory::new();
            let mut state = nnue_test_state(&net, &tt, &mut history);
            state.accumulators.refresh(0, &board, &net);
            let mut ply = 0;
            for (index, uci) in line.iter().enumerate() {
                make_uci_move(&mut state, &mut board, uci, ply);
                ply += 1;
                if index == 2 || index == 9 {
                    // a null-move branch; the line then continues from `ply`
                    let mut null = board.clone();
                    let prev_ep = null.en_passant;
                    null.side_to_move = null.side_to_move.opposite();
                    null.en_passant = None;
                    update_accumulator_for_null_move(&mut state, &null, prev_ep, ply);
                    assert_stack_matches_refresh(&mut state, &null, ply + 1);
                }
                if (index + 1) % evaluate_every == 0 {
                    assert_stack_matches_refresh(&mut state, &board, ply);
                }
            }
            assert_stack_matches_refresh(&mut state, &board, ply);
        }
    }

    #[test]
    fn long_game_cap_bounds_blitz_and_rapid_no_increment() {
        // 180+0: an over-large target is capped to time/34.
        assert_eq!(long_game_time_cap(15_000, 0, 180_000, 0), 180_000 / 34);
        // At ten minutes the phase-aware classical reserve is fully active.
        assert_eq!(long_game_time_cap(40_000, 0, 600_000, 0), 600_000 / 20);
        // 10+0 keeps it down to 400 s; 5+0 and 3+0 start at the flag-safe cap.
        assert_eq!(long_game_time_cap(40_000, 0, 450_000, 0), 450_000 / 20);
        assert_eq!(long_game_time_cap(40_000, 0, 300_000, 0), 300_000 / 34);
    }

    #[test]
    fn long_clock_opening_ramps_up_the_soft_target() {
        // Same clock and material at every ply: only the ramp differs.
        let soft_at = |ply: usize, time: u64, inc: u64| {
            let mut board = Board::starting_position();
            board.position_history = vec![0; ply];
            let params = SearchParams {
                white_time_ms: Some(time),
                black_time_ms: Some(time),
                white_inc_ms: Some(inc),
                black_inc_ms: Some(inc),
                ..Default::default()
            };
            compute_time_limit(&params, &board).0.expect("timed")
        };
        for inc in [0, 5_000] {
            let (p2, p10, p20) = (
                soft_at(2, 600_000, inc),
                soft_at(10, 600_000, inc),
                soft_at(20, 600_000, inc),
            );
            // 600+5, ply 2: 30 s before the ramp; now ~16 s.
            assert!(p2 < p10 && p10 < p20, "600+{inc}: {p2} / {p10} / {p20}");
            assert!(p2 <= 20_000, "600+{inc}: {p2}ms at ply 2");
        }
        // Blitz clocks keep the full target.
        assert_eq!(opening_ramp_permille(2, 180_000, false), 1000);
        assert_eq!(opening_ramp_permille(2, 300_000, false), 1000);
        assert_eq!(opening_ramp_permille(2, 600_000, true), 1000);
        assert_eq!(opening_ramp_permille(0, 600_000, false), 400);
        assert_eq!(opening_ramp_permille(20, 600_000, false), 1000);
    }

    #[test]
    fn ten_minute_sudden_death_spends_the_clock_without_flagging() {
        // 600+0 (QQHtFZaq, 5bjcnIB6): at time/30 the bot finished with 187 s and
        // 248 s of 600 unused. Spending the soft target on every move must now
        // reach move 20 with under 300 s left (the old cap: ~325 s), and no hard
        // limit may ever reach the clock over a 150-move game.
        let mut time = 600_000u64;
        for m in 1u64..=150 {
            let mut board = Board::starting_position();
            board.position_history = vec![0; (2 * (m - 1)) as usize];
            let params = SearchParams {
                white_time_ms: Some(time),
                black_time_ms: Some(time),
                ..Default::default()
            };
            let (soft, hard, _, _, _) = compute_time_limit(&params, &board);
            let (soft, hard) = (soft.expect("timed"), hard.expect("timed"));
            assert!(
                hard < time,
                "move {m}: hard {hard}ms with {time}ms left flags"
            );
            time -= soft;
            if m == 20 {
                assert!(
                    time < 300_000,
                    "{time}ms of 600000 still unused after 20 moves"
                );
            }
        }
        assert!(time > 0);
    }

    #[test]
    fn long_game_cap_spends_more_of_the_bank_as_the_game_advances() {
        // The increment is free, so it is spendable above the bank slice.
        assert_eq!(
            long_game_time_cap(200_000, 5_000, 1_800_000, 0),
            5_000 + 1_800_000 / 20
        );
        assert_eq!(
            long_game_time_cap(200_000, 5_000, 1_800_000, 40),
            5_000 + 1_800_000 / 17
        );
        assert_eq!(
            long_game_time_cap(200_000, 5_000, 1_800_000, 80),
            5_000 + 1_800_000 / 14
        );
        assert_eq!(
            long_game_time_cap(200_000, 5_000, 1_800_000, 120),
            5_000 + 1_800_000 / 12
        );
        // Target already under the cap is returned unchanged.
        assert_eq!(long_game_time_cap(1_000, 0, 180_000, 0), 1_000);
        assert_eq!(long_game_time_cap(3_000, 2_000, 180_000, 0), 3_000);
    }

    #[test]
    fn long_game_cap_transitions_smoothly_above_the_blitz_ceiling() {
        // Bullet and blitz with an increment keep the tuned allocation. Capping
        // them starved the bot in live play: a 180+5 game ended with 179 of its
        // 180 seconds unspent, and 10+0.1 measured -28.3 Elo (see fn docs).
        assert_eq!(long_game_time_cap(610, 100, 10_000, 0), 610); // 10+0.1 bullet
        assert_eq!(long_game_time_cap(22_500, 5_000, 180_000, 0), 22_500); // 3+5 blitz
        assert_eq!(long_game_time_cap(30_000, 5_000, 300_000, 0), 30_000);
        // The old hard switch at five minutes is replaced by a blend. Halfway
        // through the transition, the result is halfway between the tuned
        // target and the phase-aware sustainable allocation.
        assert_eq!(
            long_game_time_cap(80_000, 5_000, 450_000, 80),
            (80_000 + (5_000 + 450_000 / 14)) / 2
        );
        assert_eq!(
            long_game_time_cap(80_000, 5_000, 600_000, 80),
            5_000 + 600_000 / 14
        );
        // Sudden death is capped at ANY clock — a flag there is fatal, and the
        // 180+0 mate-at-move-90 flag is what the cap originally fixed.
        assert_eq!(long_game_time_cap(610, 0, 10_000, 120), 10_000 / 34);
        assert_eq!(long_game_time_cap(15_000, 0, 180_000, 120), 180_000 / 34);
    }

    #[test]
    fn ponder_hit_fresh_time_is_phase_aware_and_clock_safe() {
        assert_eq!(ponder_fresh_time_ms(None, 1_800_000, 0), 0);
        assert_eq!(ponder_fresh_time_ms(Some(60_000), 300_000, 120), 0);
        assert_eq!(ponder_fresh_time_ms(Some(24_000), 1_800_000, 0), 6_000);
        assert_eq!(ponder_fresh_time_ms(Some(24_000), 1_800_000, 40), 8_000);
        assert_eq!(ponder_fresh_time_ms(Some(24_000), 1_800_000, 80), 12_000);
        assert_eq!(ponder_fresh_time_ms(Some(500), 1_800_000, 0), 1_000);
        assert_eq!(ponder_fresh_time_ms(Some(100_000), 1_800_000, 120), 12_000);
    }

    #[test]
    fn successful_ponder_hit_defers_an_expired_hard_limit_for_fresh_time() {
        let tt = Arc::new(SharedTT::new(1));
        let net = NnueNetwork::embedded();
        let mut history = PersistentHistory::new();
        let mut state = SearchState::new(
            Some(0),
            Some(24_000),
            true,
            5_000,
            1_800_000,
            None,
            80,
            0,
            0,
            &tt,
            true,
            net,
            None,
            TuneParams::default(),
            &mut history,
        );
        let ponder = Arc::new(AtomicBool::new(true));
        state.set_ponder(Some(Arc::clone(&ponder)));
        let stop = AtomicBool::new(false);

        assert!(
            !state.should_stop(&stop),
            "time stops are suspended in ponder"
        );
        ponder.store(false, Ordering::Relaxed);
        assert!(state.ponder_hit_at.is_none());
        assert!(
            !state.should_stop(&stop),
            "an expired ponder budget must not stop at the instant of the hit"
        );
        assert!(state.ponder_hit_at.is_some());

        state.ponder_hit_at =
            Instant::now().checked_sub(std::time::Duration::from_millis(state.ponder_fresh_ms + 1));
        assert!(
            state.should_stop(&stop),
            "the original hard limit applies after fresh time expires"
        );
    }

    #[test]
    fn shallow_time_manager_instability_decays_after_stable_depths() {
        let mut swing = update_recent_score_swing(0, 320);
        for _ in 0..3 {
            swing = update_recent_score_swing(swing, 0);
        }
        assert_eq!(swing, 40);

        let mut pv_changes = 0;
        for _ in 0..3 {
            pv_changes = update_recent_pv_changes(pv_changes, true);
        }
        for _ in 0..3 {
            pv_changes = update_recent_pv_changes(pv_changes, false);
        }
        assert_eq!(pv_changes, 0);
    }

    #[test]
    fn allocated_time_reserves_clock_in_no_increment_blitz() {
        // 180+0 opening: the long-game cap keeps the per-move slice well
        // below the old piece-count allocation (~time/24 + bank ≈ 9.4s),
        // leaving enough clock for a long game.
        let board = Board::starting_position();
        let params = SearchParams {
            white_time_ms: Some(180_000),
            black_time_ms: Some(180_000),
            white_inc_ms: Some(0),
            black_inc_ms: Some(0),
            ..Default::default()
        };
        let alloc = allocated_move_time_ms(&params, &board).expect("timed search");
        assert!(alloc > 0);
        assert!(
            alloc <= 180_000 / 30,
            "allocated {alloc} too high for 180+0"
        );
    }

    #[test]
    fn hard_limit_never_authorises_spending_more_clock_than_we_have() {
        // A hard limit is a number the search will actually spend, so it must be
        // bounded by the clock. `max`'s low-time branch (time/8 + inc*2) is not:
        // with 6s left and a 5s increment it reaches ~10.7s. Spending that flags,
        // and it did — 3 losses on time in 21 blitz games.
        let board = Board::starting_position();
        for time in [3_000u64, 6_000, 12_000, 30_000, 90_000, 180_000] {
            let params = SearchParams {
                white_time_ms: Some(time),
                black_time_ms: Some(time),
                white_inc_ms: Some(5_000),
                black_inc_ms: Some(5_000),
                ..Default::default()
            };
            let (_, hard, _, _, _) = compute_time_limit(&params, &board);
            let hard = hard.expect("timed");
            assert!(
                hard < time,
                "with {time}ms on the clock the hard limit is {hard}ms — that flags"
            );
        }
    }

    #[test]
    fn repeating_control_spends_the_period_without_flagging() {
        // CCRL 40/15: the clock is refilled every 40 moves. The sudden-death caps
        // (long-game cap, bank term) used to apply here too and left 27-39% of every
        // period unused. Spending the soft target on every move must now use most of
        // the period, while no single move's hard limit can take the clock to zero.
        for (period_ms, inc) in [(900_000u64, 0u64), (60_000, 0), (900_000, 2_000)] {
            let mut time = period_ms;
            for m in 1u64..=120 {
                let mtg = 40 - (m - 1) % 40;
                let mut board = Board::starting_position();
                board.position_history = vec![0; (2 * (m - 1)) as usize];
                let params = SearchParams {
                    white_time_ms: Some(time),
                    black_time_ms: Some(time),
                    white_inc_ms: Some(inc),
                    black_inc_ms: Some(inc),
                    moves_to_go: Some(mtg as u32),
                    ..Default::default()
                };
                let (soft, hard, _, _, _) = compute_time_limit(&params, &board);
                let (soft, hard) = (soft.expect("timed"), hard.expect("timed"));
                assert!(
                    hard < time,
                    "move {m}: hard {hard}ms with {time}ms left flags"
                );
                time = time - soft + inc;
                if mtg == 1 {
                    let unused = time.saturating_sub(inc * 40) * 100 / period_ms;
                    assert!(
                        unused <= 20,
                        "{period_ms}+{inc}: {unused}% of the period unused after move {m}"
                    );
                    time += period_ms;
                }
            }
        }
    }

    #[test]
    fn hard_limit_leaves_room_to_think_longer_than_the_soft_target() {
        // A difficult position must be able to outspend a typical one. Before the
        // soft/hard split these were the same number, so the search was killed at
        // 1.0x and every difficulty bonus in the ID loop was dead code.
        let board = Board::starting_position();
        let params = SearchParams {
            white_time_ms: Some(180_000),
            black_time_ms: Some(180_000),
            white_inc_ms: Some(5_000),
            black_inc_ms: Some(5_000),
            ..Default::default()
        };
        let (soft, hard, use_soft, _, _) = compute_time_limit(&params, &board);
        let (soft, hard) = (soft.expect("timed"), hard.expect("timed"));
        assert!(use_soft);
        assert!(
            hard > soft,
            "hard {hard} must exceed soft {soft} or difficulty bonuses cannot fire"
        );
        // ...but a single move may not eat the clock: the ceiling stays bounded by
        // the flag-safety max, with headroom for the 1.2x fail-low extension.
        assert!(
            hard <= soft * HARD_LIMIT_SOFT_MULT,
            "hard {hard} exceeds {HARD_LIMIT_SOFT_MULT}x soft {soft}"
        );
        let max = 180_000 / 6 + 5_000; // the `max` branch for 30s < time < 120s.. see fn
        assert!(
            hard.saturating_mul(6) / 5 <= max.max(hard),
            "fail-low extension off hard {hard} must stay within the safety max"
        );
    }

    #[test]
    fn chess960_skips_opening_time_throttle() {
        let board = Board::starting_position();
        let standard = SearchParams {
            white_time_ms: Some(1_800_000),
            black_time_ms: Some(1_800_000),
            white_inc_ms: Some(2_000),
            black_inc_ms: Some(2_000),
            move_overhead_ms: 200,
            ..Default::default()
        };
        let mut frc = standard.clone();
        frc.chess960 = true;

        let (std_soft, std_hard, _, _, _) = compute_time_limit(&standard, &board);
        let (frc_soft, frc_hard, _, _, _) = compute_time_limit(&frc, &board);
        let (std_soft, std_hard) = (std_soft.unwrap(), std_hard.unwrap());
        let (frc_soft, frc_hard) = (frc_soft.unwrap(), frc_hard.unwrap());

        // 1800+2 opening: long-game cap already binds the soft target, so the
        // win is that hard can exceed soft (difficulty bonuses can fire).
        assert!(frc_soft >= std_soft);
        assert!(
            frc_hard > std_hard,
            "chess960 hard {frc_hard} should exceed standard opening hard {std_hard}"
        );
        assert!(
            frc_hard > frc_soft,
            "chess960 hard {frc_hard} must exceed soft {frc_soft}"
        );

        let blitz = SearchParams {
            white_time_ms: Some(180_000),
            black_time_ms: Some(180_000),
            white_inc_ms: Some(2_000),
            black_inc_ms: Some(2_000),
            ..Default::default()
        };
        let mut blitz_frc = blitz.clone();
        blitz_frc.chess960 = true;
        let std_blitz = allocated_move_time_ms(&blitz, &board).unwrap();
        let frc_blitz = allocated_move_time_ms(&blitz_frc, &board).unwrap();
        assert!(
            frc_blitz > std_blitz,
            "chess960 blitz opening {frc_blitz} should exceed standard {std_blitz}"
        );

        assert_eq!(opening_time_adjust(0, 1_800_000, true), 0);
        assert_eq!(opening_time_adjust(0, 1_800_000, false), -160);
        assert_eq!(opening_time_adjust(40, 1_800_000, false), 0);
    }

    #[test]
    fn allocated_time_survives_a_long_endgame_on_a_small_increment() {
        // 1800+5 (60rqL48Y): a R+B+N endgame with 10:28 left used to allocate
        // ~72s — 11% of the clock every move, which drained 30:00 down to 1:00
        // by move 57. The clock must instead sustain a long conversion: spend
        // no more than the increment plus a small slice of the bank.
        let board =
            Board::from_fen("3r4/1pk1b1p1/2b2p2/2P5/3P1KP1/P7/1P6/2R5 b - - 1 30").expect("valid");
        let params = SearchParams {
            white_time_ms: Some(628_000),
            black_time_ms: Some(628_000),
            white_inc_ms: Some(5_000),
            black_inc_ms: Some(5_000),
            ..Default::default()
        };
        let alloc = allocated_move_time_ms(&params, &board).expect("timed search");
        assert!(alloc >= 5_000, "allocated {alloc} below the free increment");
        // The increment bank slice is time/20 early on (see `long_game_time_cap`):
        // ~31s here, under half the 72s that drained the clock.
        assert!(
            alloc <= 5_000 + 628_000 / 20,
            "allocated {alloc} drains the clock faster than the increment refunds it"
        );
    }

    /// Helper: run a fixed-depth search and return the result.
    ///
    /// Spawns a thread with a large stack so that debug builds (which have much
    /// bigger stack frames than release) don't overflow on deep searches.
    /// Each `alpha_beta` frame carries ~2 KB of fixed arrays (MovePicker buffers
    /// + history move lists); 128 levels × large debug frames exceeds the
    ///   ~1 MB Windows default stack.
    fn search_position(fen: &str, max_depth: u8) -> SearchResult {
        let fen = fen.to_owned();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let board = Board::from_fen(&fen).expect("valid FEN");
                let stop = Arc::new(AtomicBool::new(false));
                let tt = Arc::new(SharedTT::new(16)); // 16 MB
                let params = SearchParams {
                    max_depth,
                    use_nnue: false, // use HCE for tests (zeroed net is useless)
                    ..Default::default()
                };
                iterative_deepening(
                    &board,
                    &params,
                    &stop,
                    &tt,
                    None,
                    0,
                    &NnueNetwork::embedded(),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            })
            .expect("failed to spawn search thread")
            .join()
            .expect("search thread panicked")
    }

    /// Helper: run a search with a specific contempt value.
    fn search_with_contempt(fen: &str, max_depth: u8, contempt: i32) -> SearchResult {
        let fen = fen.to_owned();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let board = Board::from_fen(&fen).expect("valid FEN");
                let stop = Arc::new(AtomicBool::new(false));
                let tt = Arc::new(SharedTT::new(16));
                let params = SearchParams {
                    max_depth,
                    use_nnue: false,
                    contempt,
                    ..Default::default()
                };
                iterative_deepening(
                    &board,
                    &params,
                    &stop,
                    &tt,
                    None,
                    0,
                    &NnueNetwork::embedded(),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            })
            .expect("failed to spawn search thread")
            .join()
            .expect("search thread panicked")
    }

    fn syzygy_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/syzygy")
    }

    fn search_position_with_syzygy(
        fen: &str,
        max_depth: u8,
        syzygy_tb: Option<SyzygyTB>,
    ) -> SearchResult {
        let fen = fen.to_owned();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let board = Board::from_fen(&fen).expect("valid FEN");
                let stop = Arc::new(AtomicBool::new(false));
                let tt = Arc::new(SharedTT::new(16));
                let params = SearchParams {
                    max_depth,
                    use_nnue: false,
                    ..Default::default()
                };
                let root_tb_ranking = syzygy_tb
                    .as_ref()
                    .and_then(|tb| crate::syzygy::rank_root_moves(tb, &board));
                iterative_deepening(
                    &board,
                    &params,
                    &stop,
                    &tt,
                    None,
                    0,
                    &NnueNetwork::embedded(),
                    None,
                    syzygy_tb,
                    root_tb_ranking,
                    None,
                    None,
                )
            })
            .expect("failed to spawn search thread")
            .join()
            .expect("search thread panicked")
    }

    #[test]
    fn mate_in_one() {
        // White: Kg6, Re1. Black: Kg8. Re8# is mate in 1.
        let result = search_position("6k1/8/6K1/8/8/8/8/4R3 w - - 0 1", 4);
        assert!(
            result.score.is_mate(),
            "should detect mate, got {}",
            result.score
        );
        assert_eq!(result.best_move.to_uci(), "e1e8");
    }

    #[test]
    fn mate_in_two() {
        // White: Kf5, Ra1. Black: Kh8. Mate in 2: Kg6 Kg8 Ra8#
        let result = search_position("7k/8/8/5K2/8/8/8/R7 w - - 0 1", 8);
        assert!(
            result.score.is_mate(),
            "should detect mate, got {}",
            result.score
        );
        // Optimal first move is Kg6 (approaching the king)
        assert_eq!(result.best_move.to_uci(), "f5g6");
    }

    #[test]
    fn mate_distance_pruning_efficiency() {
        // Back-rank mate: 1k6/8/1K6/8/8/8/8/7R w - - 0 1
        // White: Kb6, Rh1. Black: Kb8. Rh8# is mate in 1.
        // With MDP, the search should break after finding mate at depth 2
        // and reach high depths rapidly because all non-mating branches
        // are pruned once the short mate is known.
        let result = search_position("1k6/8/1K6/8/8/8/8/7R w - - 0 1", 12);
        assert!(
            result.score.is_mate(),
            "should detect mate, got {}",
            result.score
        );
        assert_eq!(result.best_move.to_uci(), "h1h8");
        // With MDP, the search should have broken early (iterative deepening
        // exits on finding mate). Depth should be low (≤ 3).
        assert!(
            result.depth <= 3,
            "should break early on mate, searched to depth {}",
            result.depth
        );
    }

    #[test]
    fn avoids_perpetual_trap_from_lichess_r2kln7mg() {
        // lichess.org/R2KlN7mg: Q+K vs K+a4 after 68...Ke5 — must not play Qe7+
        // into a threefold repetition while believing it is mating.
        let _guard = crate::syzygy::syzygy_test_guard();
        let path = syzygy_path();
        if !path.exists() {
            return;
        }
        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let fen = "8/8/8/4k3/p6Q/K7/8/8 w - - 3 69";
        let result = search_position_with_syzygy(fen, 16, Some(tb));
        assert_ne!(
            result.best_move.to_uci(),
            "h4e7",
            "must not play Qe7+ into the perpetual, score={}",
            result.score
        );
        assert!(
            result.score.is_mate() && result.score.0 > 0 || result.score.0 > 200,
            "should find a winning continuation, got {}",
            result.score
        );
    }

    /// Build a board from startpos + a UCI move list (populating
    /// `position_history`, so the search sees the real game history) and run
    /// a fixed-depth search with tablebases, like `search_position_with_syzygy`.
    fn search_game_with_syzygy(
        moves_uci: &str,
        max_depth: u8,
        syzygy_tb: Option<SyzygyTB>,
        use_nnue: bool,
    ) -> SearchResult {
        let moves_uci = moves_uci.to_owned();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let mut board = Board::starting_position();
                for uci in moves_uci.split_whitespace() {
                    let parsed = Move::from_uci(uci).expect("valid UCI move");
                    let legal = chess_core::generate_legal_moves(&board)
                        .as_slice()
                        .iter()
                        .copied()
                        .find(|m| {
                            m.from_sq() == parsed.from_sq()
                                && m.to_sq() == parsed.to_sq()
                                && m.flag().promotion_piece() == parsed.flag().promotion_piece()
                        })
                        .unwrap_or_else(|| panic!("illegal move in game record: {uci}"));
                    board.make_move(legal);
                }
                let stop = Arc::new(AtomicBool::new(false));
                let tt = Arc::new(SharedTT::new(16));
                let params = SearchParams {
                    max_depth,
                    use_nnue,
                    ..Default::default()
                };
                let root_tb_ranking = syzygy_tb
                    .as_ref()
                    .and_then(|tb| crate::syzygy::rank_root_moves(tb, &board));
                iterative_deepening(
                    &board,
                    &params,
                    &stop,
                    &tt,
                    None,
                    0,
                    &NnueNetwork::embedded(),
                    None,
                    syzygy_tb,
                    root_tb_ranking,
                    None,
                    None,
                )
            })
            .expect("failed to spawn search thread")
            .join()
            .expect("search thread panicked")
    }

    #[test]
    fn converts_tb_win_instead_of_repeating_from_lichess_45gemouz() {
        // lichess.org/45geMoUz: TB-won K+N+2P vs K+B, drawn by threefold at
        // move 82. The losing moment was 81.Kh5?? — after 81...Be8+ the only
        // still-winning move (82.Kg4) completed the repetition, so by move 82
        // the win was gone. Three bugs conspired: interior WDL cutoffs at
        // hmc > 0 adjudicated lines above the repetition the defender was
        // forcing; a single game-history revisit was scored as a terminal
        // draw, flattening every winning line to noise; and cp noise could
        // reorder the DTZ-ranked root moves, so the engine shuffled its
        // repetition budget away (77.Kg4, 79.Kg4, 81.Kh5) instead of
        // converting. At move 81 the DTZ-progress move is Nf4 (dtz 16, fresh
        // child); the shuffles Kg4/Kh5 (dtz 20/22) must stay behind it.
        let _guard = crate::syzygy::syzygy_test_guard();
        let path = syzygy_path();
        if !path.exists() {
            return;
        }
        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        // Full game up to and including 80...Ba4 (White to move 81).
        let moves = "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 e2e3 e7e6 b1d2 b8d7 f1d3 f8d6 e1g1 e8g8 e3e4 \
                     e6e5 c4d5 c6d5 e4d5 e5d4 d2e4 f6e4 d3e4 d7f6 d1d4 f6e4 d4e4 f8e8 e4d4 c8f5 \
                     c1g5 f7f6 g5h4 e8e4 d4d3 f5g6 h4g3 d6g3 h2g3 e4e6 d3c4 e6d6 f3d4 a8c8 c4b3 \
                     g6e4 b3e3 e4d5 d4f5 d6e6 e3a7 g7g6 f5e3 e6a6 a7d4 d5a2 d4e4 a2f7 e4b7 a6a1 \
                     f1a1 c8b8 b7f3 g8g7 a1d1 d8c7 d1d2 c7c1 f3d1 c1d1 e3d1 f6f5 d1c3 g7f6 c3e2 \
                     f7c4 e2d4 c4d5 f2f3 f6e5 g1f2 e5d6 d4e2 d6c5 f2e3 d5c4 e3f4 c4b3 f4g5 b8b7 \
                     e2g1 c5c4 g5h6 c4c5 g1h3 b3g8 h3g5 c5c4 b2b3 c4b5 d2d8 g8b3 g5h7 b7e7 d8d2 \
                     b3f7 h7g5 f7e8 g5h3 b5c6 h3f4 e7e5 d2e2 e5e2 f4e2 c6d6 e2f4 d6e5 h6g5 e8a4 \
                     f4g6 e5e6 g6f4 e6e5 g3g4 f5g4 f3g4 e5e4 f4h5 a4d1 h5g3 e4e3 g3f5 e3f2 f5h4 \
                     f2g3 h4f5 g3f2 g2g3 f2f3 g5h4 f3e4 f5d6 e4e5 d6c4 e5f6 g4g5 f6g7 c4e3 d1a4 \
                     e3d5 g7h7 h4g4 a4d7 g4h5 d7e8 h5g4 e8d7 g4h4 d7a4";
        let result = search_game_with_syzygy(moves, 16, Some(tb), false);
        assert_ne!(
            result.best_move.to_uci(),
            "h4h5",
            "81.Kh5 forfeits the win: after Be8+ the only winning reply repeats, score={}",
            result.score
        );
        assert_eq!(
            result.best_move.to_uci(),
            "d5f4",
            "must play the DTZ-progress move Nf4 instead of shuffling, score={}",
            result.score
        );
    }

    #[test]
    fn tb_order_beats_cp_argmax_from_lichess_xp85knlq() {
        // lichess.org/xp85KNLq: TB-won KQvKR, converted only after the
        // halfmove clock hit 95/100. Every root move here preserves the win,
        // so their cp scores are pure noise — but the searched argmax (a
        // queen check) was displacing the DTZ head, and with no zeroing move
        // in a pawnless ending the subtree never sees a tablebase score, so
        // the engine shuffle-checked for 28 moves without progress. At move
        // 81 the DTZ head is Kf4 (dtz 43, fresh child, the probe's own
        // recommendation); the game played Qa7+ (dtz 49). The choice must
        // follow the tablebase order, not the cp ordering.
        let _guard = crate::syzygy::syzygy_test_guard();
        let path = syzygy_path();
        if !path.exists() {
            return;
        }
        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        // Full game up to and including 80...Rd5 (White to move 81).
        let moves = "d2d4 d7d5 c2c4 d5c4 e2e3 e7e6 f1c4 g8f6 g1f3 c7c5 e1g1 c5d4 e3d4 f8e7 b1c3 e8g8 f1e1 \
                     b8c6 a2a3 d8c7 c4a2 f8d8 c1e3 c8d7 d1c2 d7e8 a1d1 c7a5 d4d5 e6d5 a2b1 g7g6 h2h3 d8d6 \
                     b1a2 a8c8 d1d3 d6d7 c2d2 c8d8 b2b4 a5a3 c3d5 a3d3 d2d3 f6d5 e3g5 e7g5 f3g5 c6b4 d3a3 \
                     b4a2 a3a2 d7d6 a2a7 e8c6 e1d1 h7h6 g5e4 d6e6 f2f3 b7b5 a7d4 b5b4 d4d2 g8h7 d1c1 c6a8 \
                     h3h4 e6b6 h4h5 b4b3 e4c5 d8e8 c1a1 b3b2 a1b1 g6h5 b1b2 b6g6 d2d4 h5h4 g1h2 a8c6 d4h4 \
                     d5e3 h4f4 h7g7 f4d4 g7g8 c5d3 g6g5 d4a7 e3c4 b2b8 c4e5 b8e8 c6e8 d3f4 e8c6 f4e2 g5h5 \
                     h2g1 h5g5 g1f1 e5g6 e2c3 g6f8 f1f2 g5g6 a7e7 f8e6 c3e2 g6g5 e7f6 g5g6 f6e5 g6g5 e5b8 \
                     g8g7 b8d6 c6a4 f3f4 g5g4 g2g3 h6h5 e2d4 g7h7 d4e6 f7e6 d6e6 a4c2 f4f5 g4g7 f5f6 g7a7 \
                     f6f7 h7g7 f7f8q g7f8 e6c8 f8f7 c8c2 a7d7 g3g4 h5g4 c2c4 f7e7 c4g4 d7d6 g4b4 e7d7 b4b7 \
                     d7d8 f2f3 d6d1 b7b6 d8d7 f3g4 d1d5";
        let result = search_game_with_syzygy(moves, 16, Some(tb), true);
        assert_eq!(
            result.best_move.to_uci(),
            "g4f4",
            "must play the DTZ head Kf4, not the cp-argmax shuffle check, score={}",
            result.score
        );
    }

    fn search_fen_with_syzygy_threads(
        fen: &str,
        max_depth: u8,
        threads: usize,
        syzygy_tb: SyzygyTB,
        ranking: Option<crate::syzygy::RootTbRanking>,
        use_nnue: bool,
    ) -> SearchResult {
        let board = Board::from_fen(fen).expect("valid FEN");
        let pool = crate::threads::ThreadPool::new(threads);
        let stop = Arc::new(AtomicBool::new(false));
        let tt = Arc::new(SharedTT::new(16));
        let params = SearchParams {
            max_depth,
            use_nnue,
            ..Default::default()
        };
        pool.search(
            &board,
            &params,
            &stop,
            &tt,
            None,
            &NnueNetwork::embedded(),
            Some(syzygy_tb),
            ranking,
            None,
            None,
        )
    }

    #[test]
    fn converts_pawnless_tb_win_without_giving_material_away_lichess_071afrc4() {
        // lichess.org/071AfrC4: K+Q+R+N vs K, then K+R+N vs K. Ascending DTZ
        // used to pick 74...Qg3+ (DTZ 1, queen hang) and 78/82...Nf1+ (unique
        // DTZ 3, mate in 12). Material-preserving ranking must keep the queen
        // and play a mate-optimal king move at the later roots. Search is
        // sticky to that order below mate-finding depth, so both 1 and 4
        // threads have to agree on the property.
        let _guard = crate::syzygy::syzygy_test_guard();
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        const M72: &str = "8/8/8/1r6/8/8/2kp2K1/q7 b - - 1 72";
        const M74: &str = "8/6q1/8/1r6/7K/8/2k5/3n4 b - - 3 74";
        const M78: &str = "8/8/8/8/6r1/4n3/2k4K/8 b - - 6 78";
        const M82: &str = "8/8/8/8/6r1/4n3/7K/2k5 b - - 14 82";
        const M78_KINGS: &[&str] = &["c2c1", "c2c3", "c2d1", "c2d2", "c2d3"];
        const M82_KINGS: &[&str] = &["c1c2", "c1d1", "c1d2"];

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let rank = |fen: &str| {
            let board = Board::from_fen(fen).expect("valid FEN");
            crate::syzygy::rank_root_moves(&tb, &board)
        };
        let r72 = rank(M72);
        let r74 = rank(M74);
        let r78 = rank(M78);
        let r82 = rank(M82);
        assert_eq!(
            r72.as_ref()
                .and_then(|r| r.winning_moves.first())
                .map(|m| m.to_uci())
                .as_deref(),
            Some("d2d1q")
        );
        assert_ne!(
            r74.as_ref()
                .and_then(|r| r.winning_moves.first())
                .map(|m| m.to_uci())
                .as_deref(),
            Some("g7g3")
        );
        assert!(
            r78.as_ref()
                .and_then(|r| r.winning_moves.first())
                .is_some_and(|m| M78_KINGS.contains(&m.to_uci().as_str()))
        );
        assert!(
            r82.as_ref()
                .and_then(|r| r.winning_moves.first())
                .is_some_and(|m| M82_KINGS.contains(&m.to_uci().as_str()))
        );

        for threads in [1, 4] {
            let m72 =
                search_fen_with_syzygy_threads(M72, 16, threads, tb.clone(), r72.clone(), true);
            assert!(
                m72.score.is_mate() && m72.score.0 > 0,
                "m72 / {threads}t should prove a short mate, got {} ({})",
                m72.score,
                m72.best_move.to_uci()
            );

            let m74 =
                search_fen_with_syzygy_threads(M74, 16, threads, tb.clone(), r74.clone(), true);
            assert_ne!(
                m74.best_move.to_uci(),
                "g7g3",
                "m74 / {threads}t must not hang the queen, score={}",
                m74.score
            );
            assert!(
                m74.score.is_mate() && m74.score.0 > 0,
                "m74 / {threads}t should prove the mate-in-2, got {} ({})",
                m74.score,
                m74.best_move.to_uci()
            );

            let m78 =
                search_fen_with_syzygy_threads(M78, 16, threads, tb.clone(), r78.clone(), true);
            assert_ne!(
                m78.best_move.to_uci(),
                "e3f1",
                "m78 / {threads}t must not play Nf1+, score={}",
                m78.score
            );
            // One thread follows the DTZ order deterministically. With helpers,
            // a thread can prove a mate for another material-keeping win first
            // (82...Kb1, also DTZ 6) and a proven mate rightly outranks an
            // unproven win, so there only the win must be kept.
            if threads == 1 {
                assert!(
                    M78_KINGS.contains(&m78.best_move.to_uci().as_str()),
                    "m78 / 1t expected a mate-optimal king move, got {} score={}",
                    m78.best_move.to_uci(),
                    m78.score
                );
            } else {
                assert!(
                    r78.as_ref()
                        .is_some_and(|r| r.winning_moves.contains(&m78.best_move)),
                    "m78 / {threads}t must keep the TB win, got {} score={}",
                    m78.best_move.to_uci(),
                    m78.score
                );
            }

            let m82 =
                search_fen_with_syzygy_threads(M82, 16, threads, tb.clone(), r82.clone(), true);
            assert_ne!(
                m82.best_move.to_uci(),
                "e3f1",
                "m82 / {threads}t must not play Nf1+, score={}",
                m82.score
            );
            // One thread follows the DTZ order deterministically. With helpers,
            // a thread can prove a mate for another material-keeping win first
            // (82...Kb1, also DTZ 6) and a proven mate rightly outranks an
            // unproven win, so there only the win must be kept.
            if threads == 1 {
                assert!(
                    M82_KINGS.contains(&m82.best_move.to_uci().as_str()),
                    "m82 / 1t expected a mate-optimal king move, got {} score={}",
                    m82.best_move.to_uci(),
                    m82.score
                );
            } else {
                assert!(
                    r82.as_ref()
                        .is_some_and(|r| r.winning_moves.contains(&m82.best_move)),
                    "m82 / {threads}t must keep the TB win, got {} score={}",
                    m82.best_move.to_uci(),
                    m82.score
                );
            }
        }
    }

    #[test]
    fn game_position_move35() {
        // Position from the game after 35.Ke2 Kd7 — Black has Q+pawns vs 2R+pawns
        // with an advanced c-pawn. Should be clearly winning for Black.
        // Key: c7 pawn blocks the Rb7 from attacking Kd7.
        let fen = "8/1Rpk4/5p2/p6p/P1pP1p1P/2q2P1R/4K1P1/8 w - - 2 36";
        let board = Board::from_fen(fen).expect("valid FEN");
        // Verify board is correct (e2 = file 4, rank 1; d7 = file 3, rank 6)
        assert_eq!(board.king_square(Color::White), Square::new(4, 1));
        assert_eq!(board.king_square(Color::Black), Square::new(3, 6));
        // Search should find a strong advantage for Black (negative score = Black winning)
        let result = search_position(fen, 10);
        eprintln!(
            "move35: score={} best={} depth={} nodes={}",
            result.score,
            result.best_move.to_uci(),
            result.depth,
            result.nodes
        );
        assert!(
            result.score.0 < -100,
            "Black should be clearly winning, got score {}",
            result.score
        );
    }

    /// Contempt: draws are scored as -contempt from the root side's perspective,
    /// whichever side is to move at the drawn node.  K vs K with halfmove_clock=99 —
    /// every White king move increments the clock to 100, triggering the 50-move-rule
    /// check at ply=1 (Black to move), so the root sees score = -contempt.
    #[test]
    fn contempt_draw_score_fifty_move() {
        // K vs K, kings far apart, halfmove_clock = 99.
        // Any White king move is non-capture and non-pawn → clock becomes 100 at ply=1.
        let fen = "8/8/8/8/8/8/1K6/7k w - - 99 1";

        let with_contempt = search_with_contempt(fen, 1, 20);
        let no_contempt = search_with_contempt(fen, 1, 0);

        // All moves lead to a 50-move draw at ply=1, which costs the root side.
        assert_eq!(
            with_contempt.score.0, -20,
            "draw with contempt=20 should score -20 from White's perspective, got {}",
            with_contempt.score
        );
        assert_eq!(
            no_contempt.score.0, 0,
            "draw with contempt=0 should score 0, got {}",
            no_contempt.score
        );
        // Either way a legal move must be returned.
        assert!(!with_contempt.best_move.is_null());
        assert!(!no_contempt.best_move.is_null());
    }

    /// Stalemate is a draw like any other: with contempt it must cost the root
    /// side, not be scored 0 while repetitions and the 50-move rule cost
    /// `contempt` (the engine would otherwise steer into stalemates as the
    /// cheapest draw).
    #[test]
    fn contempt_applies_to_stalemate() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let tt = Arc::new(SharedTT::new(1));
                let mut learning = PersistentHistory::new();
                let mut state = SearchState::new(
                    None,
                    None,
                    false,
                    0,
                    0,
                    None,
                    0,
                    20,
                    0,
                    &tt,
                    false,
                    NnueNetwork::embedded(),
                    None,
                    TuneParams::default(),
                    &mut learning,
                );
                // Black to move is stalemated. At ply 1 the opponent of the root
                // is to move, so the draw is +contempt from its perspective.
                let mut board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
                let mut pv = PvLine::new();
                let stop = AtomicBool::new(false);
                let score = alpha_beta(
                    &mut board,
                    1,
                    1,
                    -32000,
                    32000,
                    &mut pv,
                    &mut state,
                    &stop,
                    Move::NULL,
                    Move::NULL,
                );
                assert_eq!(score, 20);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// Contempt + ply limit: an engine with non-zero contempt must not overflow the ply
    /// counter or write past the end of the accumulator array, even in positions with
    /// many extensions (perpetual checks, forced repetitions).  This test ensures the
    /// search completes without panicking.
    #[test]
    fn contempt_no_crash_near_ply_limit() {
        // A complex middlegame position that can produce long check sequences.
        // Searched at depth 20 so extensions can push ply well past 64.
        let fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5";
        // Must complete without panic; score and move are irrelevant.
        let result = search_with_contempt(fen, 20, 50);
        assert!(
            !result.best_move.is_null(),
            "engine must return a legal move"
        );
    }

    /// Performance test: engine must find Nf6+ (g4f6) in a complex tactical position.
    /// After 1.Nf6+! Kh8 (forced) 2.d8=Q# it is mate in 2.
    #[test]
    fn find_best_move_nf6_mate_in_two() {
        let result = search_position("8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1", 6);
        assert_eq!(
            result.best_move.to_uci(),
            "g4f6",
            "expected Nf6+ (g4f6), got {}",
            result.best_move.to_uci()
        );
    }

    #[test]
    fn syzygy_guides_precapture_transition_to_bxc6() {
        let _guard = crate::syzygy::syzygy_test_guard();
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let result =
            search_position_with_syzygy("b7/8/P1P5/6p1/3K1k2/8/8/8 b - - 0 53", 8, Some(tb));

        assert_eq!(result.best_move.to_uci(), "a8c6");
        assert!(
            result.score.0 >= crate::syzygy::TB_WIN_SCORE,
            "expected immediate TB win score, got {}",
            result.score.0
        );
    }

    #[test]
    fn syzygy_does_not_play_losing_ka8_in_kbppvkb_draw() {
        // lichess RBmggG2n: drawn 6-man KBPPvKB, live game played 202...Ka8
        // and was mated. With TB restriction the root must stay in {Bd8, Kb7}.
        let _guard = crate::syzygy::syzygy_test_guard();
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let result =
            search_position_with_syzygy("8/k1b5/8/PP6/K1B5/8/8/8 b - - 74 202", 8, Some(tb));
        let uci = result.best_move.to_uci();
        assert!(
            uci == "c7d8" || uci == "a7b7",
            "expected a drawing move (Bd8/Kb7), got {uci} score={}",
            result.score
        );
        assert!(
            !result.score.is_mate() || result.score.0 <= 0,
            "must not report a win after a TB-draw root, got {}",
            result.score
        );
    }
}
