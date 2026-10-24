//! Playing-strength limit for `UCI_LimitStrength` / `UCI_Elo`.
//!
//! Five classes: 1000 (beginner), 1500 (hobby), 2000 (club), 2500 (strong club)
//! and full strength. `UCI_Elo` snaps to the nearest class; [`ELO_MAX`] (the
//! option's maximum and default) means full strength.
//!
//! A limited search caps the nodes per move and widens MultiPV; the move is then
//! drawn from the final-depth lines with Stockfish's skill-level rule: each line
//! gets a push of `(weakness * (top - score) + delta * rand(0..weakness)) / 128`
//! and the line with the highest `score + push` is played. With weakness 0 that
//! is always the best line; as it nears 128 the score gaps all but vanish and
//! the random term decides, so weaker classes play worse moves more often.
//!
//! Calibrated 2026-10-24 (e4fbf56) against Stockfish 19 `UCI_LimitStrength` at
//! the same `UCI_Elo` (the 1000 class vs SF 1320), 60+0.6, 100 games per class
//! (200 for 1500): 1000 → ≈ 1050, 1500 → ≈ 1515, 2000 → ≈ 1955, 2500 → ≈ 2495.
//! The search tune left the hobby class at 500 nodes ≈ 1405, hence 750 nodes.
//! Near weakness 128 the class is very sensitive (beginner at 117 ≈ 1180,
//! 121 ≈ 965, 122 ≈ 890, 125 ≈ 500); the stronger classes are tuned mainly
//! through `max_nodes`.

/// Lowest `UCI_Elo`.
pub const ELO_MIN: i32 = 1000;
/// Highest `UCI_Elo` and its default: full strength.
pub const ELO_MAX: i32 = 3000;

/// Line scores are clamped to ±this before the draw (mates included), so a mate
/// line competes as a very good move, not an infinitely good one.
const SCORE_CLAMP: i32 = 2000;

/// One strength class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrengthLevel {
    pub elo: i32,
    pub name: &'static str,
    /// Node cap per move (combined with any `go nodes` limit, the smaller wins).
    pub max_nodes: u64,
    /// Lines searched for the draw.
    pub multi_pv: usize,
    /// 0..128: how much the draw ignores score differences.
    pub weakness: i32,
}

/// The limited classes, weakest first. Full strength has no entry.
pub const LEVELS: [StrengthLevel; 4] = [
    StrengthLevel {
        elo: 1000,
        name: "beginner",
        max_nodes: 150,
        multi_pv: 6,
        weakness: 121,
    },
    StrengthLevel {
        elo: 1500,
        name: "hobby",
        max_nodes: 750,
        multi_pv: 6,
        weakness: 85,
    },
    StrengthLevel {
        elo: 2000,
        name: "club",
        max_nodes: 2000,
        multi_pv: 4,
        weakness: 55,
    },
    StrengthLevel {
        elo: 2500,
        name: "strong club",
        max_nodes: 6000,
        multi_pv: 3,
        weakness: 30,
    },
];

/// The class for a `UCI_Elo` value: the nearest of 1000/1500/2000/2500/max
/// (ties go to the stronger class). `None` = full strength.
pub fn level_for_elo(elo: i32) -> Option<&'static StrengthLevel> {
    let elo = elo.clamp(ELO_MIN, ELO_MAX);
    let nearest = LEVELS
        .iter()
        .min_by_key(|l| ((l.elo - elo).abs(), -l.elo))
        .expect("LEVELS is not empty");
    // Full strength sits at ELO_MAX for the snap.
    if (ELO_MAX - elo) <= (nearest.elo - elo).abs() {
        None
    } else {
        Some(nearest)
    }
}

/// Index of the line to play. `scores` are the final-depth MultiPV scores in
/// centipawns, best first (mates already mapped to large values); `rand`
/// supplies uniform random numbers.
pub fn pick_line(scores: &[i32], weakness: i32, mut rand: impl FnMut() -> u64) -> usize {
    if scores.len() <= 1 || weakness <= 0 {
        return 0;
    }
    let clamped: Vec<i32> = scores
        .iter()
        .map(|s| (*s).clamp(-SCORE_CLAMP, SCORE_CLAMP))
        .collect();
    let top = *clamped.iter().max().expect("non-empty");
    let lowest = *clamped.iter().min().expect("non-empty");
    let delta = (top - lowest).min(100);
    let mut best = (i32::MIN, 0);
    for (i, &s) in clamped.iter().enumerate() {
        let noise = (rand() % weakness as u64) as i32;
        let push = (weakness * (top - s) + delta * noise) / 128;
        if s + push > best.0 {
            best = (s + push, i);
        }
    }
    best.1
}

/// Small xorshift64 generator seeded from the clock, for [`pick_line`].
pub struct Rng(u64);

impl Rng {
    pub fn from_time() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| (u64::from(d.subsec_nanos()) << 20) ^ d.as_secs())
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self(seed | 1)
    }

    pub fn with_seed(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_snaps_to_nearest_class() {
        assert_eq!(level_for_elo(0).map(|l| l.elo), Some(1000));
        assert_eq!(level_for_elo(1249).map(|l| l.elo), Some(1000));
        assert_eq!(level_for_elo(1250).map(|l| l.elo), Some(1500));
        assert_eq!(level_for_elo(2000).map(|l| l.elo), Some(2000));
        assert_eq!(level_for_elo(2700).map(|l| l.elo), Some(2500));
        assert_eq!(level_for_elo(2750), None);
        assert_eq!(level_for_elo(ELO_MAX), None);
        assert_eq!(level_for_elo(9999), None);
    }

    #[test]
    fn no_weakness_plays_the_best_line() {
        let mut rng = Rng::with_seed(7);
        for _ in 0..100 {
            assert_eq!(pick_line(&[50, 40, -300], 0, || rng.next_u64()), 0);
        }
    }

    #[test]
    fn weaker_classes_pick_worse_lines_more_often() {
        let scores = [80, 40, 0, -150, -400];
        let off_best = |weakness: i32| {
            let mut rng = Rng::with_seed(12345);
            (0..10_000)
                .filter(|_| pick_line(&scores, weakness, || rng.next_u64()) != 0)
                .count()
        };
        let counts: Vec<usize> = LEVELS.iter().rev().map(|l| off_best(l.weakness)).collect();
        assert!(
            counts.windows(2).all(|w| w[0] < w[1]),
            "off-best picks should grow from 2500 down to 1000: {counts:?}"
        );
    }

    #[test]
    fn strongest_class_varies_only_between_near_equal_moves() {
        let strong = LEVELS.iter().find(|l| l.elo == 2500).unwrap();
        let picks = |scores: &[i32]| {
            let mut rng = Rng::with_seed(4242);
            let mut seen = [0usize; 3];
            for _ in 0..10_000 {
                seen[pick_line(scores, strong.weakness, || rng.next_u64())] += 1;
            }
            seen
        };
        let near_equal = picks(&[50, 45, -300]);
        assert!(
            near_equal[1] > 0,
            "a 5 cp alternative should sometimes be played: {near_equal:?}"
        );
        assert_eq!(
            near_equal[2], 0,
            "a 350 cp blunder must not be played: {near_equal:?}"
        );
        let clear_best = picks(&[150, 40, -300]);
        assert_eq!(
            clear_best[0], 10_000,
            "a 110 cp better move should always be played: {clear_best:?}"
        );
    }

    #[test]
    fn strong_classes_do_not_drop_a_mate_for_a_quiet_move() {
        let mut rng = Rng::with_seed(99);
        let strong = LEVELS.iter().find(|l| l.elo == 2500).unwrap();
        for _ in 0..10_000 {
            assert_eq!(
                pick_line(&[30_000, 300, 100], strong.weakness, || rng.next_u64()),
                0
            );
        }
    }
}
