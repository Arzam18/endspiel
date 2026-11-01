//! Hand-crafted evaluation for endgames the net cannot convert on its own.
//!
//! Without tablebases the NNUE has no gradient in these endings: it knows
//! K+B+N vs K is "won" but not that the lone king must be driven into a corner
//! of the bishop's colour, and in K+Q vs K+R it has no notion of progress, so
//! the search shuffles into a repetition or the 50-move rule. It also scores
//! K+minor vs K as a piece up, so hanging a piece in K+B+N vs K looks harmless.
//! Measured on 2026-09-29 (TB off, 1 s/move vs Stockfish 19 + Syzygy, 12 random
//! TB-won positions each): K+B+N vs K 1/12 -> 12/12 mated, K+Q vs K+R 6/12 ->
//! 12/12; the other ten classic endings (KQK, KRK, KBBK, KPK, KQKP, Lucena- and
//! Philidor-type KRPKR, KRKB, KRKN) stayed 40/40 won or held.
//!
//! Tablebases take precedence: the root is ranked by DTZ and interior nodes
//! probe WDL after zeroing moves, before any static eval. These rules only
//! fill in where no probe answers (TB off, or hmc > 0 inside a conversion).
//!
//! Each rule returns a score for the side to move and replaces the net's value
//! for that exact material; everything else returns `None`. The scores stay
//! far below the tablebase (28 000) and mate (29 000+) bands.

use chess_common::{Bitboard, Board, Color, PieceKind, Square};

/// Base value of a known, forced win (the search still has to find the mate).
const KNOWN_WIN: i32 = 2_000;

const LIGHT_SQUARES: Bitboard = Bitboard(0x55AA_55AA_55AA_55AA);

/// A dead position (FIDE 5.2.2): no sequence of legal moves can end in mate,
/// so the game is drawn whatever is played — K vs K, K+minor vs K, and any
/// number of bishops that all stand on squares of one colour. The search
/// treats it as a terminal draw.
pub fn is_dead_position(board: &Board) -> bool {
    let both = |k: PieceKind| board.pieces[0][k.index()].0 | board.pieces[1][k.index()].0;
    if both(PieceKind::Pawn) | both(PieceKind::Rook) | both(PieceKind::Queen) != 0 {
        return false;
    }
    let knights = both(PieceKind::Knight);
    let bishops = both(PieceKind::Bishop);
    if (knights | bishops).count_ones() <= 1 {
        return true;
    }
    knights == 0 && (bishops & LIGHT_SQUARES.0 == 0 || bishops & !LIGHT_SQUARES.0 == 0)
}

/// Specialized evaluation from the side-to-move perspective, or `None` when the
/// material has no rule. Cheap: bails out on the piece count first.
pub fn evaluate(board: &Board) -> Option<i32> {
    let occupied = (board.occupancy[0].0 | board.occupancy[1].0).count_ones();
    if occupied > 5 {
        return None;
    }
    let white = Material::of(board, Color::White);
    let black = Material::of(board, Color::Black);

    // Drawn material: no pawns or majors, and neither side can force mate
    // (K+minor vs K+minor, K+N+N vs K). A theoretical draw, not one by the
    // rules (a mate is still possible if the defender errs), so it scores 0
    // and stays an evaluation. Below the root the search stops at the dead
    // positions among these (K vs K, K+minor vs K, same-coloured bishops)
    // before it evaluates them, see `is_dead_position`.
    if white.pawns + black.pawns == 0
        && white.majors() + black.majors() == 0
        && white.can_not_mate()
        && black.can_not_mate()
    {
        return Some(0);
    }

    let (strong, white_score) = if white.is_bn() && black.is_bare() {
        (Color::White, kbnk(board, Color::White))
    } else if black.is_bn() && white.is_bare() {
        (Color::Black, kbnk(board, Color::Black))
    } else if white.is_queen() && black.is_rook() {
        (Color::White, kqkr(board, Color::White))
    } else if black.is_queen() && white.is_rook() {
        (Color::Black, kqkr(board, Color::Black))
    } else {
        return None;
    };
    Some(if board.side_to_move == strong {
        white_score
    } else {
        -white_score
    })
}

#[derive(Clone, Copy)]
struct Material {
    pawns: u32,
    knights: u32,
    bishops: u32,
    rooks: u32,
    queens: u32,
}

impl Material {
    fn of(board: &Board, color: Color) -> Self {
        let p = &board.pieces[color.index()];
        let n = |k: PieceKind| p[k.index()].0.count_ones();
        Self {
            pawns: n(PieceKind::Pawn),
            knights: n(PieceKind::Knight),
            bishops: n(PieceKind::Bishop),
            rooks: n(PieceKind::Rook),
            queens: n(PieceKind::Queen),
        }
    }
    fn majors(self) -> u32 {
        self.rooks + self.queens
    }
    fn minors(self) -> u32 {
        self.knights + self.bishops
    }
    fn is_bare(self) -> bool {
        self.pawns + self.minors() + self.majors() == 0
    }
    /// At most one minor, or exactly two knights (K+N+N cannot force mate).
    fn can_not_mate(self) -> bool {
        self.minors() <= 1 || (self.knights == 2 && self.bishops == 0)
    }
    fn is_bn(self) -> bool {
        self.pawns + self.majors() == 0 && self.knights == 1 && self.bishops == 1
    }
    fn is_queen(self) -> bool {
        self.queens == 1 && self.pawns + self.rooks + self.minors() == 0
    }
    fn is_rook(self) -> bool {
        self.rooks == 1 && self.pawns + self.queens + self.minors() == 0
    }
}

fn chebyshev(a: Square, b: Square) -> i32 {
    a.file().abs_diff(b.file()).max(a.rank().abs_diff(b.rank())) as i32
}

/// Manhattan distance from the centre, 1 (centre) .. 7 (corner).
fn centre_distance(sq: Square) -> i32 {
    let f = sq.file() as i32;
    let r = sq.rank() as i32;
    ((2 * f - 7).abs() + (2 * r - 7).abs()) / 2
}

/// K+B+N vs K, from the strong side's view: mate is only possible in a corner of
/// the bishop's colour, so drive the lone king there and bring our king close.
fn kbnk(board: &Board, strong: Color) -> i32 {
    let weak = strong.opposite();
    let bishop = board.pieces[strong.index()][PieceKind::Bishop.index()]
        .iter()
        .next()
        .expect("K+B+N has a bishop");
    let dark_bishop = (bishop.file() + bishop.rank()).is_multiple_of(2); // a1 is dark
    let loser = board.king_square(weak);
    let r = loser.rank() as i32;
    // Mirror the file for a light bishop so the mating corners are always a1/h8.
    let f = if dark_bishop {
        loser.file() as i32
    } else {
        7 - loser.file() as i32
    };
    // 0 on the whole a8-h1 diagonal (the wrong corners AND the centre), 7 in the
    // mating corners. A distance-to-corner measure rates the centre above the
    // wrong corner, so a king hiding at h1 was never chased (Stockfish's shape).
    let push_to_corner = (7 - r - f).abs();
    let kings = chebyshev(board.king_square(strong), loser);
    KNOWN_WIN + 150 * push_to_corner + 20 * (7 - kings)
}

/// K+Q vs K+R, from the queen's side: a win in general, made by pushing the
/// defending king to the edge with ours close behind (Stockfish's KQKR rule).
fn kqkr(board: &Board, strong: Color) -> i32 {
    let weak = strong.opposite();
    let loser = board.king_square(weak);
    let kings = chebyshev(board.king_square(strong), loser);
    KNOWN_WIN / 2 + 40 * centre_distance(loser) + 20 * (7 - kings)
}

#[cfg(test)]
mod tests {
    use super::{evaluate, is_dead_position};
    use chess_common::Board;

    fn eval(fen: &str) -> Option<i32> {
        evaluate(&Board::from_fen(fen).expect("valid FEN"))
    }

    #[test]
    fn dead_draws_score_zero() {
        assert_eq!(eval("8/8/4k3/8/8/2B5/8/4K3 w - - 0 1"), Some(0)); // KBvK
        assert_eq!(eval("8/8/4k3/8/8/2N5/8/4K3 b - - 0 1"), Some(0)); // KNvK
        assert_eq!(eval("8/8/4k3/8/8/2NN4/8/4K3 w - - 0 1"), Some(0)); // KNNvK
        assert_eq!(eval("8/8/4k3/3b4/8/2N5/8/4K3 w - - 0 1"), Some(0)); // KNvKB
        assert_eq!(eval("8/8/4k3/8/8/2BB4/8/4K3 w - - 0 1"), None); // KBBvK wins
        assert_eq!(eval("8/8/4k3/8/8/2P5/8/4K3 w - - 0 1"), None); // pawns: net
    }

    #[test]
    fn dead_positions_are_only_those_without_any_mate() {
        let dead = |fen: &str| is_dead_position(&Board::from_fen(fen).expect("valid FEN"));
        assert!(dead("8/8/4k3/8/8/8/8/4K3 w - - 0 1")); // KvK
        assert!(dead("8/8/4k3/8/8/2B5/8/4K3 w - - 0 1")); // KBvK
        assert!(dead("8/8/4k3/8/8/2N5/8/4K3 b - - 0 1")); // KNvK
        assert!(dead("8/8/4k3/8/3b4/2B5/8/4K3 w - - 0 1")); // KBvKB, c3 + d4 both dark
        assert!(dead("8/8/4k3/8/3b4/2B5/1B6/4K3 w - - 0 1")); // KBBvKB, all dark
        assert!(!dead("8/8/4k3/3b4/8/2B5/8/4K3 w - - 0 1")); // KBvKB, c3 dark + d5 light
        assert!(!dead("8/8/4k3/3b4/8/2N5/8/4K3 w - - 0 1")); // KNvKB
        assert!(!dead("8/8/4k3/8/8/2NN4/8/4K3 w - - 0 1")); // KNNvK: helpmate exists
        assert!(!dead("8/8/4k3/8/8/2P5/8/4K3 w - - 0 1")); // KPvK
        assert!(!dead("8/8/4k3/8/8/2R5/8/4K3 w - - 0 1")); // KRvK
    }

    #[test]
    fn kbnk_prefers_the_bishop_coloured_corner() {
        // Dark-squared bishop (c1): a1/h8 are the mating corners, a8/h1 are not.
        let right = eval("8/8/8/8/8/8/2K5/k1B1N3 w - - 0 1").expect("KBNK rule");
        let wrong = eval("k7/8/2K5/8/8/8/8/2B1N3 w - - 0 1").expect("KBNK rule");
        assert!(right > wrong, "a1 ({right}) must beat a8 ({wrong})");
        // Same position, defender to move: the sign flips.
        let defender = eval("8/8/8/8/8/8/2K5/k1B1N3 b - - 0 1").expect("KBNK rule");
        assert_eq!(defender, -right);
    }

    #[test]
    fn kqkr_rewards_the_defending_king_on_the_edge() {
        let edge = eval("3k4/8/3K4/8/8/8/5r2/4Q3 w - - 0 1").expect("KQKR rule");
        let centre = eval("8/8/8/3k4/8/3K4/5r2/4Q3 w - - 0 1").expect("KQKR rule");
        assert!(edge > centre, "edge ({edge}) must beat centre ({centre})");
        assert!(eval("8/8/8/3k4/8/3K4/5R2/4q3 w - - 0 1").expect("KQKR rule") < 0);
    }
}
