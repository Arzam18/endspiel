use chess_common::{Board, CastlingRights, Color, PieceKind, Square};

use crate::HIDDEN_SIZE;
use crate::features::{board_state_feature_indices, feature_index, state_feature_indices};
#[cfg(target_arch = "x86_64")]
use crate::inference::{SimdBackend, simd_backend};
use crate::network::NnueNetwork;

#[inline]
fn add_row(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => return add_row_avx512(values, row),
            SimdBackend::Avx2 => return add_row_avx2(values, row),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    add_row_scalar(values, row);
}

#[inline]
fn sub_row(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => return sub_row_avx512(values, row),
            SimdBackend::Avx2 => return sub_row_avx2(values, row),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    sub_row_scalar(values, row);
}

#[inline]
fn replace_row(
    values: &mut [i16; HIDDEN_SIZE],
    old_row: &[i16; HIDDEN_SIZE],
    new_row: &[i16; HIDDEN_SIZE],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => {
                return replace_row_avx512(values, old_row, new_row);
            }
            SimdBackend::Avx2 => return replace_row_avx2(values, old_row, new_row),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    for ((value, &old), &new) in values.iter_mut().zip(old_row).zip(new_row) {
        *value += new - old;
    }
}

#[inline]
fn add_row_scalar(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
    for (value, &delta) in values.iter_mut().zip(row) {
        *value += delta;
    }
}

#[inline]
fn sub_row_scalar(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
    for (value, &delta) in values.iter_mut().zip(row) {
        *value -= delta;
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! simd_row_op {
    ($name:ident, $feature:literal, $vector:ty, $load:ident, $store:ident, $op:ident, $lanes:expr) => {
        #[target_feature(enable = $feature)]
        unsafe fn $name(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
            use std::arch::x86_64::*;
            unsafe {
                let mut i = 0;
                while i < HIDDEN_SIZE {
                    let lhs = $load(values.as_ptr().add(i) as *const $vector);
                    let rhs = $load(row.as_ptr().add(i) as *const $vector);
                    $store(values.as_mut_ptr().add(i) as *mut $vector, $op(lhs, rhs));
                    i += $lanes;
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
simd_row_op!(
    add_row_avx512,
    "avx512f,avx512bw",
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_add_epi16,
    32
);
#[cfg(target_arch = "x86_64")]
simd_row_op!(
    sub_row_avx512,
    "avx512f,avx512bw",
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_sub_epi16,
    32
);
#[cfg(target_arch = "x86_64")]
simd_row_op!(
    add_row_avx2,
    "avx2",
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_add_epi16,
    16
);
#[cfg(target_arch = "x86_64")]
simd_row_op!(
    sub_row_avx2,
    "avx2",
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_sub_epi16,
    16
);

#[cfg(target_arch = "x86_64")]
macro_rules! simd_replace_row {
    ($name:ident, $feature:literal, $vector:ty, $load:ident, $store:ident, $add:ident, $sub:ident, $lanes:expr) => {
        #[target_feature(enable = $feature)]
        unsafe fn $name(
            values: &mut [i16; HIDDEN_SIZE],
            old_row: &[i16; HIDDEN_SIZE],
            new_row: &[i16; HIDDEN_SIZE],
        ) {
            use std::arch::x86_64::*;
            unsafe {
                let mut i = 0;
                while i < HIDDEN_SIZE {
                    let value = $load(values.as_ptr().add(i) as *const $vector);
                    let old = $load(old_row.as_ptr().add(i) as *const $vector);
                    let new = $load(new_row.as_ptr().add(i) as *const $vector);
                    $store(
                        values.as_mut_ptr().add(i) as *mut $vector,
                        $add(value, $sub(new, old)),
                    );
                    i += $lanes;
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
simd_replace_row!(
    replace_row_avx512,
    "avx512f,avx512bw",
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_add_epi16,
    _mm512_sub_epi16,
    32
);
#[cfg(target_arch = "x86_64")]
simd_replace_row!(
    replace_row_avx2,
    "avx2",
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_add_epi16,
    _mm256_sub_epi16,
    16
);

/// Feature rows one move removes from and adds to a perspective: up to three
/// pieces (mover, captured piece, castling rook) plus three state features.
#[derive(Default)]
struct RowDelta {
    sub: [usize; 6],
    add: [usize; 6],
    n_sub: usize,
    n_add: usize,
}

impl RowDelta {
    #[inline]
    fn sub(&mut self, row: usize) {
        self.sub[self.n_sub] = row;
        self.n_sub += 1;
    }

    #[inline]
    fn add(&mut self, row: usize) {
        self.add[self.n_add] = row;
        self.n_add += 1;
    }
}

/// `dst = src - Σ sub rows + Σ add rows` in a single pass. Replaces a copy plus
/// one read-modify-write pass over the accumulator per changed feature; i16
/// lanes wrap, so the result is bit-identical to the sequential updates.
#[inline]
fn apply_rows(
    dst: &mut [i16; HIDDEN_SIZE],
    src: &[i16; HIDDEN_SIZE],
    net: &NnueNetwork,
    delta: &RowDelta,
) {
    let subs = &delta.sub[..delta.n_sub];
    let adds = &delta.add[..delta.n_add];
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => {
                return apply_rows_avx512(dst, src, net, subs, adds);
            }
            SimdBackend::Avx2 => return apply_rows_avx2(dst, src, net, subs, adds),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    apply_rows_scalar(dst, src, net, subs, adds);
}

#[inline]
fn apply_rows_scalar(
    dst: &mut [i16; HIDDEN_SIZE],
    src: &[i16; HIDDEN_SIZE],
    net: &NnueNetwork,
    subs: &[usize],
    adds: &[usize],
) {
    *dst = *src;
    for &row in subs {
        for (value, &delta) in dst.iter_mut().zip(&net.ft_weights[row]) {
            *value = value.wrapping_sub(delta);
        }
    }
    for &row in adds {
        for (value, &delta) in dst.iter_mut().zip(&net.ft_weights[row]) {
            *value = value.wrapping_add(delta);
        }
    }
}

/// Registers per tile: the tile stays in registers while every row is applied.
#[cfg(target_arch = "x86_64")]
const APPLY_TILE_REGS: usize = 8;

#[cfg(target_arch = "x86_64")]
macro_rules! simd_apply_rows {
    ($name:ident, $feature:literal, $vector:ty, $load:ident, $store:ident, $add:ident, $sub:ident, $lanes:expr) => {
        #[target_feature(enable = $feature)]
        unsafe fn $name(
            dst: &mut [i16; HIDDEN_SIZE],
            src: &[i16; HIDDEN_SIZE],
            net: &NnueNetwork,
            subs: &[usize],
            adds: &[usize],
        ) {
            use std::arch::x86_64::*;
            const TILE: usize = APPLY_TILE_REGS * $lanes;
            const _: () = assert!(HIDDEN_SIZE % TILE == 0);
            unsafe {
                let mut base = 0;
                while base < HIDDEN_SIZE {
                    let mut regs: [$vector; APPLY_TILE_REGS] = std::array::from_fn(|k| {
                        $load(src.as_ptr().add(base + k * $lanes) as *const $vector)
                    });
                    for &row in subs {
                        let row = net.ft_weights[row].as_ptr().add(base);
                        for (k, reg) in regs.iter_mut().enumerate() {
                            *reg = $sub(*reg, $load(row.add(k * $lanes) as *const $vector));
                        }
                    }
                    for &row in adds {
                        let row = net.ft_weights[row].as_ptr().add(base);
                        for (k, reg) in regs.iter_mut().enumerate() {
                            *reg = $add(*reg, $load(row.add(k * $lanes) as *const $vector));
                        }
                    }
                    for (k, reg) in regs.iter().enumerate() {
                        $store(
                            dst.as_mut_ptr().add(base + k * $lanes) as *mut $vector,
                            *reg,
                        );
                    }
                    base += TILE;
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
simd_apply_rows!(
    apply_rows_avx512,
    "avx512f,avx512bw",
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_add_epi16,
    _mm512_sub_epi16,
    32
);
#[cfg(target_arch = "x86_64")]
simd_apply_rows!(
    apply_rows_avx2,
    "avx2",
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_add_epi16,
    _mm256_sub_epi16,
    16
);

/// Pieces a move removes and adds, in board terms; [`Accumulator::update_from`]
/// turns them into feature rows per perspective.
pub struct PieceDelta {
    removed: [(Color, PieceKind, Square); 3],
    added: [(Color, PieceKind, Square); 2],
    n_removed: usize,
    n_added: usize,
}

impl Default for PieceDelta {
    fn default() -> Self {
        Self::new()
    }
}

impl PieceDelta {
    pub const fn new() -> Self {
        const NONE: (Color, PieceKind, Square) = (Color::White, PieceKind::Pawn, Square::A1);
        Self {
            removed: [NONE; 3],
            added: [NONE; 2],
            n_removed: 0,
            n_added: 0,
        }
    }

    #[inline]
    pub fn remove(&mut self, color: Color, kind: PieceKind, sq: Square) {
        self.removed[self.n_removed] = (color, kind, sq);
        self.n_removed += 1;
    }

    #[inline]
    pub fn add(&mut self, color: Color, kind: PieceKind, sq: Square) {
        self.added[self.n_added] = (color, kind, sq);
        self.n_added += 1;
    }
}

/// NNUE accumulator holding feature-transformed values for both perspectives.
///
/// `needs_refresh` is set to true when the accumulator is stale (e.g. after a
/// king move changes the king bucket).  The refresh is deferred until the
/// position is actually evaluated, so pruned nodes pay no refresh cost.
#[derive(Clone)]
pub struct Accumulator {
    pub white: [i16; HIDDEN_SIZE],
    pub black: [i16; HIDDEN_SIZE],
    needs_refresh: [bool; 2],
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    /// Create a zeroed accumulator (marked as needing a refresh).
    pub const fn new() -> Self {
        Self {
            white: [0i16; HIDDEN_SIZE],
            black: [0i16; HIDDEN_SIZE],
            needs_refresh: [true; 2],
        }
    }

    /// Inherit only usable parent values. Dirty perspectives will be rebuilt
    /// before evaluation, so their destination storage can remain untouched.
    pub fn copy_clean_from(&mut self, parent: &Self, invalidate: Option<Color>) {
        self.needs_refresh = parent.needs_refresh;
        if let Some(color) = invalidate {
            self.mark_refresh(color);
        }
        if !self.needs_refresh(Color::White) {
            self.white.copy_from_slice(&parent.white);
        }
        if !self.needs_refresh(Color::Black) {
            self.black.copy_from_slice(&parent.black);
        }
    }

    /// Full recompute from scratch using the board state.
    /// Child accumulator for a move: copies the parent's refresh flags, marks
    /// `invalidate` dirty (the mover's perspective after a king move), and
    /// writes `parent − removed + added` (pieces and changed state features)
    /// for every clean perspective in one pass. Same result as
    /// [`copy_clean_from`](Self::copy_clean_from) + [`update_state`](Self::update_state)
    /// + [`sub_piece`](Self::sub_piece) / [`add_piece`](Self::add_piece).
    #[allow(clippy::too_many_arguments)]
    pub fn update_from(
        &mut self,
        parent: &Self,
        net: &NnueNetwork,
        invalidate: Option<Color>,
        white_king: Square,
        black_king: Square,
        pieces: &PieceDelta,
        old_castling: CastlingRights,
        old_en_passant: Option<Square>,
        new_castling: CastlingRights,
        new_en_passant: Option<Square>,
    ) {
        self.needs_refresh = parent.needs_refresh;
        if let Some(color) = invalidate {
            self.mark_refresh(color);
        }
        for perspective in [Color::White, Color::Black] {
            if self.needs_refresh(perspective) {
                continue;
            }
            let mut delta = RowDelta::default();
            let old = state_feature_indices(
                perspective,
                white_king,
                black_king,
                old_castling,
                old_en_passant,
            );
            let new = state_feature_indices(
                perspective,
                white_king,
                black_king,
                new_castling,
                new_en_passant,
            );
            for (old_index, new_index) in old.into_iter().zip(new) {
                if old_index != new_index {
                    delta.sub(old_index);
                    delta.add(new_index);
                }
            }
            for &(color, kind, sq) in &pieces.removed[..pieces.n_removed] {
                delta.sub(feature_index(
                    perspective,
                    white_king,
                    black_king,
                    color,
                    kind,
                    sq,
                ));
            }
            for &(color, kind, sq) in &pieces.added[..pieces.n_added] {
                delta.add(feature_index(
                    perspective,
                    white_king,
                    black_king,
                    color,
                    kind,
                    sq,
                ));
            }
            let (dst, src) = match perspective {
                Color::White => (&mut self.white, &parent.white),
                Color::Black => (&mut self.black, &parent.black),
            };
            apply_rows(dst, src, net, &delta);
        }
    }

    pub fn refresh(&mut self, board: &Board, net: &NnueNetwork) {
        self.refresh_perspective(board, net, Color::White);
        self.refresh_perspective(board, net, Color::Black);
    }

    /// Recompute only one perspective. A king move changes the moving side's
    /// feature bucket, but the opponent's bucket remains stable and can stay
    /// incrementally updated.
    pub fn refresh_perspective(&mut self, board: &Board, net: &NnueNetwork, perspective: Color) {
        let mut values = *net.ft_biases;

        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        for &color in &[Color::White, Color::Black] {
            for &kind in &PieceKind::ALL {
                let bb = board.pieces[color.index()][kind.index()];
                for sq in bb.iter() {
                    let idx = feature_index(perspective, white_king, black_king, color, kind, sq);
                    let row = &net.ft_weights[idx];
                    add_row(&mut values, row);
                }
            }
        }

        for index in board_state_feature_indices(board, perspective) {
            let row = &net.ft_weights[index];
            add_row(&mut values, row);
        }

        match perspective {
            Color::White => self.white = values,
            Color::Black => self.black = values,
        }
        self.needs_refresh[perspective.index()] = false;
    }

    #[inline]
    pub fn needs_refresh(&self, perspective: Color) -> bool {
        self.needs_refresh[perspective.index()]
    }

    #[inline]
    pub fn mark_refresh(&mut self, perspective: Color) {
        self.needs_refresh[perspective.index()] = true;
    }

    /// Apply castling/en-passant state changes to both clean perspectives.
    #[allow(clippy::too_many_arguments)]
    pub fn update_state(
        &mut self,
        net: &NnueNetwork,
        white_king: Square,
        black_king: Square,
        old_castling: CastlingRights,
        old_en_passant: Option<Square>,
        new_castling: CastlingRights,
        new_en_passant: Option<Square>,
    ) {
        for perspective in [Color::White, Color::Black] {
            if self.needs_refresh(perspective) {
                continue;
            }
            let old = state_feature_indices(
                perspective,
                white_king,
                black_king,
                old_castling,
                old_en_passant,
            );
            let new = state_feature_indices(
                perspective,
                white_king,
                black_king,
                new_castling,
                new_en_passant,
            );
            let values = match perspective {
                Color::White => &mut self.white,
                Color::Black => &mut self.black,
            };
            for (old_index, new_index) in old.into_iter().zip(new) {
                if old_index == new_index {
                    continue;
                }
                let old_row = &net.ft_weights[old_index];
                let new_row = &net.ft_weights[new_index];
                replace_row(values, old_row, new_row);
            }
        }
    }

    /// Add a piece's feature weights to both perspectives.
    #[inline]
    pub fn add_piece(
        &mut self,
        net: &NnueNetwork,
        white_king: Square,
        black_king: Square,
        color: Color,
        kind: PieceKind,
        sq: Square,
    ) {
        self.add_piece_inner(net, white_king, black_king, color, kind, sq);
    }

    /// Subtract a piece's feature weights from both perspectives.
    #[inline]
    pub fn sub_piece(
        &mut self,
        net: &NnueNetwork,
        white_king: Square,
        black_king: Square,
        color: Color,
        kind: PieceKind,
        sq: Square,
    ) {
        if !self.needs_refresh(Color::White) {
            let idx = feature_index(Color::White, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            sub_row(&mut self.white, row);
        }
        if !self.needs_refresh(Color::Black) {
            let idx = feature_index(Color::Black, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            sub_row(&mut self.black, row);
        }
    }

    #[inline]
    fn add_piece_inner(
        &mut self,
        net: &NnueNetwork,
        white_king: Square,
        black_king: Square,
        color: Color,
        kind: PieceKind,
        sq: Square,
    ) {
        if !self.needs_refresh(Color::White) {
            let idx = feature_index(Color::White, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            add_row(&mut self.white, row);
        }
        if !self.needs_refresh(Color::Black) {
            let idx = feature_index(Color::Black, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            add_row(&mut self.black, row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FEATURES_PER_BUCKET, NUM_BUCKETS, PIECE_FEATURES};

    #[test]
    fn inheriting_dirty_perspectives_remains_correct_after_refresh() {
        let board = Board::starting_position();
        let net = NnueNetwork::embedded();
        let mut parent = Accumulator::new();
        parent.refresh(&board, &net);
        let expected = parent.clone();
        parent.mark_refresh(Color::White);
        // Stale contents must never leak into a clean perspective.
        parent.white.fill(-123);
        let mut child = Accumulator::new();
        child.white.fill(456);
        child.black.fill(789);
        child.copy_clean_from(&parent, None);
        assert!(child.needs_refresh(Color::White));
        assert!(!child.needs_refresh(Color::Black));
        assert_eq!(child.black, expected.black);
        child.refresh_perspective(&board, &net, Color::White);
        assert_eq!(child.white, expected.white);

        // A king move can invalidate the other half while one is already dirty.
        child.copy_clean_from(&parent, Some(Color::Black));
        assert!(child.needs_refresh(Color::White));
        assert!(child.needs_refresh(Color::Black));
        child.refresh(&board, &net);
        assert_eq!(child.white, expected.white);
        assert_eq!(child.black, expected.black);

        child.copy_clean_from(&expected, None);
        assert!(!child.needs_refresh(Color::White));
        assert!(!child.needs_refresh(Color::Black));
    }

    #[test]
    fn dispatched_row_operations_match_scalar() {
        let original = std::array::from_fn(|i| (i as i16 % 97) - 48);
        let row = std::array::from_fn(|i| (i as i16 % 31) - 15);
        let replacement = std::array::from_fn(|i| (i as i16 % 43) - 21);

        let mut expected = original;
        add_row_scalar(&mut expected, &row);
        let mut actual = original;
        add_row(&mut actual, &row);
        assert_eq!(actual, expected);

        sub_row_scalar(&mut expected, &row);
        sub_row(&mut actual, &row);
        assert_eq!(actual, original);
        assert_eq!(actual, expected);

        for ((value, &old), &new) in expected.iter_mut().zip(&row).zip(&replacement) {
            *value += new - old;
        }
        replace_row(&mut actual, &row, &replacement);
        assert_eq!(actual, expected);
    }

    #[test]
    fn fused_update_matches_sequential_updates() {
        let net = NnueNetwork::embedded();
        let board = Board::starting_position();
        let mut parent = Accumulator::new();
        parent.refresh(&board, &net);
        let (white_king, black_king) = (Square::E1, Square::E8);
        let (old_castling, new_castling) = (board.castling, CastlingRights::NONE);
        let (old_ep, new_ep) = (None, Some(Square::new(3, 5)));
        // A capture with a castling-rook move and every state feature changing.
        let removed = [
            (Color::White, PieceKind::Knight, Square::new(6, 0)),
            (Color::Black, PieceKind::Pawn, Square::new(4, 6)),
            (Color::White, PieceKind::Rook, Square::H1),
        ];
        let added = [
            (Color::White, PieceKind::Knight, Square::new(4, 6)),
            (Color::White, PieceKind::Rook, Square::F1),
        ];
        let mut pieces = PieceDelta::new();
        removed
            .iter()
            .for_each(|&(c, k, sq)| pieces.remove(c, k, sq));
        added.iter().for_each(|&(c, k, sq)| pieces.add(c, k, sq));

        for (dirty, invalidate) in [
            (None, None),
            (Some(Color::White), None),
            (None, Some(Color::Black)),
        ] {
            let mut parent = parent.clone();
            if let Some(color) = dirty {
                parent.mark_refresh(color);
            }
            let mut expected = Accumulator::new();
            expected.copy_clean_from(&parent, invalidate);
            expected.update_state(
                &net,
                white_king,
                black_king,
                old_castling,
                old_ep,
                new_castling,
                new_ep,
            );
            for &(c, k, sq) in &removed {
                expected.sub_piece(&net, white_king, black_king, c, k, sq);
            }
            for &(c, k, sq) in &added {
                expected.add_piece(&net, white_king, black_king, c, k, sq);
            }
            let mut actual = Accumulator::new();
            actual.update_from(
                &parent,
                &net,
                invalidate,
                white_king,
                black_king,
                &pieces,
                old_castling,
                old_ep,
                new_castling,
                new_ep,
            );
            assert_eq!(actual.needs_refresh, expected.needs_refresh);
            for perspective in [Color::White, Color::Black] {
                if !expected.needs_refresh(perspective) {
                    let (a, e) = match perspective {
                        Color::White => (&actual.white, &expected.white),
                        Color::Black => (&actual.black, &expected.black),
                    };
                    assert_eq!(a, e);
                }
            }
        }
    }

    #[test]
    fn dispatched_apply_rows_match_scalar() {
        let net = NnueNetwork::embedded();
        let src: [i16; HIDDEN_SIZE] = std::array::from_fn(|i| (i as i16 % 97) - 48);
        let (subs, adds) = ([5usize, 900, 12_000], [77usize, 25_000]);
        let mut expected = [0i16; HIDDEN_SIZE];
        apply_rows_scalar(&mut expected, &src, &net, &subs, &adds);
        let mut delta = RowDelta::default();
        subs.iter().for_each(|&row| delta.sub(row));
        adds.iter().for_each(|&row| delta.add(row));
        let mut actual = [0i16; HIDDEN_SIZE];
        apply_rows(&mut actual, &src, &net, &delta);
        assert_eq!(actual, expected);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let mut actual = [0i16; HIDDEN_SIZE];
                unsafe { apply_rows_avx2(&mut actual, &src, &net, &subs, &adds) };
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn incremental_matches_refresh() {
        let net = NnueNetwork::embedded();
        let board = Board::starting_position();
        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        // Full refresh
        let mut acc_full = Accumulator::new();
        acc_full.refresh(&board, &net);

        // Incremental: start from biases, add pieces one by one
        let mut acc_inc = Accumulator::new();
        acc_inc.white = *net.ft_biases;
        acc_inc.black = *net.ft_biases;
        acc_inc.needs_refresh = [false; 2];
        for &color in &[Color::White, Color::Black] {
            for &kind in &PieceKind::ALL {
                let bb = board.pieces[color.index()][kind.index()];
                for sq in bb.iter() {
                    acc_inc.add_piece(&net, white_king, black_king, color, kind, sq);
                }
            }
        }
        for perspective in [Color::White, Color::Black] {
            let values = match perspective {
                Color::White => &mut acc_inc.white,
                Color::Black => &mut acc_inc.black,
            };
            for index in board_state_feature_indices(&board, perspective) {
                add_row(values, &net.ft_weights[index]);
            }
        }

        assert_eq!(acc_full.white, acc_inc.white);
        assert_eq!(acc_full.black, acc_inc.black);
    }

    #[test]
    fn add_sub_roundtrip() {
        let net = NnueNetwork::embedded();
        let board = Board::starting_position();
        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        let mut acc = Accumulator::new();
        acc.refresh(&board, &net);
        let original_white = acc.white;
        let original_black = acc.black;

        // Add then subtract a piece — should return to original
        acc.add_piece(
            &net,
            white_king,
            black_king,
            Color::White,
            PieceKind::Queen,
            Square::new(3, 3),
        );
        acc.sub_piece(
            &net,
            white_king,
            black_king,
            Color::White,
            PieceKind::Queen,
            Square::new(3, 3),
        );

        assert_eq!(acc.white, original_white);
        assert_eq!(acc.black, original_black);
    }

    #[test]
    fn incremental_state_change_matches_full_refresh_with_nonzero_state_weights() {
        let mut net = NnueNetwork::zeroed_for_test();
        for bucket in 0..NUM_BUCKETS {
            for slot in PIECE_FEATURES..FEATURES_PER_BUCKET {
                net.ft_weights[bucket * FEATURES_PER_BUCKET + slot]
                    .fill((slot - PIECE_FEATURES + 1) as i16);
            }
        }

        let before = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let mut after = before.clone();
        after.castling = CastlingRights::NONE;
        after.en_passant = Some(Square::new(4, 5));

        let mut incremental = Accumulator::new();
        incremental.refresh(&before, &net);
        incremental.update_state(
            &net,
            before.king_square(Color::White),
            before.king_square(Color::Black),
            before.castling,
            before.en_passant,
            after.castling,
            after.en_passant,
        );

        let mut refreshed = Accumulator::new();
        refreshed.refresh(&after, &net);
        assert_eq!(incremental.white, refreshed.white);
        assert_eq!(incremental.black, refreshed.black);
    }

    #[test]
    fn perspective_refresh_leaves_other_half_untouched() {
        let net = NnueNetwork::embedded();
        let board = Board::starting_position();

        let mut expected = Accumulator::new();
        expected.refresh(&board, &net);

        let mut acc = expected.clone();
        acc.white = [0; HIDDEN_SIZE];
        acc.black = [123; HIDDEN_SIZE];
        acc.mark_refresh(Color::White);
        acc.refresh_perspective(&board, &net, Color::White);

        assert_eq!(acc.white, expected.white);
        assert_eq!(acc.black, [123; HIDDEN_SIZE]);
        assert!(!acc.needs_refresh(Color::White));
    }
}
