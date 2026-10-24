use chess_common::{Bitboard, Board, CastlingRights, Color, PieceKind, Square};

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
            SimdBackend::Scalar | SimdBackend::Neon | SimdBackend::NeonDotprod => {}
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
            SimdBackend::Scalar | SimdBackend::Neon | SimdBackend::NeonDotprod => {}
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
            SimdBackend::Scalar | SimdBackend::Neon | SimdBackend::NeonDotprod => {}
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
    let (subs, adds) = (&delta.sub[..delta.n_sub], &delta.add[..delta.n_add]);
    // SAFETY: distinct borrows, both HIDDEN_SIZE long.
    unsafe { apply_rows_raw(dst.as_mut_ptr(), src.as_ptr(), net, subs, adds) }
}

/// In-place form of [`apply_rows`] (`values -= subs; values += adds`).
#[inline]
fn apply_rows_in_place(
    values: &mut [i16; HIDDEN_SIZE],
    net: &NnueNetwork,
    subs: &[usize],
    adds: &[usize],
) {
    let ptr = values.as_mut_ptr();
    // SAFETY: the kernels load each element before they store it.
    unsafe { apply_rows_raw(ptr, ptr, net, subs, adds) }
}

/// # Safety
/// `dst` and `src` must each be valid for HIDDEN_SIZE elements; they may be
/// equal but must not otherwise overlap.
#[inline]
unsafe fn apply_rows_raw(
    dst: *mut i16,
    src: *const i16,
    net: &NnueNetwork,
    subs: &[usize],
    adds: &[usize],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => {
                return apply_rows_avx512(dst, src, net, subs, adds);
            }
            SimdBackend::Avx2 => return apply_rows_avx2(dst, src, net, subs, adds),
            SimdBackend::Scalar | SimdBackend::Neon | SimdBackend::NeonDotprod => {}
        }
    }
    unsafe { apply_rows_scalar(dst, src, net, subs, adds) }
}

#[inline]
unsafe fn apply_rows_scalar(
    dst: *mut i16,
    src: *const i16,
    net: &NnueNetwork,
    subs: &[usize],
    adds: &[usize],
) {
    // Fixed-size tiles with row-wise inner loops: the compiler vectorizes these
    // (NEON on aarch64), which an element-wise loop over the rows defeats.
    const TILE: usize = 128;
    const _: () = assert!(HIDDEN_SIZE.is_multiple_of(TILE));
    for base in (0..HIDDEN_SIZE).step_by(TILE) {
        let mut tile = [0i16; TILE];
        tile.copy_from_slice(unsafe { std::slice::from_raw_parts(src.add(base), TILE) });
        for &row in subs {
            for (value, &delta) in tile.iter_mut().zip(&net.ft_weights[row][base..base + TILE]) {
                *value = value.wrapping_sub(delta);
            }
        }
        for &row in adds {
            for (value, &delta) in tile.iter_mut().zip(&net.ft_weights[row][base..base + TILE]) {
                *value = value.wrapping_add(delta);
            }
        }
        unsafe { std::ptr::copy_nonoverlapping(tile.as_ptr(), dst.add(base), TILE) };
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
            dst: *mut i16,
            src: *const i16,
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
                    let mut regs: [$vector; APPLY_TILE_REGS] =
                        std::array::from_fn(
                            |k| $load(src.add(base + k * $lanes) as *const $vector),
                        );
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
                        $store(dst.add(base + k * $lanes) as *mut $vector, *reg);
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
#[derive(Clone, Copy)]
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

/// One move as the accumulator sees it: the pieces it changes, the king squares
/// after it, and the castling / en-passant state before and after. A null move
/// is an empty [`PieceDelta`] that clears the en-passant square.
#[derive(Clone, Copy)]
pub struct MoveUpdate {
    pub pieces: PieceDelta,
    pub white_king: Square,
    pub black_king: Square,
    pub old_castling: CastlingRights,
    pub old_en_passant: Option<Square>,
    pub new_castling: CastlingRights,
    pub new_en_passant: Option<Square>,
}

impl MoveUpdate {
    /// Feature rows this move removes from and adds to `perspective`, whose
    /// king bucket must be the same before and after the move.
    fn rows(&self, perspective: Color) -> RowDelta {
        let (white_king, black_king) = (self.white_king, self.black_king);
        let mut delta = RowDelta::default();
        let old = state_feature_indices(
            perspective,
            white_king,
            black_king,
            self.old_castling,
            self.old_en_passant,
        );
        let new = state_feature_indices(
            perspective,
            white_king,
            black_king,
            self.new_castling,
            self.new_en_passant,
        );
        for (old_index, new_index) in old.into_iter().zip(new) {
            if old_index != new_index {
                delta.sub(old_index);
                delta.add(new_index);
            }
        }
        let pieces = &self.pieces;
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
        delta
    }
}

/// Last accumulator seen for one (perspective, own king square): the feature
/// rows depend only on that king square, so a refresh can start from here and
/// apply the pieces that differ instead of summing every piece from the biases.
struct CacheEntry {
    values: [i16; HIDDEN_SIZE],
    pieces: [[u64; PieceKind::COUNT]; 2],
    state: Option<[usize; 3]>,
}

/// Accumulator refresh cache ("Finny table"), one entry per perspective and own
/// king square. Bound to the net it was built for.
pub struct RefreshCache {
    entries: Box<[CacheEntry]>,
}

impl RefreshCache {
    pub fn new(net: &NnueNetwork) -> Self {
        let entries = (0..2 * 64)
            .map(|_| CacheEntry {
                values: *net.ft_biases,
                pieces: [[0; PieceKind::COUNT]; 2],
                state: None,
            })
            .collect();
        Self { entries }
    }

    /// Same result as [`Accumulator::refresh_perspective`] (wrapping i16 sums
    /// are order-independent), at the cost of the pieces that changed since
    /// this king square was last refreshed.
    pub fn refresh(
        &mut self,
        acc: &mut Accumulator,
        board: &Board,
        net: &NnueNetwork,
        perspective: Color,
    ) {
        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);
        let own_king = match perspective {
            Color::White => white_king,
            Color::Black => black_king,
        };
        let entry = &mut self.entries[perspective.index() * 64 + own_king.index()];

        // 32 pieces + 3 state features at most, on a cold entry
        let mut subs = [0usize; 35];
        let mut adds = [0usize; 35];
        let (mut n_sub, mut n_add) = (0, 0);
        for &color in &[Color::White, Color::Black] {
            for &kind in &PieceKind::ALL {
                let now = board.pieces[color.index()][kind.index()].0;
                let cached = &mut entry.pieces[color.index()][kind.index()];
                for sq in Bitboard(*cached & !now).iter() {
                    subs[n_sub] =
                        feature_index(perspective, white_king, black_king, color, kind, sq);
                    n_sub += 1;
                }
                for sq in Bitboard(now & !*cached).iter() {
                    adds[n_add] =
                        feature_index(perspective, white_king, black_king, color, kind, sq);
                    n_add += 1;
                }
                *cached = now;
            }
        }
        let state = board_state_feature_indices(board, perspective);
        for (k, &index) in state.iter().enumerate() {
            match entry.state {
                Some(old) if old[k] == index => {}
                Some(old) => {
                    subs[n_sub] = old[k];
                    n_sub += 1;
                    adds[n_add] = index;
                    n_add += 1;
                }
                None => {
                    adds[n_add] = index;
                    n_add += 1;
                }
            }
        }
        entry.state = Some(state);

        apply_rows_in_place(&mut entry.values, net, &subs[..n_sub], &adds[..n_add]);
        match perspective {
            Color::White => acc.white = entry.values,
            Color::Black => acc.black = entry.values,
        }
        acc.needs_refresh[perspective.index()] = false;
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

    /// Child accumulator for a move: copies the parent's refresh flags, marks
    /// `invalidate` dirty (the mover's perspective after a king move), and
    /// writes `parent − removed + added` (pieces and changed state features)
    /// for every clean perspective in one pass. Same result as
    /// [`copy_clean_from`](Self::copy_clean_from) + [`update_state`](Self::update_state)
    /// + [`sub_piece`](Self::sub_piece) / [`add_piece`](Self::add_piece).
    pub fn update_from(
        &mut self,
        parent: &Self,
        net: &NnueNetwork,
        invalidate: Option<Color>,
        update: &MoveUpdate,
    ) {
        self.needs_refresh = parent.needs_refresh;
        if let Some(color) = invalidate {
            self.mark_refresh(color);
        }
        for perspective in [Color::White, Color::Black] {
            if !self.needs_refresh(perspective) {
                self.apply_move(parent, net, perspective, update);
            }
        }
    }

    /// `self[perspective] = parent[perspective]` with `update` applied.
    #[inline]
    fn apply_move(
        &mut self,
        parent: &Self,
        net: &NnueNetwork,
        perspective: Color,
        update: &MoveUpdate,
    ) {
        let (dst, src) = match perspective {
            Color::White => (&mut self.white, &parent.white),
            Color::Black => (&mut self.black, &parent.black),
        };
        apply_rows(dst, src, net, &update.rows(perspective));
    }

    /// Full recompute from scratch using the board state.
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

/// One ply of an [`AccumulatorStack`].
#[derive(Clone)]
struct StackEntry {
    acc: Accumulator,
    /// The move from the previous ply into this one.
    update: MoveUpdate,
    /// Perspectives whose values in `acc` belong to this ply's position.
    computed: [bool; 2],
    /// Perspectives whose king moved into this ply: their king bucket changed,
    /// so they cannot be derived from the previous ply.
    king_moved: [bool; 2],
}

/// Accumulators along the search path, updated lazily. A move only records its
/// [`MoveUpdate`]; [`AccumulatorStack::get`] replays a perspective's pending
/// moves from its nearest computed ancestor when a position is evaluated, so
/// nodes that are never evaluated (TT cutoffs, in-check nodes, draws) cost no
/// accumulator work. A perspective whose king moved on the way is rebuilt from
/// the board through the [`RefreshCache`]. The values are bit-identical to
/// updating every ply eagerly: i16 lanes wrap, so the order of sums is free.
pub struct AccumulatorStack {
    entries: Box<[StackEntry]>,
}

impl AccumulatorStack {
    pub fn new(plies: usize) -> Self {
        let entry = StackEntry {
            acc: Accumulator::new(),
            update: MoveUpdate {
                pieces: PieceDelta::new(),
                white_king: Square::E1,
                black_king: Square::E8,
                old_castling: CastlingRights::NONE,
                old_en_passant: None,
                new_castling: CastlingRights::NONE,
                new_en_passant: None,
            },
            computed: [false; 2],
            king_moved: [false; 2],
        };
        Self {
            entries: vec![entry; plies].into_boxed_slice(),
        }
    }

    /// Build `ply` for `board` from scratch (the search root).
    pub fn refresh(&mut self, ply: usize, board: &Board, net: &NnueNetwork) {
        let entry = &mut self.entries[ply];
        entry.acc.refresh(board, net);
        entry.computed = [true; 2];
    }

    /// Record the move from `ply - 1` into `ply`. `king_moved` is the side
    /// that moved its king, if any.
    #[inline]
    pub fn push(&mut self, ply: usize, king_moved: Option<Color>, update: MoveUpdate) {
        let entry = &mut self.entries[ply];
        entry.update = update;
        entry.computed = [false; 2];
        entry.king_moved = [false; 2];
        if let Some(color) = king_moved {
            entry.king_moved[color.index()] = true;
        }
    }

    /// The accumulator of `ply`, brought up to date; `board` is the position
    /// at that ply.
    pub fn get(
        &mut self,
        ply: usize,
        board: &Board,
        net: &NnueNetwork,
        cache: &mut RefreshCache,
    ) -> &Accumulator {
        for perspective in [Color::White, Color::Black] {
            let side = perspective.index();
            let mut base = ply;
            while !self.entries[base].computed[side] && !self.entries[base].king_moved[side] {
                if base == 0 {
                    break;
                }
                base -= 1;
            }
            if self.entries[base].computed[side] {
                for child in base + 1..=ply {
                    let (parents, children) = self.entries.split_at_mut(child);
                    let entry = &mut children[0];
                    entry
                        .acc
                        .apply_move(&parents[child - 1].acc, net, perspective, &entry.update);
                    entry.computed[side] = true;
                }
            } else {
                let entry = &mut self.entries[ply];
                cache.refresh(&mut entry.acc, board, net, perspective);
                entry.computed[side] = true;
            }
        }
        &self.entries[ply].acc
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
            let update = MoveUpdate {
                pieces,
                white_king,
                black_king,
                old_castling,
                old_en_passant: old_ep,
                new_castling,
                new_en_passant: new_ep,
            };
            actual.update_from(&parent, &net, invalidate, &update);
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
        unsafe { apply_rows_scalar(expected.as_mut_ptr(), src.as_ptr(), &net, &subs, &adds) };
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
                unsafe { apply_rows_avx2(actual.as_mut_ptr(), src.as_ptr(), &net, &subs, &adds) };
                assert_eq!(actual, expected);
            }
            let mut in_place = src;
            apply_rows_in_place(&mut in_place, &net, &subs, &adds);
            assert_eq!(in_place, expected);
        }
    }

    #[test]
    fn refresh_cache_matches_full_refresh() {
        let net = NnueNetwork::embedded();
        let mut cache = RefreshCache::new(&net);
        // Same and different king squares, captures, castling and en-passant
        // changes, revisiting cached squares in between.
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "r3k2r/ppp2ppp/2n5/3q4/8/2N5/PPP2PPP/R3K2R b Qk - 0 9",
            "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 40",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
            "2kr3r/ppp2ppp/8/8/8/8/PPP2PPP/2KR3R w - - 4 12",
            "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 40",
        ];
        for fen in fens {
            let board = Board::from_fen(fen).unwrap();
            let mut expected = Accumulator::new();
            expected.refresh(&board, &net);
            let mut actual = Accumulator::new();
            cache.refresh(&mut actual, &board, &net, Color::White);
            cache.refresh(&mut actual, &board, &net, Color::Black);
            assert_eq!(actual.white, expected.white, "{fen}");
            assert_eq!(actual.black, expected.black, "{fen}");
            assert!(!actual.needs_refresh(Color::White) && !actual.needs_refresh(Color::Black));
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
