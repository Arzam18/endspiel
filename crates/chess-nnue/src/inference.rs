use chess_common::Color;
use std::sync::OnceLock;

use crate::accumulator::Accumulator;
use crate::network::{L1_BLOCK, L1_CHUNK, L1_CHUNKS, L1Block, NnueNetwork};
use crate::{
    FT_QUANT, HIDDEN_SIZE, MAX_L1_SIZE, MAX_L2_SIZE, NET_QUANT, OUTPUT_BUCKETS, PAIR_INPUT_SIZE,
    PAIR_SIZE,
};

/// Best x86 SIMD tier available to the current process.
///
/// Keep this in sync with the requirements used by Stockfish's universal
/// x86-64 dispatcher.  In particular, AVX512ICL means the complete feature
/// set shared by Intel Ice Lake and AMD Zen 4/5, not merely AVX-512F.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    Neon,
    Avx2,
    Avx512,
    Avx512Icl,
}

impl SimdBackend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "NEON",
            Self::Avx2 => "AVX2",
            Self::Avx512 => "AVX-512",
            Self::Avx512Icl => "AVX512ICL",
        }
    }
}

/// Report the SIMD implementation selected at runtime.
pub fn simd_backend() -> SimdBackend {
    static BACKEND: OnceLock<SimdBackend> = OnceLock::new();
    *BACKEND.get_or_init(detect_simd_backend)
}

fn detect_simd_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            if is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512dq")
                && is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("avx512ifma")
                && is_x86_feature_detected!("avx512vbmi")
                && is_x86_feature_detected!("avx512vbmi2")
                && is_x86_feature_detected!("avx512vpopcntdq")
                && is_x86_feature_detected!("avx512bitalg")
                && is_x86_feature_detected!("vpclmulqdq")
                && is_x86_feature_detected!("gfni")
                && is_x86_feature_detected!("vaes")
            {
                return SimdBackend::Avx512Icl;
            }
            return SimdBackend::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdBackend::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        SimdBackend::Neon
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        SimdBackend::Scalar
    }
}

#[inline]
pub fn output_bucket(piece_count: u32) -> usize {
    ((piece_count.saturating_sub(2) / 4) as usize).min(OUTPUT_BUCKETS - 1)
}

/// L1 input: pairwise CReLU products of both perspectives, side to move first,
/// rounded to u8: `(clamp(a, 0, 127) * clamp(b, 0, 127) + 64) >> 7` in `0..=126`.
/// Computed once per evaluation and shared by every L1 neuron.
#[repr(C, align(64))]
struct L1Input([u8; PAIR_INPUT_SIZE]);

/// The u8 L1 input is the product divided by `1 << L1_INPUT_SHIFT`.
const L1_INPUT_SHIFT: u32 = 7;

#[inline]
fn pairwise_products(acc: &[i16; HIDDEN_SIZE], out: &mut [u8]) {
    debug_assert_eq!(out.len(), PAIR_SIZE);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => {
                return pairwise_products_avx512(acc, out);
            }
            SimdBackend::Avx2 => return pairwise_products_avx2(acc, out),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { pairwise_products_neon(acc, out) };
    }
    #[allow(unreachable_code)]
    pairwise_products_scalar(acc, out)
}

#[inline]
fn pairwise_products_scalar(acc: &[i16; HIDDEN_SIZE], out: &mut [u8]) {
    for (idx, slot) in out.iter_mut().enumerate() {
        let lhs = i32::from(acc[idx].clamp(0, FT_QUANT as i16));
        let rhs = i32::from(acc[idx + PAIR_SIZE].clamp(0, FT_QUANT as i16));
        *slot = ((lhs * rhs + (1 << (L1_INPUT_SHIFT - 1))) >> L1_INPUT_SHIFT) as u8;
    }
}

// The SIMD versions round with a high multiply: mulhrs(a << 8, b) =
// (256·a·b + 2^14) >> 15 = (a·b + 64) >> 7, exactly the scalar formula.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn pairwise_products_avx512(acc: &[i16; HIDDEN_SIZE], out: &mut [u8]) {
    use std::arch::x86_64::*;

    let zero = _mm512_setzero_si512();
    let quant = _mm512_set1_epi16(FT_QUANT as i16);
    unsafe {
        let mut idx = 0;
        while idx < PAIR_SIZE {
            let lhs = _mm512_loadu_si512(acc.as_ptr().add(idx) as *const __m512i);
            let rhs = _mm512_loadu_si512(acc.as_ptr().add(idx + PAIR_SIZE) as *const __m512i);
            let lhs = _mm512_min_epi16(_mm512_max_epi16(lhs, zero), quant);
            let rhs = _mm512_min_epi16(_mm512_max_epi16(rhs, zero), quant);
            let product = _mm512_mulhrs_epi16(_mm512_slli_epi16::<8>(lhs), rhs);
            _mm256_storeu_si256(
                out.as_mut_ptr().add(idx) as *mut __m256i,
                _mm512_cvtepi16_epi8(product),
            );
            idx += 32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_products_avx2(acc: &[i16; HIDDEN_SIZE], out: &mut [u8]) {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let quant = _mm256_set1_epi16(FT_QUANT as i16);
    let product = |idx: usize| unsafe {
        let lhs = _mm256_loadu_si256(acc.as_ptr().add(idx) as *const __m256i);
        let rhs = _mm256_loadu_si256(acc.as_ptr().add(idx + PAIR_SIZE) as *const __m256i);
        let lhs = _mm256_min_epi16(_mm256_max_epi16(lhs, zero), quant);
        let rhs = _mm256_min_epi16(_mm256_max_epi16(rhs, zero), quant);
        _mm256_mulhrs_epi16(_mm256_slli_epi16::<8>(lhs), rhs)
    };
    unsafe {
        let mut idx = 0;
        while idx < PAIR_SIZE {
            // packus interleaves the 128-bit lanes; the permute restores the order
            let packed = _mm256_packus_epi16(product(idx), product(idx + 16));
            _mm256_storeu_si256(
                out.as_mut_ptr().add(idx) as *mut __m256i,
                _mm256_permute4x64_epi64::<0b11_01_10_00>(packed),
            );
            idx += 32;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pairwise_products_neon(acc: &[i16; HIDDEN_SIZE], out: &mut [u8]) {
    use std::arch::aarch64::*;

    let zero = vdupq_n_s16(0);
    let quant = vdupq_n_s16(FT_QUANT as i16);
    // vqrdmulh(a << 8, b) = (512·a·b + 2^15) >> 16, the same rounding as mulhrs
    let product = |idx: usize| unsafe {
        let lhs = vld1q_s16(acc.as_ptr().add(idx));
        let rhs = vld1q_s16(acc.as_ptr().add(idx + PAIR_SIZE));
        let lhs = vminq_s16(vmaxq_s16(lhs, zero), quant);
        let rhs = vminq_s16(vmaxq_s16(rhs, zero), quant);
        vqmovun_s16(vqrdmulhq_s16(vshlq_n_s16::<8>(lhs), rhs))
    };
    unsafe {
        let mut idx = 0;
        while idx < PAIR_SIZE {
            vst1q_u8(
                out.as_mut_ptr().add(idx),
                vcombine_u8(product(idx), product(idx + 8)),
            );
            idx += 16;
        }
    }
}

/// Writes the indices of the non-zero 4-byte input chunks, ascending, and
/// returns their count. After CReLU most chunks are zero; L1 skips them.
#[inline]
fn nonzero_chunks(input: &L1Input, out: &mut [u16; L1_CHUNKS]) -> usize {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => {
                return nonzero_chunks_avx512(input, out);
            }
            SimdBackend::Avx2 => return nonzero_chunks_avx2(input, out),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    nonzero_chunks_scalar(input, out)
}

/// For every 8-bit mask, the positions of its set bits, ascending.
static SET_BITS: [[u16; 8]; 256] = {
    let mut table = [[0u16; 8]; 256];
    let mut mask = 0;
    while mask < 256 {
        let (mut count, mut bit) = (0, 0);
        while bit < 8 {
            if mask & (1 << bit) != 0 {
                table[mask][count] = bit as u16;
                count += 1;
            }
            bit += 1;
        }
        mask += 1;
    }
    table
};

/// Appends `first + bit` for the set bits of `mask` (one mask bit per chunk
/// `first..first + 8`) without branching: always writes 8 entries, the unused
/// ones are overwritten later. `count <= first`, so the writes stay in bounds.
#[inline(always)]
fn push_chunks(out: &mut [u16; L1_CHUNKS], count: usize, first: usize, mask: u8) -> usize {
    for (slot, &bit) in out[count..count + 8]
        .iter_mut()
        .zip(&SET_BITS[usize::from(mask)])
    {
        *slot = first as u16 + bit;
    }
    count + mask.count_ones() as usize
}

#[inline]
fn nonzero_chunks_scalar(input: &L1Input, out: &mut [u16; L1_CHUNKS]) -> usize {
    let mut count = 0;
    for (group, bytes) in input.0.as_chunks::<{ 8 * L1_CHUNK }>().0.iter().enumerate() {
        let mut mask = 0u8;
        for (bit, chunk) in bytes.as_chunks::<L1_CHUNK>().0.iter().enumerate() {
            mask |= u8::from(*chunk != [0; L1_CHUNK]) << bit;
        }
        count = push_chunks(out, count, group * 8, mask);
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn nonzero_chunks_avx512(input: &L1Input, out: &mut [u16; L1_CHUNKS]) -> usize {
    use std::arch::x86_64::*;

    let mut count = 0;
    for vector in 0..PAIR_INPUT_SIZE / 64 {
        let values =
            unsafe { _mm512_load_si512(input.0.as_ptr().add(vector * 64) as *const __m512i) };
        let mask = _mm512_test_epi32_mask(values, values);
        count = push_chunks(out, count, vector * 16, mask as u8);
        count = push_chunks(out, count, vector * 16 + 8, (mask >> 8) as u8);
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nonzero_chunks_avx2(input: &L1Input, out: &mut [u16; L1_CHUNKS]) -> usize {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let mut count = 0;
    for vector in 0..PAIR_INPUT_SIZE / 32 {
        let values =
            unsafe { _mm256_load_si256(input.0.as_ptr().add(vector * 32) as *const __m256i) };
        // input bytes are <= 126, so a chunk read as i32 is non-zero iff positive
        let positive = _mm256_cmpgt_epi32(values, zero);
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(positive)) as u8;
        count = push_chunks(out, count, vector * 8, mask);
    }
    count
}

/// `out[n] = Σ input[i] * weight[n][i]` for the [`L1_BLOCK`] neurons of
/// `blocks`, over the listed non-zero chunks. The sums are exact (|sum| < 2^25,
/// no saturating step), so every backend is bit-identical.
#[inline]
fn l1_sparse_dot(input: &L1Input, chunks: &[u16], blocks: &[L1Block]) -> [i32; L1_BLOCK] {
    debug_assert_eq!(blocks.len(), L1_CHUNKS);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl => return l1_sparse_dot_vnni(input, chunks, blocks),
            SimdBackend::Avx512 => return l1_sparse_dot_avx512(input, chunks, blocks),
            SimdBackend::Avx2 => return l1_sparse_dot_avx2(input, chunks, blocks),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { l1_sparse_dot_neon_dotprod(input, chunks, blocks) };
        }
        return unsafe { l1_sparse_dot_neon(input, chunks, blocks) };
    }
    #[allow(unreachable_code)]
    l1_sparse_dot_scalar(input, chunks, blocks)
}

#[inline]
fn l1_sparse_dot_scalar(input: &L1Input, chunks: &[u16], blocks: &[L1Block]) -> [i32; L1_BLOCK] {
    let mut sums = [0i32; L1_BLOCK];
    for &chunk in chunks {
        let chunk = usize::from(chunk);
        let values = &input.0[chunk * L1_CHUNK..][..L1_CHUNK];
        for (sum, weights) in sums
            .iter_mut()
            .zip(blocks[chunk].0.as_chunks::<L1_CHUNK>().0)
        {
            for (&value, &weight) in values.iter().zip(weights) {
                *sum += i32::from(value) * i32::from(weight);
            }
        }
    }
    sums
}

/// The 4 input bytes of `chunk` as one little-endian word, for broadcasting.
#[inline(always)]
fn input_chunk(input: &L1Input, chunk: u16) -> i32 {
    let start = usize::from(chunk) * L1_CHUNK;
    i32::from_le_bytes(input.0[start..start + L1_CHUNK].try_into().unwrap())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn l1_sparse_dot_vnni(
    input: &L1Input,
    chunks: &[u16],
    blocks: &[L1Block],
) -> [i32; L1_BLOCK] {
    use std::arch::x86_64::*;

    let step = |sum: __m512i, chunk: u16| unsafe {
        let values = _mm512_set1_epi32(input_chunk(input, chunk));
        let weights = _mm512_load_si512(blocks[usize::from(chunk)].0.as_ptr() as *const __m512i);
        _mm512_dpbusd_epi32(sum, values, weights)
    };
    // two independent sums hide the dot-product latency
    let mut sums = [_mm512_setzero_si512(); 2];
    let (pairs, rest) = chunks.as_chunks::<2>();
    for &[first, second] in pairs {
        sums[0] = step(sums[0], first);
        sums[1] = step(sums[1], second);
    }
    if let [chunk] = rest {
        sums[0] = step(sums[0], *chunk);
    }
    unsafe { std::mem::transmute(_mm512_add_epi32(sums[0], sums[1])) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn l1_sparse_dot_avx512(
    input: &L1Input,
    chunks: &[u16],
    blocks: &[L1Block],
) -> [i32; L1_BLOCK] {
    use std::arch::x86_64::*;

    // maddubs: u8·i8 pairs summed to i16 (|2·126·128| < 2^15, never saturates),
    // madd with ones: the two pairs of each 4-byte lane summed to i32
    let ones = _mm512_set1_epi16(1);
    let step = |sum: __m512i, chunk: u16| unsafe {
        let values = _mm512_set1_epi32(input_chunk(input, chunk));
        let weights = _mm512_load_si512(blocks[usize::from(chunk)].0.as_ptr() as *const __m512i);
        _mm512_add_epi32(
            sum,
            _mm512_madd_epi16(_mm512_maddubs_epi16(values, weights), ones),
        )
    };
    let mut sums = [_mm512_setzero_si512(); 2];
    let (pairs, rest) = chunks.as_chunks::<2>();
    for &[first, second] in pairs {
        sums[0] = step(sums[0], first);
        sums[1] = step(sums[1], second);
    }
    if let [chunk] = rest {
        sums[0] = step(sums[0], *chunk);
    }
    unsafe { std::mem::transmute(_mm512_add_epi32(sums[0], sums[1])) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_sparse_dot_avx2(
    input: &L1Input,
    chunks: &[u16],
    blocks: &[L1Block],
) -> [i32; L1_BLOCK] {
    use std::arch::x86_64::*;

    let ones = _mm256_set1_epi16(1);
    let mut sums = [_mm256_setzero_si256(); 2];
    for &chunk in chunks {
        let values = _mm256_set1_epi32(input_chunk(input, chunk));
        let block = blocks[usize::from(chunk)].0.as_ptr();
        for (half, sum) in sums.iter_mut().enumerate() {
            let weights = unsafe { _mm256_load_si256(block.add(half * 32) as *const __m256i) };
            *sum = _mm256_add_epi32(
                *sum,
                _mm256_madd_epi16(_mm256_maddubs_epi16(values, weights), ones),
            );
        }
    }
    unsafe { std::mem::transmute(sums) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn l1_sparse_dot_neon_dotprod(
    input: &L1Input,
    chunks: &[u16],
    blocks: &[L1Block],
) -> [i32; L1_BLOCK] {
    use std::arch::aarch64::*;

    // input bytes are <= 126, so the signed dot product is exact
    let mut sums = [vdupq_n_s32(0); 4];
    for &chunk in chunks {
        let values = vreinterpretq_s8_s32(vdupq_n_s32(input_chunk(input, chunk)));
        let block = blocks[usize::from(chunk)].0.as_ptr();
        for (quarter, sum) in sums.iter_mut().enumerate() {
            let weights = unsafe { vld1q_s8(block.add(quarter * 16)) };
            *sum = vdotq_s32(*sum, weights, values);
        }
    }
    unsafe { std::mem::transmute(sums) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l1_sparse_dot_neon(
    input: &L1Input,
    chunks: &[u16],
    blocks: &[L1Block],
) -> [i32; L1_BLOCK] {
    use std::arch::aarch64::*;

    // i8·i8 products widened to i16, pairs summed (|2·126·128| < 2^15), then
    // the two pairs of each neuron accumulated into its i32 lane
    let mut sums = [vdupq_n_s32(0); 4];
    for &chunk in chunks {
        let values = vreinterpretq_s8_s32(vdupq_n_s32(input_chunk(input, chunk)));
        let block = blocks[usize::from(chunk)].0.as_ptr();
        for (quarter, sum) in sums.iter_mut().enumerate() {
            let weights = unsafe { vld1q_s8(block.add(quarter * 16)) };
            let low = vmull_s8(vget_low_s8(weights), vget_low_s8(values));
            let high = vmull_high_s8(weights, values);
            *sum = vpadalq_s16(*sum, vpaddq_s16(low, high));
        }
    }
    unsafe { std::mem::transmute(sums) }
}

/// L1 sums run over the u8 inputs (products >> 7); scale back, add the bias
/// and quantise like the other dense layers.
#[inline]
fn l1_activation(sum: i32, bias: i32) -> i16 {
    let raw = (i64::from(sum) << L1_INPUT_SHIFT) + i64::from(bias);
    (raw / i64::from(FT_QUANT * NET_QUANT)).clamp(0, i64::from(FT_QUANT)) as i16
}

#[inline]
fn dense_screlu_sum(values: &[i16], weights: &[i8]) -> i32 {
    debug_assert_eq!(values.len(), weights.len());
    values
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| {
            let value = i32::from(value).clamp(0, FT_QUANT);
            value * value * i32::from(weight)
        })
        .sum()
}

#[inline]
fn quantised_activation(raw: i32) -> i16 {
    (raw / (FT_QUANT * NET_QUANT)).clamp(0, FT_QUANT) as i16
}

/// Evaluate the layer-stacked NNUE. Dense widths come from the loaded net.
#[inline]
pub fn nnue_evaluate(
    acc: &Accumulator,
    side_to_move: Color,
    net: &NnueNetwork,
    piece_count: u32,
) -> i32 {
    let (stm_acc, opp_acc) = match side_to_move {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };
    let bucket = output_bucket(piece_count);
    let l1_size = net.l1_size;
    let l2_size = net.l2_size;

    let mut input = L1Input([0u8; PAIR_INPUT_SIZE]);
    let (stm_half, opp_half) = input.0.split_at_mut(PAIR_SIZE);
    pairwise_products(stm_acc, stm_half);
    pairwise_products(opp_acc, opp_half);
    let mut chunks = [0u16; L1_CHUNKS];
    let count = nonzero_chunks(&input, &mut chunks);
    let chunks = &chunks[..count];

    let mut l1 = [0i16; MAX_L1_SIZE];
    for block in 0..l1_size.div_ceil(L1_BLOCK) {
        let first = block * L1_BLOCK;
        let sums = l1_sparse_dot(&input, chunks, net.l1_blocks(bucket, block));
        for (k, sum) in sums.into_iter().take(l1_size - first).enumerate() {
            l1[first + k] = l1_activation(sum, net.l1_bias(bucket, first + k));
        }
    }

    let mut l2 = [0i16; MAX_L2_SIZE];
    for (neuron, slot) in l2.iter_mut().enumerate().take(l2_size) {
        let raw = dense_screlu_sum(&l1[..l1_size], net.l2_row(bucket, neuron))
            + net.l2_bias(bucket, neuron);
        *slot = quantised_activation(raw);
    }

    let raw = dense_screlu_sum(&l2[..l2_size], net.l3_row(bucket)) + net.l3_biases[bucket];
    ((i64::from(raw) * 400) / i64::from(FT_QUANT * FT_QUANT * NET_QUANT)) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_count_buckets_cover_the_board() {
        assert_eq!(output_bucket(2), 0);
        assert_eq!(output_bucket(5), 0);
        assert_eq!(output_bucket(6), 1);
        assert_eq!(output_bucket(32), 7);
    }

    /// Deterministic pseudo-random stream for the kernel tests.
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }

    #[test]
    fn pairwise_products_simd_match_scalar() {
        let mut state = 7;
        let mut acc = [0i16; HIDDEN_SIZE];
        for value in acc.iter_mut() {
            *value = (lcg(&mut state) % 400) as i16 - 150;
        }
        acc[..6].copy_from_slice(&[127, 128, i16::MAX, i16::MIN, -1, 1]);
        acc[PAIR_SIZE..PAIR_SIZE + 6].copy_from_slice(&[127, 127, 127, 127, 127, 63]);

        let mut expected = [0u8; PAIR_SIZE];
        pairwise_products_scalar(&acc, &mut expected);
        for (idx, &value) in expected.iter().enumerate() {
            let lhs = i32::from(acc[idx]).clamp(0, FT_QUANT);
            let rhs = i32::from(acc[idx + PAIR_SIZE]).clamp(0, FT_QUANT);
            assert_eq!(i32::from(value), (lhs * rhs + 64) / 128);
        }
        assert_eq!(expected[..6], [126, 126, 126, 0, 0, 0]);

        let check = |products: fn(&[i16; HIDDEN_SIZE], &mut [u8])| {
            let mut out = [0u8; PAIR_SIZE];
            products(&acc, &mut out);
            assert_eq!(out, expected);
        };
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                check(|acc, out| unsafe { pairwise_products_avx512(acc, out) });
            }
            if is_x86_feature_detected!("avx2") {
                check(|acc, out| unsafe { pairwise_products_avx2(acc, out) });
            }
        }
        #[cfg(target_arch = "aarch64")]
        check(|acc, out| unsafe { pairwise_products_neon(acc, out) });
    }

    #[test]
    fn sparse_l1_simd_matches_dense_reference() {
        type Dot = fn(&L1Input, &[u16], &[L1Block]) -> [i32; L1_BLOCK];
        type Chunks = fn(&L1Input, &mut [u16; L1_CHUNKS]) -> usize;

        let mut state = 11;
        let mut input = L1Input([0u8; PAIR_INPUT_SIZE]);
        for chunk in input.0.as_chunks_mut::<L1_CHUNK>().0 {
            if lcg(&mut state).is_multiple_of(3) {
                for value in chunk.iter_mut() {
                    *value = (lcg(&mut state) % 127) as u8;
                }
            }
        }
        let mut blocks = vec![L1Block([0; L1_CHUNK * L1_BLOCK]); L1_CHUNKS];
        for block in blocks.iter_mut() {
            for weight in block.0.iter_mut() {
                *weight = lcg(&mut state) as i8;
            }
        }
        // extreme lanes: no saturating step anywhere
        input.0[..8].fill(126);
        blocks[0].0[..L1_CHUNK].fill(-128);
        blocks[1].0[..L1_CHUNK].fill(127);

        let mut expected = [0i32; L1_BLOCK];
        for (neuron, sum) in expected.iter_mut().enumerate() {
            for (idx, &value) in input.0.iter().enumerate() {
                let weight = blocks[idx / L1_CHUNK].0[neuron * L1_CHUNK + idx % L1_CHUNK];
                *sum += i32::from(value) * i32::from(weight);
            }
        }
        let mut chunks = [0u16; L1_CHUNKS];
        let count = nonzero_chunks_scalar(&input, &mut chunks);
        assert!(count > 100 && count < L1_CHUNKS - 100, "{count}");
        assert_eq!(
            l1_sparse_dot_scalar(&input, &chunks[..count], &blocks),
            expected
        );

        let check = |nonzero: Chunks, dot: Dot| {
            let mut simd_chunks = [0u16; L1_CHUNKS];
            let simd_count = nonzero(&input, &mut simd_chunks);
            assert_eq!(simd_chunks[..simd_count], chunks[..count]);
            assert_eq!(dot(&input, &chunks[..count], &blocks), expected);
        };
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                check(
                    |input, out| unsafe { nonzero_chunks_avx512(input, out) },
                    |input, chunks, blocks| unsafe { l1_sparse_dot_avx512(input, chunks, blocks) },
                );
                if is_x86_feature_detected!("avx512vnni") {
                    check(
                        |input, out| unsafe { nonzero_chunks_avx512(input, out) },
                        |input, chunks, blocks| unsafe {
                            l1_sparse_dot_vnni(input, chunks, blocks)
                        },
                    );
                }
            }
            if is_x86_feature_detected!("avx2") {
                check(
                    |input, out| unsafe { nonzero_chunks_avx2(input, out) },
                    |input, chunks, blocks| unsafe { l1_sparse_dot_avx2(input, chunks, blocks) },
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            check(nonzero_chunks_scalar, |input, chunks, blocks| unsafe {
                l1_sparse_dot_neon(input, chunks, blocks)
            });
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                check(nonzero_chunks_scalar, |input, chunks, blocks| unsafe {
                    l1_sparse_dot_neon_dotprod(input, chunks, blocks)
                });
            }
        }
    }

    /// The evaluation with the exact i16 products as L1 input (the pre-u8 formula).
    fn exact_evaluate(
        acc: &Accumulator,
        side_to_move: Color,
        net: &NnueNetwork,
        pieces: u32,
    ) -> i32 {
        let (stm_acc, opp_acc) = match side_to_move {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };
        let bucket = output_bucket(pieces);
        let mut l1 = [0i16; MAX_L1_SIZE];
        for (neuron, slot) in l1.iter_mut().enumerate().take(net.l1_size) {
            let mut raw = i64::from(net.l1_bias(bucket, neuron));
            for (half, values) in [stm_acc, opp_acc].into_iter().enumerate() {
                for idx in 0..PAIR_SIZE {
                    let lhs = i64::from(values[idx]).clamp(0, i64::from(FT_QUANT));
                    let rhs = i64::from(values[idx + PAIR_SIZE]).clamp(0, i64::from(FT_QUANT));
                    let weight = net.l1_weight(bucket, neuron, half * PAIR_SIZE + idx);
                    raw += lhs * rhs * i64::from(weight);
                }
            }
            *slot = quantised_activation(raw as i32);
        }
        let mut l2 = [0i16; MAX_L2_SIZE];
        for (neuron, slot) in l2.iter_mut().enumerate().take(net.l2_size) {
            let raw = dense_screlu_sum(&l1[..net.l1_size], net.l2_row(bucket, neuron))
                + net.l2_bias(bucket, neuron);
            *slot = quantised_activation(raw);
        }
        let raw = dense_screlu_sum(&l2[..net.l2_size], net.l3_row(bucket)) + net.l3_biases[bucket];
        ((i64::from(raw) * 400) / i64::from(FT_QUANT * FT_QUANT * NET_QUANT)) as i32
    }

    /// Mean |Δ|, mean Δ and max |Δ| in cp of the u8-input eval vs the exact eval over `fens`.
    fn quantisation_error<'a>(fens: impl IntoIterator<Item = &'a str>) -> (f64, f64, i32) {
        let net = NnueNetwork::embedded();
        let (mut total, mut signed, mut max, mut count) = (0i64, 0i64, 0i32, 0usize);
        for fen in fens {
            let board = chess_common::Board::from_fen(fen).unwrap();
            let mut acc = Accumulator::new();
            acc.refresh(&board, &net);
            let pieces = (board.occupancy[0].0 | board.occupancy[1].0).count_ones();
            let delta = nnue_evaluate(&acc, board.side_to_move, &net, pieces)
                - exact_evaluate(&acc, board.side_to_move, &net, pieces);
            let error = delta.abs();
            signed += i64::from(delta);
            total += i64::from(error);
            max = max.max(error);
            count += 1;
        }
        let count = count as f64;
        (total as f64 / count, signed as f64 / count, max)
    }

    #[test]
    fn u8_l1_input_stays_close_to_the_exact_products() {
        let (mean, _, max) = quantisation_error([
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r2q1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9",
            "2r2rk1/1b2qppp/p3pn2/1p6/3NP3/1BN3Q1/PPP3PP/3R1R1K b - - 0 18",
            "8/5pk1/6p1/3R4/7P/6P1/r4PK1/8 b - - 0 41",
            "8/8/4k3/8/2KP4/8/8/8 w - - 0 60",
            "4r1k1/5ppp/8/3B4/8/8/5PPP/6K1 w - - 0 30",
            "r4rk1/1pp2ppp/p1np1q2/4p3/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 10",
        ]);
        assert!(mean <= 6.0 && max <= 20, "mean {mean} cp, max {max} cp");
    }

    /// `NNUE_FENS=<file with one FEN per line> cargo test --release -p chess-nnue
    /// -- --ignored --nocapture quantisation_error_on_fen_file`
    #[test]
    #[ignore]
    fn quantisation_error_on_fen_file() {
        let path = std::env::var("NNUE_FENS").expect("NNUE_FENS");
        let text = std::fs::read_to_string(path).unwrap();
        let fens: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
        let (mean, bias, max) = quantisation_error(fens.iter().copied());
        println!(
            "{} positions: mean |Δ| {mean:.3} cp, mean Δ {bias:+.3} cp, max {max} cp",
            fens.len()
        );
    }
}
