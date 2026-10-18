use chess_common::Color;
use std::sync::OnceLock;

use crate::accumulator::Accumulator;
use crate::network::NnueNetwork;
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

/// Pairwise CReLU products of both perspectives, side to move first: the L1
/// input. Computed once per evaluation and shared by every L1 neuron.
/// `clamp(a, 0, 127) * clamp(b, 0, 127) <= 16129` fits in i16.
#[repr(C, align(64))]
struct L1Input([i16; PAIR_INPUT_SIZE]);

#[inline]
fn pairwise_products(acc: &[i16; HIDDEN_SIZE], out: &mut [i16]) {
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
fn pairwise_products_scalar(acc: &[i16; HIDDEN_SIZE], out: &mut [i16]) {
    for (idx, slot) in out.iter_mut().enumerate() {
        let lhs = acc[idx].clamp(0, FT_QUANT as i16);
        let rhs = acc[idx + PAIR_SIZE].clamp(0, FT_QUANT as i16);
        *slot = lhs * rhs;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn pairwise_products_avx512(acc: &[i16; HIDDEN_SIZE], out: &mut [i16]) {
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
            _mm512_storeu_si512(
                out.as_mut_ptr().add(idx) as *mut __m512i,
                _mm512_mullo_epi16(lhs, rhs),
            );
            idx += 32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_products_avx2(acc: &[i16; HIDDEN_SIZE], out: &mut [i16]) {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let quant = _mm256_set1_epi16(FT_QUANT as i16);
    unsafe {
        let mut idx = 0;
        while idx < PAIR_SIZE {
            let lhs = _mm256_loadu_si256(acc.as_ptr().add(idx) as *const __m256i);
            let rhs = _mm256_loadu_si256(acc.as_ptr().add(idx + PAIR_SIZE) as *const __m256i);
            let lhs = _mm256_min_epi16(_mm256_max_epi16(lhs, zero), quant);
            let rhs = _mm256_min_epi16(_mm256_max_epi16(rhs, zero), quant);
            _mm256_storeu_si256(
                out.as_mut_ptr().add(idx) as *mut __m256i,
                _mm256_mullo_epi16(lhs, rhs),
            );
            idx += 16;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pairwise_products_neon(acc: &[i16; HIDDEN_SIZE], out: &mut [i16]) {
    use std::arch::aarch64::*;

    let zero = vdupq_n_s16(0);
    let quant = vdupq_n_s16(FT_QUANT as i16);
    unsafe {
        let mut idx = 0;
        while idx < PAIR_SIZE {
            let lhs = vld1q_s16(acc.as_ptr().add(idx));
            let rhs = vld1q_s16(acc.as_ptr().add(idx + PAIR_SIZE));
            let lhs = vminq_s16(vmaxq_s16(lhs, zero), quant);
            let rhs = vminq_s16(vmaxq_s16(rhs, zero), quant);
            vst1q_s16(out.as_mut_ptr().add(idx), vmulq_s16(lhs, rhs));
            idx += 8;
        }
    }
}

/// `out[n] = Σ input[i] * rows[n][i]` for N L1 neurons in one pass over the
/// input. i32 sums wrap like the SIMD lanes, so every backend is bit-identical.
#[inline]
fn l1_dot<const N: usize>(input: &L1Input, rows: [&[i8; PAIR_INPUT_SIZE]; N]) -> [i32; N] {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => return l1_dot_avx512(input, rows),
            SimdBackend::Avx2 => return l1_dot_avx2(input, rows),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { l1_dot_neon(input, rows) };
    }
    #[allow(unreachable_code)]
    l1_dot_scalar(input, rows)
}

#[inline]
fn l1_dot_scalar<const N: usize>(input: &L1Input, rows: [&[i8; PAIR_INPUT_SIZE]; N]) -> [i32; N] {
    rows.map(|row| {
        input
            .0
            .iter()
            .zip(row)
            .fold(0i32, |sum, (&value, &weight)| {
                sum.wrapping_add(i32::from(value) * i32::from(weight))
            })
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn l1_dot_avx512<const N: usize>(
    input: &L1Input,
    rows: [&[i8; PAIR_INPUT_SIZE]; N],
) -> [i32; N] {
    use std::arch::x86_64::*;

    let mut sums = [_mm512_setzero_si512(); N];
    unsafe {
        let mut idx = 0;
        while idx < PAIR_INPUT_SIZE {
            let values = _mm512_load_si512(input.0.as_ptr().add(idx) as *const __m512i);
            for (sum, row) in sums.iter_mut().zip(rows) {
                let packed = _mm256_loadu_si256(row.as_ptr().add(idx) as *const __m256i);
                let weights = _mm512_cvtepi8_epi16(packed);
                *sum = _mm512_add_epi32(*sum, _mm512_madd_epi16(values, weights));
            }
            idx += 32;
        }
    }
    sums.map(|sum| _mm512_reduce_add_epi32(sum))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_dot_avx2<const N: usize>(
    input: &L1Input,
    rows: [&[i8; PAIR_INPUT_SIZE]; N],
) -> [i32; N] {
    use std::arch::x86_64::*;

    let mut sums = [_mm256_setzero_si256(); N];
    unsafe {
        let mut idx = 0;
        while idx < PAIR_INPUT_SIZE {
            let values = _mm256_load_si256(input.0.as_ptr().add(idx) as *const __m256i);
            for (sum, row) in sums.iter_mut().zip(rows) {
                let packed = _mm_loadu_si128(row.as_ptr().add(idx) as *const __m128i);
                let weights = _mm256_cvtepi8_epi16(packed);
                *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(values, weights));
            }
            idx += 16;
        }
    }
    sums.map(|sum| {
        let sum128 = _mm_add_epi32(
            _mm256_extracti128_si256(sum, 1),
            _mm256_castsi256_si128(sum),
        );
        let sum64 = _mm_hadd_epi32(sum128, sum128);
        _mm_cvtsi128_si32(_mm_hadd_epi32(sum64, sum64))
    })
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l1_dot_neon<const N: usize>(
    input: &L1Input,
    rows: [&[i8; PAIR_INPUT_SIZE]; N],
) -> [i32; N] {
    use std::arch::aarch64::*;

    let mut low = [vdupq_n_s32(0); N];
    let mut high = [vdupq_n_s32(0); N];
    unsafe {
        let mut idx = 0;
        while idx < PAIR_INPUT_SIZE {
            let values = vld1q_s16(input.0.as_ptr().add(idx));
            for n in 0..N {
                let weights = vmovl_s8(vld1_s8(rows[n].as_ptr().add(idx)));
                low[n] = vmlal_s16(low[n], vget_low_s16(values), vget_low_s16(weights));
                high[n] = vmlal_high_s16(high[n], values, weights);
            }
            idx += 8;
        }
    }
    std::array::from_fn(|n| vaddvq_s32(vaddq_s32(low[n], high[n])))
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

    let mut input = L1Input([0i16; PAIR_INPUT_SIZE]);
    let (stm_half, opp_half) = input.0.split_at_mut(PAIR_SIZE);
    pairwise_products(stm_acc, stm_half);
    pairwise_products(opp_acc, opp_half);

    // Four neurons per pass: each input block is loaded once for all four rows.
    let mut l1 = [0i16; MAX_L1_SIZE];
    let mut neuron = 0;
    while neuron + 4 <= l1_size {
        let rows = std::array::from_fn(|k| net.l1_row(bucket, neuron + k));
        for (k, raw) in l1_dot::<4>(&input, rows).into_iter().enumerate() {
            l1[neuron + k] =
                quantised_activation(raw.wrapping_add(net.l1_bias(bucket, neuron + k)));
        }
        neuron += 4;
    }
    while neuron < l1_size {
        let [raw] = l1_dot::<1>(&input, [net.l1_row(bucket, neuron)]);
        l1[neuron] = quantised_activation(raw.wrapping_add(net.l1_bias(bucket, neuron)));
        neuron += 1;
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

    #[test]
    fn l1_simd_matches_scalar() {
        let mut acc = [0i16; HIDDEN_SIZE];
        let mut rows = vec![[0i8; PAIR_INPUT_SIZE]; 4];
        for (idx, value) in acc.iter_mut().enumerate() {
            *value = ((idx * 97 + 31) % 401) as i16 - 150;
        }
        for (n, row) in rows.iter_mut().enumerate() {
            for (idx, weight) in row.iter_mut().enumerate() {
                *weight = ((idx * 29 + n * 13 + 7) % 255) as i8;
            }
        }
        let rows: [&[i8; PAIR_INPUT_SIZE]; 4] = std::array::from_fn(|n| &rows[n]);
        let mut expected = L1Input([0i16; PAIR_INPUT_SIZE]);
        let (stm, opp) = expected.0.split_at_mut(PAIR_SIZE);
        pairwise_products_scalar(&acc, stm);
        pairwise_products_scalar(&acc, opp);
        let expected_sums = l1_dot_scalar(&expected, rows);
        // reference: the pre-batching per-neuron formula (wrapping, like the SIMD lanes)
        for (n, row) in rows.iter().enumerate() {
            let mut sum = 0i64;
            for half in 0..2 {
                for idx in 0..PAIR_SIZE {
                    let lhs = i64::from(acc[idx]).clamp(0, i64::from(FT_QUANT));
                    let rhs = i64::from(acc[idx + PAIR_SIZE]).clamp(0, i64::from(FT_QUANT));
                    sum += lhs * rhs * i64::from(row[half * PAIR_SIZE + idx]);
                }
            }
            assert_eq!(expected_sums[n], sum as i32);
        }

        let check =
            |products: fn(&[i16; HIDDEN_SIZE], &mut [i16]),
             dot: fn(&L1Input, [&[i8; PAIR_INPUT_SIZE]; 4]) -> [i32; 4]| {
                let mut input = L1Input([0i16; PAIR_INPUT_SIZE]);
                let (stm, opp) = input.0.split_at_mut(PAIR_SIZE);
                products(&acc, stm);
                products(&acc, opp);
                assert_eq!(input.0, expected.0);
                assert_eq!(dot(&input, rows), expected_sums);
            };
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                check(
                    |acc, out| unsafe { pairwise_products_avx512(acc, out) },
                    |input, rows| unsafe { l1_dot_avx512(input, rows) },
                );
            }
            if is_x86_feature_detected!("avx2") {
                check(
                    |acc, out| unsafe { pairwise_products_avx2(acc, out) },
                    |input, rows| unsafe { l1_dot_avx2(input, rows) },
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        check(
            |acc, out| unsafe { pairwise_products_neon(acc, out) },
            |input, rows| unsafe { l1_dot_neon(input, rows) },
        );
        check(pairwise_products_scalar, l1_dot_scalar);
    }
}
