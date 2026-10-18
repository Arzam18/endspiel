pub mod accumulator;
pub mod features;
pub mod inference;
pub mod network;

pub const NUM_BUCKETS: usize = 32;
pub const PIECE_FEATURES: usize = 768;
pub const STATE_FEATURES: usize = 17;
pub const FEATURES_PER_BUCKET: usize = PIECE_FEATURES + STATE_FEATURES;
pub const INPUT_SIZE: usize = FEATURES_PER_BUCKET * NUM_BUCKETS;
pub const HIDDEN_SIZE: usize = 1536;
pub const PAIR_SIZE: usize = HIDDEN_SIZE / 2;
pub const PAIR_INPUT_SIZE: usize = PAIR_SIZE * 2;
/// Production dense widths (embedded `default.nnue`). EvalFile nets may use
/// any L1/L2 in `1..=MAX_L1_SIZE` / `1..=MAX_L2_SIZE` as recorded in the header.
pub const L1_SIZE: usize = 16;
pub const L2_SIZE: usize = 32;
pub const MAX_L1_SIZE: usize = 64;
pub const MAX_L2_SIZE: usize = 64;
/// Material-keyed output buckets: bucket = (piece_count - 2) / 4, matching
/// bullet's MaterialCount<8> (divisor = ceil(32/8) = 4).
pub const OUTPUT_BUCKETS: usize = 8;
pub const FT_QUANT: i32 = 127; // activation quantization (QA)
pub const NET_QUANT: i32 = 64; // dense-layer weight quantization (QB)

pub use accumulator::{Accumulator, PieceDelta, RefreshCache};
pub use inference::nnue_evaluate;
pub use inference::{SimdBackend, simd_backend};
pub use network::NnueNetwork;
