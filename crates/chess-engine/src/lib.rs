pub mod endgame;
pub mod eval;
pub mod polyglot;
pub mod search;
pub mod see;
pub mod strength;
pub mod syzygy;
pub mod threads;
pub mod tt;

use chess_common::{Board, Move, Score};
use chess_nnue::NnueNetwork;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use polyglot::OpeningBook;
use syzygy::SyzygyTB;
use threads::ThreadPool;
use tt::SharedTT;

pub use chess_nnue::{SimdBackend, simd_backend};

/// Search result returned by the engine.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: Score,
    pub depth: u8,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

/// Search parameters.
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub max_depth: u8,
    pub max_nodes: Option<u64>,
    pub move_time_ms: Option<u64>,
    pub white_time_ms: Option<u64>,
    pub black_time_ms: Option<u64>,
    pub white_inc_ms: Option<u64>,
    pub black_inc_ms: Option<u64>,
    pub moves_to_go: Option<u32>,
    pub infinite: bool,
    /// Whether to use NNUE evaluation instead of HCE.
    pub use_nnue: bool,
    /// Safety margin (ms) subtracted from allocated time to avoid flagging.
    pub move_overhead_ms: u64,
    /// Time management scaling factor (percentage). 100 = normal.
    pub slow_mover: u64,
    /// Contempt factor in centipawns. A positive value makes the engine
    /// slightly prefer avoiding draws (scored as -contempt from the perspective
    /// of the side to move at the root). 0 = neutral.
    pub contempt: i32,
    /// Singular extension mode.
    /// 0 = off, 1 = conservative (default), 2 = aggressive.
    pub singular_ext_mode: u8,
    /// Number of best lines to search and report (MultiPV). 1 = normal PV.
    pub multi_pv: usize,
    /// Internal search constants (LMR, pruning margins, corrhist).
    pub tune: TuneParams,
    /// Set while this is a `go ponder` search. While the flag is true every
    /// time-based stop is suspended (the UCI layer holds the move back
    /// anyway); `ponderhit`/`stop` clear it, which switches the running
    /// search to normal soft/hard time management. Elapsed time is measured
    /// from the search start, so time already pondered is credited against
    /// the budget automatically.
    pub ponder: Option<Arc<AtomicBool>>,
    /// UCI_Chess960. Opening time throttle is skipped: there is no book, and
    /// the first moves are the ones that need the long think.
    pub chess960: bool,
    /// UCI `go searchmoves`: search only these root moves (empty = all).
    pub search_moves: Vec<Move>,
}

/// Declares the search constants once: field, default, and the SPSA range.
/// Defaults are the production values; the range only matters to a `tune`
/// build, which exposes every field as a UCI spin option.
macro_rules! tune_params {
    ($($(#[doc = $doc:literal])* $name:ident: $default:literal, $min:literal..=$max:literal;)*) => {
        /// Internal search constants (LMR, pruning margins, extensions, history,
        /// aspiration, corrhist). Not exposed as UCI options in release builds;
        /// the `tune` feature exposes them for SPSA.
        #[derive(Clone, Debug)]
        pub struct TuneParams {
            $($(#[doc = $doc])* pub $name: i32,)*
        }

        impl Default for TuneParams {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }

        impl TuneParams {
            /// Every parameter as `(name, value, min, max)`.
            pub fn spec(&self) -> Vec<(&'static str, i32, i32, i32)> {
                vec![$((stringify!($name), self.$name, $min, $max),)*]
            }

            /// Set one parameter by name, clamped to its range. Returns false
            /// if the name is unknown.
            pub fn set(&mut self, name: &str, value: i32) -> bool {
                match name {
                    $(stringify!($name) => {
                        self.$name = value.clamp($min, $max);
                        true
                    })*
                    _ => false,
                }
            }
        }
    };
}

tune_params! {
    /// LMR base ×100
    lmr_base: 24, 0..=150;
    /// LMR divisor ×100
    lmr_div: 156, 100..=400;
    /// Divisor for history score's influence on LMR reduction
    hist_lmr_div: 1055, 300..=8000;
    /// LMR: eval this far below alpha at non-PV nodes reduces one more ply
    lmr_eval_margin: 150, 30..=400;
    /// Reverse futility pruning margin per depth when improving
    rfp_margin_imp: 70, 20..=200;
    /// Reverse futility pruning margin per depth when not improving
    rfp_margin_noimp: 98, 20..=250;
    /// Futility pruning margin per depth when improving
    fut_margin_imp: 41, 10..=200;
    /// Futility pruning margin per depth when not improving
    fut_margin_noimp: 67, 10..=200;
    /// Futility pruning base margin
    fut_base: 50, 0..=200;
    /// Futility: history divisor (bigger = less history influence)
    fut_hist_div: 64, 16..=256;
    /// SEE quiet-move pruning margin per depth
    see_quiet_margin: 13, 0..=100;
    /// SEE quiet-move pruning: history divisor
    see_quiet_hist_div: 64, 16..=256;
    /// SEE capture pruning margin per depth
    see_capture_margin: 30, 0..=150;
    /// History pruning threshold per depth
    hist_prune_margin: 3000, 500..=10000;
    /// Null move: eval-above-beta per extra ply of reduction
    nmp_eval_div: 200, 50..=600;
    /// Probcut margin above beta
    probcut_margin: 200, 50..=400;
    /// Probcut: static eval may sit this far below the probcut beta
    probcut_eval_slack: 100, 0..=300;
    /// Razoring base margin
    razor_base: 300, 50..=800;
    /// Razoring margin per depth
    razor_depth_mult: 200, 50..=500;
    /// Singular beta = TT score − se_beta_mult × depth / 16
    se_beta_mult: 32, 8..=96;
    /// Singular extension: TT score must exceed alpha by this much (non-PV)
    se_tt_margin: 50, 0..=150;
    /// Double extension when every alternative fails this far below singular beta
    double_ext_margin: 25, 0..=100;
    /// Quiet/continuation history bonus = mult × depth² / 16
    hist_bonus_mult: 16, 4..=64;
    /// Quiet/continuation history malus = mult × depth² / 16
    hist_malus_mult: 16, 4..=64;
    /// Capture history bonus/malus = mult × depth² / 16
    cap_hist_mult: 16, 4..=64;
    /// Aspiration window half-width at the start of an iteration
    asp_window: 30, 8..=100;
    /// Aspiration widening step after the first fail
    asp_delta: 60, 10..=200;
    /// Quiescence delta pruning margin above a queen
    qs_delta_margin: 200, 0..=600;
    /// Corrhist correction multiplier ×100 (0 disables)
    corrhist_mult: 82, 0..=300;
    /// Corrhist: cap on the summed correction (cp)
    corr_total_cap: 64, 16..=256;
    /// Corrhist: cap per facet (cp)
    corr_facet_cap: 32, 8..=128;
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_nodes: None,
            move_time_ms: None,
            white_time_ms: None,
            black_time_ms: None,
            white_inc_ms: None,
            black_inc_ms: None,
            moves_to_go: None,
            infinite: false,
            use_nnue: true,
            move_overhead_ms: 20,
            slow_mover: 100,
            contempt: 20,
            singular_ext_mode: 1,
            multi_pv: 1,
            tune: TuneParams::default(),
            ponder: None,
            chess960: false,
            search_moves: Vec::new(),
        }
    }
}

/// Callback for reporting search info (UCI info lines).
pub type InfoCallback = Box<dyn Fn(&SearchInfo) + Send>;

/// Search info reported during search.
#[derive(Debug, Clone)]
pub struct SearchInfo {
    pub depth: u8,
    pub seldepth: u8,
    pub score: Score,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
    pub hashfull: u16,
    /// Which MultiPV line this info belongs to (1-based). Always 1 in single-PV mode.
    pub multipv_line: usize,
}

/// The main engine that performs search.
pub struct Engine {
    stop: Arc<AtomicBool>,
    tt: Arc<SharedTT>,
    thread_pool: ThreadPool,
    hash_mb: usize,
    num_threads: usize,
    /// Safety margin (ms) subtracted from allocated time to avoid flagging
    /// due to communication/USB lag. Tournament default: 20ms.
    move_overhead_ms: u64,
    /// Time management scaling factor (percentage). 100 = normal,
    /// >100 = think longer, <100 = think faster. Range: 10..300.
    slow_mover: u64,
    /// Contempt factor in centipawns (draw score = -contempt for the root side).
    contempt: i32,
    /// Singular extension mode: 0 = off (default), 1 = conservative, 2 = aggressive.
    singular_ext_mode: u8,
    /// Whether to use NNUE evaluation.
    use_nnue: bool,
    /// The active NNUE network (embedded or loaded from file).
    nnue_net: Arc<NnueNetwork>,
    /// Syzygy tablebase handle. `None` when no path has been set.
    syzygy_tb: Option<SyzygyTB>,
    /// Opening book loaded via `BookFile`. `None` means no book.
    book: Option<Arc<OpeningBook>>,
    /// Internal search constants (LMR, pruning margins, corrhist).
    tune: TuneParams,
}

impl Engine {
    pub fn new() -> Self {
        // Start with a minimal TT; the UCI handler will resize to the
        // configured (or default) size at `isready` time, after all
        // `setoption` commands have been processed.  This avoids
        // allocating 4 GB × concurrency at startup before fastchess
        // has a chance to send `setoption name Hash value N`.
        Self::with_hash(1)
    }

    /// Create an engine with a specific hash size (MB).
    /// Prefer this over `new()` + `set_hash_mb()` when the desired hash size
    /// is known up front — avoids allocating and immediately discarding the
    /// default 4 GB transposition table.
    pub fn with_hash(hash_mb: usize) -> Self {
        let hash_mb = hash_mb.clamp(1, 131072);
        // Device-adaptive default: performance-core count on Android,
        // min(available_parallelism, 16) elsewhere.
        // Can be overridden via UCI "setoption name Threads value N".
        let num_threads = chess_common::platform::default_threads();
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            tt: Arc::new(SharedTT::new(hash_mb)),
            thread_pool: ThreadPool::new(num_threads),
            hash_mb,
            num_threads,
            move_overhead_ms: 20,
            slow_mover: 100,
            contempt: 20,
            singular_ext_mode: 1,
            nnue_net: NnueNetwork::embedded(),
            use_nnue: NnueNetwork::embedded().is_trained(),
            syzygy_tb: None,
            book: None,
            tune: TuneParams::default(),
        }
    }

    /// Get a handle to stop the search.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    /// Set the transposition table size in MB.
    pub fn set_hash_mb(&mut self, mb: usize) {
        let mb = mb.clamp(1, 131072);
        if mb != self.hash_mb {
            self.hash_mb = mb;
            self.tt = Arc::new(SharedTT::new(mb));
        }
    }

    /// Get the current hash size in MB.
    pub fn hash_mb(&self) -> usize {
        self.hash_mb
    }

    /// Set the number of search threads.
    pub fn set_threads(&mut self, threads: usize) {
        self.num_threads = threads.clamp(1, 256);
        self.thread_pool = ThreadPool::new(self.num_threads);
    }

    /// Get the current number of threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Set the move overhead in milliseconds.
    pub fn set_move_overhead(&mut self, ms: u64) {
        self.move_overhead_ms = ms.clamp(0, 5000);
    }

    /// Get the current move overhead in ms.
    pub fn move_overhead_ms(&self) -> u64 {
        self.move_overhead_ms
    }

    /// Set the slow mover percentage (time management scaling).
    pub fn set_slow_mover(&mut self, pct: u64) {
        self.slow_mover = pct.clamp(10, 300);
    }

    /// Get the current slow mover percentage.
    pub fn slow_mover(&self) -> u64 {
        self.slow_mover
    }

    /// Set the contempt factor in centipawns. Draws are scored as -contempt
    /// for the side to move at the search root. Range: 0..=100.
    pub fn set_contempt(&mut self, cp: i32) {
        self.contempt = cp.clamp(0, 100);
    }

    /// Get the current contempt factor.
    pub fn contempt(&self) -> i32 {
        self.contempt
    }

    /// Get singular extension mode (0 = off).
    pub fn singular_ext_mode(&self) -> u8 {
        self.singular_ext_mode
    }

    /// Load an opening book from `path` (Polyglot .bin, EPD, or PGN).
    /// An empty path clears any loaded book.
    pub fn set_book_file(&mut self, path: &str) -> Result<(), String> {
        let path = path.trim();
        if path.is_empty() {
            self.book = None;
            log::info!("BookFile cleared");
            return Ok(());
        }
        match OpeningBook::open(std::path::Path::new(path)) {
            Some(b) => {
                self.book = Some(Arc::new(b));
                log::info!("BookFile loaded from '{path}'");
                Ok(())
            }
            None => Err(format!("Failed to load opening book from '{path}'")),
        }
    }

    /// Return the loaded book handle, if any.
    pub fn book(&self) -> Option<Arc<OpeningBook>> {
        self.book.clone()
    }

    /// Check if NNUE evaluation is enabled.
    pub fn use_nnue(&self) -> bool {
        self.use_nnue
    }

    /// Load a NNUE network from a file path. Pass an empty string to revert to the embedded net.
    pub fn set_nnue_file(&mut self, path: &str) -> Result<(), String> {
        let net = NnueNetwork::from_path(path)?;
        self.use_nnue = net.is_trained();
        self.nnue_net = net;
        // TT entries cache the previous net's static evals
        self.clear_tt();
        Ok(())
    }

    /// Return the active NNUE network.
    pub fn nnue_net(&self) -> &Arc<NnueNetwork> {
        &self.nnue_net
    }

    /// Set the Syzygy tablebase handle directly from a pre-created instance.
    ///
    /// Use this when the tablebase was already initialized elsewhere (e.g. a
    /// shared handle cloned across threads) to avoid concurrent `tb_init`/
    /// `tb_free` races that can occur when each thread calls `set_syzygy_path`.
    pub fn set_syzygy_tb(&mut self, tb: Option<SyzygyTB>) {
        self.syzygy_tb = tb;
    }

    /// Temporarily take ownership of the loaded Syzygy handle.
    pub fn take_syzygy_tb(&mut self) -> Option<SyzygyTB> {
        self.syzygy_tb.take()
    }

    /// Load Syzygy tablebases from the given directory path.
    ///
    /// An empty path clears any previously loaded tablebases.
    /// Multiple directories can be separated by colons (`:`) on Unix or
    /// semicolons (`;`) on Windows.
    pub fn set_syzygy_path(&mut self, path: &str) -> Result<(), String> {
        // Drop existing handle first so the global singleton is freed before
        // re-initializing with the new path.
        self.syzygy_tb = None;

        let path = path.trim();
        if path.is_empty() {
            log::info!("Syzygy tablebases cleared");
            return Ok(());
        }

        match SyzygyTB::new(path) {
            Ok(tb) => {
                // A "successful" init can still expose zero usable tables when the
                // directory is empty or unreadable: pyrrhic returns Ok and
                // `max_pieces()` stays pinned at the compiled cap, so it cannot tell
                // a real load from a useless one. The only reliable check is a
                // functional probe of a canonical position the tables must cover.
                const TB_SANITY_FEN: &str = "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"; // KQ vs K (legal, no check)
                let usable = chess_common::Board::from_fen(TB_SANITY_FEN)
                    .ok()
                    .and_then(|b| crate::syzygy::probe_wdl(&tb, &b))
                    .is_some();
                if !usable {
                    return Err(format!(
                        "Syzygy path '{path}' loaded no usable tablebases (empty or unreadable directory?)"
                    ));
                }
                let max = tb.max_pieces();
                log::info!(
                    "Syzygy tablebases loaded from '{}' (max {} pieces)",
                    path,
                    max
                );
                self.syzygy_tb = Some(tb);
                Ok(())
            }
            Err(e) => Err(format!(
                "Failed to load Syzygy tablebases from '{path}': {e:?}"
            )),
        }
    }

    /// Return the loaded tablebase handle, if any.
    pub fn syzygy_tb(&self) -> Option<&SyzygyTB> {
        self.syzygy_tb.as_ref()
    }

    /// Get the current tune params.
    pub fn tune(&self) -> &TuneParams {
        &self.tune
    }

    /// Set one search constant by name (SPSA). Returns false if the name is unknown.
    pub fn set_tune_param(&mut self, name: &str, value: i32) -> bool {
        self.tune.set(name, value)
    }

    /// Run a search on the given position.
    pub fn search(
        &self,
        board: &Board,
        params: &SearchParams,
        info_callback: Option<InfoCallback>,
    ) -> SearchResult {
        // Inject current tune params into search params so threads get them.
        let mut p = params.clone();
        p.tune = self.tune.clone();
        self.stop.store(false, Ordering::SeqCst);
        self.thread_pool.search(
            board,
            &p,
            &self.stop,
            &self.tt,
            info_callback,
            &self.nnue_net,
            self.syzygy_tb.clone(),
            None,
            self.book.clone(),
            None,
        )
    }

    /// Signal the engine to stop searching.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Clear the transposition table (e.g., on ucinewgame).
    pub fn clear_tt(&self) {
        self.tt.clear();
    }

    /// Bump the TT generation without clearing entries (cheap alternative to
    /// `clear_tt` between successive games — old entries remain valid since the
    /// XOR key check already guarantees correctness).
    pub fn new_search_tt(&self) {
        self.tt.new_search();
    }

    /// Get a shared reference to the transposition table.
    pub fn shared_tt(&self) -> Arc<SharedTT> {
        Arc::clone(&self.tt)
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
