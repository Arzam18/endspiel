//! Device-capability probing used to pick sensible default `Hash` and
//! `Threads` values when the GUI/user hasn't set them explicitly.
//!
//! Both defaults adapt to the running machine on every platform:
//!
//! * **Hash** scales to ~1/16 of currently-available RAM (clamped to
//!   16–1024 MB), read via [`sysinfo`] so a single code path covers
//!   Linux/Android/macOS/Windows.
//! * **Threads** uses the count of "performance" cores on Linux/Android —
//!   detected from `/sys/.../cpufreq/cpuinfo_max_freq`, which is the only
//!   reliable source of per-core *max* frequency and thus the prime/big vs
//!   LITTLE tier split. `sysinfo` only exposes *current* per-core MHz, so on
//!   platforms without that sysfs interface we fall back to
//!   `available_parallelism` capped at [`MAX_AUTO_THREADS`].
//!
//! Detection is best-effort: every path falls back gracefully when the
//! relevant source is missing, and the results are cached so the probing cost
//! is paid once. The pure scaling/tier logic is factored into free functions
//! and unit-tested on any host.

use std::sync::OnceLock;

/// Hash fallback (MB) used as the RAM ceiling when RAM can't be probed.
const FIXED_HASH_MB: usize = 256;

/// Upper bound on the auto-selected thread count.
const MAX_AUTO_THREADS: usize = 16;

/// The RAM ceiling for auto-`Hash` is ~`1 / HASH_DIVISOR` of available RAM.
const HASH_DIVISOR: u64 = 16;
/// Floor for the auto-selected hash size (MB).
const HASH_MIN_MB: usize = 16;
/// Per-thread transposition-table budget (MB) — see [`default_hash_mb`].
const HASH_MB_PER_THREAD: usize = 128;

/// Cores whose max frequency is at least this percentage of the fastest core
/// are treated as "performance" cores. Tuned to group the prime + big tiers
/// together (big cores commonly clock ~80% of the prime core) while excluding
/// the much slower LITTLE tier (~50–60%) on big.LITTLE phones.
#[allow(dead_code)] // only consulted on Linux/Android
const PERF_CORE_FREQ_PCT: u64 = 70;

/// Number of search threads to use when the user hasn't set `Threads`.
///
/// On Linux/Android this is the count of "performance" cores (the top CPU
/// frequency tier), which avoids loading the slow LITTLE cores and the thermal
/// throttling that follows. Elsewhere it is `available_parallelism` capped at
/// [`MAX_AUTO_THREADS`]. Computed once and cached.
pub fn default_threads() -> usize {
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| performance_cores().unwrap_or_else(fallback_threads))
}

fn fallback_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(MAX_AUTO_THREADS))
        .unwrap_or(1)
}

/// Default transposition-table size (MB) for a search using `threads` threads,
/// when the user hasn't set `Hash`.
///
/// Adapts to the two things that actually matter for table sizing:
///
/// * **Throughput** — more threads generate nodes faster and saturate the
///   table sooner, so each is budgeted [`HASH_MB_PER_THREAD`].
/// * **RAM** — the result is capped at ~1/16 of currently-available RAM
///   (falling back to [`FIXED_HASH_MB`] if RAM can't be read).
///
/// The smaller of the two wins, with a floor of [`HASH_MIN_MB`]. The table is
/// shared across threads, so this scales the *useful* size — there is no
/// per-thread memory multiplier.
pub fn default_hash_mb(threads: usize) -> usize {
    hash_for(threads, available_ram_mb())
}

/// Pure: combine the per-thread throughput budget with the RAM ceiling.
fn hash_for(threads: usize, available_mb: Option<u64>) -> usize {
    let thread_budget = threads.max(1) * HASH_MB_PER_THREAD;
    let ram_cap = available_mb
        .map(|mb| (mb / HASH_DIVISOR) as usize)
        .unwrap_or(FIXED_HASH_MB);
    thread_budget.min(ram_cap).max(HASH_MIN_MB)
}

/// Log the detected device capabilities and the defaults derived from them.
/// Call once at startup; makes it easy to see, from a bot's logs, why a given
/// machine chose a particular `Hash`/`Threads` default.
pub fn log_summary() {
    log::info!(
        "device tuning: Threads={} (of {} logical), Hash={} MB",
        default_threads(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        default_hash_mb(default_threads()),
    );
    let (total, avail) = mem_total_avail_mb();
    log::info!("system memory: {avail} MB available of {total} MB total");
    if let Some(brand) = cpu_brand() {
        log::info!("cpu: {brand}");
    }
}

/// Pure: count "performance" cores given each core's max frequency (kHz).
/// Returns at least 1 and at most [`MAX_AUTO_THREADS`]; `None` if no usable
/// frequency data was supplied.
#[allow(dead_code)] // only invoked on Linux/Android, but always compiled for tests
fn count_performance_cores(max_freqs: &[u64]) -> Option<usize> {
    let max = max_freqs.iter().copied().max()?;
    if max == 0 {
        return None;
    }
    let threshold = max * PERF_CORE_FREQ_PCT / 100;
    let count = max_freqs.iter().filter(|&&f| f >= threshold).count();
    Some(count.clamp(1, MAX_AUTO_THREADS))
}

/// Best-effort usable RAM (MB) via sysinfo. `None` if it reads as zero.
fn available_ram_mb() -> Option<u64> {
    let avail = mem_total_avail_mb().1;
    (avail > 0).then_some(avail)
}

/// `(total_mb, available_mb)` from sysinfo. Either may be 0 if unavailable.
fn mem_total_avail_mb() -> (u64, u64) {
    use sysinfo::System;
    let Ok(mut sys) = System::new() else {
        return (0, 0);
    };
    sys.refresh_memory();
    const MB: u64 = 1024 * 1024;
    (sys.total_memory() / MB, sys.available_memory() / MB)
}

/// First CPU brand string reported by sysinfo, for logging.
fn cpu_brand() -> Option<String> {
    use sysinfo::System;
    let mut sys = System::new().ok()?;
    sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
    let brand = sys.cpus().first()?.brand().trim().to_string();
    (!brand.is_empty()).then_some(brand)
}

/// Probe per-core max frequencies from sysfs and count the performance tier.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn performance_cores() -> Option<usize> {
    use std::fs;

    let mut freqs = Vec::new();
    for entry in fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Match cpuN directories only (skip "cpufreq", "cpuidle", "cpu" etc.).
        let Some(digits) = name.strip_prefix("cpu") else {
            continue;
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let path = entry.path().join("cpufreq/cpuinfo_max_freq");
        if let Ok(s) = fs::read_to_string(&path)
            && let Ok(f) = s.trim().parse::<u64>()
        {
            freqs.push(f);
        }
    }
    if freqs.is_empty() {
        return None;
    }
    count_performance_cores(&freqs)
}

/// No reliable per-core max-frequency source outside Linux/Android.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn performance_cores() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_caps_by_ram_on_low_memory_devices() {
        // Phone: 4 perf cores would want 512 MB but ~2.5 GB RAM caps to 2500/16.
        assert_eq!(hash_for(4, Some(2500)), 156);
    }

    #[test]
    fn hash_scales_with_threads_when_ram_is_ample() {
        // 64 GB box: the throughput budget binds, not RAM.
        assert_eq!(hash_for(16, Some(64_000)), 16 * HASH_MB_PER_THREAD);
        assert_eq!(hash_for(1, Some(64_000)), HASH_MB_PER_THREAD);
    }

    #[test]
    fn hash_respects_floor_and_ram_fallback() {
        // Tiny RAM clamps up to the floor.
        assert_eq!(hash_for(1, Some(64)), HASH_MIN_MB);
        // RAM unreadable -> fall back to the fixed ceiling.
        assert_eq!(hash_for(8, None), FIXED_HASH_MB);
        assert_eq!(hash_for(1, None), HASH_MB_PER_THREAD);
    }

    #[test]
    fn performance_cores_excludes_little_tier() {
        // Snapdragon-style 1 prime + 3 big + 4 little (kHz).
        let freqs = [
            2_995_200, 2_419_200, 2_419_200, 2_419_200, 1_804_800, 1_804_800, 1_804_800, 1_804_800,
        ];
        // Prime (100%) + big (~81%) clear the 70% bar; little tier (~60%) excluded.
        assert_eq!(count_performance_cores(&freqs), Some(4));
    }

    #[test]
    fn performance_cores_homogeneous_uses_all() {
        let freqs = [2_400_000; 8];
        assert_eq!(count_performance_cores(&freqs), Some(8));
    }

    #[test]
    fn performance_cores_capped_and_floored() {
        assert_eq!(count_performance_cores(&[]), None);
        assert_eq!(count_performance_cores(&[0, 0]), None);
        let many = [3_000_000u64; 32];
        assert_eq!(count_performance_cores(&many), Some(MAX_AUTO_THREADS));
    }

    #[test]
    fn defaults_are_sane_on_this_host() {
        // Whatever this machine reports, the defaults must stay sane.
        assert!(default_hash_mb(default_threads()) >= HASH_MIN_MB);
        assert!(default_threads() >= 1);
    }
}
