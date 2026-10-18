# Endspiel

[![CI](https://github.com/thilo11/endspiel/actions/workflows/ci.yml/badge.svg)](https://github.com/thilo11/endspiel/actions/workflows/ci.yml)
[![Release](https://github.com/thilo11/endspiel/actions/workflows/release.yml/badge.svg)](https://github.com/thilo11/endspiel/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/thilo11/endspiel?logo=github&label=release)](https://github.com/thilo11/endspiel/releases/latest)
[![License](https://img.shields.io/github/license/thilo11/endspiel)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org)
[![Lichess](https://img.shields.io/badge/lichess-Endspiel%20%28Pi5%29-green?logo=lichess)](https://lichess.org/@/endspiel-pi)
[![Lichess endspiel-engine](https://img.shields.io/badge/lichess-endspiel--engine-green?logo=lichess)](https://lichess.org/@/endspiel-engine)

> **Endspiel** /ˈɛnt.ʃpiːl/ *n.* (German) &nbsp; **1.** the final, decisive game. &nbsp; **2.** the final phase of a chess game.

A UCI chess engine written in Rust. Bitboards, move generation, search,
NNUE inference, and the UCI front end are hand-written — no external chess
libraries. The eval is a layer-stacked, king-bucketed HalfKA net with
castling and en-passant inputs, trained mostly on Endspiel's own self-play,
not a borrowed network file. It ships
as a self-contained binary (embedded net, Pi / Android / universal x86-64 builds).

Rust NNUE bots on Lichess are common. What is less common is a from-scratch
chess stack, a modern king-bucketed net with castling and en passant in the
features, and packaging that is meant to be installed rather than compiled
on a VPS. Strength sits well above human play and below the top of the
engine lists; details in [ABOUT.md](ABOUT.md).

## Features

- **Built from scratch** — move generation, search, NNUE inference, and the
  UCI layer are hand-rolled in Rust with no external chess libraries. Net
  training uses [Bullet](https://github.com/jw1912/bullet); Syzygy probing
  uses `pyrrhic-rs`. Those are the only third-party pieces in the pipeline.
- **Full UCI compliance** — works in any UCI GUI (Arena, CuteChess, Fritz, Banksia, Scid, …)
- **Chess960** — Fischer Random / Freestyle chess and Double Fischer Random (different setups for
  White and Black): X-FEN and Shredder-FEN, king-takes-own-rook castling with `UCI_Chess960`.
  Support is basic: the rules are complete, but the net was trained on standard-chess openings, so
  expect Chess960 opening play to be weaker than its standard play (see **Notes**)
- **NNUE evaluation** (default) — king-bucketed HalfKA with game state: 32 king buckets (board
  mirrored so the king is on files a–d) × (768 piece features, both kings included, + 17
  castling/en-passant features) → 1536-wide feature transformer per side → pairwise CReLU product
  (768 per side, both sides concatenated) → 16 → 32 → 1 (SCReLU), with a separate 16/32/1 stack for
  each of 8 piece-count buckets. Trained on Endspiel self-play (search scores as labels) and a public eval dump; since 2026-09 the mix also includes score labels from [Leela Chess Zero](https://lczero.org)'s public T80 training data (see [CREDITS.md](CREDITS.md)). The net is embedded in the binary. `EvalFile` nets must match the 1536-wide feature transformer (1024-wide and piece-only nets no longer load). `EvalFile` nets may use dense L1/L2 widths other than 16/32 (up to 64), taken from the ESPNNUE2 header.
- **HCE fallback** — tapered hand-crafted evaluation used when no trained
  NNUE net is available, with pawn hash, mobility, king safety, pawn
  structure, threats, space, and endgame scaling
- **Modern search** — alpha-beta + PVS with iterative deepening and
  aspiration windows, null move, reverse futility, futility, razoring,
  ProbCut, SEE pruning, LMR, LMP, IIR, singular extensions (with double
  extensions and multi-cut) and passed-pawn extensions,
  1- and 2-ply continuation history, capture history, and multi-facet
  correction history (pawn, non-pawn, minor/major, and continuation keys).
  History and correction tables persist across moves within a game
  (reset on `ucinewgame`)
- **Pondering** — thinks on the opponent's time (`go ponder` / `ponderhit`);
  enable via the `Ponder` UCI option in your GUI. A legal ponder move is
  still advertised when the PV is only one ply (book, tablebase, forced lines)
- **Multi-threading** — Lazy SMP with depth diversity (`Threads` UCI option)
- **MultiPV** — up to 256 principal variations for analysis (`MultiPV` UCI option)
- **Adjustable strength** — `UCI_LimitStrength` + `UCI_Elo` play as a beginner (1000), hobby player
  (1500), club player (2000) or strong club player (2500); full strength by default
- **Syzygy tablebases** — WDL/DTZ probing (up to 7-man) via `pyrrhic-rs`; at the root the engine stays on win- or draw-preserving moves instead of drifting into a loss
- **Opening books** — load Polyglot `.bin`, EPD (`.epd`, with `bm` opcodes), or PGN (`.pgn`) at runtime; format is auto-detected by extension
- **WDL output** — optional `wdl W D L` annotation on each `info` line
  (`UCI_ShowWDL`), with the win/draw/loss mapping fit per net
- **Contempt** and configurable time management (`Move Overhead`, `Slow Mover`); repeating time controls (`movestogo`, e.g. 40 moves in 15 min) are budgeted per period
- **Performance** — one universal x86-64 executable detects the host CPU at
  startup and dispatches its hot NNUE inference and accumulator operations to
  AVX512ICL, AVX-512, AVX2, or the portable fallback. AArch64 uses NEON;
  selected release builds also use profile-guided optimisation and fat LTO
- **Cross-platform** — Linux x86_64/ARM64, Windows x86_64/ARM64, macOS Apple Silicon
- **Self-contained binary** — no runtime dependencies, no external net file

## Download

Prebuilt binaries are on the [Releases](https://github.com/thilo11/endspiel/releases/latest) page.

> The universal filenames below begin with the release containing this change.
> v1.6.0 and earlier still expose the old `-v2`, `-v3`, and `-v4` x86-64
> assets; use `-v3` as the normal AVX2 choice for those older releases.

**x86-64 now ships as one universal binary per operating system.** Download
`endspiel-linux-x64` or `endspiel-win-x64.exe`; there is no longer a CPU tier
to choose. These replace the former `-v2`, `-v3`, and `-v4` release assets.

The executable itself has an x86-64-v2 baseline (SSE4.2 + POPCNT), while its
hot NNUE code contains several implementations. At startup it detects CPU and
OS support once, then selects the highest usable tier in this order:

1. `AVX512ICL` — the complete Ice Lake-class feature set, also available on
   suitable AMD Zen 4/5 CPUs
2. `AVX-512` — AVX-512F + AVX-512BW
3. `AVX2`
4. portable fallback

This lets the same file run on older v2-compatible machines and use
AVX512ICL automatically on a machine such as Beast. Run the binary's `bench`
command to verify the selected backend; it is printed on the first line.

| Platform | Binary | CPU requirement |
|----------|--------|-----------------|
| Linux x86-64 | `endspiel-linux-x64` | x86-64-v2 baseline; NNUE SIMD selected automatically |
| Windows x86-64 | `endspiel-win-x64.exe` | x86-64-v2 baseline; NNUE SIMD selected automatically |

**Other platforms:**

| Platform | Binary | Notes |
|----------|--------|-------|
| macOS Apple Silicon | `endspiel-mac-arm64` | `apple-m1`, PGO-optimised |
| Windows ARM64 | `endspiel-win-arm64.exe` | generic ARM64 |
| Raspberry Pi 5 | `endspiel-linux-arm64-pi5` | `cortex-a76`, fat LTO + PGO; needs Raspberry Pi OS (Trixie / Debian 13) or newer — glibc ≥ 2.39 |
| Android arm64 | `endspiel-android-arm64.apk` | `arm64-v8a`, minSdk 24 — installs like an app; pick "Endspiel" as a UCI engine in DroidFish / Chess for Android |

**Raspberry Pi 5.** Any RAM tier runs the engine; hash size is the only
thing that scales with it. The default is 128 MB per thread, capped at 1/16 of
the RAM available at startup — on a Pi 5 typically 200–500 MB — so set `Hash`
in your GUI to use more:

| Pi 5 RAM | Recommended `Hash` | Notes |
|----------|--------------------|-------|
| 4 GB | 512–1024 MB | usable for blitz/rapid; leave ≥2 GB for the desktop/browser |
| 8 GB | 2048–4096 MB | sweet spot — full Pi-5 strength at all time controls |
| 16 GB | 4096–8192 MB | only worth it for deep analysis or running other workloads alongside |

Expect NPS roughly 5–10× lower than a modern x86 desktop, so plan for a
noticeable Elo drop at fixed time controls and compensate by giving the
engine more thinking time. Active cooling is recommended: under
sustained engine load the SoC will thermally throttle without a fan.

## Usage

Endspiel is a command-line program used through a UCI-compatible chess GUI (Arena, CuteChess, Fritz, Banksia, etc.). Point the GUI at the binary — no further setup is required.

**Bench** — runs a fixed-depth search across a small set of positions and
prints the total node count, elapsed time, and NPS. Useful as a sanity
check that the binary runs end-to-end:

```bash
./endspiel-linux-x64 bench       # also prints the selected NNUE SIMD backend
./endspiel-linux-x64 bench 18    # deeper, for performance tuning
```

On Windows, use `endspiel-win-x64.exe bench`. A current universal build prints
an opening line such as `Running bench: ... (NNUE: AVX512ICL)`.

> **macOS users — run this once from a terminal before pointing a chess
> GUI at the binary.** The release binaries are not code-signed, so
> macOS Gatekeeper will quarantine them. If that's the case, this
> command surfaces the error clearly instead of leaving the GUI to
> silently fail to launch the engine. To clear the quarantine flag:
>
> ```bash
> chmod +x endspiel-mac-arm64
> xattr -d com.apple.quarantine endspiel-mac-arm64
> ```
>
> Then re-run `./endspiel-mac-arm64 bench` — once you get a Nodes/NPS
> line, the GUI will also be able to launch it.

## UCI Options

| Option | Default | Description |
|--------|---------|-------------|
| `Hash` | auto (RAM + threads) | Transposition table size in MB |
| `Threads` | auto (cores) | Search threads |
| `Move Overhead` | 20 | Time safety margin in ms |
| `Slow Mover` | 100 | Time usage scaling (%) — >100 thinks longer, <100 plays faster |
| `Ponder` | false | Think on the opponent's time; the GUI toggles this and drives `go ponder` / `ponderhit` |
| `Contempt` | 20 | Draw avoidance in centipawns |
| `EvalFile` | *(embedded)* | Path to an external `.nnue` / `quantised.bin` net |
| `BookFile` | *(disabled)* | Path to an opening book: Polyglot `.bin`, EPD `.epd`, or PGN `.pgn` (auto-detected by extension) |
| `SyzygyPath` | *(disabled)* | Path to Syzygy tablebase directory |
| `MultiPV` | 1 | Number of principal variations to report (1–256) |
| `OpeningVariety` | 0 | Opening spice: 0 = off; otherwise pick at random among MultiPV moves within this many centipawns of the best, for the first 8 plies of standard chess (ignored in Chess960) |
| `UCI_ShowWDL` | false | Append `wdl <win> <draw> <loss>` (0–1000) to each info line |
| `UCI_Chess960` | false | Chess960 move notation: castling is sent and read as king-takes-own-rook (GUIs set this for Chess960 games) |
| `UCI_LimitStrength` | false | Play at the `UCI_Elo` class instead of full strength |
| `UCI_Elo` | 3000 | Strength class when `UCI_LimitStrength` is on: snaps to the nearest of 1000 / 1500 / 2000 / 2500; 3000 = full strength |

Set `BookFile` or `SyzygyPath` to a valid path to enable; clear to disable. No separate toggle is needed.

### Notes

- **`Hash`** — increase for long time controls or analysis; watch `hashfull` in engine output (permille, so 950 = 95%). The default adapts to the machine: ~128 MB per search thread (more threads fill the table faster), capped at ~1/16 of available RAM, floor 16 MB. So it grows with both core count and RAM, and tracks an explicit `Threads` setting.
- **`Threads`** — Lazy SMP; scaling is sub-linear. Stick to physical core count. On Linux/Android the default is the performance-core count (the top CPU-frequency tier), which avoids the slow LITTLE cores and the thermal throttling they invite; elsewhere it's `min(available, 16)`.
- **`EvalFile`** — load an alternate net at runtime without rebuilding. Clear to revert to the embedded net.
- **`UCI_LimitStrength` / `UCI_Elo`** — see *Playing at reduced strength* below.
- **`UCI_Chess960`** — Chess960 positions (including Double Fischer Random) are accepted as X-FEN or
  Shredder-FEN either way; the option only changes how castling is written. The net has no Chess960
  opening training, so its evaluations right after the start are less reliable than in standard
  chess. `OpeningVariety` is ignored in Chess960, and a standard-chess `BookFile` finds no moves there.
- **`SyzygyPath`** — WDL/DTZ probing for up to 7-man endgames. Multiple directories: `:` on Linux/macOS, `;` on Windows.
- **`OpeningVariety`** — only affects the first 8 plies of standard chess; 0 (default) always plays the best move. Chess960 always plays the best move, and also skips the early-opening time throttle so the first moves use the normal clock budget.

> **Fritz 20 (Windows):** Fritz manages Syzygy and opening books through its own systems.
> Set the tablebase path in Fritz's settings — it forwards it to Endspiel automatically.
> To let Endspiel use its own `BookFile`, disable Fritz's opening book in the match settings.
> Verify loading via View → Engine Output: a successful load prints `info string BookFile loaded from '...'`.

### Playing at reduced strength

For practice games, switch on `UCI_LimitStrength` and pick a class with `UCI_Elo`:

| `UCI_Elo` | Class |
|-----------|-------|
| 1000 | beginner |
| 1500 | hobby player |
| 2000 | club / tournament player |
| 2500 | strong club / tournament player |
| 3000 (default) | full strength |

The classes are intended to roughly match human Elo; values in between snap to the nearest class. Most GUIs show both options in the engine settings;
from a terminal:

```
setoption name UCI_LimitStrength value true
setoption name UCI_Elo value 1500
```

Weaker classes miss tactics and occasionally blunder — the beginner can overlook a mate in one.
At reduced strength the engine moves quickly whatever the clock, uses a single thread and no
tablebases, and ignores `OpeningVariety`.

## Build from Source

Requires Rust 1.99.1+.

```bash
cargo build --release
# binary: target/release/endspiel
```

### Android (arm64)

Cross-compile a standalone CLI binary for 64-bit Android. It speaks UCI over
stdin/stdout like the desktop build and runs under Termux or via `adb shell`.

```bash
rustup target add aarch64-linux-android
cargo install cargo-ndk
export ANDROID_NDK_HOME=/path/to/android-ndk   # r23+

scripts/build-android.sh
# binary: target/aarch64-linux-android/release/endspiel
```

On-device the engine auto-tunes its defaults: `Threads` is the performance-core
count of the SoC, and `Hash` is ~128 MB per such core, capped at ~1/16 of
available RAM (floor 16 MB). Both remain overridable via UCI `setoption`.

The raw binary is only convenient under Termux or `adb` — a non-rooted device
can't `exec` a binary off shared storage. For normal use, install the **Open
Exchange (OEX) engine APK** instead (`endspiel-android-arm64.apk` on the release
page): it installs like an app and is auto-discovered as a UCI engine by GUIs
such as DroidFish and Chess for Android. The APK wraps the same binary; the
release workflow builds it, and `android/oex/` documents building it by hand.

> **DroidFish doesn't list Endspiel?** DroidFish only scans for installed OEX
> engines when it has storage permission. Grant it under **Settings → Apps →
> DroidFish → Permissions → Files and media (Storage) → Allow**, then reopen the
> engine selector (**⋮ → Set engine…**) — Endspiel will appear next to Stockfish
> and CuckooChess.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Credits

Third-party software, tablebase/training tooling, and technique inspirations are
documented in [CREDITS.md](CREDITS.md).
