# Endspiel

[![CI](https://github.com/thilo11/endspiel/actions/workflows/ci.yml/badge.svg)](https://github.com/thilo11/endspiel/actions/workflows/ci.yml)
[![Release](https://github.com/thilo11/endspiel/actions/workflows/release.yml/badge.svg)](https://github.com/thilo11/endspiel/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/thilo11/endspiel?logo=github&label=release)](https://github.com/thilo11/endspiel/releases/latest)
[![License](https://img.shields.io/github/license/thilo11/endspiel)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org)
[![Lichess](https://img.shields.io/badge/lichess-Endspiel%20%28Pi5%29-green?logo=lichess)](https://lichess.org/@/endspiel-pi)

> **Endspiel** /ˈɛnt.ʃpiːl/ *n.* (German) &nbsp; **1.** the final, decisive game. &nbsp; **2.** the final phase of a chess game.

A free, open-source chess engine for analysing your games, practising at your own
level and running engine matches. **Endspiel 2.0.0 is rated 3508 Elo on the CCRL Blitz
list** (805 games) and 3455 in its first games on CCRL 40/15. That is far beyond any
human player, and it runs on a desktop, a laptop, a Raspberry Pi 5 or an Android phone.

Download one file, point your chess GUI at it, and play. There is no net file to fetch
and nothing to configure: the neural network is built into the binary, and hash size and
thread count adapt to your machine. You can also challenge it on Lichess as
[endspiel-pi](https://lichess.org/@/endspiel-pi), which runs on a Raspberry Pi 5.

## Features

- **Works in any UCI GUI**: open-source ones like En Croissant, CuteChess, Scid, ChessX and
  DroidFish (Android), free ones like Arena, Banksia and Chess for Android, and commercial ones
  like ChessBase and Fritz
- **Analysis**: up to 256 lines at once (`MultiPV`), win/draw/loss percentages
  (`UCI_ShowWDL`), multi-threaded search (`Threads`) and pondering on the opponent's time
  (`Ponder`)
- **Practice at your level**: `UCI_LimitStrength` + `UCI_Elo` play as a beginner (1000), hobby
  player (1500), club player (2000) or strong club player (2500); full strength by default
  (see [Playing at reduced strength](#playing-at-reduced-strength))
- **Endgame tablebases**[^optional]: Syzygy WDL/DTZ up to 7 pieces (`SyzygyPath`). The engine
  converts won endings and keeps to drawing moves in drawn ones
- **Opening books**[^optional]: Polyglot `.bin`, EPD (`.epd`, with `bm` opcodes) or PGN (`.pgn`)
  (`BookFile`); `OpeningVariety` varies the first moves when no book is used
- **Time management for any clock**: bullet to classical, including repeating controls such
  as 40 moves in 15 minutes; tune with `Move Overhead` and `Slow Mover`. `Contempt` sets how
  hard it avoids draws
- **Runs everywhere**: one universal x86-64 binary per OS for Linux and Windows that detects
  your CPU and uses its fastest instructions; macOS Apple Silicon, Windows ARM64, a tuned
  Raspberry Pi 5 build, and an Android app (APK) that GUIs find automatically
- **Self-contained**: no runtime dependencies and no external net file
- **Chess960 too**: Fischer Random / Freestyle and Double Fischer Random are fully supported
  (`UCI_Chess960`, X-FEN and Shredder-FEN). Standard chess is the focus, though: the net was
  trained on standard openings, so expect weaker Chess960 opening play (see **Notes**)

[^optional]: Tablebases and opening books are optional and not included in the download.
    Endspiel plays without them; to use them, install the files yourself and point `SyzygyPath`
    or `BookFile` at them. Syzygy tables can be downloaded from the Lichess mirror
    (<https://tablebase.lichess.ovh/tables/standard/>): the 3–5-piece set is about 1 GB, 6 pieces
    about 150 GB and 7 pieces about 17 TB.

## Download

Get the file for your device from the
[latest release](https://github.com/thilo11/endspiel/releases/latest):

| Your device | File |
|-------------|------|
| Windows PC or laptop (Intel / AMD) | `endspiel-win-x64.exe` |
| Windows on ARM (e.g. Snapdragon laptops) | `endspiel-win-arm64.exe` |
| Mac with Apple Silicon (M1 or newer) | `endspiel-mac-arm64` |
| Linux PC (Intel / AMD) | `endspiel-linux-x64` |
| Raspberry Pi 5 (Raspberry Pi OS Trixie / Debian 13 or newer) | `endspiel-linux-arm64-pi5` |
| Android phone or tablet (Android 7 or newer) | `endspiel-android-arm64.apk` |

There is no installer and nothing else to download: the file is the whole engine.

- **Windows**: save the `.exe` in a folder of your choice, e.g. next to your chess GUI.
- **macOS**: the files are not code-signed, so macOS blocks them at first. Open a terminal in
  the download folder once and run:

  ```bash
  chmod +x endspiel-mac-arm64
  xattr -d com.apple.quarantine endspiel-mac-arm64
  ```

- **Linux / Raspberry Pi**: make the file executable with `chmod +x endspiel-linux-x64` (or
  `endspiel-linux-arm64-pi5`).
- **Android**: open the `.apk` to install it like an app (allow installing from your browser or
  file manager if Android asks). It does nothing on its own; your chess app finds it as an
  engine (see below).

## Usage

Endspiel has no board of its own. You use it through a chess GUI that supports UCI engines,
for example:

- **Open source**: En Croissant, CuteChess, Scid, ChessX; DroidFish on Android
- **Free**: Arena, Banksia; Chess for Android
- **Commercial**: ChessBase (load Endspiel as a Kibitzer to have it analyse alongside your
  game), Fritz

1. In your GUI, look for **Add engine**, **Install new engine** or **New UCI engine**.
2. Select the Endspiel file you downloaded.
3. Done. Endspiel picks its threads and memory for your machine; the defaults are fine.

Useful settings in the GUI's engine options:

- **`MultiPV`**: show several best moves at once when analysing.
- **`UCI_LimitStrength`** and **`UCI_Elo`**: play against a weaker Endspiel (see
  [Playing at reduced strength](#playing-at-reduced-strength)).
- **`Ponder`**: let it think on your time too.
- **`SyzygyPath`** / **`BookFile`**: point at tablebases or an opening book, if you have them.

**Android**: in DroidFish open **⋮ → Set engine…** and pick Endspiel. If it is not listed,
DroidFish lacks storage permission, which it needs to find installed engines: allow it under
**Settings → Apps → DroidFish → Permissions → Files and media (Storage)**, then reopen the engine
list.

**The GUI can't start the engine?** On macOS, run the two commands from
[Download](#download) first. On Linux, check that the file is executable. Running the file once
from a terminal with `bench` (below) shows whether it works.

### Details for experts

**One x86-64 binary per OS.** `endspiel-linux-x64` and `endspiel-win-x64.exe` run on any
x86-64-v2 CPU (SSE4.2 + POPCNT). At startup they detect the CPU and pick the fastest NNUE
code path, in this order:

1. `AVX512ICL`: the full Ice Lake-class feature set (with VNNI), also on AMD Zen 4/5
2. `AVX-512`: AVX-512F + AVX-512BW
3. `AVX2`
4. portable fallback

(v1.6.0 and earlier had separate `-v2`, `-v3` and `-v4` assets; `-v3` is the AVX2 build there.)

| File | Build |
|------|-------|
| `endspiel-linux-x64`, `endspiel-win-x64.exe` | x86-64-v2 baseline, runtime SIMD dispatch, PGO |
| `endspiel-mac-arm64` | `apple-m1`, PGO |
| `endspiel-win-arm64.exe` | generic ARM64 |
| `endspiel-linux-arm64-pi5` | `cortex-a76`, fat LTO + PGO; glibc ≥ 2.39 |
| `endspiel-android-arm64.apk` | `arm64-v8a` generic, fat LTO, minSdk 24; NEON or NEON+dotprod selected at runtime; Open Exchange (OEX) engine package |

**Bench** runs a fixed-depth search over a small set of positions and prints the node count,
time and NPS. Its first line names the selected SIMD tier, e.g.
`Running bench: ... (NNUE: AVX512ICL)`:

```bash
./endspiel-linux-x64 bench       # quick check that the binary works
./endspiel-linux-x64 bench 18    # deeper, for performance tuning
```

On Windows: `endspiel-win-x64.exe bench`. On macOS, a `bench` run is also the quickest way to
see a Gatekeeper error that a GUI would hide.

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

## Under the hood

Endspiel is a from-scratch chess stack in Rust with its own neural network, not a borrowed
net file. More on the project in [ABOUT.md](ABOUT.md).

- **Built from scratch**: bitboards, move generation, search, NNUE inference and the UCI layer
  are hand-written, with no external chess libraries. Net training uses
  [Bullet](https://github.com/jw1912/bullet); Syzygy probing uses `pyrrhic-rs`. Those are the
  only third-party pieces in the pipeline.
- **NNUE evaluation**: king-bucketed HalfKA with game state: 32 king buckets (board mirrored so
  the king is on files a–d) × (768 piece features, both kings included, + 17 castling/en-passant
  features) → 1536-wide feature transformer per side → pairwise CReLU product (768 per side, both
  sides concatenated) → 16 → 32 → 1 (SCReLU), with a separate 16/32/1 stack for each of 8
  piece-count buckets. The first dense layer takes u8 inputs through a sparse int8 kernel that
  skips zero inputs. Accumulators are updated lazily, only for positions that are actually
  evaluated, king moves refresh from a per-square cache, the static eval is cached in the
  transposition table, and the feature transformer sits on transparent huge pages where the OS
  offers them.
- **Training data**: Endspiel's own self-play (its search scores as labels), positions from the
  Lichess evaluation database, and, since 2026-09, score labels from
  [Leela Chess Zero](https://lczero.org)'s public T80 training data (see [CREDITS.md](CREDITS.md)). The net is embedded in the binary. `EvalFile` nets
  must match the 1536-wide feature transformer (1024-wide and piece-only nets no longer load) and
  may use dense L1/L2 widths other than 16/32 (up to 64), taken from the ESPNNUE2 header.
- **Search**: alpha-beta + PVS with iterative deepening and aspiration windows, null move,
  reverse futility, futility, razoring, ProbCut, SEE pruning, LMR, LMP, IIR, singular extensions
  (with double extensions and multi-cut) and passed-pawn extensions, 1- and 2-ply continuation
  history, capture history, and multi-facet correction history (pawn, non-pawn, minor/major, and
  continuation keys). History and correction tables persist across moves within a game (reset on
  `ucinewgame`). The search constants are SPSA-tuned
- **50-move rule awareness**: the eval is damped toward a draw as the 50-move counter runs, so
  the engine prefers moves that make progress over shuffling in a better position
- **Endgame knowledge without tablebases**: hand-written rules for K+B+N vs K, K+Q vs K+R and
  dead draws replace the net where it has no useful gradient
- **Tablebases at the root**: the engine stays on win- or draw-preserving moves instead of
  drifting into a loss, and orders winning moves so it converts within the 50-move rule
- **Multi-threading**: Lazy SMP with depth diversity
- **Hand-crafted eval fallback**: a tapered evaluation (pawn hash, mobility, king safety, pawn
  structure, threats, space, endgame scaling) used when no trained net is loaded
- **WDL output**: the win/draw/loss mapping is fit per net
- **Performance**: one universal x86-64 executable detects the host CPU at startup and
  dispatches its hot NNUE inference and accumulator operations to AVX512ICL (with VNNI),
  AVX-512, AVX2, or the portable fallback. AArch64 uses NEON, with the dotprod L1 kernel on cores
  that have it (`bench` prints the tier). All release builds except Windows ARM64 and Android use
  profile-guided optimisation; the Pi 5 and Android builds also use fat LTO

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
export ANDROID_NDK_HOME=/path/to/android-ndk   # r23+ (default: newest under ~/Android/Sdk/ndk)

scripts/build-android.sh
# binary: target/aarch64-linux-android/release/endspiel
```

One portable ARMv8-A binary: the NNUE picks its NEON or NEON+dotprod kernels at
runtime (`endspiel bench` prints which). The build uses fat LTO; `PGO=1` adds PGO
(training run under `qemu-aarch64`), which measured slower on a Snapdragon 8 Elite.

On-device the engine auto-tunes its defaults: `Threads` is the performance-core
count of the SoC, and `Hash` is ~128 MB per such core, capped at ~1/16 of
available RAM (floor 16 MB). Both remain overridable via UCI `setoption`.

The raw binary is only convenient under Termux or `adb` — a non-rooted device
can't `exec` a binary off shared storage. For normal use, install the **Open
Exchange (OEX) engine APK** instead (`endspiel-android-arm64.apk` on the release
page): it installs like an app and is auto-discovered as a UCI engine by GUIs
such as DroidFish and Chess for Android. The APK wraps the same binary; the
release workflow builds it, and `android/oex/` documents building it by hand.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Credits

Third-party software, tablebase/training tooling, and technique inspirations are
documented in [CREDITS.md](CREDITS.md).
