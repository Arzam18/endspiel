# Endspiel — Developer Guide

## Layout

```
├── Cargo.toml                    # Workspace root + endspiel binary
├── src/main.rs                   # Entry point
├── crates/
│   ├── chess-common/             # Shared types: Board, Move, Bitboard, FEN
│   ├── chess-core/               # Move generation, attack tables, validation
│   ├── chess-engine/             # Search, HCE evaluation, Syzygy WDL probing
│   ├── chess-nnue/               # NNUE inference + embedded net (build.rs)
│   └── chess-uci/                # UCI protocol handler
├── scripts/                      # build & setup helpers (Android, Syzygy download)
└── assets/                       # Gitignored — local resources
```

## Architecture

### Search (`chess-engine`)

Alpha-beta with iterative deepening and PVS:

- **Pruning**: null move, reverse futility, futility, razoring, SEE (captures + quiets), history pruning, LMP, ProbCut, singular multi-cut
- **Extensions**: singular extension, passed-pawn push (7th rank / promotion). No check extension: an in-check node is only floored at depth 1, and checking moves are merely exempt from LMR and quiet pruning. A 2026-09 probe of a real check extension was mixed on the 47...Rh1 family and grew bench nodes ~58%. Singular extension runs in conservative mode (`singular_ext_mode = 1`: non-PV nodes, depth ≥ 8, lower-bound TT entry) and has two follow-ups, both SPRT-tested on 2026-10-01: **multi-cut** (a non-PV node returns `se_beta` when the singular search also fails high above beta, +6.1 ± 3.9 Elo) and **double extensions** (+2 plies when every alternative fails ≥ 25 cp below `se_beta`, at most 8 per root path via `SearchState::double_exts`, +9.5 ± 5.2 Elo). A negative extension (−1 when not singular and `tt_score ≥ beta`) and four LMR terms (cut-node, tt-pv, tt-capture, late-capture LMR) failed or showed no gain
- **Reductions**: LMR, IIR
- **Move ordering**: TT → good captures (MVV-LVA) → killers → counter → history-sorted quiets → bad captures
- **Quiescence**: SEE-based pruning; **SMP**: Lazy SMP with depth diversity
- **Time management** (`compute_time_limit`, `long_game_time_cap` in `search.rs`): two budgets per move — a soft target (a typical position; the ID loop stops at ~0.35–3× of it depending on PV stability, score drops and node concentration) and a hard limit (≤ 3× soft, ≤ 80% of the clock). Clock controls are capped per move: above 5 min at `inc + time/20` early (17/14/12 later in the game), sudden death keeping that slice down to 400 s and blending to `time/34` by 300 s, so blitz (≤ 5 min) keeps the flag-safe cap. Repeating controls (`movestogo`, e.g. CCRL 40/15) skip those sudden-death caps and spread the clock over the moves left (`time/(movestogo+1)`); at 40/60 that used 79% of each period instead of 56% and measured +19.1 ± 11.1 Elo (782 games). On clocks of 400 s and more (not Chess960, not `movestogo`), `opening_ramp_permille` scales the soft target from 40% to 100% over plies 0–20, so the bank slice is not spent on known opening moves. Regression tests: `repeating_control_spends_the_period_without_flagging`, `ten_minute_sudden_death_spends_the_clock_without_flagging`

### Evaluation

Two backends:

- **NNUE** (default): king-bucketed HalfKA with state features (`crates/chess-nnue`). Input per perspective: 32 king buckets (half-board king square; the board is mirrored when the king is on files e–h) × 785 features = 768 piece-square (12 piece types incl. both kings, unmerged) + 17 state (4 friendly + 4 enemy castling-rights combinations, 9 en-passant categories). Feature transformer 1536 (i16, QA 127) → pairwise CReLU product of the two halves, rounded to u8 (`(a·b + 64) >> 7`) → 768 per side, side to move first → sparse int8 L1 16 (only non-zero 4-byte input chunks; VNNI / AVX-512 / AVX2 / NEON dotprod kernels) → L2 32 → 1, SCReLU on L1/L2 (QB 64), one L1/L2/output stack per output bucket `min((pieces − 2) / 4, 7)`; output ×400. Embedded at compile time via `include_bytes!`; dense L1/L2 read from the net header (`1..=64`) so architecture-trial nets load without a rebuild.
- **HCE**: tapered MG/EG with pawn, mobility, king safety, pawn structure, threat, center, connectivity, space, and material-imbalance terms. Fallback when the embedded net is zeroed by `build.rs`. Superseded by NNUE; HCE parameter work is out of scope for new PRs.

### NNUE net embedding

`crates/chess-nnue/build.rs` copies `nets/default.nnue` into `OUT_DIR`. Missing/wrong size → zero buffer (HCE fallback). Always `cargo build --release` after replacing the net.

### Chess960

Rook origins live on `Board::castle_rooks` (one per colour and side, so Double Fischer Random works) unless a FEN overrides them; internal castling still moves king→c/g and rook→d/f. FEN parsing accepts KQkq, X-FEN, and Shredder (`AHah`); emission is X-FEN. With `UCI_Chess960=true`, the UCI layer prints and parses king-takes-own-rook moves. Chess960 skips `OpeningVariety` and the opening time ramp. The net has no Chess960 opening data: a capped SF-labelled 960-opening overlay (endspiel-tools `chess960_opening_campaign.sh`) cut the net's 960 static error from 95 to 63 cp but lost 35 Elo in 960 games (2026-09-29, parked).

### Strength limit (`chess-engine/src/strength.rs`)

`UCI_LimitStrength` + `UCI_Elo` (1000–3000, default 3000 = full strength) snap to five classes. A limited class caps nodes per move, widens MultiPV, and draws the move among the final-depth lines with Stockfish's skill rule (push = `(weakness·(top − score) + delta·rand(weakness)) / 128`, scores clamped to ±2000); it searches with one thread, without tablebases and without `OpeningVariety`.

| Class | `max_nodes` | `multi_pv` | `weakness` | Measured vs SF19 `UCI_Elo` (60+0.6, 100 games) |
|-------|-------------|------------|------------|-----------------------------------------------|
| 1000 | 150 | 6 | 121 | 11.5% vs SF 1320 → ≈ 965 |
| 1500 | 500 | 6 | 85 | 46.5% → ≈ 1475 |
| 2000 | 2000 | 4 | 55 | 42.5% → ≈ 1945 |
| 2500 | 6000 | 3 | 30 | 47.5% → ≈ 2480 |

Recalibrate after a net or major search change with endspiel-tools `scripts/strength_calibrate.sh <binary>` (about an hour at 16 cores). Weakness near 128 is very sensitive (beginner: 117 ≈ 1180, 121 ≈ 965, 125 ≈ 500); the stronger classes are tuned mainly through `max_nodes`.

### Syzygy (`chess-engine/src/syzygy.rs`)

WDL probing via `pyrrhic-rs` at alpha-beta nodes when castling rights are gone and piece count ≤ loaded range. Empty-king-bitboard guard exists because move generation is pseudo-legal and `panic = "abort"` is set.

## Build

```bash
cargo build --release              # endspiel binary
```

### Native CPU optimisation / release contract

`.cargo/config.toml` is gitignored. For machine-local AVX/AVX-512 builds:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

**Release x86-64 binaries must stay on `-C target-cpu=x86-64-v2`** (SSE4.2 + POPCNT baseline). Advancing the whole binary to v3/v4/native defeats its portability floor. Only individually dispatched kernels (`#[target_feature]`) may use AVX2/AVX-512, with a portable reference path kept SSE4.2/POPCNT-clean and a scalar-equivalence test for each dispatched kernel. NNUE follows this pattern (baseline + AVX2 + AVX-512 variants, runtime dispatch; `endspiel bench` prints the selected tier).

### Release matrix

CI (`release.yml`, manual `workflow_dispatch` on tag):

| Artifact | target | LTO | PGO |
|----------|--------|-----|-----|
| linux-x64 / win-x64 | x86-64-v2 | thin | yes |
| win-arm64 | generic | thin | no (cross-built) |
| mac-arm64 | apple-m1 | thin | yes |
| linux-arm64-pi5 (Raspberry Pi 5, glibc ≥ 2.39) | cortex-a76 | fat | yes |
| android-arm64.apk | generic | thin | no (see `android/oex/`) |

### Releasing a version

Post-release `-dev` bump: main is always *not* a release.

1. On main, `workspace.package.version` → drop `-dev` (`1.0.1-dev` → `1.0.1`).
2. Commit `chore: release 1.0.1`, tag `v1.0.1`, push tag. **Tag push does NOT trigger CI** — dispatch manually with the tag as input:
   `gh workflow run release.yml -f tag=v1.0.1` (or Actions > Release > Run workflow).
3. Immediately bump to next patch `-dev` (`1.0.1` → `1.0.2-dev`) as a separate commit.

Pick a minor bump only when the queued work for the next cycle is known to be minor-worthy.

## Testing / Lint

```bash
cargo test --release --workspace          # required before merge
cargo clippy --workspace --all-targets    # zero warnings; #[allow] only for proven false positives, with a comment
cargo fmt                                 # on change
```

## Commit / PR conventions

[Conventional Commits](https://www.conventionalcommits.org) **one-liners only** — a single subject line, no body or footers: `<type>(<scope>): <summary>` with types `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`; scope is the crate short name (`engine`, `nnue`, `uci`, …).

PR checklist:
- Tests green, clippy clean
- `Bench: <number>` in description when search logic changed (see below)
- One logical change per PR; title matches commit format

### Bench as a search-change diff

`endspiel bench` = depth-14 over 7 pinned positions, 1 thread, fixed hash → **deterministic** node count, used to detect whether the search tree changed:

1. `endspiel bench` on parent commit → `Nodes: X`; build branch, same → `Nodes: Y`.
2. `X == Y`: behaviour-neutral or dead/guarded code (investigate for features). `X != Y`: tree changed — improvement still needs game testing (see **Promotion gate**).

## Promotion gate (nets and engine changes)

Every candidate — a replacement `crates/chess-nnue/nets/default.nnue` or a search/eval
change — is gated by one fastchess run: **candidate vs Stockfish 19 at a fixed
`nodes=10000` per move, 1000 games, no opening book** (startpos only; the candidate uses
`OpeningVariety=110` for variety since Stockfish is deterministic at fixed nodes).
`tc=10+0.1` for the candidate, `Hash=64`, `Threads=1` on both, concurrency 16.
**Promote only if the candidate scores ≥ 45%.** Reference: `c5e2aaf` scored 55.55%
(+38.7 ± 13.9 Elo) on 2026-09-27; `d665032` had scored 42.5% on 2026-09-24. Single search
changes are filtered first by SPRT against their parent (endspiel-tools `scripts/sprt.sh`).

```bash
fastchess \
  -engine cmd=target/release/endspiel name=candidate option.Hash=64 option.Threads=1 option.OpeningVariety=110 \
  -engine cmd=stockfish name=sf19-n10000 nodes=10000 option.Hash=64 option.Threads=1 \
  -each tc=10+0.1 -rounds 500 -games 2 -concurrency 16 -recover
```

Paste the final `Games/Wins/Losses/Draws/Points/Elo` line in the commit or PR. For a net,
note any architecture size change (`build.rs` checks it) and include updated
`WDL_A`/`WDL_B` if the win-rate ↔ centipawn mapping shifted.

## Syzygy Tablebases

```bash
bash scripts/download_syzygy.sh             # WDL + DTZ (~350 MB)
bash scripts/download_syzygy.sh --wdl-only  # ~150 MB
```
Lands in `assets/syzygy/` (gitignored). KRK probe sanity check (`go movetime 500` should return 28000 cp from depth 1):

```bash
(printf "uci\nisready\nsetoption name SyzygyPath value assets/syzygy\nucinewgame\nposition fen 8/8/8/8/4K3/8/4R3/7k w - - 0 1\ngo movetime 500\n"; sleep 2) \
  | ./target/release/endspiel
```
