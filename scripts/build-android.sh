#!/usr/bin/env bash
#
# Cross-compile the endspiel UCI engine for 64-bit Android (arm64-v8a).
#
# Produces a standalone CLI binary that speaks UCI over stdin/stdout, runnable
# under Termux or via `adb push` + `adb shell`. At runtime the engine auto-tunes
# its defaults to the device — Hash from /proc/meminfo (~1/16 of available RAM,
# clamped to 16–1024 MB) and Threads from the performance-core count — both still
# overridable via UCI `setoption`. Like the x86-64 release, it is one portable
# binary (ARMv8.0-A) whose NNUE kernels pick NEON or NEON+dotprod at runtime.
#
# Fat LTO (+0.9% bench NPS on a Snapdragon 8 Elite vs thin). PGO is opt-in
# (PGO=1): unlike on the Pi 5 (+5–9%), it cost 3–4% on that tablet whether the
# profile was recorded under qemu or on the device (2026-10-24, bench 20, pinned
# to a 4.32 GHz Oryon core). A PGO profile only matches a build for the same
# target triple, so the training run uses an instrumented, statically linked
# aarch64-linux-android build of the same code: it needs no bionic linker and runs
# under qemu-aarch64 on an x86 host (natively on an aarch64 Linux host).
#
# Requirements:
#   - Android NDK r23+    (ANDROID_NDK_HOME / ANDROID_NDK_ROOT / ANDROID_NDK_LATEST_HOME,
#                          else the newest under $ANDROID_HOME/ndk or ~/Android/Sdk/ndk)
#   - rustup target:      rustup target add aarch64-linux-android
#   - cargo-ndk:          cargo install cargo-ndk
#   - for PGO=1:          rustup component add llvm-tools, and qemu-aarch64 (or
#                          qemu-aarch64-static) on PATH or as $QEMU_AARCH64
#
# Usage:
#   scripts/build-android.sh                 # API 24, target-cpu=generic, fat LTO
#   API=29 scripts/build-android.sh          # raise the minSdk floor
#   PGO=1 scripts/build-android.sh           # add PGO (bench under qemu-aarch64)
#   ANDROID_TARGET_CPU=cortex-a76 scripts/build-android.sh   # tune for one device
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API="${API:-24}"
TARGET="aarch64-linux-android"
ABI="arm64-v8a"
# Default to a portable ARMv8-A baseline so the binary runs on any 64-bit
# Android device (NEON is mandatory on AArch64, so the SIMD eval path is always
# available). Override ANDROID_TARGET_CPU to tune for a specific SoC.
CPU="${ANDROID_TARGET_CPU:-generic}"
LTO="${LTO:-fat}"
PGO="${PGO:-0}"
FLAGS="${RUSTFLAGS:-} -C target-cpu=$CPU"

if ! command -v cargo-ndk >/dev/null 2>&1; then
  echo "error: cargo-ndk not found. Install it with: cargo install cargo-ndk" >&2
  exit 1
fi

NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${ANDROID_NDK_LATEST_HOME:-}}}"
if [[ -z "$NDK" ]]; then
  for sdk in "${ANDROID_HOME:-}" "$HOME/Android/Sdk"; do
    if [[ -n "$sdk" && -d "$sdk/ndk" ]]; then
      NDK="$(find "$sdk/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)"
      [[ -n "$NDK" ]] && break
    fi
  done
fi
if [[ -z "$NDK" ]]; then
  echo "error: Android NDK not found; set ANDROID_NDK_HOME" >&2
  exit 1
fi
export ANDROID_NDK_HOME="$NDK"

rustup target add "$TARGET" >/dev/null 2>&1 || true

# --- PGO prerequisites ------------------------------------------------------
pgo_fail() {
  echo "error: PGO=1 but $1" >&2
  exit 1
}
if [[ "$PGO" != 0 ]]; then
  PROFDATA="$(find "$(rustc --print sysroot)" -type f -name llvm-profdata | head -1)"
  [[ -n "$PROFDATA" ]] || pgo_fail "llvm-profdata not found (rustup component add llvm-tools)"
fi
if [[ "$PGO" != 0 ]]; then
  if [[ "$(uname -m)" == aarch64 ]]; then
    RUNNER=()
  else
    QEMU="${QEMU_AARCH64:-$(command -v qemu-aarch64 || command -v qemu-aarch64-static || true)}"
    if [[ -n "$QEMU" ]]; then
      RUNNER=("$QEMU")
    else
      pgo_fail "qemu-aarch64 not found to run the training workload"
    fi
  fi
fi
if [[ "$PGO" != 0 ]]; then
  BUILTINS="$(find "$NDK/toolchains/llvm/prebuilt" -name libclang_rt.builtins-aarch64-android.a | head -1)"
  [[ -n "$BUILTINS" ]] || pgo_fail "libclang_rt.builtins-aarch64-android.a not found in $NDK"
fi

ndk_build() {
  CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_LTO="$LTO" \
    cargo ndk --target "$ABI" --platform "$API" \
    build --release --locked --bin endspiel
}

cd "$ROOT"
if [[ "$PGO" != 0 ]]; then
  PGO_DIR="$ROOT/target/android-pgo"
  rm -rf "$PGO_DIR"
  mkdir -p "$PGO_DIR"
  # Static linking against NDK r30's libc.a pulls in its bundled Rust std
  # (duplicate rust_eh_personality) and needs compiler-rt (__arm_za_disable),
  # which rustc's -nodefaultlibs leaves out. Only the profiling binary links this way.
  STATIC="-C target-feature=+crt-static -C link-arg=-Wl,--allow-multiple-definition -C link-arg=$BUILTINS"

  echo "Building instrumented endspiel (static, PGO gen) for $ABI, minSdk $API, target-cpu=$CPU ..."
  CARGO_TARGET_DIR="$ROOT/target/android-pgo-gen" \
    RUSTFLAGS="$FLAGS $STATIC -Cprofile-generate=$PGO_DIR/raw" ndk_build

  echo "Running the PGO training workload (bench) ${RUNNER[*]:+under ${RUNNER[*]}} ..."
  "${RUNNER[@]}" "$ROOT/target/android-pgo-gen/$TARGET/release/endspiel" bench
  if [[ ${#RUNNER[@]} -gt 0 ]]; then
    # qemu's default CPU has dotprod; profile the plain NEON kernels too, so they
    # are not optimised as cold code for dotprod-less cores (Cortex-A53 phones)
    QEMU_CPU=cortex-a53 "${RUNNER[@]}" "$ROOT/target/android-pgo-gen/$TARGET/release/endspiel" bench
  fi
  "$PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR/raw"
  FLAGS="$FLAGS -Cprofile-use=$PGO_DIR/merged.profdata"
fi

echo "Building endspiel for $ABI ($TARGET), minSdk $API, target-cpu=$CPU, LTO $LTO, PGO $([[ "$PGO" != 0 ]] && echo on || echo off) ..."
RUSTFLAGS="$FLAGS" ndk_build

OUT="target/${TARGET}/release/endspiel"
echo
echo "Built: $OUT  ($(du -h "$OUT" 2>/dev/null | cut -f1 || echo '?'))"
file "$OUT" 2>/dev/null || true
echo
echo "Run on a device, e.g.:"
echo "  adb push $OUT /data/local/tmp/endspiel && adb shell chmod +x /data/local/tmp/endspiel"
echo "  adb shell /data/local/tmp/endspiel bench"
