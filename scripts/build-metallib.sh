#!/bin/zsh
# build-metallib.sh — put mlx.metallib next to the `swift build` products.
#
# `swift build` does not produce the Metal library that mlx-swift's Cmlx needs
# at runtime (Xcode builds do; the app's sidecar script copies it out of
# DerivedData). Without it every MLX call dies with
# "Failed to load the default metallib". This compiles the same kernel subset the
# test support (Tests/MLXCatTests/Support/MLXMetalRuntime.swift) compiles from
# the mlx-swift checkout and installs it beside the binaries; MLX JIT-compiles
# the rest from source at runtime.
#
#   scripts/build-metallib.sh                 # → .build/release/mlx.metallib (+ debug)
#   scripts/build-metallib.sh .build/release  # explicit product dir(s)

set -euo pipefail
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
SRC="$REPO_ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
AIR="$REPO_ROOT/.build/mlxcat-metal-air"
OUT="$AIR/mlx.metallib"
typeset -a DESTS
if (( $# )); then DESTS=("$@"); else DESTS=("$REPO_ROOT/.build/release" "$REPO_ROOT/.build/debug"); fi

[[ -d "$SRC" ]] || { echo "mlx-swift Metal sources not found at $SRC — run 'swift build' first" >&2; exit 1; }
mkdir -p "$AIR"

SOURCES=(
  arg_reduce.metal conv.metal gemv.metal layer_norm.metal random.metal rms_norm.metal rope.metal
  scaled_dot_product_attention.metal steel/attn/kernels/steel_attention.metal
)
typeset -a AIRS
for rel in $SOURCES; do
  air="$AIR/${${rel//\//_}%.metal}.air"
  if [[ ! -f "$air" || "$SRC/$rel" -nt "$air" ]]; then
    xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions \
      -mmacosx-version-min=14.0 -c "$SRC/$rel" -I "$SRC" -o "$air"
  fi
  AIRS+=("$air")
done
xcrun -sdk macosx metallib "${AIRS[@]}" -o "$OUT"

for dest in $DESTS; do
  [[ -d "$dest" ]] || continue
  mkdir -p "$dest/Resources"
  cp "$OUT" "$dest/mlx.metallib"
  cp "$OUT" "$dest/Resources/mlx.metallib"
  cp "$OUT" "$dest/Resources/default.metallib"
  echo "installed mlx.metallib → $dest"
done
