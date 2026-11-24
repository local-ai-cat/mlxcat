#!/bin/zsh
# Install the same macOS Metal library that the embedded package builds,
# beside standalone binaries for distribution without SwiftPM resources.
#
#   scripts/build-metallib.sh                 # .build/release and .build/debug
#   scripts/build-metallib.sh .build/release  # explicit destination

set -euo pipefail
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
CHECKOUT="$REPO_ROOT/.build/checkouts/mlx-swift"
SRC="$CHECKOUT/Source/Cmlx/mlx-generated/metal"
BUILDER="$CHECKOUT/tools/build-macos-metallib.sh"
OUT="$REPO_ROOT/.build/mlxcat-metal-air/mlx.metallib"
typeset -a DESTS
if (( $# )); then DESTS=("$@"); else DESTS=("$REPO_ROOT/.build/release" "$REPO_ROOT/.build/debug"); fi

[[ -f "$BUILDER" && -d "$SRC" ]] || {
  echo "Missing the pinned mlx-swift Metal builder. Run swift build first." >&2
  exit 1
}
mkdir -p "${OUT:h}"
/bin/zsh "$BUILDER" "$SRC" "$OUT"

for dest in $DESTS; do
  [[ -d "$dest" ]] || continue
  mkdir -p "$dest/Resources"
  cp "$OUT" "$dest/mlx.metallib"
  cp "$OUT" "$dest/Resources/mlx.metallib"
  cp "$OUT" "$dest/Resources/default.metallib"
  echo "installed mlx.metallib → $dest"
done
