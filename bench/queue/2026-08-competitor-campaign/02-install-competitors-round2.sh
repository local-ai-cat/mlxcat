#!/bin/zsh
source ~/.zshenv
export PATH="$HOME/.local/bin:$PATH"
echo "--- omlx from source (no brew trust needed)"
pipx install "git+https://github.com/jundot/omlx.git" 2>&1 | tail -3
echo "--- mlx-serve from source (zig; avoids brew trust)"
cd ~/src
[ -d mlx-serve ] || git clone -q --depth 1 https://github.com/ddalcu/mlx-serve.git
cd mlx-serve
brew bundle install --file=Brewfile 2>&1 | tail -2
zig build -Doptimize=ReleaseFast 2>&1 | tail -5
echo "--- results"
for c in omlx vllm-mlx; do printf "%s: %s\n" "$c" "$(command -v $c || echo MISSING)"; done
printf "mlx-serve binary: %s\n" "$(ls ~/src/mlx-serve/zig-out/bin/mlx-serve 2>/dev/null || echo MISSING)"
echo INSTALL2_DONE
