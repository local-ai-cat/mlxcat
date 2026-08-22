#!/bin/zsh
source ~/.zshenv
echo "--- omlx (tap)"
brew tap jundot/omlx 2>&1 | tail -1
brew install omlx 2>&1 | tail -2
echo "--- mlx-serve (tap)"
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve 2>&1 | tail -1
brew install mlx-serve 2>&1 | tail -2
echo "--- vllm-mlx (pipx)"
pipx install vllm-mlx 2>&1 | tail -2
echo "--- results"
for c in omlx mlx-serve vllm-mlx; do printf "%s: %s\n" "$c" "$(command -v $c || echo MISSING)"; done
echo COMPETITORS_DONE
