#!/bin/zsh
source ~/.zshenv
export PATH="$HOME/.local/bin:$PATH"
echo "--- omlx via python3.13 (upstream requires <3.14)"
pipx install --python /opt/homebrew/bin/python3.13 "git+https://github.com/jundot/omlx.git" 2>&1 | tail -3
echo "--- mlx-serve: official prebuilt arm64 binary (no brew trust, no nightly zig)"
mkdir -p ~/src/mlx-serve-bin && cd ~/src/mlx-serve-bin
gh release download --repo ddalcu/mlx-serve --pattern "mlx-serve-bin-macos-arm64.tar.gz" --clobber 2>&1 | tail -2
tar xzf mlx-serve-bin-macos-arm64.tar.gz 2>&1 | tail -2
find . -type f -perm -111 -name "mlx-serve*" | head -3
echo "--- results"
printf "omlx: %s\n" "$(command -v omlx || echo MISSING)"
printf "mlx-serve: %s\n" "$(find ~/src/mlx-serve-bin -type f -perm -111 -name mlx-serve | head -1 || echo MISSING)"
echo INSTALL3_DONE
