#!/bin/zsh
source ~/.zshenv
export PATH="$HOME/Library/Python/3.9/bin:$HOME/.local/bin:$PATH"
root=$HOME/Library/Caches/models/mlx-community
HF=$(command -v hf || command -v huggingface-cli || echo "python3 -m huggingface_hub.commands.huggingface_cli")
for m in Qwen3.5-4B-MLX-4bit gemma-4-E2B-it-qat-4bit Qwen3.8-27B-4bit Qwen3-Coder-30B-A3B-Instruct-4bit; do
  echo "== downloading mlx-community/$m"
  if ${=HF} download "mlx-community/$m" --local-dir "$root/$m" 2>&1 | tail -2; then
    rm -rf "$root/$m/.cache"
    touch "$root/.done-$m"
    echo "== DONE $m ($(du -sh $root/$m | cut -f1))"
  else
    echo "== FAILED $m"
  fi
done
echo FETCH_ALL_DONE
