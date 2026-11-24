"""Capture mlx-lm text parity fixtures for the native survey qualification."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache


def capture_code_trace(model, tokens, path):
    """Capture the first recurrent visit and subsequent decoder boundaries."""
    text = model.language_model.model
    cache = make_prompt_cache(model)
    hidden = text.embed_tokens(tokens[None, :])
    normalized = text.layers[0].input_layernorm(hidden)
    stages = {"embed_tokens": hidden, "input_layernorm": normalized}
    first = text.layers[0].linear_attn
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        stages[name] = getattr(first, name)(normalized)
    stages["linear_attn"] = first(normalized)

    attention_mask = create_attention_mask(hidden, cache[text.fa_idx])
    recurrent_mask = create_ssm_mask(hidden, cache[text.ssm_idx])
    for index, layer in enumerate(text.layers):
        mask = recurrent_mask if layer.is_linear else attention_mask
        hidden = layer(hidden, mask=mask, cache=cache[index])
        stages[f"layer_{index}"] = hidden
    mx.save_safetensors(str(path), stages)


def capture(model, tokenizer, case, name, output, chunking="remainder"):
    messages = case["messages"]
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    settings = case["settings"]
    thinking = settings.get("reasoning_effort", "none") != "none"
    tools = settings.get("tools")
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    if hasattr(prompt, "input_ids"):
        prompt = prompt.input_ids
    prompt = list(prompt)
    tokens = mx.array(prompt)
    cache = make_prompt_cache(model)

    # Match Swift's ceiling and reserve the same final token for sampling.
    step = 1024
    prefill = len(prompt) - 1 if len(prompt) > step else 0
    if chunking == "balanced" and prefill > step:
        windows = (prefill + step - 1) // step
        step = (prefill + windows - 1) // windows
    for start in range(0, prefill, step):
        model(tokens[None, start : min(start + step, prefill)], cache=cache)
        mx.eval([entry.state for entry in cache])
    row = model(tokens[None, prefill:], cache=cache)[0, -1].astype(mx.float32)

    generated, sampled = [], []
    for step in range(64):
        next_id = int(mx.argmax(row).item())
        generated.append(next_id)
        if step < 10:
            indices = sorted(set(mx.argpartition(-row, 7)[:8].tolist()))
            sampled.append({"ids": indices, "scores": row[mx.array(indices)].tolist()})
        if next_id in tokenizer.eos_token_ids or step == 63:
            break
        row = model(mx.array([[next_id]]), cache=cache)[0, -1].astype(mx.float32)

    if name == "code":
        capture_code_trace(model, tokens, output.with_suffix(".safetensors"))
    return {
        "name": name,
        "messages": messages,
        "thinking": thinking,
        "tools": tools,
        "prompt": prompt,
        "generated": generated,
        "sampled": sampled,
        "prefillChunking": chunking,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Verified local Qwen3.5 survey checkpoint")
    parser.add_argument("inkling_fixtures", type=Path, help="Directory containing the Inkling fixtures")
    parser.add_argument("output", type=Path, help="JSON receipt path")
    parser.add_argument("--chunking", choices=("remainder", "balanced"), default="remainder")
    args = parser.parse_args()

    model, tokenizer = load(str(args.checkpoint))
    tokenizer.chat_template = (args.checkpoint / "chat_template.jinja").read_text()
    prefix = "Inkling-2-Mini-MLX-4-8bit-"
    records = []
    for path in sorted(args.inkling_fixtures.glob(f"{prefix}*.json")):
        case = json.loads(path.read_text())
        if case.get("image_sha256"):
            continue
        record = capture(
            model, tokenizer, case, path.stem.removeprefix(prefix), args.output, args.chunking
        )
        records.append(record)
        print(record["name"], len(record["prompt"]), len(record["generated"]), flush=True)
        mx.clear_cache()

    if len(records) != 11:
        raise ValueError(f"Expected 11 text fixtures, found {len(records)}")
    args.output.write_text(json.dumps(records))


if __name__ == "__main__":
    main()
