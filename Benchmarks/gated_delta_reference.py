"""Export real Qwen3.5 layer operands for the native gated delta parity probe."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache


def capture(checkpoint, fixture, destination, layer_index=0):
    model, _ = load(str(checkpoint))
    case = next(row for row in json.loads(fixture.read_text()) if row["name"] == "code")
    text = model.language_model.model
    layer = text.layers[layer_index]
    tokens = mx.array([case["prompt"]])
    hidden = text.embed_tokens(tokens)
    cache = make_prompt_cache(model)
    attention_mask = create_attention_mask(hidden, cache[text.fa_idx])
    recurrent_mask = create_ssm_mask(hidden, cache[text.ssm_idx])
    for index, previous_layer in enumerate(text.layers[:layer_index]):
        mask = recurrent_mask if previous_layer.is_linear else attention_mask
        hidden = previous_layer(hidden, mask=mask, cache=cache[index])

    inputs = layer.input_layernorm(hidden)
    linear = layer.linear_attn
    arrays = {
        "x": inputs,
        "in_proj_qkv": linear.in_proj_qkv(inputs),
        "in_proj_z": linear.in_proj_z(inputs),
        "in_proj_b": linear.in_proj_b(inputs),
        "in_proj_a": linear.in_proj_a(inputs),
    }
    original = qwen3_5.gated_delta_update

    def record(q, k, v, a, b, a_log, dt_bias, state, mask, **options):
        if state is not None or mask is not None:
            raise ValueError("The gated-delta probe requires an unmasked cold prompt")
        output, state = original(q, k, v, a, b, a_log, dt_bias, state, mask, **options)
        arrays.update(q=q, k=k, v=v, a=a, b=b, aLog=a_log, dtBias=dt_bias, y=output, state=state)
        return output, state

    qwen3_5.gated_delta_update = record
    try:
        arrays["attention"] = linear(inputs)
        arrays["post_norm"] = layer.post_attention_layernorm(hidden + arrays["attention"])
        arrays["mlp"] = layer.mlp(arrays["post_norm"])
    finally:
        qwen3_5.gated_delta_update = original
    mx.save_safetensors(str(destination), arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    capture(args.checkpoint, args.fixture, args.destination, args.layer)
