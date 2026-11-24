"""Export real Qwen3.5 layer-zero operands for the native gated delta parity probe."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_5


def capture(checkpoint, fixture, destination):
    model, _ = load(str(checkpoint))
    case = next(row for row in json.loads(fixture.read_text()) if row["name"] == "code")
    text = model.language_model.model
    layer = text.layers[0]
    inputs = layer.input_layernorm(text.embed_tokens(mx.array([case["prompt"]])))
    arrays = {"x": inputs}
    original = qwen3_5.gated_delta_update

    def record(q, k, v, a, b, a_log, dt_bias, state, mask, **options):
        if state is not None or mask is not None:
            raise ValueError("The layer-zero probe requires an unmasked cold prompt")
        output, state = original(q, k, v, a, b, a_log, dt_bias, state, mask, **options)
        arrays.update(q=q, k=k, v=v, a=a, b=b, aLog=a_log, dtBias=dt_bias, y=output, state=state)
        return output, state

    qwen3_5.gated_delta_update = record
    try:
        arrays["attention"] = layer.linear_attn(inputs)
    finally:
        qwen3_5.gated_delta_update = original
    mx.save_safetensors(str(destination), arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    capture(args.checkpoint, args.fixture, args.destination)
