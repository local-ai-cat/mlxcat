"""Capture or evaluate a Qwen3.5 layer-0 affine projection in isolation."""

import argparse
import json
from pathlib import Path

import mlx.core as mx


PROJECTIONS = ("in_proj_qkv", "in_proj_z")


def capture(checkpoint, fixture, output):
    from mlx_lm import load

    model, _ = load(str(checkpoint))
    code = next(case for case in json.loads(fixture.read_text()) if case["name"] == "code")
    text = model.language_model.model
    layer = text.layers[0]
    inputs = layer.input_layernorm(text.embed_tokens(mx.array([code["prompt"]])))
    arrays = {"x": inputs}
    for name in PROJECTIONS:
        projection = getattr(layer.linear_attn, name)
        if (projection.group_size, projection.bits) != (64, 4):
            raise ValueError(f"{name} must use group-64, 4-bit affine weights")
        for field in ("weight", "scales", "biases"):
            arrays[f"{name}_{field}"] = getattr(projection, field)
    mx.save_safetensors(str(output), arrays)


def evaluate(source, output):
    tensors = mx.load(str(source))
    x = tensors["x"]
    results = {}
    for name in PROJECTIONS:
        keys = ("weight", "scales", "biases")
        weights, scales, biases = (tensors[f"{name}_{key}"] for key in keys)
        for rows in (1, 4, 5, 8, 16, x.shape[1]):
            for dtype, suffix in ((mx.bfloat16, "bf16"), (mx.float32, "f32")):
                results[f"{name}_{rows}_{suffix}"] = mx.quantized_matmul(
                    x[:, :rows].astype(dtype),
                    weights,
                    scales.astype(dtype),
                    biases.astype(dtype),
                    transpose=True,
                    group_size=64,
                    bits=4,
                )
    mx.save_safetensors(str(output), results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("checkpoint", type=Path)
    capture_parser.add_argument("fixture", type=Path)
    capture_parser.add_argument("output", type=Path)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("source", type=Path)
    evaluate_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.checkpoint, args.fixture, args.output)
    else:
        evaluate(args.source, args.output)


if __name__ == "__main__":
    main()
