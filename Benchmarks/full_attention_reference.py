"""Export layer-three operands from the Qwen3.5 mlx-lm reference."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_next
from mlx_lm.models.cache import make_prompt_cache


def capture(checkpoint, fixtures, out, layer_index):
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load(str(checkpoint))
    layers = model.language_model.model.layers
    target = layers[layer_index].self_attn
    attention_type = type(target)
    original = attention_type.__call__
    captures = {}
    boundaries = {}
    layer_type = type(layers[0])
    layer_call = layer_type.__call__
    first_layers = {id(layer): index for index, layer in enumerate(layers[:layer_index + 1])}

    def trace_layer(self, *args, **kwargs):
        result = layer_call(self, *args, **kwargs)
        if id(self) in first_layers:
            boundaries[f"layer_{first_layers[id(self)]}"] = result
        return result

    def trace(self, x, mask=None, cache=None):
        if self is not target:
            return original(self, x, mask, cache)
        batch, length, _ = x.shape
        projected = self.q_proj(x)
        q, gate = mx.split(projected.reshape(batch, length, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(batch, length, -1)
        kp, vp = self.k_proj(x), self.v_proj(x)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(kp.reshape(batch, length, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        v = vp.reshape(batch, length, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        qr, kr = self.rope(q, offset=offset), self.rope(k, offset=offset)
        ck, cv = (kr, v) if cache is None else cache.update_and_fetch(kr, v)
        attended = qwen3_next.scaled_dot_product_attention(qr, ck, cv, cache=cache, scale=self.scale, mask=mask)
        gated = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1) * mx.sigmoid(gate)
        y = self.o_proj(gated)
        values = dict(
            x=x, projected=projected, kp=kp, vp=vp, q=q, k=k, v=v,
            qr=qr, kr=kr, ck=ck, cv=cv, gate=gate, attended=attended,
            gated=gated, y=y, offset=mx.array(offset),
        )
        values["attended_contiguous"] = qwen3_next.scaled_dot_product_attention(
            mx.contiguous(qr), mx.contiguous(ck), mx.contiguous(cv),
            cache=cache, scale=self.scale, mask=mask,
        )
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unexpected attention mask {mask}")
            values["causal"] = mx.array(True)
        elif mask is not None:
            values["mask"] = mask
        captures.clear()
        captures.update(values)
        return y

    cases = {f["name"]: f for f in json.loads(fixtures.read_text())}
    names = [("code", None)]
    if "Kestrel" in checkpoint.name:
        names += [("reasoning", 10), ("short-chat", 3)]
    layer_type.__call__ = trace_layer
    attention_type.__call__ = trace
    try:
        for name, index in names:
            case = cases[name]
            cache = make_prompt_cache(model)
            tokens = mx.array([case["prompt"]])
            logits = model(tokens, cache=cache)[0, -1]
            mx.eval(logits)
            mx.save_safetensors(str(out / f"{name}-prefill.safetensors"), captures | boundaries | {"tokens": tokens})
            if index is not None:
                for token in case["generated"][:index]:
                    logits = model(mx.array([[token]]), cache=cache)[0, -1]
                    mx.eval(logits)
                captures["tokens"] = tokens
                captures["prefix"] = mx.array(case["generated"][:index])
                captures["logits"] = logits
                mx.save_safetensors(str(out / f"{name}-flip.safetensors"), captures | boundaries)
        print(checkpoint, names)

        long_tokens = mx.array([cases["long-context"]["prompt"]])
        cache = make_prompt_cache(model)
        for chunk in range(2):
            tokens = long_tokens[:, chunk * 1024 : (chunk + 1) * 1024]
            model(tokens, cache=cache)
            mx.eval([entry.state for entry in cache])
            mx.save_safetensors(
                str(out / f"long-chunk-{chunk}.safetensors"),
                captures | boundaries | ({"tokens": tokens} if chunk == 0 else {}),
            )

    finally:
        attention_type.__call__ = original
        layer_type.__call__ = layer_call


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args()
    capture(args.checkpoint, args.fixtures, args.output, args.layer)
