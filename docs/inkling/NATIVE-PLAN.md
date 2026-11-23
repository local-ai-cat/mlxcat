# Inkling 2 Mini native implementation plan

## Entry points and modules

- **Configuration and visit plan** decode `inkling2_loop`, validate the fixed loop and enumerate visits with a physical layer index, pass, and independent cache slot. The Mini plan has six prelude visits, three passes through layers 6–17, and six coda visits. It has 24 physical layers and 48 visits.
- **Text model** implements normalized embeddings, attention residual mixing, attention, short convolution, routed experts, and FP32 output projection with softcapping and vocabulary truncation. Use one parameter set per physical layer. Follow mlx-vlm `models/inkling2/language.py:576–782` at `466069f`.
- **Cache** pairs attention state with four convolution histories for each visit. Verify absolute positions, sliding-window rollover, sink handling, chunked prefill and continuation before turning on prefix persistence or ragged batching. Reusing weights across passes saves weight storage. Reusing a previous pass's KV would change the forward computation.
- **Checkpoint loading** validates the shard index and resolves exact per-module quantization modes and packed tensor names. Preserve FP32 tensors where the reference does and materialize in bounded steps. Validate each community build independently.
- **Vision and processor** implement the patch grid, normalization, vision encoder and adapter, then replace image placeholders before text prefill. Load the checkpoint tokenizer and reproduce the supplied Jinja template for system, user, assistant and tool turns. Verify prompt token IDs and image placeholder counts before comparing logits.
- **Native serving** integrates at `NativeModelLoader.loadModel`, supplying the model, tokenizer and processor to the existing native engine used by the embedder and `mlxcat-http`. Reuse streaming and HTTP dialect handling. Add the Inkling reasoning and tool delimiters and cover delimiter splits across streamed chunks.
- **Parity** compares same-weight, same-input greedy generation on this Mac. Record token IDs, reference top-k scores, error and tolerance, prefill chunk size, sampling settings and memory. Compare cold/warm runs and two conversations on one loaded model only after correctness gates. Run the fixed Whisker task set beside Qwen3.8-27B-4bit and record TTFT, throughput, peak memory, tool failures and task acceptance.

## Loop design

Snapshot the attention-residual stream immediately before the first core visit. At each restart, form an FP32 RMS-normalized message bank from completed core blocks and the last partial, excluding the prelude partial appended at the first core boundary. Restore the prelude snapshot. For every core layer on that pass, apply its four learned queries to the bank with scaled dot products and softmax. Feed the resulting messages into the attention, key, value and MLP controllers. After the final core pass, continue directly into the coda. The visit selects request state and the physical layer selects weights.

## References and implementation choices

| Reference | Practice |
| --- | --- |
| mlx-vlm `models/inkling2/language.py:721–782,945–955` | Canonical message-bank boundaries, pass restart and per-visit cache ownership. |
| mlx-lm `mlx_lm/models/cache.py`, `mlx_lm/generate.py` | Reuse mlxcat's existing chunked generation and batching seams. No Inkling 2 model file in the inspected tree. |
| mlx-swift-lm `Libraries/MLXLMCommon/KVCache.swift` and existing MoE models | Follow the native model/cache contracts and use the same module registration and packed projection patterns. No Inkling 2 model file in the inspected tree. |
| mlx-swift `Source/MLX/Ops.swift:1124,1126,1539,2520` | Use native MLX operations for dense and indexed packed matmul. |
| oMLX `omlx/patches/inkling2/model.py:1032–1071` | Describe the cache layout explicitly for each visit, including recurrent state. |
| vLLM `vllm/models/inkling2/nvidia/loop.py:342–389` | Cross-check explicit layer/pass visits and prelude reset. Its fused CUDA operators are architecture-specific. |
| Ollama `mlxrunner/model/inkling2/model.go:133–192` | Cross-check cache ownership and test split versus unsplit prefill because low-precision expert routing can vary with row count. |

## MLX operation gates

The Swift reference tree exposes NVFP4 and MXFP8 modes for both quantized matmul and indexed gather. Verify the pinned mlxcat build with synthetic packed round trips, including NVFP4 group 16, MXFP8 group 32 and without affine biases. Check per-expert scale semantics against shard headers and Python before using the fast path. The relative-position attention term, dynamic short convolution, attention residual mixing and controllers require model code. Start from composable MLX operations, then measure any specialized Metal replacements against the same fixtures. The public operation signatures alone are insufficient evidence that the weight formats run on the pinned Metal kernels.
