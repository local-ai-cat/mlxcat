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
| mlx-swift-lm `Libraries/MLXLMCommon/KVCache.swift`, `Libraries/MLXLMCommon/SwitchLayers.swift`, and `Libraries/MLXLLM/Models/Qwen3MoE.swift` | Follow the native model/cache contracts and use the same module registration and packed projection patterns. No Inkling 2 model file in the inspected tree. |
| mlx-swift `Source/MLX/Ops.swift:1124,1126,1539,2520` | Use native MLX operations for dense and indexed packed matmul. |
| oMLX `omlx/patches/inkling2/model.py:1032–1071` | Describe the cache layout explicitly for each visit, including recurrent state. |
| vLLM `vllm/models/inkling2/nvidia/loop.py:342–389` | Cross-check explicit layer/pass visits and prelude reset. Its fused CUDA operators are architecture-specific. |
| Ollama `mlxrunner/model/inkling2/model.go:133–192` | Cross-check cache ownership and test split versus unsplit prefill because low-precision expert routing can vary with row count. |

## MLX operation gates

The Swift reference tree exposes NVFP4 and MXFP8 modes for both quantized matmul and indexed gather. Verify the pinned mlxcat build with synthetic packed round trips, including NVFP4 group 16, MXFP8 group 32 and without affine biases. Check per-expert scale semantics against shard headers and Python before using the fast path. The relative-position attention term, dynamic short convolution, attention residual mixing and controllers require model code. Start from composable MLX operations, then measure any specialized Metal replacements against the same fixtures. The public operation signatures alone are insufficient evidence that the weight formats run on the pinned Metal kernels.

## Implementation notes

The pinned MLX core instantiates Metal kernels for NVFP4 group 16 and MXFP8 group 32 in FP32, BF16 and FP16. The pinned Swift gather wrapper lacks an outer-scale argument. For NVFP4 vector routing, use the mlx-vlm FP32 fallback and apply each selected expert's stored `weight_scale_2` before rounding to the input dtype. For grouped matrix routing, MLX 0.32 scales the unpacked tile before rounding the weights to BF16. The pinned implementation materializes one equivalently scaled projection, performs native grouped matmul, and evaluates it before allocating the next workspace. Checkpoint weights stay packed.

The pinned MXFP8 vector dispatcher uses 512-element alignment. The reference uses 256-element alignment, which selects a different reduction for the 2,304-wide gate and up projections. Add zero columns to those packed tensors once and append corresponding zero activations. The synthetic Metal tests cover both formats, expert-specific scales, and the MXFP8 reduction at the older kernel's alignment.

The first native serving path serializes requests and keeps the paired attention and convolution states per visit. It uses the publisher's typed chat protocol and ports the Mini chat template to Swift because the pinned Jinja `tojson` filter rejects the publisher's `sort_keys` and `separators` arguments. The verifier fixture lane uses the mlx-vlm introduction revision `111ac668`. Its loop boundaries and residual equations agree with the reviewed `466069f` snapshot. The parity harness uses the fixture's prefill chunk size and reserves the last prompt token for the sampling forward as in that revision.

## Download metadata

The catalogue identity is [netprojects/Inkling-2-Mini-MLX-4-8bit](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4-8bit).
The checkpoint is a separate download. Its source model is
[Thinking Machines Lab's Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), with the publisher's
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
