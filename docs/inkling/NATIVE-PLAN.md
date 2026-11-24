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
| Ollama `mlxrunner/model/inkling2/model.go:133–192` and `model/parsers/inkling2.go` | Cross-check cache ownership and split prefill. Use the typed output framing for streamed reasoning, text and tool calls. |

## MLX operation gates

The Swift reference tree exposes NVFP4 and MXFP8 modes for both quantized matmul and indexed gather. Verify the pinned mlxcat build with synthetic packed round trips, including NVFP4 group 16, MXFP8 group 32 and without affine biases. Check per-expert scale semantics against shard headers and Python before using the fast path. The relative-position attention term, dynamic short convolution, attention residual mixing and controllers require model code. Start from composable MLX operations, then measure any specialized Metal replacements against the same fixtures. The public operation signatures alone are insufficient evidence that the weight formats run on the pinned Metal kernels.

## Implementation notes

The pinned MLX core instantiates Metal kernels for NVFP4 group 16 and MXFP8 group 32 in FP32, BF16 and FP16. The pinned Swift gather wrapper lacks an outer-scale argument. For NVFP4 vector routing, use the mlx-vlm FP32 fallback and apply each selected expert's stored `weight_scale_2` before rounding to the input dtype. For grouped matrix routing, MLX 0.32 scales the unpacked tile before rounding the weights to BF16. The pinned implementation fuses unpacking, scaling and conversion into one projection workspace, performs native grouped matmul, and evaluates it before allocating the next workspace. Checkpoint weights stay packed.

The pinned MXFP8 vector dispatcher uses 512-element alignment. The reference uses 256-element alignment, which selects a different reduction for the 2,304-wide gate and up projections. Add zero columns to those packed tensors once and append corresponding zero activations. The synthetic Metal tests cover both formats, expert-specific scales, and the MXFP8 reduction at the older kernel's alignment.

The first native serving path serializes requests and keeps the paired attention and convolution states per visit. It uses the publisher's typed chat protocol and ports the Mini chat template to Swift because the pinned Jinja `tojson` filter rejects the publisher's `sort_keys` and `separators` arguments. The verifier fixture lane uses the mlx-vlm introduction revision `111ac668`. Its loop boundaries and residual equations agree with the reviewed `466069f` snapshot. The parity harness uses the fixture's prefill chunk size and reserves the last prompt token for the sampling forward as in that revision.

## Download metadata

The catalogue identity is [netprojects/Inkling-2-Mini-MLX-4-8bit](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4-8bit).
The checkpoint is a separate download with 19,525,880,832 bytes of tensor
payload. The runtime supports text and image input, reasoning output, and
tool calls. Its source model is
[Thinking Machines Lab's Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), with the publisher's
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).

## Prefill and decode follow-up

- **Prefix reuse.** oMLX `04e6e8e` records rolling KV and convolution
  state at known boundaries in `omlx/models/inkling2/cache.py` and restores the
  complete visit state in `model.py`. The Swift scheduler stores all six visit
  arrays at the last fixed prefill boundary and uses its existing byte-bounded
  session store. It keeps the original chunking and one-token sample boundary.
  Image requests use the model's own prefill because token IDs alone cannot
  identify image features.
- **Packed prefill.** mlx-vlm `111ac668` sorts routed rows before the grouped
  multiply and supplies the tensor scale to the packed kernel. The Metal trial
  uses upstream MLX `b835fb3`'s SIMD prefix scheduling from
  `mlx/backend/metal/kernels/utils.h` and applies the tensor scale within each
  16×32 output tile. The existing staged path remains available for A/B runs.
- **Vocabulary bandwidth.** MTPLX `3890be5` reads 16-bit vocabulary weights into
  FP32 accumulators in `mtplx/models/inkling/head.py`. The Swift trial follows
  the pinned MLX GEMV lane and reduction order for the singleton shape and
  retains the standard FP32 matmul for other shapes.
- **Affine checkpoint.** VisualInference `443e30e` retains the first 20
  layers' NVFP4 expert codes, stores the last four expert layers in 4-bit
  affine groups, and stores large dense projections in 8-bit affine groups.
  Swift reads the per-tensor metadata, keeps both affine layouts packed, and
  only dequantizes selected embedding rows. It uses 256-token prefill chunks
  for this checkpoint by default, following the publisher's memory sweep.
- **MTP.** MTPLX `fast/step.py` and `fast/round.py` use row-stable
  verification and commit only accepted state. The Swift verification probe
  compares a panel with ordinary singleton decoding on the same known tokens
  and cache. A throughput claim or a match on a few generated completions alone
  cannot establish equivalence of a new verifier.
