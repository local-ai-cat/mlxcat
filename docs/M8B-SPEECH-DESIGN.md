# M8b — Speech engine registry: design of record

Decided 2026-07-05 with the operator. Context: `docs-site/pages/mlxcat-omlx-parity.mdx`
(Local-AI-Chat repo) — audio gets done properly in-engine, not deferred to cat-integration.

## Shape

One registry, engines as peer adapters, consumers on top:

- **`MLXCatSpeech`** (new dependency-free target): the protocol layer —
  `SpeechEngineAdapter`, `SpeechStreamSession`, capabilities, `SpeechEngineRegistry`.
- **Adapter packages/targets** (one per engine, each owning its heavy dependency):
  WhisperKit (ANE) first; Parakeet/FluidAudio (ANE), whisper.cpp (CPU), MLX-ASR (GPU)
  later. App-side engines are **wrapped, not rewritten** — rewrites allowed only after
  the conformance battery is green against the wrapped version.
- **Consumers**: the existing `/v1/audio/transcriptions` route (via a registry-backed
  `AudioTranscriptionBackend` bridge), the app UI picker, the studio `TranscriptionTap`,
  and the M8c consensus dispatcher.

## The studio contract (why streaming is first-class from day one)

The Local-AI-Chat streaming/recording plan (Phase 2: audio lane + session clock)
consumes this registry through a `TranscriptionTap`. Its requirements are protocol
requirements:

1. **Timestamped PCM in, engine-relative word times out** — the tap maps words back
   onto the session clock; a session that returns bare text is useless to it.
2. **Partial → final segments with stable identity** — captions burn partials; the
   recording bundle wants finals; both from one session.
3. **Latency stats** — per-session `arrival − capture` style stats the studio's delay
   budget can consume.
4. **Silicon placement as capability + preference** — during a live stream the GPU is
   compositing; transcription must be steerable to ANE engines.

## Capabilities model

`SpeechEngineCapabilities`: `supportsFileTranscription`, `supportsStreaming`,
`silicon` (.ane/.gpu/.cpu/.multiple), `wordTimestamps`, `confidence`, `languages`
(nil = open). Lane mismatches are solved **inside adapters** (stream-only engines
bridge file→PCM-push; file-only engines buffer), never by consumers.

## Lifecycle (pool citizenship — phase 2)

Adapters expose `loadModel`/`unloadModel`/`loadedFootprint` so speech models can
appear beside LLMs as `audio_stt` model types (omlx's taxonomy) in `/v1/models`
and `/v1/models/status`. Load/unload routes delegate to the owning speech adapter,
but speech models are deliberately not enrolled in the LLM `EnginePool` LRU or
memory-ceiling eviction yet; that needs a dedicated cross-engine admission policy.
`loadedFootprint` is reported as `actual_size`. For CoreML/ANE-backed adapters this
is an approximate adapter working-set signal, not a precise per-model MLX tensor
allocation. When multiple models are loaded inside one adapter, the adapter
footprint is attributed once to the first loaded model in that adapter's status
rows so aggregate status memory does not double count the shared working set.

## Phasing

- **M8b-1 (this branch):** protocol + registry + registry-backed HTTP bridge +
  WhisperKit adapter (file lane real; stream session implemented via WhisperKit's
  streaming transcribe) + conformance test suite with a fake adapter + gated live
  tests. `/v1/audio/transcriptions` stops 501ing when an adapter is registered.
- **M8b-2:** pool citizenship (`audio_stt`), Parakeet adapter (wraps the app's
  FluidAudio engine at cat-integration), `models/status` integration.
- **M8b-3 — parallelism & "batching" (ruling 2026-07-06, revised same day):**
  audio batching is PLACEMENT first, scheduling last — Whisper-class/ANE
  engines can't merge audios into one forward pass (fixed-shape CoreML), and
  the user ruled AGAINST queue-jumping as the primary mechanism: run things
  genuinely in parallel instead. Mechanism order:
  1. **Multi-instance scale-out (primary)** — a busy engine spins up / reuses
     a second instance of the same model, memory-budget permitting (instances
     are pool citizens; LRU counts each). Instances get **compute-unit
     placement**: CoreML lets an instance pin `.cpuAndNeuralEngine` vs
     `.cpuAndGPU`, so whisper×2 runs ANE+GPU concurrently — parallel, not
     contending. Short-while-long = both just run.
  2. **Cross-engine busy-aware routing** — adapters expose queue-depth/busy to
     `resolveCandidates`; a short job routes to an idle engine on other
     silicon (whisper + Parakeet/CoreML) when the preferred one is grinding.
     Same signal M8c consensus needs.
  3. **Window-quantum yield (constrained fallback ONLY)** — when the memory
     budget forbids another instance AND every capable engine is busy, long
     jobs yield between decode windows (the sliding-window `trimmedSeconds`
     machinery already resumes) so a short job isn't stuck behind a 2h file.
     Not the design centerpiece; the escape hatch.
  **Priority classes** (`interactive` | `background`, HTTP param, surface
  bindings map to it) inform PLACEMENT — background jobs take non-preferred
  silicon and never trigger scale-out beyond budget — rather than queue
  position. Per-adapter contract additions: `maxConcurrentSessions`,
  `computeUnitPlacement`, `preemptionQuantum` (fallback), `supportsTrueBatch`
  (MLX-ASR on GPU later — the only lane where real batch inference exists).
- **M8c:** consensus dispatcher — fan one input to N adapters (ANE+GPU+CPU run
  truly parallel), fuse: confidence pick → ROVER voting → LLM fusion via the
  server's constrained-JSON decode. Post-hoc (raw-first recordings) before live.

## Dispatch control plane + default configuration (ALIGNED 2026-07-06)

Policy/mechanism split — this is the implementation contract for M8b-3:

- **`SpeechDispatchPolicy` — pure function** `decide(request, fleetSnapshot,
  config) -> Decision` where Decision ∈ {useInstance, spawnInstance(model,
  computeUnit), routeToEngine, enqueue, windowYield}. No side effects/state/
  clock — the policy is a reviewable table and every row gets a unit test with
  fake snapshots. No routing heuristics may live anywhere else (adapters are
  dumb executors).
- **`SpeechDispatchController` — actor** beside the registry: builds
  snapshots, executes decisions, accounts every instance in the shared pool
  (LRU/footprint from M8b-2 — this also enforces the `release()` discipline
  centrally; see the FluidAudio jetsam-OOM scar). Emits one structured trace
  per decision (decision + reason) and counters (spawns/fallbacks/yields).
  Config read/write over `/v1/audio/dispatch` (GET/PATCH) — app settings and
  the later UI drive the same object.
- **Busy = estimated wait** (windows remaining × s/window), never queue depth.

Default config (user-aligned 2026-07-06):

| key | default | rationale |
|---|---|---|
| `global.audioPoolShare` | 0.20 | audio burst can't evict the chat LLM |
| `global.maxTotalInstances` | 4 | 2 engines × 2 compute units |
| `global.scaleOut` | on | |
| `engine.*.maxInstancesPerModel` | 2 | ANE+GPU placement; a 3rd contends |
| `defaults.interactive` | parakeet → appleSpeech (iOS) / whisperkit (macOS) → wait | **Parakeet is the overall interactive default** (user ruling: ~600MB CoreML, streaming partials → live-reveal UI, powers translate). **Download-aware**: skip engines whose model isn't on disk (never auto-download on resolve); auto-promote to Parakeet once its model is present |
| `defaults.file` | whisperkit → parakeet | word-level timestamps (studio/editor + API contract) + long-file accuracy |
| `background.mayScaleOut` | false | background takes leftover capacity; only interactive spawns |
| `surfaces.translate.*` | pinned parakeet | motivating surface |

## Surface bindings (ruling 2026-07-06)

Engine selection is **per app surface**, layered ABOVE the registry:

- **Registry layer (headless — never changes for this):** always the full
  multi-engine catalog. API callers and services name `engineID:model`
  explicitly or pass capability preferences and get candidate resolution +
  fallback. No pinning lives here — a surface config can never remove options
  from the headless API.
- **Surface-binding layer (app-side):** each surface (translate-live,
  translate-conversation, transcribe-file, global transcription, studio tap, …)
  gets a `SurfaceEngineConfig` — a pinned `engineID` or a preference set
  ("stream-capable, prefer ANE"). Backend/settings storage first; UI exposure
  later is a per-surface picker reading the same config. At call time the
  binding resolves to registry preferences — sugar over `resolveCandidates`,
  never a parallel path. Translate screens are the motivating case: they need
  specific engines per screen without affecting any other consumer.

## Out of scope, permanently

Capture. Mic ownership, AVAudioSession, permissions stay app-side. The registry's
contract starts at "audio in" (file bytes or PCM buffers), ends at "text out".
