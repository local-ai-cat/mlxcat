# Standalone shared model accounting

The standalone server uses `EnginePool` as the shared admission, residency,
lease, and eviction owner for chat, WhisperKit speech, native rerank, and embeddings.
All four configured standalone model classes use the pool. **S15 remains open for
precise pressure attribution, hardware acceptance, and the consuming application.**

## Units and provenance

All pool charges and status `accounted_bytes` values are bytes. Speech footprint
is no longer a loaded-model count. `EnginePoolFootprint` distinguishes measured,
disk-estimated, configured-estimated, and unknown sizes. Unknown is `nil`, whereas
known zero is zero. Finite ceilings refuse an unknown auxiliary footprint before
calling its factory. An intentionally unbounded pool continues to allow unknown
sizes. Status exposes `footprint_source` and `accounted_bytes`; estimates never
populate `actual_size`. Legacy `estimated_size` retains its numeric wire shape,
so use the added fields to distinguish unknown from known zero.

WhisperKit's model-directory sum is a disk estimate, not measured CoreML resident
RAM. Empty/unreadable directories remain unknown. Rerank and embeddings use
discovery's disk estimate. Chat replaces its loading estimate with the loader's MLX active-memory
delta when available. That process-wide delta can include concurrent generation
or freeing; it is not exact per-model attribution. Registered descriptors are
fixed for a pool lifetime; changed model files/estimates require rebuilding that
registration/pool rather than assuming stale estimates are refreshed.

## Admission and lifetimes

Pending chat and auxiliary loads charge the existing pool before awaiting their
factory. A shared load gate serializes decisions. Same auxiliary resource loads
share one owned task; cancellation of one waiter cannot cancel another waiter's
allocation. A cancelled waiter receives no lease after the shared task completes.
The waiter currently waits for shared load completion before cancellation returns.
Failure cleans partial factory state before removing the pending charge.

Chat projected KV growth has idempotent reservation tokens in the same owner.
Model and request reservations compete for one ceiling. Chat factories remain
lazy until the reservation and pressure check succeed. Failed or cancelled
admission never starts generation and rolls back its reservation.

Every auxiliary acquire checks live pressure before work, including already-resident
models. Its waiter protects the model from eviction across the pressure callback;
denial or cancellation unwinds that waiter without issuing a lease. The check adds
held request projections, not the already-resident model charge. Initial loads also
retain the existing pre-allocation pressure check.

Active chat, file transcription, streaming speech, rerank scoring, and embedding evaluation hold
leases. Mixed LRU eviction considers idle resources across all four classes and
preserves chat pins. Pending/waiting/active/unloading resources cannot be selected.
Charges remain held until unload callbacks return. Current native owners remove
their model/pipeline references through nonthrowing callbacks; the interface does
not prove OS deallocation or support a fallible unload acknowledgement. An adapter
must not return from unload while still owning an active allocation.

Speech stream cancellation holds its lease through already-running push/finish
calls and raw cancellation acknowledgement. Chat cancellation waits for the
scheduler's actual terminal removal path, GPU synchronization when required, and
decoder task completion. It does not use a delay as an acknowledgement. HTTP chat
and legacy completion, Anthropic Messages, and Responses consumers use `withLifetime` around their complete post-start
operation, so an early stop, initial header failure, or subsequent write failure
also awaits teardown. Closed chat handles drop their retained model lease and
teardown closure. Repeated cancellation releases once.

## Two accounting scopes and an open pressure limitation

The pool is the logical owner of model charges plus projected request growth.
The watchdog separately samples observed MLX active/cache bytes plus the pool's
non-MLX charges, including pending speech. MLX pending model bytes and retained
request-growth estimates remain additive during pressure admission. Non-MLX
pending bytes are already in the sampler and are not added a second time.

This is intentionally conservative: materialized chat KV appears in the MLX
sample while its full logical growth reservation is still added. For example,
weights40 + actual KV30 + held reservation30 + new growth10 may be rejected under
a hard watermark92 even though actual projected use80 would fit. This false-denial
case remains **open**. A factory-start event is not allocation acknowledgement,
and `max(observed, logical reservations)` is unsafe: observed40 plus pending20
must not fit a ceiling50. Neither shortcut is used. Resolving this requires
trustworthy scheduler allocation/teardown telemetry, including batched/shared KV
and retained prefix caches, before subtracting materialized bytes. Pressure and
pool estimates do not establish an exact physical-memory bound.

## Verification and remaining acceptance

`SharedModelAccountingTests` uses injected CPU loaders, scoring, streams, and
barriers to exercise actual pool/adapter orchestration and HTTP handler entrypoints:
concurrent admission, pending/unloading charges, deduplication and cancelled
waiters, mixed LRU and pins, factory failure/retry, active scoring/stream leases,
lazy chat factories, acknowledged chat teardown, and conservative pressure
counterexamples. The scheduler/MLX/CoreML production modules are compiled
separately; CPU tests do not execute GPU synchronization or inference. Real mixed
model memory calibration, cancellation and HTTP disconnect acceptance remain
required on an authorized Apple Silicon runtime.

The configured embedding directory is excluded from chat discovery even if it also
lives under the chat model root (canonicalized paths include symlinks). Discovery
otherwise does not infer every model architecture's capability; unconfigured
embedding directories are not automatically promoted to embedding backends.
`--pin` remains chat-only, as its help now states; auxiliary status exposes pinned
false and load/unload does not imply pin support.

Embeddings, configured rerank models, and discovered speech models register their
descriptors at startup without loading weights. Speech catalog refresh also registers
newly discovered descriptors before listing them. Registration does not call factories.
Canonical lifecycle IDs are `embedding://<model-id>`, `rerank://<model-id>`, and
`speech://<engine-id>/<model-id>`; URL-encode them when used in route paths. The same
existing embedding factory, tokenizer, model call, normalization, evaluation, and
CPU-vector conversion run only after a pool lease is admitted. Loading failure
cleans up before releasing the charge; cancellation waits for the scoring call to
return before releasing its lease. Numerical embedding behavior and HTTP encoding/
dimension handling are unchanged. Real embedding model output and memory calibration
still require authorized hardware acceptance.

Existing generic model load/unload routes resolve registered pool resources through
the same descriptors used by status. Use the URL-encoded canonical resource ID
`embedding://<model-id>` to disambiguate ownership. Bare IDs resolve only when unique,
and chat and resolvable speech-registry references keep their existing precedence.
Existing speech fallback remains for unregistered registry models. An explicit
unload while scoring returns busy; idle unload waits for the cache owner to release its container. These routes do not claim
instant OS memory reclamation.

Additional embedding CPU regressions exercise all four classes active together,
shared factory cancellation, pending bytes, cleanup/retry, explicit lifecycle,
active scoring cancellation, unload acknowledgement and unknown/oversized rejection.
The injected cache returns deterministic vectors solely to prove orchestration and
exact result/request forwarding; it does not substitute for production inference.

The consuming app's `LocalAIResourceBudget` bridge is also open. This change does
not alter advertised app capabilities or prove an app-wide memory ceiling.

## Helper source migration and unchanged prompt boundary

The completion response builder now accepts `CompletionResponseMetadata(id:created:)`
in place of separate `id` and `created` arguments. This is a source-breaking Swift
helper signature change; the HTTP response schema, values, token accounting, and
runtime routes are unchanged. Known standalone callers and tests are migrated;
no consuming-app package or app source caller was found.

Native prompt history still does not carry prior assistant tool-call records in
`OpenAIChatMessage`. The structural cleanup does not deliver that missing history
capability or change current template input.


## Optional enclosing-process budget

`EnginePoolParentBudget` receives an allocation UUID and byte provenance before
model, auxiliary, or lazy request factories run. One returned handle follows each
allocation generation; pool usage leases do not add another parent charge. The
parent must not independently reclaim these handles. Pool unload remains the
resource owner, and concurrent request-release callers join the same acknowledgement.
The app adapter and global app/speech budget migration are still pending.

With a parent attached, native requests pass their existing allocation ownership
through SchedulerPrefixCache into actual SessionPrefixKVStore slots and borrowed
hits. Completion releases transient ownership, while retained slots keep that
same charge. Last-slot/hit eviction or explicit model cache clear retires it.
Repeated generations therefore do not accumulate retired charges. A slot first
discards its arrays, then releases ownership. No second parent registry or cache
allocation token is created. Standalone nil-parent behavior is unchanged.

The retained charge is conservative: it keeps the original request projection,
or a larger high-water sum of observed slot logical extents. Extents may share
underlying tensor storage and therefore are not exact unique allocator bytes;
`logicalExtent` labels this provenance. Larger observed extents reconcile as debt
in both pool and parent before request completion returns, even above the ceiling.
Synchronous stores may discover an estimate overshoot after generation allocated
it; this is debt accounting, not a claim of pre-admitted measured growth. Exact
per-allocation materialization and transfer telemetry remain open. Unknown native
KV profiles reach parent admission as unknown, never as a known zero estimate.

NativeModelEngine explicitly invalidates its ModelContext and MLXCatEngine.
MLXCatEngine drains registration and active operations, cancels its demultiplexed
streams through scheduler acknowledgement, joins the pump, and releases both
scheduler owners. Retaining an old public engine/lease handle cannot keep retired
weights or prefix arrays resident. The Foundation lifetime owner rejects new work
after invalidation starts. CPU reference tests prove explicit payload retirement
before parent release; authentic module builds prove native wiring. Current source
still needs serialized real-tensor/model hardware acceptance.
