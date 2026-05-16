# KV Cache

The KV cache holds per-layer K and V tensors so subsequent decode
steps don't re-compute attention over the entire prefix. FFAI ships
one cache implementation today (raw fp16 / bf16); compressed variants
land in Phase 5.

## What's supported today

| Algorithm | When to use | Memory ratio | Status |
|---|---|---|---|
| **Raw fp16 / bf16** (`KVCache`, default) | All current models. | 1× | ✅ Shipped (Phase 2). |
| **Affine quantization** | Memory-constrained; modest decode-tok/s tax. | ~3.5× at 4-bit | ⏳ Planned (Phase 5). |
| **TurboQuant** | Best memory ratio at minimal quality loss. | ~6–8× at `turbo4v2` | ⏳ Planned (Phase 5). |
| **SSM / Hybrid** | Mamba / GatedDeltaNet (Qwen 3.5, NemotronH) | n/a — stores recurrent + conv state | ⏳ Planned (Phase 5). |
| **Batched** | Multi-stream decode (speculative, B>1 serving) | linear in B | ⏳ Planned (Phase 8+). |

The shipped raw cache is what every demo / test exercises today.
Compressed variants are deliberately deferred — the goal of Phase 4
(perf) was to nail the dispatch path; the goal of Phase 5 is to add
cache compression on top.

## How the cache works

Each layer holds its own `KVCache` instance. During the forward pass:

1. `Q`, `K`, `V` are projected from the post-RMSNorm hidden state.
2. RoPE is applied to `Q` and `K`.
3. **`kv_cache_update`** kernel appends the new `K`/`V` rows into the
   per-layer cache buffer **on the GPU**. No CPU↔GPU sync — the
   append enqueues onto the same `MTLCommandBuffer` as the rest of
   the layer.
4. `sdpa_decode` kernel scores the single query row against the full
   cached `K`/`V` slice up to the current position.

The cache buffer is allocated once per layer at the configured max
context length; appends bump an `offset` rather than reallocating.
This is the same shape MLX uses, minus the Metal compile latency.

```swift
let caches = model.engine.makeKVCache()  // [KVCache], one per layer
```

`makeKVCache()` is on the `LanguageModel` protocol — `LlamaModel` and
`Qwen3Model` both implement it. The user owns the cache lifetime;
keep it across `forward(...)` / `forwardSample(...)` calls for
multi-turn or streaming.

## Choosing a configuration

Today, only the raw cache exists, so the choice is implicit. The API
surface for picking variants is in place via `LoadOptions.kvCache`:

```swift
public enum KVCacheKind: Sendable {
    case raw                // unquantized fp16 / bf16  (shipped)
    // .affineQuantized — Phase 5
    // .turbo            — Phase 5
}
```

When the Phase 5 variants land, the same field selects them:

```swift
let model = try await Model.load(
    "mlx-community/Qwen3-4B-4bit",
    options: LoadOptions(kvCache: .affineQuantized)  // Phase 5
)
```

## Multi-turn / streaming

For multi-turn or streaming UIs, drive the loop yourself and reuse
the cache across calls (see [quickstart.md § Lower-level
API](quickstart.md#lower-level-api)):

```swift
let caches = model.engine.makeKVCache()

func respond(_ prompt: String, position: inout Int) -> String {
    var pos = position
    var nextToken = 0
    for t in model.tokenizer.encode(text: prompt) {
        nextToken = model.engine.forwardSample(tokenId: t, position: pos, caches: caches)
        pos += 1
    }
    var generated: [Int] = []
    while !isStop(nextToken) {
        generated.append(nextToken)
        nextToken = model.engine.forwardSample(tokenId: nextToken, position: pos, caches: caches)
        pos += 1
    }
    position = pos
    return model.tokenizer.decode(tokens: generated)
}
```

`pos` keeps advancing across calls; the cache holds every K / V row
appended so far.

## What's coming (Phase 5+)

From [`planning/plan.md`](../planning/plan.md):

- **Affine quantized KV cache** — 4 / 6 / 8-bit affine group-quant
  for K and V. Self-transitions raw → quantized at `startOffset` so
  prefill stays fast. ~3.5× memory at 4-bit; modest decode-tok/s tax.
- **TurboQuant cache** — block-wise MSE codec with asymmetric K/V
  bits (e.g. 4-bit K, 2-bit V — `turbo4v2`). Two attention paths:
  TurboFlash compressed-domain Metal kernel (default) or
  bulk-dequant → MLXFast SDPA (opt-in). ~6-8× memory.
- **`SSMStateCache`** — for Mamba / GatedDeltaNet families
  (Qwen 3.5 / NemotronH / Jamba). Stores conv + recurrent state
  instead of K/V; composes with attention layers via `CacheList`.
- **Batched cache** — slot-based admission for fixed-size batches;
  enables speculative decoding and multi-stream serving.

Each lands in its own commit with the corresponding kernels in
`metaltile`. Affine and TurboQuant are the highest-priority Phase 5
deliverables; SSM/GDN follow.

## See also

- [Architecture](architecture.md) — where the cache sits in the
  per-token dispatch loop.
- [Performance](performance.md) — current `tok/s` numbers, including
  what `kv_cache_update` (Phase 4 wave 1) bought us vs the original
  CPU-memcpy append.
- [Quantization](quantization.md) — weight quantization (a different
  axis from KV cache compression).
