# FFAI performance — status, numbers, and the #1 bottleneck

## Current throughput (GPT-2-124M, incremental KV cache, correctness-first)

| platform | prefill | decode | one-time | output |
|---|---|---|---|---|
| CUDA (GB10 sm_121) | 52.5 tok/s | 24.6 tok/s (41 ms/tok) | 16s upload + 42s JIT | exact HF match |
| Metal (Apple GPU) | 11 tok/s | 9.4 tok/s (107 ms/tok) | 11s upload + 32s JIT | exact HF match |

These are **correctness-first** numbers, ~10–40× off production. They are NOT
bandwidth-bound: GPT-2 reads ~500 MB of weights/token, a ~2.5 ms floor at Metal
bandwidth, yet we measure 107 ms — **42× over**.

## Root cause: the Metal `Device` is a host-shadow copy-in/copy-out shim

`crates/backends/ffai-metal/src/lib.rs::dispatch` (and `synchronize`, which is a
no-op "dispatch is synchronous") does, **per op**:

1. read **every input operand's bytes from a host shadow** (`shadow(buf).data.read().clone()`),
2. hand them to `Context::dispatch_with_grid` as a `BTreeMap<String, Vec<u8>>` (re-uploaded),
3. run + wait,
4. copy each output back into the host shadow.

There are **no persistent device buffers**. Every dispatch re-stages all
operands through host memory and synchronizes. A token = ~120 ops, so weights
and activations cross the bus ~120×/token, each a commit+wait. That is the
107 ms, not compute and not host round-trips in the *test* (those barely
mattered — cutting them 3× changed nothing).

This also explains why "device-resident weights" gave 26× earlier: that win was
from **not recomputing host-side weight transposes** (`conv_t`) every step, not
from any actual device residency — the backend re-uploads regardless.

The CUDA backend already wraps real device buffers (`CudaDevice` alloc/htod/dtoh),
which is why GB10 is ~2.6× faster and has more headroom.

## The fix (highest-leverage perf item)

Make `MetalDevice` hold **persistent `MTLBuffer`s** keyed by the `DeviceBuffer`
handle, and a `dispatch` that **binds device buffers by handle** instead of
re-uploading bytes — upload once on `alloc`/`upload`, never copy back unless
`download` is called. Requires the `metaltile_runtime::Context` Metal face to
accept pre-resident buffer handles (a small Context API addition) rather than a
bytes map. Then:

- weights upload once, stay resident; activations stay resident across layers;
- `synchronize` becomes a real fence; dispatches enqueue into one command buffer
  per token and sync once (kills the per-op commit+wait).

Expected: the dispatch-bound 42× overhead largely collapses; F16 weights then
become worthwhile (halves the resident footprint + the now-dominant bandwidth).

## Secondary perf items (after the above)

- Cache the compiled metallib / cubin to kill the 32–42 s one-time JIT.
- Fuse the decode path (fewer dispatches) — esp. QKV proj + the SwiGLU/GELU MLP.
- Device-side KV cache (append + attend without host reorg/re-upload).
- Batched decode (batch > 1).
