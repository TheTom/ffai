# FFAI performance — status, numbers, and the #1 bottleneck

## Current throughput (GPT-2-124M, incremental KV cache)

| platform | prefill | decode | one-time | output |
|---|---|---|---|---|
| CUDA (GB10 sm_121) | 52.5 tok/s | 24.6 tok/s (41 ms/tok) | 16s upload + 42s JIT | exact HF match |
| Metal — resident bufs | 24 tok/s | **17.6 tok/s (57 ms/tok)** | 11s upload + 32s JIT | exact HF match |
| Metal — old shim | 11 tok/s | 9.4 tok/s (107 ms/tok) | (re-uploaded weights/dispatch) | exact HF match |

## Per-dispatch micro-bench — Rust vs Swift, same Apple GPU (settles the FFI-overhead question)

gemv (2048×2048), per-commit = one command buffer per op + wait:

| path | per-dispatch | batched (N ops / 1 cmd buffer) |
|---|---|---|
| Rust → Metal (resident weights) | **178 µs** | (via `dispatch_chain`) |
| Rust → Metal (old host-shadow shim) | 1868 µs | — |
| Swift → Metal (native `Ops.gemv`) | **170 µs** | **22 µs/op** |

**Rust and Swift per-dispatch are within ~4%** — both pay the same Metal
`commit()`+wait, so the host language is not the variable (there's no Swift→Rust
FFI in the inference path anyway). The earlier 10× Rust gap was purely the
host-shadow shim re-uploading weights every dispatch, now fixed.

The real lever is **batching**: Swift's 22 µs/op (7.6× faster) comes from
encoding many ops into one command buffer (its `Ops.gemv(on: cmd)` API). Rust
gets the same via `dispatch_chain` — the Device trait currently commits per op;
batching a whole layer into one command buffer is the next win, identical in
kind to what Swift does. tiny add: 188 µs (shim) → 161 µs (resident).

## Fixed: resident-buffer fast path (was a host-shadow copy-in/out shim)

`ffai-metal` now caches a GPU-resident `ResidentBuffer` on each `MetalBuffer`
(via the runtime's `upload_resident`), so pure inputs (weights) upload **once**
and bind resident across every dispatch instead of being re-staged from a host
shadow each call. Outputs still flow through host bytes (preserves in-place
reads + readback). Verified correct across elementwise / reduction / multi-output
/ in-place / MLA-composite kernels and all real-model forwards. Result: gemv 10×,
decode 1.9× (107→57 ms/tok).

### Original root cause (for the record)

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
