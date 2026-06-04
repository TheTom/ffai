# Adding a backend (ROCm, Vulkan, …)

The engine is built so a new GPU backend needs **only two things**; everything
above the `Device` trait — ops, models, loader, runtime, the whole verified
model suite — comes for free.

```
              ┌──────────────────────────────────────────────┐
  SHARED      │  ffai-models / ffai-modeltests  (model graphs) │
  (no per-    │  ffai-ops        (the op seam, → Kernel IR)    │
   backend    │  ffai-loader     (GGUF / SafeTensors + dequant)│
   code)      │  ffai-core       (Device trait · Tensor)       │
              └───────────────────────┬──────────────────────┘
                                      │  the ONLY boundary
              ┌───────────────────────▼──────────────────────┐
  PER-BACKEND │  1. impl Device  (alloc/upload/download/        │
  (all you    │       dispatch/synchronize)  — ~150 lines      │
   write)     │  2. metaltile codegen target  (emit the        │
              │       backend's kernel language)               │
              └────────────────────────────────────────────────┘
```

## The two pieces

**1. A `Device` impl** (`crates/backends/ffai-<name>/src/lib.rs`). Mirror
`ffai-cuda` / `ffai-metal`: wrap the backend's buffers + module/PSO cache, and
implement the five `ffai_core::Device` methods. `alloc`/`upload`/`download` move
bytes; `dispatch(kernel, bindings, grid)` binds the operands and launches.
`ffai-ops` only ever calls these — it never names a concrete backend.

**2. A metaltile codegen target.** `metaltile-codegen` already emits **MSL**
(`src/msl/`) and **CUDA** (`src/cuda/`) from one `#[kernel]` DSL.
- **ROCm/HIP** is the cheapest: HIP is ~source-compatible with CUDA C, so the
  `cuda` backend is the starting point — most kernels compile under `hipcc`
  unchanged; differences are wavefront size (64 vs 32) and a few intrinsics.
- **Vulkan** needs an SPIR-V/GLSL compute emit (a new `src/vulkan/`).

## Wiring the tests (one file, not N)

The model forwards live **once** in `ffai-modeltests` as `verify_*(&dyn Device)`,
collected by `run_all(dev, plat)`. Each backend has a **single** test file:

```rust
// crates/backends/ffai-<name>/tests/all_models.rs
use ffai_<name>::<Name>Device;
#[test]
fn all_models() {
    let Some(dev) = <Name>Device::create().expect("<name>") else { return };
    ffai_modeltests::run_all(dev.as_ref(), "<Name> GPU");
}
```

That's the whole verification surface. The 10-model suite (GPT-2, Pythia,
GPT-Neo, OLMo-2, Gemma-2, Phi-1.5, StableLM-2, OLMoE, Mamba2, Falcon-H1 —
LayerNorm/parallel-residual/post-norm/geGLU/MoE/SSM/hybrid families) runs on
the new backend with **zero model code**, checked against the same HF oracles.

## Checklist for a new backend

1. `crates/backends/ffai-rocm/` — `Device` impl + `unsafe impl Send/Sync` as needed.
2. metaltile codegen target (HIP: fork `src/cuda/`; tune wavefront 64).
3. `tests/all_models.rs` (copy the template above).
4. `cargo test -p ffai-rocm --test all_models` → the full suite, vs HF.
5. Add the crate to `[workspace].members`.

No changes to `ffai-ops`, `ffai-models`, `ffai-loader`, or any model — those are
shared. If a kernel diverges per backend, it diverges in metaltile's codegen,
not in the engine.
