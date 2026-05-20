# metaltile `ek/aura-port` → upstream-synced `dev` Reconciliation Plan

| Field | Value |
|---|---|
| Date | 2026-05-21 |
| metaltile `dev` HEAD | `cd85c66` (= `0xClandestine/metaltile` upstream/dev) |
| metaltile `ek/aura-port` HEAD | `01b5c5f` |
| merge-base | `69a0101` |
| `ek/aura-port` ahead of `dev` | 38 commits |
| `dev` ahead of `ek/aura-port` | 20 commits |
| Scope | Survey only — no git state touched, no code changed |

## Executive finding

The convergent re-implementation is **far cleaner than feared**. Every AURA codec kernel,
every sdpa head_dim variant, the `ffai_` prefix, `mt_softplus`, and all of the codegen /
DSL work on `ek/aura-port` are a **functional subset or formatting-only variant of
upstream's**. The only genuinely divergent kernel is `gated_delta`. The only must-port
deltas are (1) a `vectorize.rs` SSA-remap fix and (2) the AURA non-identity-rotation test
coverage. Most of our 38 commits **drop cleanly** during rebase because upstream landed an
equivalent (or strictly better) version.

**Critical AURA stride-fix check (task item 3):** RESOLVED — *upstream `#97` already
carries the `tokens` stride parameter.* `aura_dequant_rotated` on both branches indexes
`(bh * tokens + t) * packed_width` with `tokens` as a `#[constexpr]`. Heads past 0 land
correctly on upstream. No stride fix needs porting.

---

## Overlap-area classification table

Classification key: **(a)** identical — git drops our dup cleanly · **(b)** different,
upstream canonical — take upstream, drop ours · **(c)** our version carries a delta
upstream lacks — take upstream base + port our delta · **(d)** genuinely unique — keep ours.

| # | Overlap area | Our commits | Upstream | Class | Resolution | FFAI re-pointing impact |
|---|---|---|---|---|---|---|
| 1 | DSL primitives (`simd_shuffle_xor`/`simd_broadcast`/atomics/`_tg`/`threadgroup_alloc` dtype/`stack_alloc`) | `dc56af2`,`901d65d`,`170c2b8`,`0382b0a` | `#95` `344c90c` | **a** | Drop ours — upstream `#95` is a superset (also adds `simdgroup_load`). `body_parser.rs` / `ir.rs` diffs are ours merely *lacking* upstream's `simdgroup_load` op. | None — DSL is metaltile-internal. |
| 2 | Codegen fixes — `const_fold` entry-block, `unroll` loop-body clone, GELU NaN, CSE/value_sink/tgid SSA | `5b18f85`,`326abf5`,`ba18f6e`,`dc2afc6` | `#94` `858af16` | **a** | Drop ours — `#94` is the same fix set (`const_fold.rs`/`unroll.rs`/`cse.rs`/`value_sink.rs`/`remap.rs`/`type_check.rs` diff to **near-zero** vs upstream; remaining lines are ours lacking `#60`'s `SimdgroupLoad`). | None. |
| 2b | `vectorize.rs` `DeclareLocal`/`SetLocal` remap arms + test | part of `01b5c5f` area / vectorize work | **absent upstream** | **c** | **MUST PORT.** Upstream `vectorize.rs` lacks the `Op::DeclareLocal{value}` / `Op::SetLocal{value}` arms in `remap_values_in_op` (+43 lines incl. `remap_rewires_declare_and_set_local` test). Without it a `let mut x = load(...)` coalesced into a `VectorExtract` emits an undeclared-identifier MSL error. Upstream's `ir.rs` HAS both ops, so the fix applies cleanly. | None directly — but prevents MSL miscompiles in AURA flash kernels. |
| 3 | AURA codec kernels — `aura_encode`/`aura_dequant_rotated`/`aura_score`/`aura_value`/`aura_flash_p1`/`aura_flash_pass2` | `bfa830b`,`37d2962`,`4058f08`,`553d1ad`,`5f40689`,`3cd48d4`,`b678799`,`4c28dc4`,`6f1f724` | `#97` `ac1a722` | **a** | Drop ours — kernels are **functionally identical**; diffs are pure rustfmt whitespace. Kernel names, variant lists (`int2/3/4/6/8`, flash `kb4_vb{2,4}_d{64,128}`, pass2 `d64..d512`, dequant `_clean`/`_odd`), and the `tokens` stride param all match. | **None** — `MetalTileKernels.aura_encode_int4_f32` etc. names are byte-identical. `Ops.auraEncode/auraDequantRotated/auraScore/auraValue` unchanged. `AURAQuantizedKVCache` unaffected. |
| 3b | AURA non-identity-rotation test coverage + `aura_value_gpu_correctness` + `aura_msl_snapshots.rs` | `4306f85`,`bf1f418`,`01b5c5f` | partial in `#99` | **c/d** | **PORT the delta.** Our `aura_*_gpu_correctness.rs` carry ~317 extra lines: `aura_encode_to_dequant_round_trip_srht_rotation_f32`, SRHT-rotation helpers, encode +98 / dequant +145 / flash +59 / score +15. `aura_value_gpu_correctness.rs` and `aura_msl_snapshots.rs` (6 `.snap` fixtures) are **ours-only** → keep. | None — tests only. |
| 4 | gated-delta | `2513c03` `gated_delta_step.rs` + `gated_delta_step_gpu_correctness.rs` | `#112` `00e5125` `gated_delta.rs` | **b** | **Take upstream, drop ours.** Upstream `mt_gated_delta_step` + `mt_gated_delta_chunk` is strictly more complete (chunked-prefill via `mt_gated_delta_chunk` with `t_len` tensor). Signatures **diverge**: upstream is plain generic `<T>` with `dk/dv/hv/hk` as **runtime** `#[constexpr]` (single kernel name); ours is macro-generated with 6 baked variants + `t_steps` constexpr. Param order also differs (`...,state_out,y` upstream vs `...,y,state_out` ours). | **HIGH — see §FFAI impact.** Largest re-pointing item. |
| 5 | `ffai_` prefix + `mt_softplus` | `8a78d24`,`6f61df9` + test fixups `228c50a`,`141a60b`,`7038508` | `#96` `d0f3242` | **a** | Drop ours — upstream `#96` does the identical rename + identical `mt_softplus` (`unary.rs` diffs to zero). | **None** — FFAI `Ops.swift` already calls `ffai_gather_*`, `ffai_rope_llama_*`, `ffai_sdpa_decode_*`, `ffai_argmax_*` (already on this branch). Confirm regenerated kernels keep these names — they do. |
| 6 | sdpa head_dim variants — `sdpa_decode_d64` / `sdpa_decode_d256` | `04a964d`,`770c625`,`3bd7859` | `#98` `405a1a2` | **a** | Drop ours — `ffai_sdpa_decode_d64` / `ffai_sdpa_decode_d256` kernel names + `sdpa_decode.rs` + `rms_norm.rs` are **identical**. `#98` also adds rms_norm dispatch guards (superset of our `ada128a`). | **None** — `Ops.sdpaDecode` already calls `ffai_sdpa_decode_d64_*`/`_d256_*`. |
| 7 | GPU correctness tests — A2.x reduction/Grid3D suite | `1f47359`,`3bd7859`,`bf1f418`,`4306f85` | `#99` `cd85c66` | **a/c** | Mostly drop ours — `#99` lands `argmax`/`gemv`/`mt_qmv`/`rms_norm`/`conv1d_causal_step`/`ssm_step`/`aura_*` correctness tests. **Port delta:** the non-identity-rotation cases (3b) + ours-only `aura_value`/`aura_msl_snapshots`. | None — tests only. |
| 8 | MoE kernels | none (FFAI does MoE CPU-side: routing in Swift, no metaltile kernel) | `#60`/`#61`/`#65` `mt_qmm_mma`/`mt_moe_*` | **d (upstream)** | **Adopt-as-is, no action.** Upstream's MoE metaltile kernels arrive free with the rebase; FFAI does not call them today. **Wiring FFAI MoE onto `mt_moe_permute`/`mt_moe_router_topk`/`mt_moe_unpermute`/`mt_qmm_mma` is a follow-up, out of scope for the rebase.** File as a new FFAI feature issue. |
| 9 | Genuinely unique non-overlap commits | `9772ab6` (docs), `ff3d178`+`a07faf5` (Makefile), `f2c28e2`+`a4186fe` (KERNEL_AUDIT docs), `ce551fd` (`MLX_COMMIT`→alpha), `7d29c71` (sdpa DISPATCH INVARIANTS doc) | n/a | **d** | **Keep — rebase on top cleanly.** Check `7d29c71` doesn't conflict with `#98`'s sdpa_decode invariants block; check `ce551fd` `MLX_COMMIT` vs upstream's pin. `a07faf5`/`ff3d178` Makefile may conflict with upstream `build.rs`/`#123` changes — resolve in favour of upstream's progress-reporting `build.rs`. | None. |

---

## Ordered rebase strategy

`git rebase cd85c66 ek/aura-port` (after the user creates a fresh backup ref).
Of the 38 commits, walked chronologically (`git log --reverse origin/dev..ek/aura-port`):

**Drop / will auto-drop as empty (≈30 commits)** — upstream landed an equivalent. Git
marks most empty after a 3-way merge; the rest become trivial no-ops. Use
`git rebase --empty=drop`:

- `5b18f85`, `326abf5`, `ba18f6e`, `dc2afc6` — codegen fixes ⊂ `#94`.
- `dc56af2`, `901d65d`, `170c2b8`, `0382b0a` — DSL primitives ⊂ `#95`.
- `bfa830b`, `37d2962`, `4058f08`, `553d1ad`, `5f40689`, `3cd48d4`, `b678799`,
  `4c28dc4`, `6f1f724` — AURA kernels = `#97` (whitespace-only delta).
- `8a78d24`, `6f61df9` — `ffai_` prefix + `mt_softplus` = `#96`.
- `04a964d`, `770c625`, `3bd7859`, `ada128a` — sdpa variants + rms_norm guards ⊂ `#98`.
- `1f47359` — GPU correctness suite ⊂ `#99`.
- `228c50a`, `141a60b`, `7038508` — `ffai_`-rename test fixups, already done upstream.

**Keep (≈6 commits — genuinely unique, no upstream counterpart):**

- `9772ab6` docs(macros), `f2c28e2` + `a4186fe` KERNEL_AUDIT docs, `7d29c71` sdpa
  DISPATCH INVARIANTS doc, `ff3d178` + `a07faf5` Makefile. **Squash the two Makefile
  commits into one**; resolve any conflict with upstream's `#123` `build.rs` in favour of
  upstream. `ce551fd` `MLX_COMMIT` — keep only if it still points somewhere upstream
  doesn't already pin (verify; likely drop).

**Squash + re-point as a delta-port (3 commits → 1 new commit):**

- `4306f85` + `bf1f418` (AURA non-identity-rotation tests, `aura_value` test) and the
  `aura_msl_snapshots.rs` portion of `01b5c5f` — these will **conflict** against `#97`/`#99`'s
  versions of the same test files. Resolve by taking upstream's file as the base and
  re-applying our extra `srht_rotation` / non-identity test functions on top. Squash into
  one commit: `test(aura): non-identity SRHT rotation coverage + MSL snapshots`.

**Resolve as a take-upstream conflict (1 commit → drop):**

- `2513c03` (`gated_delta_step`) — conflicts conceptually with `#112`. **Drop our
  `gated_delta_step.rs` + `gated_delta_step_gpu_correctness.rs`; keep upstream's
  `gated_delta.rs` + `gated_delta_gpu_correctness.rs`.** The real work moves FFAI-side.

**Add as one new commit on top (the must-port delta):**

- `vectorize.rs` `DeclareLocal`/`SetLocal` remap arms + `remap_rewires_declare_and_set_local`
  test (item 2b). New commit: `fix(vectorize): rewire DeclareLocal/SetLocal value operand on
  load coalescing`. Verify it applies on upstream's `vectorize.rs` (upstream `ir.rs` already
  has both ops, so it does).

Net post-rebase: ~6 kept docs/Makefile commits + 1 squashed AURA-test commit + 1 new
`vectorize.rs` fix commit ≈ **8 commits ahead of `dev`**, down from 38.

---

## Judgment calls needing a human decision

1. **gated_delta adoption shape (HIGH).** Upstream `mt_gated_delta_step` uses runtime
   `#[constexpr]` for `dk/dv/hv/hk` (one kernel) vs our 6 macro-baked variants. Runtime
   constexprs are simpler for FFAI but may cost a recompiled PSO per config or lose a
   perf specialization. **Decide:** accept upstream as-is, or open an upstream PR to add
   our baked-variant fast path. Recommendation: accept upstream as-is for the rebase;
   benchmark later.

2. **gated_delta chunked-prefill mapping.** FFAI's `tSteps>1` path (a `tSteps` constexpr
   on the single step kernel) has no upstream equivalent — upstream splits it into a
   separate `mt_gated_delta_chunk` taking a `t_len` *runtime tensor* and `[B,T,H,D]`
   layouts. **Decide:** does FFAI `StateReplayCache`/`Qwen35` adopt `mt_gated_delta_chunk`
   (layout change), or keep looping `mt_gated_delta_step` per token? Affects whether the
   FFAI re-point is medium or large.

3. **`MLX_COMMIT` pin (`ce551fd`).** Confirm whether upstream `dev` already pins
   `ekryski/mlx@alpha` (or a compatible commit). If upstream pins a different MLX, keep
   `ce551fd`; if same, drop it. Needs a one-line check of upstream `build.rs`/Cargo.

4. **Makefile reconciliation.** Our `a07faf5`/`ff3d178` "normalize on make" vs upstream
   `#123`'s `build.rs` MLX-fetch-progress work. Likely orthogonal but both touch build
   plumbing — human eyes on the merge.

5. **MoE follow-up scope.** Upstream MoE kernels (`#60`/`#61`/`#65`) arrive unused.
   Confirm wiring FFAI MoE onto them is explicitly a **post-rebase** issue, not a rebase
   blocker. Recommendation: yes, follow-up.

---

## FFAI-side re-pointing & re-test impact

FFAI calls metaltile kernels by generated name via `MetalTileKernels.*` (generated under
`Sources/MetalTileSwift/Generated/` by `make regenerate-kernels`). Per area:

| Area | FFAI change required | Effort |
|---|---|---|
| AURA codec (item 3) | **None.** Kernel names identical. `Ops.auraEncode/auraDequantRotated/auraScore/auraValue/auraFlash*`, `AURAQuantizedKVCache`, `KVCacheEviction`, `Models/Qwen3.swift`, `Models/Llama.swift` unchanged. | none |
| `ffai_` prefix + softplus (5) | **None.** `Ops.swift` already on `ffai_*` names. | none |
| sdpa d64/d256 (6) | **None.** `Ops.sdpaDecode` already calls `ffai_sdpa_decode_d64_*`/`_d256_*`. | none |
| **gated_delta (4)** | **Required.** `Ops.gatedDeltaStep` currently dispatches one of six `MetalTileKernels.gated_delta_step_{Dk}_{Dv}_{Hk}_{Hv}_f32` variants — these names **vanish** post-rebase, replaced by a single `mt_gated_delta_step` taking `dk/dv/hv/hk` as runtime constexpr args. Re-point: (a) collapse the 6-way `switch` in `Ops.gatedDeltaStep` to one call passing the dims as args; (b) fix param order (`...,stateOut,y` upstream vs current `...,y,stateOut`); (c) update `OpsValidation.validateGatedDeltaStep` doc-refs from `gated_delta_step.rs` → `gated_delta.rs`; (d) re-point `tSteps>1` either onto `mt_gated_delta_chunk` (layout rework) or a per-token loop; (e) update `GDNStateCache`, `StateReplayCache`, `Models/Qwen35.swift` call sites. Affects Qwen3.5/3.6 linear-attention layers. | **Medium–Large** |
| Generated bindings | After rebase, run `make regenerate-kernels` from FFAI root — regenerates `MetalTileSwift/Generated/`. The `gated_delta_step_*` symbols disappear; `mt_gated_delta_step` / `mt_gated_delta_chunk` appear. Any stale reference is a compile error (good — surfaces every site). | small |

**Estimated FFAI re-pointing + re-test effort: ~1–1.5 days.**

- ~0.5 day: `Ops.gatedDeltaStep` rewrite + `OpsValidation` + `OpsTests` (`OpsTests.swift`
  must exercise `mt_gated_delta_step` at production Qwen3.5/3.6 shapes per CLAUDE.md
  wrapper checklist).
- ~0.25 day: `tSteps>1` decision implementation (longer if adopting `mt_gated_delta_chunk`).
- ~0.25 day: `make regenerate-kernels`, fix fallout, `make test-unit`.
- ~0.25–0.5 day: `make test-integration` — Qwen3.5/3.6 (`Qwen35IntegrationTests`,
  `DeterminismSmokeTests`) coherent-output gate; AURA model coherence on Qwen3/Llama as a
  no-regression check (should pass untouched since AURA names are stable).

**No re-test risk for AURA / sdpa / MoE.** AURA is byte-identical; sdpa is identical; MoE
is unused. The entire FFAI risk surface is the `gated_delta` re-point.

---

## Summary of buckets

- **(a) identical — drop ours:** areas 1, 2, 3, 5, 6, 7-base ≈ **30 commits** auto-drop.
- **(b) upstream canonical — drop ours, take upstream:** area 4 (gated_delta) — **1 commit**.
- **(c) port our delta:** area 2b (`vectorize.rs` fix) + 3b (AURA non-identity tests) —
  **2 deltas → 2 new/squashed commits**.
- **(d) genuinely unique — keep:** area 9 docs/Makefile/`MLX_COMMIT` — **~5–6 commits**;
  area 8 (MoE) is upstream-unique, adopted free.
