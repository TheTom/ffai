# Nemotron-Nano-30B Decode Optimization — Research & Synthesis (GB10 / sm_121)

**Date:** 2026-06-05
**Author:** research synthesis (web + reasoning), no code touched, spark untouched.
**Question:** vLLM gets **55.8 tok/s @ 20.9 GB**; we get **54.2 tok/s @ 18.4 GB** at byte-matched NVFP4 recipe. vLLM reads MORE bytes yet runs faster → its kernels are more byte-efficient. What is it doing that we aren't, and what else can push decode higher?

---

## TL;DR — the one finding that matters

The gap is **not** in the dense GEMV path and **not** in precision. At batch=1 the dense layers (lm_head, attn projections, Mamba in/out proj, shared expert) are **memory-bound and tensor cores do NOT help** — this is now firmly established (see §A). Our scalar-dequant GEMV is the structurally correct choice there, and our `lm_head` already runs at 99% of roofline.

The gap is concentrated in **two places vLLM does structurally differently**:

1. **The MoE expert GEMM.** With top-6 experts × intermediate rows, the MoE GEMM has *enough M* to use Blackwell **tensor cores via TMA-warp-specialized grouped GEMM** (FlashInfer/CUTLASS `b12x`). This is the ONE matmul in the decode where M>1, so it's the one place tensor cores legitimately win — and it's also our single biggest token share (45.6%) and our worst-efficiency kernels (shared-expert 48–53%, gather 71–74%). The public CUTLASS data shows the *exact same FP4 MoE grouped-GEMM* going **14.6 → 39 tok/s purely by switching `compute_120a → compute_120f` to enable TMA warp-specialized tactics** — a 2.7× kernel-efficiency swing with zero precision change [cutlass#3096].
2. **Kernel fusion that collapses dispatch count.** vLLM fuses (a) the whole MoE — *token dispatch + W1 GEMM + SwiGLU + W2 GEMM into a single kernel* (FlashInfer `b12x_fused_moe`), and (b) RoPE+Quant+Q-write on the attention side, "eliminating two intermediate memory round-trips" [vllm-blackwell-blog, flashinfer-docs]. We run ~390 kernels/token with our gather-up / silu / gather-down as *separate* dispatches, each round-tripping the intermediate activation through DRAM.

Everything else (FP8-vs-INT8, vectorized loads, cp.async, persistent kernels, megakernel) is either already-proven-flat for us or structurally a non-win at batch=1. Speculative decode (EAGLE3/MTP) is the one *additional* lever that is **net-positive at batch=1** and that we appear to have prematurely dismissed (see §3) — but it's a quality/infra cost, not a pure-speed freebie.

---

## §A — The settled fact: tensor cores do NOT help batch-1 dense GEMV

This kills a whole class of "are we leaving the FP8/FP4 MMA path on the table?" ideas for the *dense* layers, so it's worth nailing down before the ranked list.

- A dedicated NVFP4 batch-1 GEMV study (M=7168, K=16384, L=1) found the best hand-written CUDA-core kernel hit **26.7 µs vs 8.6 µs speed-of-light (~32% of SoL)**, and the author's explicit conclusion: *"The kernel is fundamentally memory-bound. FP4 data is tiny (4 bits per element)… FMA latency hiding is irrelevant when every cycle is waiting on memory."* Even the top competitors only reached ~2× SoL, and a plain PyTorch call scored 22.4 µs — i.e. custom tensor-core tricks bought almost nothing [amandeepsp-nvfp4-gemv].
- Blackwell's `tcgen05.mma` accumulates in **TMEM** and *always needs ≥4 epilogue warps to drain a 128-wide tile*; the practical minimum tile is **BLOCK_M=128**. At M=1 you pad 1→128 and waste ~99% of the tensor-core ALU [gau-nernst-tcgen05, vllm batch-1 GEMV note]. vLLM itself ships a runtime-dispatched custom op specifically because *"Triton's GEMV pads M=1 up to the tensor-core tile and wastes most of the ALU at batch size 1."*

**Implication for us:** our `lm_head` at 99% and the dense projections near roofline are *already at the ceiling*. Do not chase an FP8/FP4 tensor-core GEMV for lm_head / attn-proj / shared-up/down / Mamba-proj. That door is closed by physics, and we'd be re-proving what vLLM and the GEMV-hackathon already proved.

The MoE expert GEMM is the **exception**: top-6 routing gives it real M, so tensor cores are legitimately on the table there (§1).

---

## LIKELY-REAL WINS (ranked)

### 1. Fuse the MoE into a single grouped-GEMM kernel (dispatch + W1 + SwiGLU + W2) — **HIGHEST PAYOFF**
**What:** Replace our separate `moe_gather_up` → `silu` → `moe_gather_down` dispatches with one fused kernel that does token-dispatch, the up-GEMM, SwiGLU, and the down-GEMM in a single launch, keeping the intermediate activation in shared memory / registers instead of round-tripping it to DRAM. This is exactly FlashInfer `b12x_fused_moe` / `cutlass_fused_moe` [flashinfer-docs, vllm-pr#40082].

**Why it helps given our profiling:** MoE is **45.6% of our token** and contains our two worst-efficiency kernels (shared 48–53%, gather 71–74%). The 71–74% gather efficiency is precisely the "scatter penalty + intermediate round-trip" that fusion removes — the up-output never hits DRAM, so the down-GEMM doesn't re-read it, and the dispatch/scatter is amortized inside one kernel. The vLLM fused path also uses a `block_id → (expert_id, offset)` indirection table so variable-sized expert batches run *in one launch without padding waste* — directly attacking our dynamic-scatter cost.

**Expected impact:** This is where vLLM's "reads more bytes but goes faster" advantage lives. Removing the intermediate activation round-trip on up→down (0.69 GB of gather traffic + the silu pass) plus dispatch fusion is plausibly **5–10%** end-to-end. The profiling map's own "path-to-75" math already credits ~1.0 ms (≈7%) to fixing gather efficiency alone; fusion is the mechanism that gets there.

**Effort:** HIGH. This is the big structural rewrite — a grouped-GEMM with in-kernel SwiGLU in the `#[kernel]` DSL. Note the playbook already shows `moe_down_swiglu_accum_int4_chain8` exists (PR #201), so a fused down+swiglu is partially precedented; the harder part is the up+swiglu+down single-launch grouped form.
**Risk:** MEDIUM. Correctness of grouped-GEMM offset math + SwiGLU numerics (playbook: silu wants 2e-4 / f16 5e-2 tols). CUDA-graph-safety must be preserved.
**Sources:** [vllm-pr#40082], [flashinfer-docs cutlass_fused_moe], [vllm-fused-moe-design].

### 2. Multi-warp the shared-expert (already #1 on the in-tree list) + apply the *same* coalescing to gather
**What:** The profiling map's #1 ranked item — `shared_up_q4` / `shared_down_acc` are single-warp-per-row at 48–53%; mirror the multi-warp `rows_per_tg` coalesced kernel to target 80%+. Blocked earlier by an f32-vs-f16 scale-type mismatch in the accum fusion — fix the dtype (playbook §"f32-vs-f16 scale type" is a recurring trap).

**Why it helps:** Pure kernel-efficiency, no precision change, clearest measured headroom (the *weakest* kernels in the whole token). This is the lowest-risk real win and is *independent* of the big MoE fusion in §1 — do it first as a down-payment.
**Expected impact:** −0.4–0.5 ms/tok (≈3%) per the in-tree estimate.
**Effort:** LOW–MEDIUM. **Risk:** LOW (the dtype fix is understood).
**Source:** in-tree `PROFILING_32K_DECODE.md` ranked item #1.

### 3. Use TMA-warp-specialized tactics for the MoE grouped-GEMM (the `compute_120f` lesson)
**What:** This is the *mechanism* behind why vLLM's MoE is byte-efficient, and it's CUDA-toolchain-level, not algorithm-level. The public CUTLASS thread shows the identical FP4 MoE grouped GEMM on desktop Blackwell going **14.6 tok/s (`compute_120a`, CUDA 12.8) → 39 tok/s (`compute_120f`, CUDA 13.0)** — a **2.7×** swing — purely because `compute_120f` lets the autotuner select **TMA warp-specialized grouped-GEMM kernels** instead of falling back to non-TMA variants [cutlass#3096]. For us this means: if/when our MoE GEMM uses tensor cores at all (per §1, M is large enough there), the build must target the `f`-suffix arch and use TMA (async bulk copy) to feed the MMA. Note: **TMA/cp.async is only worth it for the tensor-core MoE tile** — the playbook already proved `simdgroup_async_copy` / cp.async-style prefetch is *dead* for batch-1 GEMV (amortizes only over kilo-FLOP tiles), so do NOT apply it to the dense GEMVs.

**Why it helps:** This is the single largest *documented* kernel-efficiency multiplier on this exact hardware+model class, and it's specific to the one matmul where tensor cores apply.
**Expected impact:** Large *if* we adopt a tensor-core MoE GEMM (it's the enabling toolchain piece for §1's grouped GEMM, not a standalone lever). On its own, on our current scalar-MoE, it does nothing.
**Effort:** Coupled to §1 (need a tensor-core grouped GEMM first). **Risk:** sm_121 toolchain fragility — note vLLM hit `nvidia-cutlass-dsl==4.5.0 generates faulty PTX for SM121`, must pin 4.4.2; and `compute_120f` needs CUDA 13.0 [vllm-pr#40082, cutlass#3096].
**Sources:** [cutlass#3096], [vllm-pr#40082].

### 4. Fuse the attention pre-amble: RoPE + Quant + (K/Q write) into one kernel
**What:** vLLM fuses "RoPE + Quant + Q Write (Decode)" into a single kernel, "eliminating two intermediate memory round-trips" [vllm-blackwell-blog]. Our equivalent rope/quant/kv-write are separate tiny dispatches.

**Why it helps:** Each is "≈0 ms real work" per our profiler, BUT we run ~390 kernels/token and the *dispatch + DRAM round-trip* of the intermediate is the real cost, not the FLOPs. Collapsing 3 dispatches→1 removes two activation round-trips and two launch latencies per layer. With CUDA graphs the launch cost is already low, so the win here is the round-trip elimination, modest but cheap.
**Expected impact:** 1–3% (small per-op, but it's per-layer × 23+ layers).
**Effort:** LOW–MEDIUM. **Risk:** LOW. **Source:** [vllm-blackwell-blog].

---

## NET-POSITIVE BUT QUALITY/INFRA-COSTED (re-examine — we may have been wrong)

### 5. Speculative decode (EAGLE3 / MTP) — **contradicts our "net-loss at batch=1" prior**
**What:** A small draft proposes k tokens; the target verifies all k in one forward pass. At B=1 the GPU is bandwidth-bound with **idle compute**, so the verification pass costs ~the same as one normal token while yielding >1 accepted token.

**Why our prior may be wrong:** The evidence says spec-decode is **net-positive precisely at B=1** and *loses* benefit as batch grows (opposite of the usual intuition): *"At B=1, even a modest acceptance rate yields meaningful latency gains because the GPU has substantial idle capacity"* [hf-eagle3]. Concrete measured single-stream numbers on a comparable hybrid/MoE-scale model: **120 → 168 tok/s (1.39×) at 40% acceptance, 2.4 accepted/verify, k=6**, with a 277 MB draft against a 30B-class target [hf-eagle3]. The "7× weight reads" overhead (6 draft + 1 target) is dwarfed by the tiny draft size and absorbed by spare bandwidth.

**The catch (why it's costed, not free):**
- Needs a **trained draft head / MTP module** for *this* model (Nemotron-3-Nano ships MTP-style heads in some configs — worth checking the checkpoint). Without one, you train/distill it = real infra cost.
- **Acceptance rate is the whole ballgame.** At 40% accept you get 1.39×; below ~25% it can go net-negative. Reasoning models with long deterministic spans tend to accept *well*, which is favorable here.
- Verification pass on a **hybrid Mamba+MoE** model is more complex than on a dense transformer: the **Mamba2 state must be advanced/rolled-back per draft token** (the SSM is sequential), which is the non-trivial engineering piece and a place our hybrid architecture is harder than vLLM's dense-transformer spec path.
- At our 273 GB/s (vs the 3.35 TB/s box in the cited result) the *bandwidth slack* is tighter, so the realized speedup will be lower than 1.39× — but still plausibly net-positive at decent acceptance.

**Expected impact:** **+15–35%** single-stream tok/s *if* acceptance ≥40% and the Mamba-state rollback is solved — by far the largest single lever, but the only one that changes the token-count math instead of the per-token cost.
**Effort:** HIGH (draft head + Mamba2 state checkpoint/rollback in the decode loop). **Risk:** MEDIUM–HIGH (acceptance-rate dependent; Mamba rollback correctness).
**Recommendation:** **Re-open this.** Verify whether the NVFP4 checkpoint ships an MTP head, and prototype acceptance-rate measurement before committing — that single number decides go/no-go.
**Sources:** [hf-eagle3], [modular-specdec], [sglang-specdec].

### 6. Sub-4-bit experts (Q3 / NVFP4-with-tighter-grouping) — the path past 75, quality-costed
**What:** Drop experts to ~3-bit. The experts dominate MoE bytes; trimming 4→3 bit cuts the largest byte share.
**Why/impact:** Mechanically ~+15–20% (matches the in-tree note that 75 needs Q3). **Quality cost is real** and no engine ships it at 32K for this model. Mark as a quality-traded option, not a clean win.
**Effort:** MEDIUM (quant recipe + QAD-style recovery). **Risk:** HIGH on quality. **Source:** in-tree map; [nvidia-nemotron-nvfp4] (NVIDIA themselves stop at 4-bit experts + 8-bit elsewhere via QAD, i.e. they judged sub-4-bit not worth the quality hit).

---

## PROBABLY FLAT / DEAD-ENDS (do NOT re-litigate — with the *why*)

| Idea | Why it's flat for us |
|---|---|
| **FP8/FP4 tensor-core GEMV for dense layers** (lm_head, attn-proj, shared, Mamba-proj) | Batch-1 dense GEMV is memory-bound; tensor cores need M≥128, you pad 1→128 and waste the ALU. Proven by the NVFP4-GEMV study (~32% SoL even hand-tuned) and tcgen05's 4-epilogue-warp/BLOCK_M=128 floor. Our lm_head is already 99%. §A. |
| **cp.async / TMA prefetch on dense GEMVs** | Amortizes only over kilo-FLOP tensor-core tiles. Playbook already measured `simdgroup_async_copy` dead at 2–4 KB matvec X-loads. Only valid inside the §1/§3 tensor-core MoE tile. |
| **Persistent / megakernel for the whole decode** | Already tried: cooperative-launch megakernel can't fit GB10's 48 SMs. Persistent-kernel for *all* of decode is the same class. (Fusing *within* MoE per §1 is different and fine.) |
| **Vectorized uint4 loads, MT_GEMV_2ROW/VEC, rpt sweeps, multiwarp-shared(prev attempt), gather cache-streaming hints** | All measured flat or worse in-tree (uint4 −34%, gather rpt flat because bottleneck is scatter not warp count). |
| **TG-memory cache for X-broadcast in matvec** | Playbook: −33pp on M-series; L1 already absorbs the redundancy. Same logic on GB10's L1/L2. |
| **INT8 → FP8(E4M3) for the Q8 layers** | Both are 8-bit = same byte budget = same bandwidth at batch-1. NVIDIA's E4M3 vs our INT8/f32-scale is a numerical/quality nuance, **not** a speed lever (the baseline doc already says "minor diff"). Don't expect tok/s from it. |
| **`concat_mla_k` / MLA KV tricks** | vLLM's MLA-specific KV kernel — Nemotron-Nano is **not** MLA (it's hybrid Mamba2 + standard GQA attention on a minority of layers). Not applicable. |

---

## On the Mamba2 decode step specifically (§4 of the brief)
Mamba is only **15% of our token** and our in/out-proj already run 64–76%. The public Mamba2 fusion wins (PyTorch blog, vLLM SSD) are **all prefill chunked-scan** — there is **no published faster single-step selective-scan kernel** for decode; the decode path is already a cheap conv1d + per-channel state update [pytorch-mamba2, vllm-mamba2]. The only decode-relevant note: vLLM split conv1d into separate prefill/decode kernels to avoid mixed-batch slowdown — not relevant to our pure-decode bench. **Verdict: low priority.** The Mamba in/out-proj headroom (64–76%) is ordinary GEMV multi-warp tuning, same bucket as §2, not a Mamba-algorithm change. (Caveat in §5: if we do spec-decode, the Mamba *state rollback* becomes the hard part — that's the one place Mamba decode work pays off.)

---

## Recommended sequence (most-confident first)
1. **§2 multi-warp shared-expert** (fix the f16 scale dtype) — low risk, clear headroom, ~3%, down-payment.
2. **§4 fuse rope+quant+kv-write** — low risk, ~1–3%, cheap.
3. **§1 fused MoE grouped-GEMM (dispatch+up+swiglu+down)** + **§3 TMA/`compute_120f` toolchain** — the big structural win that mirrors vLLM; ~5–10%. This is the one that closes the "vLLM reads more, goes faster" gap.
4. **§5 spec-decode acceptance-rate spike** — measure acceptance on the real checkpoint *before* building; if ≥40% and Mamba-rollback tractable, it's the largest lever (+15–35%) and the realistic path *past* parity to a genuine win.
5. **§6 Q3 experts** — only if a quality budget exists; the documented route to ~75 but quality-costed.

§1+§3 together should plausibly take us from 54.2 to **parity-plus** at matched precision (closing the 3% and adding a few). §5 is what turns "at parity" into "beats vLLM" on single-stream. The dense-GEMV tensor-core fantasy (§A) is dead — stop looking there.

---

## Sources
- vLLM Blackwell WideEP blog (RoPE+Quant fusion, NVFP4 MoE dispatch, FlashInfer TRTLLM-Gen GEMM): https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1
- vLLM PR #40082 — FlashInfer b12x fused MoE + FP4 dense GEMM for SM120/121 (fuses dispatch+W1+SwiGLU+W2; CuTe-DSL warp-MMA adaptive tiling for small-M decode; +1.8% 1P / +6.0% 8P on DGX Spark Qwen3-30B-A3B-NVFP4; pin cutlass-dsl 4.4.2): https://github.com/vllm-project/vllm/pull/40082
- CUTLASS issue #3096 — FP4 MoE grouped GEMM on desktop Blackwell, 14.6→39 tok/s via compute_120f TMA warp-specialized tactics; Marlin W4A16 baseline 46–49 tok/s: https://github.com/NVIDIA/cutlass/issues/3096
- "Twelve Attempts at an FP4 Kernel" (batch-1 NVFP4 GEMV is memory-bound, ~32% SoL, tensor cores don't help): https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/
- tcgen05 for dummies (TMEM, ≥4 epilogue warps / BLOCK_M=128 floor → unsuitable for M=1): https://gau-nernst.github.io/tcgen05/
- FlashInfer cutlass_fused_moe docs (single-call fused MoE, block_id→expert mapping, variable expert batches no padding): https://docs.flashinfer.ai/generated/flashinfer.fused_moe.cutlass_fused_moe.html
- vLLM fused-MoE design / moe_align_block_size (persistent grouped GEMM, indirect token addressing, padding-to-block): https://docs.vllm.ai/en/latest/design/moe_kernel_features/
- EAGLE3 single-stream B=1 (1.39× @ 40% accept, 120→168 tok/s; net-positive at B=1, shrinks with batch): https://huggingface.co/blog/lujangusface/tw-eagle3-gpu
- Modular spec-decode / SGLang spec-decode (B=1 bandwidth-bound idle-compute rationale): https://docs.modular.com/max/serve/speculative-decoding/ , https://sgl-project.github.io/advanced_features/speculative_decoding.html
- PyTorch "Accelerating Mamba2 with Kernel Fusion" (prefill SSD only, 1.5–2.5×; no decode single-step win): https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/
- vLLM Mamba2 conv1d prefill/decode split (PR #17146), SM121 Triton crash note (#37431): https://github.com/vllm-project/vllm/pull/17146
- Nemotron-3-Nano-30B-A3B-NVFP4 model card / paper (128 experts, top-6 + 1 shared, 3.5B active, QAD 4-bit experts): https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 , https://arxiv.org/pdf/2512.20848
