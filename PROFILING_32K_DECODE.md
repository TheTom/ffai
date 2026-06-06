# Nemotron-Nano-30B-A3B — 32K Decode Profiling Map (GB10 / ASUS GX10)

**Goal:** 75+ tok/s decode @ 32,768 ctx. **Current:** 68.2 tok/s graph-batched (14.63 ms/token), full-quality Q4, argmax 1234.

**Config:** `NEMOTRON_FAKECTX=32768 NEMOTRON_GRAPH=1 NEMOTRON_DEVROUTER=1 NEMOTRON_Q4CACHE=1 NEMOTRON_F16KV=1`, GB10 sm_121, LPDDR5X 273 GB/s peak.

**Method:** per-op CUDA-event profiler (`NEMOTRON_PROFILE=1`, eager, sync-around-each-op). ⚠️ The profiler synchronizes per op, so **ms/tok is sync-inflated for tiny elementwise ops** (silu/rope/rms_norm/conv read ~0 bytes → their "8 ms" is sync overhead, real ≈0). **Trust the `GB/s` column and the ablation, not the per-op ms.**

## The reference: achievable bandwidth = ~189 GB/s
`lm_head` is a big contiguous Q4 GEMV and runs at **187 GB/s ≈ 99% of the achievable roofline**. So **189 GB/s is reachable on this hardware** — any kernel below it has headroom that is *kernel efficiency*, not a hardware wall.

## Per-kernel map (real, bandwidth-bound work)

| kernel | GB read/tok | eff GB/s | % of 189 achievable | calls/tok | verdict |
|---|---|---|---|---|---|
| **lm_head** | 0.176 | **187** | **99%** | 1 | ✅ saturated — no headroom |
| m_in_proj (Mamba) | 0.319 | 144 | 76% | 23 | 🟡 mild headroom |
| moe_gather_up | 0.345 | 140 | 74% | 23 | 🟡 **scatter penalty** |
| moe_gather_down | 0.346 | 134 | 71% | 23 | 🟡 **scatter penalty** |
| m_out_proj (Mamba) | 0.127 | 120 | 64% | 23 | 🟡 headroom |
| shared_up_q4 | 0.115 | 101 | 53% | 23 | 🔴 **single-warp, big headroom** |
| shared_down_acc | 0.115 | 90 | 48% | 23 | 🔴 **single-warp, big headroom** |
| sdpa_2pass | 0.202 | 78 | 41% | 6 | 🟡 KV-read, latency-bound |
| silu/rope/rms_norm/conv/ssm/router | ~0 | — | — | 12–52 | sync-artifact, real ≈0 ms |

**host overhead:** 0.75 ms/tok (eager; graphs remove most of it).

**Confirmation run (warmer box) — ranking reproduced, efficiencies thermal-sensitive:**
`lm_head` **186.8 GB/s = 99%** (stable), moe_gather_up/down **119/115 = 63%/61%**, m_in_proj **121 = 64%**, m_out_proj **118 = 63%**, shared_up_q4 **77 = 41%**, shared_down_acc **68 = 36%**, sdpa_2pass **58 = 31%**. Absolute % drifts with temperature but `lm_head`≈99% and the under-performer ranking are invariant. **Headroom is real and likely *larger* than the first table suggests.**

## Category ablation (skip-based ground truth)
- **MoE total: 7.3 ms (45.6%)** — gather up/down + gate + shared experts
- **Attn + lm_head + norms: 6.3 ms (39.4%)** — q/k/v/o proj, sdpa, lm_head
- **Mamba: 2.4 ms (15.0%)** — in/out proj, conv1d, ssm_step

## KEY INSIGHT — this is NOT a hardware wall
`lm_head` proves 189 GB/s is achievable. The dominant kernels run well under it:
- **shared-expert (up 53% / down 48%)**: weakest. Cause = single-warp-per-row (no multi-warp `rows_per_tg`). Fix = mirror the multi-warp coalesced kernel → target 80%+.
- **moe_gather up/down (74%/71%)**: the top-6-of-128 **dynamic scatter** (6 experts at different offsets) costs ~25% vs lm_head's contiguous read. Levers: cache-streaming hints (`ld.global.cs`), better expert-block locality, or compaction.
- **m_out_proj (64%) / m_in_proj (76%)**: mild headroom.

## Path-to-75 math (clean, no precision loss)
Token = 14.63 ms. Pulling the underperformers toward the proven 189 GB/s:
- shared-expert 0.23 GB @ ~95 → @ 170 GB/s: saves ~1.0 ms
- moe_gather 0.69 GB @ ~137 → @ 175 GB/s: saves ~1.0 ms
- → ~12.6 ms = **~79 tok/s**. **75 is reachable on efficiency alone.**

## Ruled out (measured, no gain)
- SDPA 2-pass BC4 / TILED variants: **much worse** (56 ms vs 14 ms)
- SDPA split-K block sweep (64–512): flat
- `MT_MOE_RPT` 1–4 on gather: flat (gather bottleneck is scatter, not warp count)
- `--use_fast_math`: no change
- `MT_GEMV_2ROW`, `MT_GEMV_VEC`: crash
- uint4 vectorized loads: −34% (starves the pipe)
- f16-KV: +2 tok/s (banked)

## Ranked opportunities (next work)
1. **shared_up_q4 + shared_down_acc → multi-warp (`rows_per_tg`)** — 48–53% → 80%+, est. **−0.4–0.5 ms/tok**. Lowest risk, clearest headroom. *(blocked earlier by f32-vs-f16 scale type mismatch in the accum fusion — fix the dtype.)*
2. **moe_gather scatter efficiency** — 71–74% → 85%+. Try `ld.global.cs` cache-streaming hints (inline PTX) + expert-block locality. Biggest share of the token (45%), so highest absolute payoff if the scatter penalty is partly cache-pollution.
3. **m_out_proj** (64%) — multi-warp / config tune.

## Banked optimizations (in the 68.2)
CUDA graphs (+6.5%), Q4 disk cache (setup 120 s→20 s), MoE rpt2 default, parallel dequant, f16-KV, FMAD-on, `__ldg`/`__restrict__`/`__expf` codegen.

---
*Generated from the in-tree `NEMOTRON_PROFILE=1` per-op profiler (ffai-modeltests/src/lib.rs). Re-run: `NEMOTRON_PROFILE=1 NEMOTRON_FAKECTX=32768 NEMOTRON_DECODE=24 ... nemotron_decode_bench`.*
