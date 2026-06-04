# DeepSeek-V4-Flash → shared Rust engine — implementation spec

Status: **groundwork / roadmap.** DSv4-Flash is a research-grade architecture
with several novel subsystems; this maps the exact forward, weights, kernels,
and config so the Rust port is a build problem, not a discovery problem.

> ⚠️ The Swift reference is itself **partial**: full-attention layers work,
> but CSA/HCA compression + the Lightning Indexer are WIP (greedy decode NaNs
> on those layers). A fully-correct end-to-end DSv4 reference does not yet
> exist on either side — port the full-attention path first and treat
> CSA/HCA as a parallel research track.

## Config (DSv4-Flash)

hidden 4096 · layers 43 · vocab 129280 · n_heads 64 · n_kv_heads 1 ·
head_dim 512 · qk_rope_head_dim 64 (nope 448 / rope 64) · q_lora_rank 1024 ·
o_lora_rank 1024 · o_groups 8 · rope_theta 10000 (compress 160000) ·
sliding_window 128 · n_routed_experts 256 · experts_per_tok 6 ·
n_shared_experts 1 · moe_intermediate 2048 · scoring `sqrtsoftplus` ·
routed_scaling 1.5 · swiglu_limit 10.0 · rms_norm_eps 1e-6 ·
mHC: hc_mult 4, sinkhorn_iters 20 · **activations f32** (f16 drifts ~0.08
logits over 43 layers).

## Per-layer forward (decode)

mHC sinkhorn-split → mHC collapse (4ch→1) → attn_norm → **MLA**:
q_a → q_a_norm → q_b → per-head unit-RMS q-norm → partial-RoPE(q tail) ;
kv → kv_a_norm → partial-RoPE(kv tail) → sliding-window cache append →
SDPA d512 + per-head sink → inverse partial-RoPE → grouped O-LoRA (8×) →
mHC expand. Then mHC split → collapse → ffn_norm → **DSv4-MoE**
(sqrtsoftplus router + bias → top-6 → per-expert clamped-SwiGLU + 1 shared)
→ mHC expand.

## metaltile kernels (exist, pass the CUDA corpus)

`ffai_dsv4_partial_rope` (qk,out,head_dim,n_nope,half_rot,position,theta,
inverse,freq_scale,ext,corr_low,corr_high) · `ffai_dsv4_indexer_score` ·
`ffai_dsv4_indexer_topk_block` · `ffai_dsv4_mhc_sinkhorn_split` ·
`ffai_dsv4_mhc_collapse` · `ffai_dsv4_mhc_expand` · `ffai_dsv4_compressor_pool` ·
`ffai_sdpa_decode_d512_sink` (reused MLX) · `ffai_moe_router_sqrtsoftplus` ·
`ffai_dsv4_swiglu_limit` · dequant: `ffai_dsv4_mxfp4_dequant`,
`ffai_dsv4_fp8_block_dequant`.

## GGUF weights (blk.N.*) + quant

q_a/q_b/kv/output_a/output_b/shexp: **q8_0** · gate_exps/up_exps: **iq2_xxs**
· down_exps: **q2_K** · ffn_gate_inp/mHC/compressor/indexer: **f16** ·
norms/sinks/scales/biases: **f32**. token_embd f16, output q8_0.
(See Swift `Models/Text/DeepSeekV4Text.swift` loader + `Loader/GGUF/`.)

## Build order (each its own focused session)

1. **GGUF v3 loader + dequant** (q8_0, q2_K, iq2_xxs, f16, f32) — gate for everything.
2. **MLA attention builder** — partial-RoPE + sink-SDPA ops wired into ffai-ops; validate the op-composition vs CPU on both platforms (kernels already pass the corpus).
3. **DSv4-MoE** — sqrtsoftplus router + clamped-SwiGLU + shared expert (extends the verified `ffai_models::moe`).
4. **mHC** — sinkhorn-split / collapse / expand (the 4-channel residual).
5. **Full-attention layers end-to-end** on CUDA, diff vs Swift full-attn output.
6. **CSA/HCA + indexer** — research track; reference is WIP.

## Real checkpoint — reverse-engineered layout (2026-06, IQ2XXS 86 GB, CUDA box)

GGUF dequant verified vs gguf-py on the real checkpoint **on the GB10 box**
(43 blocks, 1328 tensors): Q8_0 Δ4.0e-7, Q2_K Δ4.9e-7, IQ2_XXS Δ4.7e-7.
Composites (MLA/MoE/mHC/swiglu_limit/router) all pass on CUDA vs CPU.

**Config (`deepseek4.*` metadata):** hidden 4096 · 64 heads · head_dim
(key/value_length) 512 · q_lora 1024 · output_lora 1024 · output_groups 8 ·
rope_dim 64 (⇒ n_nope 448, half_rot 32) · eps 1e-6 · experts 256 (used 6,
shared 1, ffn 2048, weights_scale 1.5, norm true) · swiglu_clamp 10 · mHC
count 4, sinkhorn_iters 20.

**blk.N tensor → struct map** (ggml `[in,out]` dims = row-major `[out,in]` =
struct layout, no transpose):

| GGUF tensor | dims | → struct field |
|---|---|---|
| `attn_norm.weight` | [4096] | `MlaWeights.attn_norm` |
| `attn_q_a.weight` | [4096,1024] | `q_a` [1024,4096] |
| `attn_q_a_norm.weight` | [1024] | `q_a_norm` |
| `attn_q_b.weight` | [1024,32768] | `q_b` [32768,1024] |
| `attn_kv.weight` | [4096,512] | `kv` [512,4096] |
| `attn_kv_a_norm.weight` | [512] | `kv_a_norm` |
| `attn_sinks.weight` | [64] | `sink` (per-head) |
| `attn_output_a.weight` | [4096,8192] | `output_a` = 8 groups × [1024,4096] |
| `attn_output_b.weight` | [8192,4096] | `output_b` [4096,8192] |
| `hc_attn_fn.weight` | [16384,24] | `MhcWeights.hc_fn` [24,16384] |
| `hc_attn_base.weight` | [24] | `hc_base` |
| `hc_attn_scale.weight` | [3] | `hc_scale` (pre/post/comb — a tensor, not a const) |
| `ffn_gate_inp.weight` | [4096,256] | MoE router |
| `ffn_{gate,up}_exps.weight` | [4096,2048,256] | 256 experts (IQ2_XXS) |
| `ffn_down_exps.weight` | [2048,4096,256] | 256 experts (Q2_K) |
| `ffn_{gate,up,down}_shexp.weight` | … | shared expert |
| `ffn_gate_tid2eid.weight` | [6,129280] | hash-routing table |

**Full-forward blocker (now named precisely):** the `attention.indexer`
(head_count 64, key_length 128, top_k 512) + `sliding_window 128` — the CSA
sparse-attention indexer, WIP in the reference itself ⇒ no correct oracle for
the full 43-layer forward. Everything *else* (dense MLA + mHC + MoE per layer)
is op-verified; a real-weight single-layer forward (GPU-vs-CPU) is turnkey from
the map above.
