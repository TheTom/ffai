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
