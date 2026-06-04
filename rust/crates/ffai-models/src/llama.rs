// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Transformer-LLM builder (Llama / Qwen / Mistral / Yi / Phi / SmolLM …).
//! One decode layer assembled entirely from [`ffai_ops`] — so it runs on any
//! backend that implements [`ffai_core::Device`]. Config-parameterized: the
//! same code serves every model in this family by varying [`LlamaConfig`].

use ffai_core::{Device, Result, Tensor};
use ffai_ops as ops;

/// Architecture config shared by the transformer-LLM family.
#[derive(Debug, Clone, Copy)]
pub struct LlamaConfig {
    pub hidden: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

impl LlamaConfig {
    pub fn heads_per_group(&self) -> u32 {
        (self.n_q_heads / self.n_kv_heads) as u32
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Per-layer weights (dense). Row-major: a projection `[out, in]` is applied
/// as `gemv(W, x)`.
pub struct LayerWeights {
    pub attn_norm: Tensor, // [hidden]
    pub wq: Tensor,        // [n_q_heads*head_dim, hidden]
    pub wk: Tensor,        // [n_kv_heads*head_dim, hidden]
    pub wv: Tensor,        // [n_kv_heads*head_dim, hidden]
    pub wo: Tensor,        // [hidden, n_q_heads*head_dim]
    pub mlp_norm: Tensor,  // [hidden]
    pub w_gate: Tensor,    // [intermediate, hidden]
    pub w_up: Tensor,      // [intermediate, hidden]
    pub w_down: Tensor,    // [hidden, intermediate]
}

/// One transformer decode step for a single token, attending only to itself
/// (single-position KV — `n_kv = 1`). This exercises the full layer pipeline
/// — RMSNorm → QKV proj → RoPE → SDPA → O proj → residual → RMSNorm →
/// SwiGLU MLP → residual — through the shared op layer. The multi-position
/// KV cache (write new k/v at `pos`, attend over `[0, pos]`) is the model
/// loop's responsibility; this is the per-layer compute.
///
/// `x` is the `[hidden]` residual stream for the current token.
pub fn decode_layer_self(
    dev: &dyn Device,
    cfg: &LlamaConfig,
    w: &LayerWeights,
    x: &Tensor,
    pos: u32,
) -> Result<Tensor> {
    let hd = cfg.head_dim;
    let theta = cfg.rope_theta;

    // ── attention ────────────────────────────────────────────────────
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;
    let q = ops::gemv(dev, &w.wq, &h)?;
    let k = ops::gemv(dev, &w.wk, &h)?;
    let v = ops::gemv(dev, &w.wv, &h)?;

    // RoPE on Q and K (vanilla — freq-band scaling disabled).
    let q = ops::rope_llama(dev, &q.reshaped(vec![cfg.n_q_heads, hd]), pos, theta, 1.0, 1.0, 1.0, 1e9)?;
    let k = ops::rope_llama(dev, &k.reshaped(vec![cfg.n_kv_heads, hd]), pos, theta, 1.0, 1.0, 1.0, 1e9)?;

    // Single-position attention: KV cache is just this token (n_kv=1, stride=1).
    let attn = ops::sdpa_decode(
        dev,
        &q,
        &k,
        &v.reshaped(vec![cfg.n_kv_heads, hd]),
        hd,
        1,
        1,
        cfg.heads_per_group(),
        cfg.attn_scale(),
    )?;

    let o = ops::gemv(dev, &w.wo, &attn.reshaped(vec![cfg.n_q_heads * hd]))?;
    let x1 = ops::add(dev, x, &o)?;

    // ── MLP (SwiGLU) ─────────────────────────────────────────────────
    let h2 = ops::rms_norm(dev, &x1, &w.mlp_norm, cfg.eps)?;
    let gate = ops::gemv(dev, &w.w_gate, &h2)?;
    let up = ops::gemv(dev, &w.w_up, &h2)?;
    let act = ops::swiglu(dev, &gate, &up)?;
    let down = ops::gemv(dev, &w.w_down, &act)?;
    let x2 = ops::add(dev, &x1, &down)?;

    Ok(x2)
}
