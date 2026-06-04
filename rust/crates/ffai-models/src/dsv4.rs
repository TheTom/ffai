// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 Multi-head Latent Attention (decode), assembled from the
//! verified DSv4 ops in [`ffai_ops`]. Single-token, single-position KV
//! (`n_kv = 1`). Covers the full-attention layers; the CSA/HCA sparse path
//! (compressor + Lightning indexer) is a separate track (WIP in the
//! reference too). The mHC residual wrapping is the model loop's job.

use ffai_core::{DType, Device, Result, Tensor};
use ffai_ops as ops;

#[derive(Debug, Clone, Copy)]
pub struct MlaConfig {
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize, // 512 (the d512 sink-SDPA kernel)
    pub q_lora_rank: usize,
    pub n_nope: usize,  // 448
    pub half_rot: usize, // 32  (rope tail = 64)
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

/// MLA attention weights (one full-attention layer).
pub struct MlaWeights {
    pub attn_norm: Tensor,        // [hidden]
    pub q_a: Tensor,              // [q_lora_rank, hidden]
    pub q_a_norm: Tensor,         // [q_lora_rank]
    pub q_b: Tensor,              // [n_heads*head_dim, q_lora_rank]
    pub kv: Tensor,               // [head_dim, hidden]
    pub kv_a_norm: Tensor,        // [head_dim]
    pub sink: Tensor,             // [n_heads] f32
    pub output_a: Vec<Tensor>,    // o_groups × [o_lora_rank, gsize]
    pub output_b: Tensor,         // [hidden, o_groups*o_lora_rank]
}

fn fb(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn tb(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Run MLA attention for a single token: `x [hidden]` → `blockOut [hidden]`.
pub fn mla_attention(
    dev: &dyn Device,
    cfg: &MlaConfig,
    w: &MlaWeights,
    x: &Tensor,
    position: u32,
) -> Result<Tensor> {
    let hd = cfg.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    let xn = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;

    // Q low-rank path + per-head unit-RMS norm + partial RoPE.
    let qa = ops::gemv(dev, &w.q_a, &xn)?;
    let qan = ops::rms_norm(dev, &qa, &w.q_a_norm, cfg.eps)?;
    let q = ops::gemv(dev, &w.q_b, &qan)?; // [n_heads*head_dim]
    let ones = Tensor::new(dev.upload(&tb(&vec![1.0f32; hd]))?, vec![hd], DType::F32);
    let q = ops::rms_norm(dev, &q.reshaped(vec![cfg.n_heads, hd]), &ones, cfg.eps)?;
    let q = ops::dsv4_partial_rope(
        dev, &q, cfg.n_heads as u32, hd as u32, cfg.n_nope as u32, cfg.half_rot as u32,
        position, cfg.rope_theta, false,
    )?;

    // KV latent path + norm + partial RoPE (single kv head).
    let kv = ops::gemv(dev, &w.kv, &xn)?; // [head_dim]
    let kvn = ops::rms_norm(dev, &kv, &w.kv_a_norm, cfg.eps)?;
    let kvn = ops::dsv4_partial_rope(
        dev, &kvn.reshaped(vec![1, hd]), 1, hd as u32, cfg.n_nope as u32, cfg.half_rot as u32,
        position, cfg.rope_theta, false,
    )?;

    // MQA sink-SDPA over the single-position cache (n_kv=1), then inverse RoPE.
    let attn =
        ops::sdpa_decode_sink(dev, &q, &kvn, &kvn, &w.sink, 1, 1, cfg.n_heads as u32, scale)?;
    let attn = ops::dsv4_partial_rope(
        dev, &attn, cfg.n_heads as u32, hd as u32, cfg.n_nope as u32, cfg.half_rot as u32,
        position, cfg.rope_theta, true,
    )?;

    // Grouped O-LoRA: attn [n_heads*head_dim] → o_groups groups; per-group
    // low-rank, concat, then the dense up-projection. The group slicing is
    // host-side (single token) since gemv consumes whole buffers.
    let qd = cfg.n_heads * hd;
    let gsize = qd / cfg.o_groups;
    let mut ab = vec![0u8; qd * 4];
    dev.synchronize()?;
    dev.download(attn.buffer.as_ref(), &mut ab)?;
    let attn_h = fb(&ab);

    let mut o_low = vec![0.0f32; cfg.o_groups * cfg.o_lora_rank];
    for g in 0..cfg.o_groups {
        let slice = &attn_h[g * gsize..(g + 1) * gsize];
        let tslice = Tensor::new(dev.upload(&tb(slice))?, vec![gsize], DType::F32);
        let og = ops::gemv(dev, &w.output_a[g], &tslice)?; // [o_lora_rank]
        dev.synchronize()?;
        let mut ogb = vec![0u8; cfg.o_lora_rank * 4];
        dev.download(og.buffer.as_ref(), &mut ogb)?;
        o_low[g * cfg.o_lora_rank..(g + 1) * cfg.o_lora_rank].copy_from_slice(&fb(&ogb));
    }
    let t_olow = Tensor::new(dev.upload(&tb(&o_low))?, vec![o_low.len()], DType::F32);
    ops::gemv(dev, &w.output_b, &t_olow)
}
