// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! A full transformer decode layer (Qwen2-0.5B geometry) assembled from the
//! shared ffai-ops, run on CUDA through the Device trait, and checked against
//! an independent CPU reference. This is the proof that a model — not just an
//! op — runs on the shared layer: one builder (ffai_models::llama) + the op
//! seam produce a correct forward on real GB10 hardware.
#![cfg(feature = "cuda")]

use ffai_core::{DType, Device, Tensor};
use ffai_cuda::CudaDevice;
use ffai_models::llama::{LayerWeights, LlamaConfig, decode_layer_self};

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
/// Deterministic small values in ~[-0.04, 0.04], varied by salt.
fn fill(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + salt * 131) % 97) as f32 - 48.0) * 0.0008).collect()
}
fn tens(dev: &dyn Device, data: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::new(dev.upload(&to_bytes(data)).unwrap(), shape, DType::F32)
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
/// out[m] = Σ_k mat[m*K + k] * vec[k]
fn matvec(mat: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m).map(|r| (0..k).map(|c| mat[r * k + c] * vec[c]).sum()).collect()
}
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let s = 1.0 / (ms + eps).sqrt();
    (0..n).map(|i| x[i] * s * w[i]).collect()
}

#[test]
fn qwen2_decode_layer_on_cuda_matches_cpu() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    // Qwen2-0.5B geometry.
    let cfg = LlamaConfig {
        hidden: 896,
        n_q_heads: 14,
        n_kv_heads: 2,
        head_dim: 64,
        intermediate: 4864,
        rope_theta: 1_000_000.0,
        eps: 1e-6,
    };
    let h = cfg.hidden;
    let qd = cfg.n_q_heads * cfg.head_dim; // 896
    let kd = cfg.n_kv_heads * cfg.head_dim; // 128
    let im = cfg.intermediate;

    // Weights (deterministic).
    let attn_norm = fill(h, 1);
    let wq = fill(qd * h, 2);
    let wk = fill(kd * h, 3);
    let wv = fill(kd * h, 4);
    let wo = fill(h * qd, 5);
    let mlp_norm = fill(h, 6);
    let w_gate = fill(im * h, 7);
    let w_up = fill(im * h, 8);
    let w_down = fill(h * im, 9);
    let x = fill(h, 10);

    let weights = LayerWeights {
        attn_norm: tens(dev.as_ref(), &attn_norm, vec![h]),
        wq: tens(dev.as_ref(), &wq, vec![qd, h]),
        wk: tens(dev.as_ref(), &wk, vec![kd, h]),
        wv: tens(dev.as_ref(), &wv, vec![kd, h]),
        wo: tens(dev.as_ref(), &wo, vec![h, qd]),
        mlp_norm: tens(dev.as_ref(), &mlp_norm, vec![h]),
        w_gate: tens(dev.as_ref(), &w_gate, vec![im, h]),
        w_up: tens(dev.as_ref(), &w_up, vec![im, h]),
        w_down: tens(dev.as_ref(), &w_down, vec![h, im]),
    };
    let tx = tens(dev.as_ref(), &x, vec![h]);

    // ── GPU: the shared-layer forward ────────────────────────────────
    let out = decode_layer_self(dev.as_ref(), &cfg, &weights, &tx, 0).unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; h * 4];
    dev.download(out.buffer.as_ref(), &mut ob).unwrap();
    let got = from_bytes(&ob);

    // ── CPU reference (pos=0 → RoPE identity, n_kv=1 → attn=v per group) ─
    let hh = rmsnorm(&x, &attn_norm, cfg.eps);
    let q = matvec(&wq, &hh, qd, h);
    let _k = matvec(&wk, &hh, kd, h);
    let v = matvec(&wv, &hh, kd, h);
    let _ = &q; // rope identity at pos 0; q/k unchanged, attn ignores q for n_kv=1
    // attn[q_head, d] = v[kv_head, d], kv_head = q_head / heads_per_group
    let hpg = cfg.n_q_heads / cfg.n_kv_heads;
    let mut attn = vec![0.0f32; qd];
    for qh in 0..cfg.n_q_heads {
        let kvh = qh / hpg;
        for d in 0..cfg.head_dim {
            attn[qh * cfg.head_dim + d] = v[kvh * cfg.head_dim + d];
        }
    }
    let o = matvec(&wo, &attn, h, qd);
    let x1: Vec<f32> = (0..h).map(|i| x[i] + o[i]).collect();
    let h2 = rmsnorm(&x1, &mlp_norm, cfg.eps);
    let gate = matvec(&w_gate, &h2, im, h);
    let up = matvec(&w_up, &h2, im, h);
    let act: Vec<f32> = (0..im).map(|i| silu(gate[i]) * up[i]).collect();
    let down = matvec(&w_down, &act, h, im);
    let want: Vec<f32> = (0..h).map(|i| x1[i] + down[i]).collect();

    let mut err = 0.0f32;
    let mut worst = 0usize;
    for i in 0..h {
        let d = (got[i] - want[i]).abs();
        if d > err {
            err = d;
            worst = i;
        }
    }
    eprintln!(
        "transformer decode layer on CUDA vs CPU: max|Δ|={err:.3e} at [{worst}] (got {:.5}, want {:.5})",
        got[worst], want[worst]
    );
    assert!(err <= 5e-3, "decode layer mismatch: max|Δ|={err:.3e}");
    eprintln!("✅ Qwen2-0.5B-shaped decode layer runs on GB10 through the shared op layer, matches CPU.");
}
