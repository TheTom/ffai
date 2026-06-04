// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real Pythia-160m (GPT-NeoX) single-token forward, verified vs HF.
//! Exercises the GPT-NeoX path: **parallel residual** (attn and MLP both read
//! the same layer input, summed together), **interleaved per-head QKV** (one
//! query_key_value proj packed [head, 3·head_dim]), partial rotary (rotary_pct
//! 0.25 → identity at pos 0, so skipped), LayerNorm(+bias), exact-erf GELU
//! (host-side), untied embed_out.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{add, gemv, layer_norm, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn erf(x: f32) -> f32 {
    let s = x.signum(); let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}
fn gelu_erf(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) }

#[test]
fn pythia_160m_full_forward_vs_hf() {
    let dir = std::env::var("PYTHIA_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (hid, nh, hd, inter, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 3072usize, 12usize, 50304usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };

    let token = 5usize;
    let embed = g("gpt_neox.embed_in.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();

    for l in 0..n_layers {
        let p = format!("gpt_neox.layers.{l}");
        // both sublayers read the SAME x (parallel residual)
        let ln1 = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]),
            &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        let qkv = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.attention.query_key_value.weight")), vec![3 * hid, hid]), &ln1).unwrap(),
                          &up(&g(&format!("{p}.attention.query_key_value.bias")), vec![3 * hid])).unwrap(), 3 * hid);
        // interleaved per head: [h*3hd + {0,hd,2hd} + d]
        let mut q = vec![0.0f32; hid]; let mut k = vec![0.0f32; hid]; let mut v = vec![0.0f32; hid];
        for h in 0..nh { for dd in 0..hd {
            q[h * hd + dd] = qkv[h * 3 * hd + dd];
            k[h * hd + dd] = qkv[h * 3 * hd + hd + dd];
            v[h * hd + dd] = qkv[h * 3 * hd + 2 * hd + dd];
        }}
        // partial rotary at pos 0 = identity → skip
        let attn = sdpa_decode(d, &up(&q, vec![nh, hd]), &up(&k, vec![nh, hd]), &up(&v, vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let ao = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.attention.dense.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(),
                         &up(&g(&format!("{p}.attention.dense.bias")), vec![hid])).unwrap(), hid);

        // MLP reads x (not the attn output) — parallel
        let ln2 = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]),
            &up(&g(&format!("{p}.post_attention_layernorm.bias")), vec![hid]), eps).unwrap();
        let mut h1 = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.mlp.dense_h_to_4h.weight")), vec![inter, hid]), &ln2).unwrap(),
                            &up(&g(&format!("{p}.mlp.dense_h_to_4h.bias")), vec![inter])).unwrap(), inter);
        for vv in h1.iter_mut() { *vv = gelu_erf(*vv); }
        let m = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.mlp.dense_4h_to_h.weight")), vec![hid, inter]), &up(&h1, vec![inter])).unwrap(),
                        &up(&g(&format!("{p}.mlp.dense_4h_to_h.bias")), vec![hid])).unwrap(), hid);

        for i in 0..hid { x[i] += ao[i] + m[i]; } // parallel residual
    }

    let xf = layer_norm(d, &up(&x, vec![hid]),
        &up(&g("gpt_neox.final_layer_norm.weight"), vec![hid]),
        &up(&g("gpt_neox.final_layer_norm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("embed_out.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
    eprintln!("Pythia-160m full forward on CUDA: argmax = {argmax} (HF = 285)");
    assert_eq!(argmax, 285, "Pythia argmax != HF 285");
    eprintln!("✅ Full real Pythia-160m (GPT-NeoX) forward matches HF on the shared engine (GB10 sm_121) — parallel-residual path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--EleutherAI--pythia-160m/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
