// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real Phi-1.5 single-token forward, verified vs HF. Phi path: a SINGLE
//! shared input_layernorm feeding **parallel** attn + MLP (both read the same
//! normed input, summed with the residual), separate q/k/v/dense projections
//! all with bias, partial rotary (factor 0.5 → identity at pos 0, skipped),
//! gelu_new (= tanh-approx, our device `gelu`), untied lm_head with bias.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{add, gelu, gemv, layer_norm, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn phi_1_5_full_forward_vs_hf() {
    let dir = std::env::var("PHI_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (hid, nh, hd, inter, n_layers, vocab, eps) = (2048usize, 32usize, 64usize, 8192usize, 24usize, 51200usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let proj = |w: &str, b: &str, x: &Tensor, m: usize| -> Tensor {
        add(d, &gemv(d, &up(&g(w), vec![m, hid_in(w, hid, inter)]), x).unwrap(), &up(&g(b), vec![m])).unwrap()
    };

    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();

    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]),
            &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        // parallel attention
        let q = proj(&format!("{p}.self_attn.q_proj.weight"), &format!("{p}.self_attn.q_proj.bias"), &h, hid);
        let k = proj(&format!("{p}.self_attn.k_proj.weight"), &format!("{p}.self_attn.k_proj.bias"), &h, hid);
        let v = proj(&format!("{p}.self_attn.v_proj.weight"), &format!("{p}.self_attn.v_proj.bias"), &h, hid);
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&proj(&format!("{p}.self_attn.dense.weight"), &format!("{p}.self_attn.dense.bias"), &attn.reshaped(vec![hid]), hid), hid);
        // parallel MLP (from the same h)
        let f = proj(&format!("{p}.mlp.fc1.weight"), &format!("{p}.mlp.fc1.bias"), &h, inter);
        let act = gelu(d, &f).unwrap();
        let m = dl(&proj(&format!("{p}.mlp.fc2.weight"), &format!("{p}.mlp.fc2.bias"), &act, hid), hid);
        for i in 0..hid { x[i] += o[i] + m[i]; } // parallel residual
    }

    let xf = layer_norm(d, &up(&x, vec![hid]),
        &up(&g("model.final_layernorm.weight"), vec![hid]), &up(&g("model.final_layernorm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&add(d, &gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), &up(&g("lm_head.bias"), vec![vocab])).unwrap(), vocab);
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top3 = &idx[..3];
    eprintln!("Phi-1.5 full forward on CUDA: top3 = {top3:?} (HF = [11, 13, 546])");
    assert_eq!(top3, &[11usize, 13, 546], "Phi top-3 != HF");
    eprintln!("✅ Full real Phi-1.5 forward matches HF on the shared engine (GB10 sm_121) — parallel single-norm path verified.");
}

// fc1 has in_dim = hid, fc2 has in_dim = inter; q/k/v/dense have in_dim = hid.
fn hid_in(w: &str, hid: usize, inter: usize) -> usize { if w.ends_with("fc2.weight") { inter } else { hid } }

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--microsoft--phi-1_5/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
