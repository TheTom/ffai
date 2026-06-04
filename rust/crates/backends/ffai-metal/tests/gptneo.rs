// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real GPT-Neo-125M single-token forward, verified vs HF. Exercises the
//! GPT-Neo path: GPT-2-style **learned position embedding** + **LayerNorm(+bias)**
//! pre-norm sequential residual, but with SEPARATE q/k/v Linear projections
//! (real `[out,in]` weights — NOT GPT-2's transposed Conv1D), **no bias on
//! q/k/v** (bias only on out_proj), no rotary, gelu_new (= tanh-approx, device
//! `gelu`), and **tied** lm_head (= wte). GPT-Neo uniquely does NOT scale the
//! attention logits by 1/√head_dim — but at pos 0 attention is over a single
//! key (softmax-of-one = 1), so the scale is irrelevant and the value passes
//! through. Alternating local/global attention layers also collapse to
//! self-attention at a single token. 12 heads, head_dim 64.
use ffai_core::{DType, Tensor};
use ffai_metal::MetalDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{add, gelu, gemv, layer_norm, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn gptneo_125m_full_forward_vs_hf() {
    let dir = std::env::var("GPTNEO_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();

    let (hid, nh, hd, inter, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 3072usize, 12usize, 50257usize, 1e-5f32);
    // softmax over a single key ⇒ scale is irrelevant at pos 0 (GPT-Neo uses no 1/√d anyway)
    let scale = 1.0f32;

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };

    let token = 5usize;
    let wte = g("transformer.wte.weight"); // [vocab, hid], tied to lm_head
    let wpe = g("transformer.wpe.weight"); // [n_pos, hid]
    let mut x: Vec<f32> = (0..hid).map(|i| wte[token * hid + i] + wpe[i]).collect(); // pos 0

    for l in 0..n_layers {
        let p = format!("transformer.h.{l}");
        // ── attention (pre-LN; q/k/v no bias, out_proj has bias) ──
        let h = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.ln_1.weight")), vec![hid]),
            &up(&g(&format!("{p}.ln_1.bias")), vec![hid]), eps).unwrap();
        let q = gemv(d, &up(&g(&format!("{p}.attn.attention.q_proj.weight")), vec![hid, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.attn.attention.k_proj.weight")), vec![hid, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.attn.attention.v_proj.weight")), vec![hid, hid]), &h).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = add(d, &gemv(d, &up(&g(&format!("{p}.attn.attention.out_proj.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(),
                    &up(&g(&format!("{p}.attn.attention.out_proj.bias")), vec![hid])).unwrap();
        let o = dl(&o, hid);
        for i in 0..hid { x[i] += o[i]; }

        // ── MLP (pre-LN, gelu_new = tanh) ──
        let h2 = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.ln_2.weight")), vec![hid]),
            &up(&g(&format!("{p}.ln_2.bias")), vec![hid]), eps).unwrap();
        let f = add(d, &gemv(d, &up(&g(&format!("{p}.mlp.c_fc.weight")), vec![inter, hid]), &h2).unwrap(),
                    &up(&g(&format!("{p}.mlp.c_fc.bias")), vec![inter])).unwrap();
        let act = gelu(d, &f).unwrap();
        let m = add(d, &gemv(d, &up(&g(&format!("{p}.mlp.c_proj.weight")), vec![hid, inter]), &act).unwrap(),
                    &up(&g(&format!("{p}.mlp.c_proj.bias")), vec![hid])).unwrap();
        let m = dl(&m, hid);
        for i in 0..hid { x[i] += m[i]; }
    }

    // final LN + tied lm_head
    let xf = layer_norm(d, &up(&x, vec![hid]),
        &up(&g("transformer.ln_f.weight"), vec![hid]), &up(&g("transformer.ln_f.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&wte, vec![vocab, hid]), &xf).unwrap(), vocab); // tied
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top3 = &idx[..3];
    eprintln!("GPT-Neo-125M full forward on Metal: top3 = {top3:?} (HF = [28, 59, 91])");
    assert_eq!(top3, &[28usize, 59, 91], "GPT-Neo top-3 != HF");
    eprintln!("✅ Full real GPT-Neo-125M forward matches HF on the shared engine (Apple GPU) — learned-pos + LayerNorm + separate-qkv path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--EleutherAI--gpt-neo-125m/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
