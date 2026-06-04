// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real StableLM-2-1.6B single-token forward, verified vs HF. Exercises the
//! StableLM path: standard **sequential pre-norm** residual but with
//! **LayerNorm(+bias)** (not RMSNorm), **bias on q/k/v** projections, **partial
//! rotary** (factor 0.25 → identity at pos 0, skipped), SwiGLU MLP, and a final
//! LayerNorm(+bias). Untied lm_head (no bias). MHA (32 heads, head_dim 64),
//! n_kv=1 ⇒ softmax-over-one passthrough. Note: BF16 weights → loaded as f32 by
//! the loader's `tensor_f32`.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{add, gemv, layer_norm, sdpa_decode, swiglu};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn stablelm2_1_6b_full_forward_vs_hf() {
    let dir = std::env::var("STABLELM2_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (hid, nh, hd, inter, n_layers, vocab, eps) =
        (2048usize, 32usize, 64usize, 5632usize, 24usize, 100352usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let bproj = |w: &str, b: &str, x: &Tensor, m: usize| -> Tensor {
        add(d, &gemv(d, &up(&g(w), vec![m, hid]), x).unwrap(), &up(&g(b), vec![m])).unwrap()
    };

    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();

    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        // ── attention (sequential pre-norm, LayerNorm+bias, qkv bias) ──
        let h = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]),
            &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        let q = bproj(&format!("{p}.self_attn.q_proj.weight"), &format!("{p}.self_attn.q_proj.bias"), &h, hid);
        let k = bproj(&format!("{p}.self_attn.k_proj.weight"), &format!("{p}.self_attn.k_proj.bias"), &h, hid);
        let v = bproj(&format!("{p}.self_attn.v_proj.weight"), &format!("{p}.self_attn.v_proj.bias"), &h, hid);
        // partial rotary (0.25) at pos 0 = identity → skip
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]),
            hd, 1, 1, 1, scale).unwrap();
        let o = gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap();
        let o = dl(&o, hid);
        for i in 0..hid { x[i] += o[i]; }

        // ── SwiGLU MLP (post_attention_layernorm) ──
        let h2 = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]),
            &up(&g(&format!("{p}.post_attention_layernorm.bias")), vec![hid]), eps).unwrap();
        let gate = gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = swiglu(d, &gate, &upp).unwrap();
        let m = dl(&gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap(), hid);
        for i in 0..hid { x[i] += m[i]; }
    }

    let xf = layer_norm(d, &up(&x, vec![hid]),
        &up(&g("model.norm.weight"), vec![hid]), &up(&g("model.norm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab); // untied
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top3 = &idx[..3];
    eprintln!("StableLM-2-1.6B full forward on CUDA: top3 = {top3:?} (HF = [341, 11, 280])");
    assert_eq!(top3, &[341usize, 11, 280], "StableLM-2 top-3 != HF");
    eprintln!("✅ Full real StableLM-2-1.6B forward matches HF on the shared engine (GB10 sm_121) — LayerNorm+bias / qkv-bias / partial-rotary path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--stabilityai--stablelm-2-1_6b/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
