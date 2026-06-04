// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real Gemma-2-2b-it single-token forward, verified vs HF. Exercises the
//! Gemma path: embedding scaled by √hidden, RMSNorm with **(1+weight)** gain,
//! **four** norms per layer (pre+post around both attn and MLP), **geGLU** MLP
//! (gelu_tanh(gate)·up, not SwiGLU), GQA head_dim 256. Attn logit-softcap and
//! final logit-softcap are both monotone / irrelevant to argmax+top-k at a
//! single position (softmax over one key = 1; tanh preserves ordering), so the
//! argmax + top-3 match HF exactly. RoPE identity at pos 0 → skipped.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{gelu, gemv, mul, rms_norm, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn gemma2_2b_full_forward_vs_hf() {
    let dir = std::env::var("GEMMA2_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (hid, nq, nkv, hd, inter, n_layers, vocab, eps) =
        (2304usize, 8usize, 4usize, 256usize, 9216usize, 26usize, 256000usize, 1e-6f32);
    let qdim = nq * hd; let kvdim = nkv * hd; // 2048, 1024
    let scale = 1.0 / (256.0f32).sqrt(); // query_pre_attn_scalar = 256
    let embed_scale = (hid as f32).sqrt(); // √2304 = 48

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let g1 = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0.iter().map(|w| w + 1.0).collect() }; // (1+weight)
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };

    let token = 9707usize;
    let embed = g("model.embed_tokens.weight"); // tied
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].iter().map(|v| v * embed_scale).collect();

    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        // attention (input norm → attn → post-attn norm → residual)
        let h = rms_norm(d, &up(&x, vec![hid]), &up(&g1(&format!("{p}.input_layernorm.weight")), vec![hid]), eps).unwrap();
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![qdim, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![kvdim, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![kvdim, hid]), &h).unwrap();
        // RoPE identity at pos 0 → skip
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, hd]), &k.reshaped(vec![nkv, hd]), &v.reshaped(vec![nkv, hd]),
            hd, 1, 1, (nq / nkv) as u32, scale).unwrap();
        let o = gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, qdim]), &attn.reshaped(vec![qdim])).unwrap();
        let o = rms_norm(d, &o, &up(&g1(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), eps).unwrap();
        let o = dl(&o, hid);
        for i in 0..hid { x[i] += o[i]; }

        // geGLU MLP (pre-ffn norm → geglu → post-ffn norm → residual)
        let h2 = rms_norm(d, &up(&x, vec![hid]), &up(&g1(&format!("{p}.pre_feedforward_layernorm.weight")), vec![hid]), eps).unwrap();
        let gate = gelu(d, &gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &h2).unwrap()).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = mul(d, &gate, &upp).unwrap();
        let down = gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap();
        let down = rms_norm(d, &down, &up(&g1(&format!("{p}.post_feedforward_layernorm.weight")), vec![hid]), eps).unwrap();
        let down = dl(&down, hid);
        for i in 0..hid { x[i] += down[i]; }
    }

    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g1("model.norm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&embed, vec![vocab, hid]), &xf).unwrap(), vocab); // tied
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top3 = &idx[..3];
    eprintln!("Gemma-2-2b full forward on CUDA: top3 = {top3:?} (HF = [9707, 235265, 110])");
    assert_eq!(top3, &[9707usize, 235265, 110], "Gemma top-3 != HF");
    eprintln!("✅ Full real Gemma-2-2b forward matches HF on the shared engine (GB10 sm_121) — geGLU + RMSNorm(1+w) + 4-norm path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--unsloth--gemma-2-2b-it/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
