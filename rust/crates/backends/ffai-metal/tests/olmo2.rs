// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real OLMo-2-0425-1B single-token forward, verified vs HF. Exercises the
//! OLMo-2 path: **post-norm** placement — each sublayer reads the RAW residual
//! (no pre-norm), and the RMSNorm is applied to the sublayer OUTPUT before the
//! residual add (`residual + post_attention_layernorm(attn(x))`, likewise for
//! MLP). Plus **QK-norm**: a plain RMSNorm over the FULL q (2048) and k (2048)
//! projections before the head reshape. SwiGLU MLP, GQA=MHA here (16 heads,
//! head_dim 128), plain RMSNorm (no 1+w), untied lm_head. RoPE (theta 500000)
//! is identity at pos 0 → skipped; n_kv=1 ⇒ softmax-over-one = passthrough.
use ffai_core::{DType, Tensor};
use ffai_metal::MetalDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{gemv, rms_norm, sdpa_decode, swiglu};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn olmo2_1b_full_forward_vs_hf() {
    let dir = std::env::var("OLMO2_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();

    let (hid, nq, nkv, hd, inter, n_layers, vocab, eps) =
        (2048usize, 16usize, 16usize, 128usize, 8192usize, 16usize, 100352usize, 1e-6f32);
    let qdim = nq * hd; let kvdim = nkv * hd; // 2048, 2048
    let scale = 1.0 / (hd as f32).sqrt();

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };

    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();

    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        // ── attention: reads RAW x (no pre-norm) ──
        let xin = up(&x, vec![hid]);
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![qdim, hid]), &xin).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![kvdim, hid]), &xin).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![kvdim, hid]), &xin).unwrap();
        // QK-norm over the FULL projection (before head reshape), RoPE identity at pos 0
        let q = rms_norm(d, &q, &up(&g(&format!("{p}.self_attn.q_norm.weight")), vec![qdim]), eps).unwrap();
        let k = rms_norm(d, &k, &up(&g(&format!("{p}.self_attn.k_norm.weight")), vec![kvdim]), eps).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, hd]), &k.reshaped(vec![nkv, hd]), &v.reshaped(vec![nkv, hd]),
            hd, 1, 1, (nq / nkv) as u32, scale).unwrap();
        let o = gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, qdim]), &attn.reshaped(vec![qdim])).unwrap();
        // post-norm: norm the attn output, THEN add to residual
        let o = rms_norm(d, &o, &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), eps).unwrap();
        let o = dl(&o, hid);
        for i in 0..hid { x[i] += o[i]; }

        // ── SwiGLU MLP: reads RAW x, norm applied to output ──
        let xin2 = up(&x, vec![hid]);
        let gate = gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &xin2).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &xin2).unwrap();
        let act = swiglu(d, &gate, &upp).unwrap();
        let down = gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap();
        let down = rms_norm(d, &down, &up(&g(&format!("{p}.post_feedforward_layernorm.weight")), vec![hid]), eps).unwrap();
        let down = dl(&down, hid);
        for i in 0..hid { x[i] += down[i]; }
    }

    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g("model.norm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab); // untied
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top3 = &idx[..3];
    eprintln!("OLMo-2-1B full forward on Metal: top3 = {top3:?} (HF = [198, 8, 13])");
    assert_eq!(top3, &[198usize, 8, 13], "OLMo-2 top-3 != HF");
    eprintln!("✅ Full real OLMo-2-1B forward matches HF on the shared engine (Apple GPU) — post-norm + QK-norm path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
