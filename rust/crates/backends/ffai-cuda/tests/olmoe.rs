// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real MoE model (OLMoE-1B-7B-0924) single-token forward on the shared
//! engine, verified vs HF transformers. Turns MoE from block-verified into a
//! complete real-model-vs-HF family. 64 experts, top-8, no shared expert,
//! norm_topk_prob=false, qk-norm over the *full* 2048 projection (not per-head
//! like Qwen3), MHA head_dim 128. Sharded BF16 checkpoint (3 files) via the
//! mmap sharded loader. Single token at pos 0 → RoPE is identity → skipped.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{gemv, rms_norm, sdpa_decode, swiglu};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn bf16(b: &[u8]) -> Vec<f32> { b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect() }

#[test]
fn olmoe_1b7b_full_forward_vs_hf() {
    let dir = std::env::var("OLMOE_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    // config (OLMoE-1B-7B-0924)
    let (hid, nh, hd, inter, n_exp, top_k, n_layers, vocab, eps) =
        (2048usize, 16usize, 128usize, 1024usize, 64usize, 8usize, 16usize, 50304usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();

    // host f32 of a tensor (BF16 or F32 on disk)
    let g = |name: &str| -> Vec<f32> {
        let (b, dt, _s) = st.tensor(name).unwrap();
        match dt { DType::F32 => fb(b), DType::BF16 => bf16(b), o => panic!("dtype {o:?} for {name}") }
    };
    let up = |v: &[f32]| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };

    let token = 5usize;
    let embed = g("model.embed_tokens.weight"); // [vocab, hid]
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();

    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        // ── attention ──
        let xn = rms_norm(d, &up(&x), &up(&g(&format!("{p}.input_layernorm.weight"))), eps).unwrap();
        let q = gemv(d, &upm(&g(&format!("{p}.self_attn.q_proj.weight")), vec![hid, hid]), &xn).unwrap();
        let k = gemv(d, &upm(&g(&format!("{p}.self_attn.k_proj.weight")), vec![hid, hid]), &xn).unwrap();
        let v = gemv(d, &upm(&g(&format!("{p}.self_attn.v_proj.weight")), vec![hid, hid]), &xn).unwrap();
        // qk-norm over the FULL 2048 projection (q_norm/k_norm weight is [2048])
        let q = rms_norm(d, &q, &up(&g(&format!("{p}.self_attn.q_norm.weight"))), eps).unwrap();
        let k = rms_norm(d, &k, &up(&g(&format!("{p}.self_attn.k_norm.weight"))), eps).unwrap();
        // pos 0 → RoPE identity → skip. MHA: 16 q heads = 16 kv heads, n_kv=1.
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]),
            &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = gemv(d, &upm(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, hid]),
            &attn.reshaped(vec![nh * hd])).unwrap();
        let o_h = dl(&o, hid);
        for i in 0..hid { x[i] += o_h[i]; }

        // ── MoE MLP ──
        let xn2 = rms_norm(d, &up(&x), &up(&g(&format!("{p}.post_attention_layernorm.weight"))), eps).unwrap();
        let logits = dl(&gemv(d, &upm(&g(&format!("{p}.mlp.gate.weight")), vec![n_exp, hid]), &xn2).unwrap(), n_exp);
        // softmax over all experts (HF order), then top-k, NO renorm (norm_topk_prob=false)
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&z| (z - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        let mut idx: Vec<usize> = (0..n_exp).collect();
        idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let chosen = &idx[..top_k];

        let mut acc = vec![0.0f32; hid];
        for &e in chosen {
            let w = probs[e];
            let ep = format!("{p}.mlp.experts.{e}");
            let ge = gemv(d, &upm(&g(&format!("{ep}.gate_proj.weight")), vec![inter, hid]), &xn2).unwrap();
            let ue = gemv(d, &upm(&g(&format!("{ep}.up_proj.weight")), vec![inter, hid]), &xn2).unwrap();
            let act = swiglu(d, &ge, &ue).unwrap();
            let de = dl(&gemv(d, &upm(&g(&format!("{ep}.down_proj.weight")), vec![hid, inter]), &act).unwrap(), hid);
            for i in 0..hid { acc[i] += w * de[i]; }
        }
        for i in 0..hid { x[i] += acc[i]; }
    }

    // final norm + untied lm_head
    let xf = rms_norm(d, &up(&x), &up(&g("model.norm.weight")), eps).unwrap();
    let logits = dl(&gemv(d, &upm(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
    eprintln!("OLMoE-1B-7B full forward on CUDA: argmax = {argmax} (HF = 310)");
    assert_eq!(argmax, 310, "OLMoE argmax != HF 310");
    eprintln!("✅ Full real OLMoE-1B-7B (64-expert MoE) forward matches HF on the shared engine (GB10 sm_121).");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
