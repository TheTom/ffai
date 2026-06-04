// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real GPT-2 (124M) single-token forward on the shared engine, verified
//! vs HF transformers. Exercises the LayerNorm-LLM path: learned position
//! embedding, LayerNorm (+bias) rather than RMSNorm, gelu_new (= tanh-approx,
//! our device `gelu`), and Conv1D weights (stored transposed `[in, out]` — we
//! transpose to the `[out, in]` our `gemv` expects). Tied lm_head. Causal at
//! pos 0 ⇒ attend-to-self (n_kv=1), exactly HF's 1-token forward.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{add, gelu, gemv, layer_norm, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn gpt2_124m_full_forward_vs_hf() {
    let dir = std::env::var("GPT2_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (hid, nh, hd, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 12usize, 50257usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();

    let g = |name: &str| -> Vec<f32> { let (b, dt, _s) = st.tensor(name).unwrap(); assert_eq!(dt, DType::F32, "{name}"); fb(b) };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    // Conv1D weight is [in, out]; gemv wants [out, in] → transpose.
    let conv_t = |w: &[f32], nin: usize, nout: usize| -> Vec<f32> {
        let mut o = vec![0.0f32; nin * nout];
        for i in 0..nin { for j in 0..nout { o[j * nin + i] = w[i * nout + j]; } }
        o
    };

    let token = 5usize;
    let wte = g("wte.weight"); // [vocab, hid], tied to lm_head
    let wpe = g("wpe.weight"); // [n_pos, hid]
    let mut x: Vec<f32> = (0..hid).map(|i| wte[token * hid + i] + wpe[i]).collect(); // pos 0

    for l in 0..n_layers {
        let p = format!("h.{l}");
        // ── attention (pre-LN) ──
        let h = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.ln_1.weight")), vec![hid]),
            &up(&g(&format!("{p}.ln_1.bias")), vec![hid]), eps).unwrap();
        // c_attn: [hid, 3*hid] Conv1D → qkv [3*hid]
        let cattn_w = conv_t(&g(&format!("{p}.attn.c_attn.weight")), hid, 3 * hid);
        let qkv = add(d, &gemv(d, &up(&cattn_w, vec![3 * hid, hid]), &h).unwrap(),
                      &up(&g(&format!("{p}.attn.c_attn.bias")), vec![3 * hid])).unwrap();
        let qkv = dl(&qkv, 3 * hid);
        let q = up(&qkv[0..hid], vec![nh, hd]);
        let k = up(&qkv[hid..2 * hid], vec![nh, hd]);
        let v = up(&qkv[2 * hid..3 * hid], vec![nh, hd]);
        let attn = sdpa_decode(d, &q, &k, &v, hd, 1, 1, 1, scale).unwrap();
        let cproj_w = conv_t(&g(&format!("{p}.attn.c_proj.weight")), hid, hid);
        let o = add(d, &gemv(d, &up(&cproj_w, vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(),
                    &up(&g(&format!("{p}.attn.c_proj.bias")), vec![hid])).unwrap();
        let o = dl(&o, hid);
        for i in 0..hid { x[i] += o[i]; }

        // ── MLP (pre-LN, gelu_new = tanh) ──
        let h2 = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.ln_2.weight")), vec![hid]),
            &up(&g(&format!("{p}.ln_2.bias")), vec![hid]), eps).unwrap();
        let fc_w = conv_t(&g(&format!("{p}.mlp.c_fc.weight")), hid, 4 * hid);
        let f = add(d, &gemv(d, &up(&fc_w, vec![4 * hid, hid]), &h2).unwrap(),
                    &up(&g(&format!("{p}.mlp.c_fc.bias")), vec![4 * hid])).unwrap();
        let act = gelu(d, &f).unwrap();
        let proj_w = conv_t(&g(&format!("{p}.mlp.c_proj.weight")), 4 * hid, hid);
        let m = add(d, &gemv(d, &up(&proj_w, vec![hid, 4 * hid]), &act).unwrap(),
                    &up(&g(&format!("{p}.mlp.c_proj.bias")), vec![hid])).unwrap();
        let m = dl(&m, hid);
        for i in 0..hid { x[i] += m[i]; }
    }

    // final LN + tied lm_head
    let xf = layer_norm(d, &up(&x, vec![hid]),
        &up(&g("ln_f.weight"), vec![hid]), &up(&g("ln_f.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&wte, vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
    eprintln!("GPT-2-124M full forward on CUDA: argmax = {argmax} (HF = 198)");
    assert_eq!(argmax, 198, "GPT-2 argmax != HF 198");
    eprintln!("✅ Full real GPT-2 forward matches HF on the shared engine (GB10 sm_121) — LayerNorm-LLM path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--gpt2/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
