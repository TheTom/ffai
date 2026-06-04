// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real Mamba2 (mamba2-130m) single-token forward on the shared engine,
//! verified vs HF transformers. Turns SSM from component-verified into a
//! complete real-model-vs-HF family. Single token → zero initial state →
//! conv1d step + SSD scan both start from zero (= HF's 1-token forward).
use ffai_core::{DType, Device as _, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{conv1d_causal_step, gemv, rms_norm, silu, ssm_step};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn mamba2_130m_full_forward_vs_hf() {
    let dir = std::env::var("MAMBA2_DIR").unwrap_or_else(|_| {
        let g = glob_snap();
        g.unwrap_or_default()
    });
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    // config (mamba2-130m)
    let (hid, di, nh, dh, ds, ng, kc, vocab, eps) = (768usize, 1536usize, 24usize, 64usize, 128usize, 1usize, 4usize, 50288usize, 1e-5f32);
    let conv_dim = di + 2 * ng * ds; // 1792
    let n_layers = 24usize;

    // host f32 of a tensor
    let g = |name: &str| -> Vec<f32> { let (b, dt, _s) = st.tensor(name).unwrap(); assert_eq!(dt, DType::F32); fb(b) };
    let up = |v: &[f32]| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };

    let token = 5usize;
    let embed = g("backbone.embeddings.weight"); // [vocab, hid]
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();

    for l in 0..n_layers {
        let p = format!("backbone.layers.{l}");
        // norm → in_proj
        let xn = rms_norm(d, &up(&x), &up(&g(&format!("{p}.norm.weight"))), eps).unwrap();
        let in_proj = upm(&g(&format!("{p}.mixer.in_proj.weight")), vec![3352, hid]);
        let proj = dl(&gemv(d, &in_proj, &xn).unwrap(), 3352);
        let z = &proj[0..di];
        let xbc = &proj[di..di + conv_dim];
        let dt_raw = &proj[di + conv_dim..di + conv_dim + nh];

        // conv1d (transpose HF [d,1,k] → [k,d]) + silu, zero state
        let cw_hf = g(&format!("{p}.mixer.conv1d.weight")); // [conv_dim*1*kc] = [d*kc + k]
        let mut cw = vec![0.0f32; kc * conv_dim];
        for ch in 0..conv_dim { for k in 0..kc { cw[k * conv_dim + ch] = cw_hf[ch * kc + k]; } }
        let cb = g(&format!("{p}.mixer.conv1d.bias"));
        let state0 = vec![0.0f32; (kc - 1) * conv_dim];
        let yc = conv1d_causal_step(d, &up(xbc), &up(&cw), &up(&cb), &up(&state0), conv_dim as u32, kc as u32).unwrap();
        let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
        let x_ssm = &xbc_act[0..di];
        let bmat = &xbc_act[di..di + ng * ds];
        let cmat = &xbc_act[di + ng * ds..di + 2 * ng * ds];

        // dt = softplus(dt_raw + dt_bias)
        let dt_bias = g(&format!("{p}.mixer.dt_bias"));
        let dt: Vec<f32> = (0..nh).map(|i| softplus(dt_raw[i] + dt_bias[i])).collect();

        // SSD scan from zero state
        let a_log = g(&format!("{p}.mixer.A_log"));
        let dsk = g(&format!("{p}.mixer.D"));
        let state_in = vec![0.0f32; nh * dh * ds];
        let (_so, y_t) = ssm_step(d, &up(x_ssm), &up(&a_log), &up(bmat), &up(cmat), &up(&dsk), &up(&dt), &up(&state_in), dh as u32, ds as u32, nh as u32, (nh / ng) as u32).unwrap();
        let y = dl(&y_t, di);

        // gated RMSNorm: rmsnorm(y * silu(z), norm.weight)
        let sz = dl(&silu(d, &up(z)).unwrap(), di);
        let y_gated: Vec<f32> = (0..di).map(|i| y[i] * sz[i]).collect();
        let y_normed = rms_norm(d, &up(&y_gated), &up(&g(&format!("{p}.mixer.norm.weight"))), eps).unwrap();

        // out_proj + residual
        let out_proj = upm(&g(&format!("{p}.mixer.out_proj.weight")), vec![hid, di]);
        let out = dl(&gemv(d, &out_proj, &y_normed).unwrap(), hid);
        for i in 0..hid { x[i] += out[i]; }
    }

    // final norm + tied lm_head
    let xf = rms_norm(d, &up(&x), &up(&g("backbone.norm_f.weight")), eps).unwrap();
    let lm = upm(&embed, vec![vocab, hid]);
    let logits = dl(&gemv(d, &lm, &xf).unwrap(), vocab);
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
    eprintln!("Mamba2-130m full forward on CUDA: argmax = {argmax} (HF = 310)");
    assert_eq!(argmax, 310, "Mamba2 argmax != HF 310");
    eprintln!("✅ Full real Mamba2-130m forward matches HF on the shared engine (GB10 sm_121).");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
