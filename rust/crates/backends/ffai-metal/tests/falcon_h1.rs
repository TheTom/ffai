// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real Falcon-H1-0.5B single-token forward, verified vs HF. The unique
//! **hybrid** family: every layer runs a Mamba2 SSM mixer AND a GQA attention
//! in PARALLEL off one shared input norm, summed into the residual, then a
//! SwiGLU FFN. Reuses the whole shared op set — `conv1d_causal_step` + `ssm_step`
//! (the Mamba2 mixer; gate is y·silu(z), no rms-norm here) alongside
//! `sdpa_decode` — in one model. Falcon's µP scalar multipliers are handled
//! exactly: ssm_in folds into the mup vector, gate_mult into the gate weight,
//! ssm_out/attn_out/down scale their outputs; key_mult (softmax-of-1 at one
//! position) and lm_head_mult (monotone) don't affect argmax/top-k and are
//! omitted. RoPE identity at pos 0 → skipped.
use ffai_core::{DType, Tensor};
use ffai_metal::MetalDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{conv1d_causal_step, gemv, rms_norm, sdpa_decode, silu, ssm_step};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn falcon_h1_0_5b_full_forward_vs_hf() {
    let dir = std::env::var("FALCON_H1_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();

    // config (Falcon-H1-0.5B)
    let (hid, nq, nkv, ahd, inter, n_layers, vocab, eps) =
        (1024usize, 8usize, 2usize, 64usize, 2048usize, 36usize, 32784usize, 1e-5f32); // attn head_dim=64 (explicit, not hid/nq)
    // mamba
    let (d_ssm, m_nh, m_dh, d_state, n_groups, d_conv) = (1536usize, 24usize, 64usize, 128usize, 1usize, 4usize);
    let conv_dim = d_ssm + 2 * n_groups * d_state; // 1792
    let proj_dim = 2 * d_ssm + 2 * n_groups * d_state + m_nh; // 3352
    let ascale = 1.0 / (ahd as f32).sqrt();
    // µP multipliers
    let (ssm_in, ssm_out, attn_out) = (1.25f32, 0.23570226039551587f32, 0.9375f32);
    let (gate_mult, down_mult, embed_mult) = (0.8838834764831844f32, 0.5859375f32, 5.656854249492381f32);
    let ssm_m = [0.3535533905932738f32, 0.25, 0.3535533905932738, 0.5, 0.3535533905932738];

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };

    // mup vector with ssm_in folded in (proj = in_proj(h·ssm_in)·mup = in_proj(h)·(mup·ssm_in))
    let mut mup = vec![0.0f32; proj_dim];
    for i in 0..proj_dim {
        let m = if i < d_ssm { ssm_m[0] }                         // z / gate
            else if i < 2 * d_ssm { ssm_m[1] }                    // x
            else if i < 2 * d_ssm + n_groups * d_state { ssm_m[2] } // B
            else if i < 2 * d_ssm + 2 * n_groups * d_state { ssm_m[3] } // C
            else { ssm_m[4] };                                    // dt
        mup[i] = m * ssm_in;
    }

    let token = 9707usize; // token 5 is a reserved/zero-embedding slot — use a real token
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].iter().map(|v| v * embed_mult).collect();

    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = rms_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), eps).unwrap();

        // ── Mamba2 mixer ──
        let mut proj = dl(&gemv(d, &up(&g(&format!("{p}.mamba.in_proj.weight")), vec![proj_dim, hid]), &h).unwrap(), proj_dim);
        for i in 0..proj_dim { proj[i] *= mup[i]; }
        let z = &proj[0..d_ssm];
        let xbc = &proj[d_ssm..d_ssm + conv_dim];
        let dt_raw = &proj[d_ssm + conv_dim..proj_dim];
        // conv1d (transpose HF [d,1,k] → [k,d]) + silu, zero state
        let cw_hf = g(&format!("{p}.mamba.conv1d.weight"));
        let mut cw = vec![0.0f32; d_conv * conv_dim];
        for ch in 0..conv_dim { for k in 0..d_conv { cw[k * conv_dim + ch] = cw_hf[ch * d_conv + k]; } }
        let cb = g(&format!("{p}.mamba.conv1d.bias"));
        let state0 = vec![0.0f32; (d_conv - 1) * conv_dim];
        let yc = conv1d_causal_step(d, &up(xbc, vec![conv_dim]), &up(&cw, vec![d_conv, conv_dim]), &up(&cb, vec![conv_dim]), &up(&state0, vec![(d_conv-1)*conv_dim]), conv_dim as u32, d_conv as u32).unwrap();
        let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
        let x_ssm = &xbc_act[0..d_ssm];
        let bmat = &xbc_act[d_ssm..d_ssm + n_groups * d_state];
        let cmat = &xbc_act[d_ssm + n_groups * d_state..conv_dim];
        let dt_bias = g(&format!("{p}.mamba.dt_bias"));
        let dt: Vec<f32> = (0..m_nh).map(|i| softplus(dt_raw[i] + dt_bias[i])).collect();
        let a_log = g(&format!("{p}.mamba.A_log"));
        let dsk = g(&format!("{p}.mamba.D"));
        let state_in = vec![0.0f32; m_nh * m_dh * d_state];
        let (_so, y_t) = ssm_step(d, &up(x_ssm, vec![d_ssm]), &up(&a_log, vec![m_nh]), &up(bmat, vec![n_groups*d_state]), &up(cmat, vec![n_groups*d_state]), &up(&dsk, vec![m_nh]), &up(&dt, vec![m_nh]), &up(&state_in, vec![m_nh*m_dh*d_state]), m_dh as u32, d_state as u32, m_nh as u32, (m_nh / n_groups) as u32).unwrap();
        let y = dl(&y_t, d_ssm);
        // gate: scan = y * silu(z)
        let sz = dl(&silu(d, &up(z, vec![d_ssm])).unwrap(), d_ssm);
        let scan: Vec<f32> = (0..d_ssm).map(|i| y[i] * sz[i]).collect();
        let mamba_out = dl(&gemv(d, &up(&g(&format!("{p}.mamba.out_proj.weight")), vec![hid, d_ssm]), &up(&scan, vec![d_ssm])).unwrap(), hid);

        // ── GQA attention (parallel, attn_in_mult = 1.0) ──
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![nq * ahd, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![nkv * ahd, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![nkv * ahd, hid]), &h).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, ahd]), &k.reshaped(vec![nkv, ahd]), &v.reshaped(vec![nkv, ahd]), ahd, 1, 1, (nq / nkv) as u32, ascale).unwrap();
        let attn_out_v = dl(&gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, nq * ahd]), &attn.reshaped(vec![nq * ahd])).unwrap(), hid);

        for i in 0..hid { x[i] += mamba_out[i] * ssm_out + attn_out_v[i] * attn_out; }

        // ── SwiGLU FFN (gate_mult folded into gate weight, down_mult on output) ──
        let h2 = rms_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.pre_ff_layernorm.weight")), vec![hid]), eps).unwrap();
        let gate_w: Vec<f32> = g(&format!("{p}.feed_forward.gate_proj.weight")).iter().map(|w| w * gate_mult).collect();
        let gate = silu(d, &gemv(d, &up(&gate_w, vec![inter, hid]), &h2).unwrap()).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.feed_forward.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = dl(&gate, inter).iter().zip(dl(&upp, inter)).map(|(gg, uu)| gg * uu).collect::<Vec<f32>>();
        let ff = dl(&gemv(d, &up(&g(&format!("{p}.feed_forward.down_proj.weight")), vec![hid, inter]), &up(&act, vec![inter])).unwrap(), hid);
        for i in 0..hid { x[i] += ff[i] * down_mult; }
    }

    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g("model.final_layernorm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab); // lm_head_mult > 0: argmax-invariant
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top3 = &idx[..3];
    eprintln!("Falcon-H1-0.5B full forward on Metal: top3 = {top3:?} (HF = [593, 531, 587])");
    assert_eq!(top3, &[593usize, 531, 587], "Falcon-H1 top-3 != HF");
    eprintln!("✅ Full real Falcon-H1-0.5B forward matches HF on the shared engine (Apple GPU) — hybrid Mamba2+attention path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--tiiuae--Falcon-H1-0.5B-Base/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
