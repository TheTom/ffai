// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Shared, backend-agnostic model forwards + HF-reference verification.
//!
//! Each `verify_*(dev: &dyn Device)` holds a model's forward and its HF oracle
//! ONCE. The per-backend test files (`ffai-metal/tests/*`, `ffai-cuda/tests/*`)
//! are thin wrappers that build their device and call these — so a model's
//! logic lives in exactly one place, not a Metal test + a sed'd CUDA twin.
use ffai_core::{DType, Device, Tensor};
use ffai_loader::SafeTensors;
use ffai_ops::{add, gelu, gemv, layer_norm, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

/// Resolve a model dir from `$ENV` or an HF-cache snapshot glob.
fn model_dir(env: &str, hub: &str) -> Option<String> {
    if let Ok(d) = std::env::var(env) {
        return Some(d);
    }
    let base = format!("{}/.cache/huggingface/hub/{hub}/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}

/// GPT-2-124M single-token forward (LayerNorm-LLM, Conv1D weights, learned-pos,
/// gelu_new, tied) vs HF argmax 198. Runs on whatever `Device` is passed.
pub fn verify_gpt2(d: &dyn Device, plat: &str) {
    let dir = model_dir("GPT2_DIR", "models--gpt2").unwrap_or_default();
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };

    let (hid, nh, hd, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 12usize, 50257usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let conv_t = |w: &[f32], nin: usize, nout: usize| -> Vec<f32> { let mut o = vec![0.0f32; nin * nout]; for i in 0..nin { for j in 0..nout { o[j * nin + i] = w[i * nout + j]; } } o };

    let token = 5usize;
    let wte = g("wte.weight");
    let wpe = g("wpe.weight");
    let mut x: Vec<f32> = (0..hid).map(|i| wte[token * hid + i] + wpe[i]).collect();

    for l in 0..n_layers {
        let p = format!("h.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.ln_1.weight")), vec![hid]),
            &up(&g(&format!("{p}.ln_1.bias")), vec![hid]), eps).unwrap();
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

    let xf = layer_norm(d, &up(&x, vec![hid]),
        &up(&g("ln_f.weight"), vec![hid]), &up(&g("ln_f.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&wte, vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    eprintln!("GPT-2-124M full forward on {plat}: argmax = {argmax} (HF = 198)");
    assert_eq!(argmax, 198, "GPT-2 argmax != HF 198");
    eprintln!("✅ Full real GPT-2 forward matches HF on the shared engine ({plat}) — one shared forward, both backends.");
}

/// Mamba2-130m single-token forward (SSM: conv1d + SSD scan + gated RMSNorm)
/// vs HF argmax 310. Same shared op layer, any `Device`.
pub fn verify_mamba2(d: &dyn Device, plat: &str) {
    use ffai_ops::{conv1d_causal_step, rms_norm, silu, ssm_step};
    let dir = model_dir("MAMBA2_DIR", "models--AntonV--mamba2-130m-hf").unwrap_or_default();
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };

    let (hid, di, nh, dh, ds, ng, kc, vocab, eps) = (768usize, 1536usize, 24usize, 64usize, 128usize, 1usize, 4usize, 50288usize, 1e-5f32);
    let conv_dim = di + 2 * ng * ds;
    let n_layers = 24usize;
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32]| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };

    let token = 5usize;
    let embed = g("backbone.embeddings.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();

    for l in 0..n_layers {
        let p = format!("backbone.layers.{l}");
        let xn = rms_norm(d, &up(&x), &up(&g(&format!("{p}.norm.weight"))), eps).unwrap();
        let in_proj = upm(&g(&format!("{p}.mixer.in_proj.weight")), vec![3352, hid]);
        let proj = dl(&gemv(d, &in_proj, &xn).unwrap(), 3352);
        let z = &proj[0..di];
        let xbc = &proj[di..di + conv_dim];
        let dt_raw = &proj[di + conv_dim..di + conv_dim + nh];
        let cw_hf = g(&format!("{p}.mixer.conv1d.weight"));
        let mut cw = vec![0.0f32; kc * conv_dim];
        for ch in 0..conv_dim { for k in 0..kc { cw[k * conv_dim + ch] = cw_hf[ch * kc + k]; } }
        let cb = g(&format!("{p}.mixer.conv1d.bias"));
        let state0 = vec![0.0f32; (kc - 1) * conv_dim];
        let yc = conv1d_causal_step(d, &up(xbc), &up(&cw), &up(&cb), &up(&state0), conv_dim as u32, kc as u32).unwrap();
        let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
        let x_ssm = &xbc_act[0..di];
        let bmat = &xbc_act[di..di + ng * ds];
        let cmat = &xbc_act[di + ng * ds..di + 2 * ng * ds];
        let dt_bias = g(&format!("{p}.mixer.dt_bias"));
        let dt: Vec<f32> = (0..nh).map(|i| softplus(dt_raw[i] + dt_bias[i])).collect();
        let a_log = g(&format!("{p}.mixer.A_log"));
        let dsk = g(&format!("{p}.mixer.D"));
        let state_in = vec![0.0f32; nh * dh * ds];
        let (_so, y_t) = ssm_step(d, &up(x_ssm), &up(&a_log), &up(bmat), &up(cmat), &up(&dsk), &up(&dt), &up(&state_in), dh as u32, ds as u32, nh as u32, (nh / ng) as u32).unwrap();
        let y = dl(&y_t, di);
        let sz = dl(&silu(d, &up(z)).unwrap(), di);
        let y_gated: Vec<f32> = (0..di).map(|i| y[i] * sz[i]).collect();
        let y_normed = rms_norm(d, &up(&y_gated), &up(&g(&format!("{p}.mixer.norm.weight"))), eps).unwrap();
        let out_proj = upm(&g(&format!("{p}.mixer.out_proj.weight")), vec![hid, di]);
        let out = dl(&gemv(d, &out_proj, &y_normed).unwrap(), hid);
        for i in 0..hid { x[i] += out[i]; }
    }

    let xf = rms_norm(d, &up(&x), &up(&g("backbone.norm_f.weight")), eps).unwrap();
    let lm = upm(&embed, vec![vocab, hid]);
    let logits = dl(&gemv(d, &lm, &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    eprintln!("Mamba2-130m full forward on {plat}: argmax = {argmax} (HF = 310)");
    assert_eq!(argmax, 310, "Mamba2 argmax != HF 310");
    eprintln!("✅ Full real Mamba2-130m forward matches HF on the shared engine ({plat}).");
}

/// Run the whole shared model suite against any backend. A new backend (ROCm,
/// Vulkan, …) implements `Device`, then calls this from one test file — and
/// inherits every model with zero model code.
pub fn run_all(d: &dyn Device, plat: &str) {
    verify_gpt2(d, plat);
    verify_pythia(d, plat);
    verify_gptneo(d, plat);
    verify_olmo2(d, plat);
    verify_gemma2(d, plat);
    verify_phi(d, plat);
    verify_stablelm2(d, plat);
    verify_olmoe(d, plat);
    verify_mamba2(d, plat);
    verify_falcon_h1(d, plat);
}

// exact-erf GELU (Abramowitz-Stegun) — shared by GPT-NeoX / Whisper-style nets
fn erf(x: f32) -> f32 {
    let s = x.signum(); let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}
fn gelu_erf(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) }

/// Pythia-160m (GPT-NeoX): parallel residual, interleaved per-head QKV, partial
/// rotary (identity@pos0), exact-erf GELU. vs HF argmax 285.
pub fn verify_pythia(d: &dyn Device, plat: &str) {
    let dir = model_dir("PYTHIA_DIR", "models--EleutherAI--pythia-160m").unwrap_or_default();
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
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
        let ln1 = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        let qkv = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.attention.query_key_value.weight")), vec![3*hid, hid]), &ln1).unwrap(), &up(&g(&format!("{p}.attention.query_key_value.bias")), vec![3*hid])).unwrap(), 3*hid);
        let (mut q, mut k, mut v) = (vec![0.0f32; hid], vec![0.0f32; hid], vec![0.0f32; hid]);
        for h in 0..nh { for dd in 0..hd { q[h*hd+dd]=qkv[h*3*hd+dd]; k[h*hd+dd]=qkv[h*3*hd+hd+dd]; v[h*hd+dd]=qkv[h*3*hd+2*hd+dd]; }}
        let attn = sdpa_decode(d, &up(&q, vec![nh, hd]), &up(&k, vec![nh, hd]), &up(&v, vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let ao = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.attention.dense.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(), &up(&g(&format!("{p}.attention.dense.bias")), vec![hid])).unwrap(), hid);
        let ln2 = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.post_attention_layernorm.bias")), vec![hid]), eps).unwrap();
        let mut h1 = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.mlp.dense_h_to_4h.weight")), vec![inter, hid]), &ln2).unwrap(), &up(&g(&format!("{p}.mlp.dense_h_to_4h.bias")), vec![inter])).unwrap(), inter);
        for vv in h1.iter_mut() { *vv = gelu_erf(*vv); }
        let m = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.mlp.dense_4h_to_h.weight")), vec![hid, inter]), &up(&h1, vec![inter])).unwrap(), &up(&g(&format!("{p}.mlp.dense_4h_to_h.bias")), vec![hid])).unwrap(), hid);
        for i in 0..hid { x[i] += ao[i] + m[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![hid]), &up(&g("gpt_neox.final_layer_norm.weight"), vec![hid]), &up(&g("gpt_neox.final_layer_norm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("embed_out.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    eprintln!("Pythia-160m on {plat}: argmax = {argmax} (HF = 285)");
    assert_eq!(argmax, 285, "Pythia argmax != HF 285");
    eprintln!("✅ Pythia-160m matches HF ({plat}).");
}

/// OLMo-2-1B: post-norm placement + qk-norm over full proj, SwiGLU. vs HF top-3 [198,8,13].
pub fn verify_olmo2(d: &dyn Device, plat: &str) {
    use ffai_ops::{rms_norm, swiglu};
    let dir = model_dir("OLMO2_DIR", "models--allenai--OLMo-2-0425-1B").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nq, nkv, hd, inter, n_layers, vocab, eps) = (2048usize, 16usize, 16usize, 128usize, 8192usize, 16usize, 100352usize, 1e-6f32);
    let (qdim, kvdim) = (nq*hd, nkv*hd); let scale = 1.0/(hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].to_vec();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let xin = up(&x, vec![hid]);
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![qdim, hid]), &xin).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![kvdim, hid]), &xin).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![kvdim, hid]), &xin).unwrap();
        let q = rms_norm(d, &q, &up(&g(&format!("{p}.self_attn.q_norm.weight")), vec![qdim]), eps).unwrap();
        let k = rms_norm(d, &k, &up(&g(&format!("{p}.self_attn.k_norm.weight")), vec![kvdim]), eps).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, hd]), &k.reshaped(vec![nkv, hd]), &v.reshaped(vec![nkv, hd]), hd, 1, 1, (nq/nkv) as u32, scale).unwrap();
        let o = gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, qdim]), &attn.reshaped(vec![qdim])).unwrap();
        let o = dl(&rms_norm(d, &o, &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), eps).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let xin2 = up(&x, vec![hid]);
        let gate = gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &xin2).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &xin2).unwrap();
        let act = swiglu(d, &gate, &upp).unwrap();
        let down = gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap();
        let down = dl(&rms_norm(d, &down, &up(&g(&format!("{p}.post_feedforward_layernorm.weight")), vec![hid]), eps).unwrap(), hid);
        for i in 0..hid { x[i] += down[i]; }
    }
    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g("model.norm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("OLMo-2-1B on {plat}: top3 = {:?} (HF = [198,8,13])", &idx[..3]);
    assert_eq!(&idx[..3], &[198usize, 8, 13], "OLMo-2 top-3 != HF");
    eprintln!("✅ OLMo-2-1B matches HF ({plat}).");
}

/// GPT-Neo-125M: learned-pos + LayerNorm, separate q/k/v (no bias), no attn
/// scaling, gelu_new, tied. vs HF top-3 [28,59,91].
pub fn verify_gptneo(d: &dyn Device, plat: &str) {
    let dir = model_dir("GPTNEO_DIR", "models--EleutherAI--gpt-neo-125m").unwrap_or_default();
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let (hid, nh, hd, inter, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 3072usize, 12usize, 50257usize, 1e-5f32);
    let scale = 1.0f32;
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 5usize;
    let wte = g("transformer.wte.weight"); let wpe = g("transformer.wpe.weight");
    let mut x: Vec<f32> = (0..hid).map(|i| wte[token*hid+i] + wpe[i]).collect();
    for l in 0..n_layers {
        let p = format!("transformer.h.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.ln_1.weight")), vec![hid]), &up(&g(&format!("{p}.ln_1.bias")), vec![hid]), eps).unwrap();
        let q = gemv(d, &up(&g(&format!("{p}.attn.attention.q_proj.weight")), vec![hid, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.attn.attention.k_proj.weight")), vec![hid, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.attn.attention.v_proj.weight")), vec![hid, hid]), &h).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.attn.attention.out_proj.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(), &up(&g(&format!("{p}.attn.attention.out_proj.bias")), vec![hid])).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let h2 = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.ln_2.weight")), vec![hid]), &up(&g(&format!("{p}.ln_2.bias")), vec![hid]), eps).unwrap();
        let f = add(d, &gemv(d, &up(&g(&format!("{p}.mlp.c_fc.weight")), vec![inter, hid]), &h2).unwrap(), &up(&g(&format!("{p}.mlp.c_fc.bias")), vec![inter])).unwrap();
        let act = gelu(d, &f).unwrap();
        let m = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.mlp.c_proj.weight")), vec![hid, inter]), &act).unwrap(), &up(&g(&format!("{p}.mlp.c_proj.bias")), vec![hid])).unwrap(), hid);
        for i in 0..hid { x[i] += m[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![hid]), &up(&g("transformer.ln_f.weight"), vec![hid]), &up(&g("transformer.ln_f.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&wte, vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("GPT-Neo-125M on {plat}: top3 = {:?} (HF = [28,59,91])", &idx[..3]);
    assert_eq!(&idx[..3], &[28usize, 59, 91], "GPT-Neo top-3 != HF");
    eprintln!("✅ GPT-Neo-125M matches HF ({plat}).");
}

/// Gemma-2-2b: √hidden embed-scale, RMSNorm(1+w), 4 norms/layer, geGLU, GQA
/// hd256, softcaps (argmax-invariant at 1 pos). vs HF top-3 [9707,235265,110].
pub fn verify_gemma2(d: &dyn Device, plat: &str) {
    use ffai_ops::{mul, rms_norm};
    let dir = model_dir("GEMMA2_DIR", "models--unsloth--gemma-2-2b-it").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nq, nkv, hd, inter, n_layers, vocab, eps) = (2304usize, 8usize, 4usize, 256usize, 9216usize, 26usize, 256000usize, 1e-6f32);
    let (qdim, kvdim) = (nq*hd, nkv*hd); let scale = 1.0/(256.0f32).sqrt(); let embed_scale = (hid as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let g1 = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0.iter().map(|w| w+1.0).collect() };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].iter().map(|v| v*embed_scale).collect();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = rms_norm(d, &up(&x, vec![hid]), &up(&g1(&format!("{p}.input_layernorm.weight")), vec![hid]), eps).unwrap();
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![qdim, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![kvdim, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![kvdim, hid]), &h).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, hd]), &k.reshaped(vec![nkv, hd]), &v.reshaped(vec![nkv, hd]), hd, 1, 1, (nq/nkv) as u32, scale).unwrap();
        let o = gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, qdim]), &attn.reshaped(vec![qdim])).unwrap();
        let o = dl(&rms_norm(d, &o, &up(&g1(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), eps).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let h2 = rms_norm(d, &up(&x, vec![hid]), &up(&g1(&format!("{p}.pre_feedforward_layernorm.weight")), vec![hid]), eps).unwrap();
        let gate = gelu(d, &gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &h2).unwrap()).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = mul(d, &gate, &upp).unwrap();
        let down = gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap();
        let down = dl(&rms_norm(d, &down, &up(&g1(&format!("{p}.post_feedforward_layernorm.weight")), vec![hid]), eps).unwrap(), hid);
        for i in 0..hid { x[i] += down[i]; }
    }
    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g1("model.norm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&embed, vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("Gemma-2-2b on {plat}: top3 = {:?} (HF = [9707,235265,110])", &idx[..3]);
    assert_eq!(&idx[..3], &[9707usize, 235265, 110], "Gemma top-3 != HF");
    eprintln!("✅ Gemma-2-2b matches HF ({plat}).");
}

/// Phi-1.5: single shared norm → parallel attn+MLP, separate q/k/v+bias,
/// partial rotary, gelu_new. vs HF top-3 [11,13,546].
pub fn verify_phi(d: &dyn Device, plat: &str) {
    let dir = model_dir("PHI_DIR", "models--microsoft--phi-1_5").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nh, hd, inter, n_layers, vocab, eps) = (2048usize, 32usize, 64usize, 8192usize, 24usize, 51200usize, 1e-5f32);
    let scale = 1.0/(hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let proj = |w: &str, b: &str, x: &Tensor, m: usize, inn: usize| -> Tensor {
        add(d, &gemv(d, &up(&g(w), vec![m, inn]), x).unwrap(), &up(&g(b), vec![m])).unwrap() };
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].to_vec();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        let q = proj(&format!("{p}.self_attn.q_proj.weight"), &format!("{p}.self_attn.q_proj.bias"), &h, hid, hid);
        let k = proj(&format!("{p}.self_attn.k_proj.weight"), &format!("{p}.self_attn.k_proj.bias"), &h, hid, hid);
        let v = proj(&format!("{p}.self_attn.v_proj.weight"), &format!("{p}.self_attn.v_proj.bias"), &h, hid, hid);
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&proj(&format!("{p}.self_attn.dense.weight"), &format!("{p}.self_attn.dense.bias"), &attn.reshaped(vec![hid]), hid, hid), hid);
        let f = proj(&format!("{p}.mlp.fc1.weight"), &format!("{p}.mlp.fc1.bias"), &h, inter, hid);
        let act = gelu(d, &f).unwrap();
        let m = dl(&proj(&format!("{p}.mlp.fc2.weight"), &format!("{p}.mlp.fc2.bias"), &act, hid, inter), hid);
        for i in 0..hid { x[i] += o[i] + m[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![hid]), &up(&g("model.final_layernorm.weight"), vec![hid]), &up(&g("model.final_layernorm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&add(d, &gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), &up(&g("lm_head.bias"), vec![vocab])).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("Phi-1.5 on {plat}: top3 = {:?} (HF = [11,13,546])", &idx[..3]);
    assert_eq!(&idx[..3], &[11usize, 13, 546], "Phi top-3 != HF");
    eprintln!("✅ Phi-1.5 matches HF ({plat}).");
}

/// StableLM-2-1.6B: LayerNorm(+bias), q/k/v bias, partial rotary, SwiGLU.
/// vs HF top-3 [341,11,280].
pub fn verify_stablelm2(d: &dyn Device, plat: &str) {
    use ffai_ops::swiglu;
    let dir = model_dir("STABLELM2_DIR", "models--stabilityai--stablelm-2-1_6b").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nh, hd, inter, n_layers, vocab, eps) = (2048usize, 32usize, 64usize, 5632usize, 24usize, 100352usize, 1e-5f32);
    let scale = 1.0/(hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let bproj = |w: &str, b: &str, x: &Tensor, m: usize| -> Tensor { add(d, &gemv(d, &up(&g(w), vec![m, hid]), x).unwrap(), &up(&g(b), vec![m])).unwrap() };
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].to_vec();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        let q = bproj(&format!("{p}.self_attn.q_proj.weight"), &format!("{p}.self_attn.q_proj.bias"), &h, hid);
        let k = bproj(&format!("{p}.self_attn.k_proj.weight"), &format!("{p}.self_attn.k_proj.bias"), &h, hid);
        let v = bproj(&format!("{p}.self_attn.v_proj.weight"), &format!("{p}.self_attn.v_proj.bias"), &h, hid);
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let h2 = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.post_attention_layernorm.bias")), vec![hid]), eps).unwrap();
        let gate = gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = swiglu(d, &gate, &upp).unwrap();
        let m = dl(&gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap(), hid);
        for i in 0..hid { x[i] += m[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![hid]), &up(&g("model.norm.weight"), vec![hid]), &up(&g("model.norm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("StableLM-2-1.6B on {plat}: top3 = {:?} (HF = [341,11,280])", &idx[..3]);
    assert_eq!(&idx[..3], &[341usize, 11, 280], "StableLM-2 top-3 != HF");
    eprintln!("✅ StableLM-2-1.6B matches HF ({plat}).");
}

/// OLMoE-1B-7B: 64-expert MoE, top-8 no-renorm, no shared expert, qk-norm over
/// full proj, MHA hd128. vs HF argmax 310.
pub fn verify_olmoe(d: &dyn Device, plat: &str) {
    use ffai_ops::{rms_norm, swiglu};
    let dir = model_dir("OLMOE_DIR", "models--allenai--OLMoE-1B-7B-0924").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nh, hd, inter, n_exp, top_k, n_layers, vocab, eps) = (2048usize, 16usize, 128usize, 1024usize, 64usize, 8usize, 16usize, 50304usize, 1e-5f32);
    let scale = 1.0/(hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32]| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 5usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].to_vec();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let xn = rms_norm(d, &up(&x), &up(&g(&format!("{p}.input_layernorm.weight"))), eps).unwrap();
        let q = gemv(d, &upm(&g(&format!("{p}.self_attn.q_proj.weight")), vec![hid, hid]), &xn).unwrap();
        let k = gemv(d, &upm(&g(&format!("{p}.self_attn.k_proj.weight")), vec![hid, hid]), &xn).unwrap();
        let v = gemv(d, &upm(&g(&format!("{p}.self_attn.v_proj.weight")), vec![hid, hid]), &xn).unwrap();
        let q = rms_norm(d, &q, &up(&g(&format!("{p}.self_attn.q_norm.weight"))), eps).unwrap();
        let k = rms_norm(d, &k, &up(&g(&format!("{p}.self_attn.k_norm.weight"))), eps).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&gemv(d, &upm(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, hid]), &attn.reshaped(vec![nh*hd])).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let xn2 = rms_norm(d, &up(&x), &up(&g(&format!("{p}.post_attention_layernorm.weight"))), eps).unwrap();
        let rl = dl(&gemv(d, &upm(&g(&format!("{p}.mlp.gate.weight")), vec![n_exp, hid]), &xn2).unwrap(), n_exp);
        let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = rl.iter().map(|&z| (z-mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e/sum).collect();
        let eidx = ffai_runtime::topk(&probs, top_k);
        let mut acc = vec![0.0f32; hid];
        for &e in &eidx {
            let w = probs[e]; let ep = format!("{p}.mlp.experts.{e}");
            let ge = gemv(d, &upm(&g(&format!("{ep}.gate_proj.weight")), vec![inter, hid]), &xn2).unwrap();
            let ue = gemv(d, &upm(&g(&format!("{ep}.up_proj.weight")), vec![inter, hid]), &xn2).unwrap();
            let act = swiglu(d, &ge, &ue).unwrap();
            let de = dl(&gemv(d, &upm(&g(&format!("{ep}.down_proj.weight")), vec![hid, inter]), &act).unwrap(), hid);
            for i in 0..hid { acc[i] += w*de[i]; }
        }
        for i in 0..hid { x[i] += acc[i]; }
    }
    let xf = rms_norm(d, &up(&x), &up(&g("model.norm.weight")), eps).unwrap();
    let logits = dl(&gemv(d, &upm(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    eprintln!("OLMoE-1B-7B on {plat}: argmax = {argmax} (HF = 310)");
    assert_eq!(argmax, 310, "OLMoE argmax != HF 310");
    eprintln!("✅ OLMoE-1B-7B (64-expert MoE) matches HF ({plat}).");
}

/// Falcon-H1-0.5B: hybrid Mamba2 mixer ∥ GQA attention per layer + µP. vs HF
/// top-3 [593,531,587]. (token 5 is a reserved zero-embedding slot — use 9707.)
pub fn verify_falcon_h1(d: &dyn Device, plat: &str) {
    use ffai_ops::{conv1d_causal_step, rms_norm, silu, ssm_step};
    let dir = model_dir("FALCON_H1_DIR", "models--tiiuae--Falcon-H1-0.5B-Base").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nq, nkv, ahd, inter, n_layers, vocab, eps) = (1024usize, 8usize, 2usize, 64usize, 2048usize, 36usize, 32784usize, 1e-5f32);
    let (d_ssm, m_nh, m_dh, d_state, n_groups, d_conv) = (1536usize, 24usize, 64usize, 128usize, 1usize, 4usize);
    let conv_dim = d_ssm + 2*n_groups*d_state; let proj_dim = 2*d_ssm + 2*n_groups*d_state + m_nh;
    let ascale = 1.0/(ahd as f32).sqrt();
    let (ssm_in, ssm_out, attn_out) = (1.25f32, 0.23570226039551587f32, 0.9375f32);
    let (gate_mult, down_mult, embed_mult) = (0.8838834764831844f32, 0.5859375f32, 5.656854249492381f32);
    let ssm_m = [0.3535533905932738f32, 0.25, 0.3535533905932738, 0.5, 0.3535533905932738];
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    let mut mup = vec![0.0f32; proj_dim];
    for i in 0..proj_dim {
        let m = if i < d_ssm { ssm_m[0] } else if i < 2*d_ssm { ssm_m[1] }
            else if i < 2*d_ssm + n_groups*d_state { ssm_m[2] }
            else if i < 2*d_ssm + 2*n_groups*d_state { ssm_m[3] } else { ssm_m[4] };
        mup[i] = m * ssm_in;
    }
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].iter().map(|v| v*embed_mult).collect();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = rms_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), eps).unwrap();
        let mut proj = dl(&gemv(d, &up(&g(&format!("{p}.mamba.in_proj.weight")), vec![proj_dim, hid]), &h).unwrap(), proj_dim);
        for i in 0..proj_dim { proj[i] *= mup[i]; }
        let z = &proj[0..d_ssm]; let xbc = &proj[d_ssm..d_ssm+conv_dim]; let dt_raw = &proj[d_ssm+conv_dim..proj_dim];
        let cw_hf = g(&format!("{p}.mamba.conv1d.weight"));
        let mut cw = vec![0.0f32; d_conv*conv_dim];
        for ch in 0..conv_dim { for k in 0..d_conv { cw[k*conv_dim+ch] = cw_hf[ch*d_conv+k]; } }
        let cb = g(&format!("{p}.mamba.conv1d.bias"));
        let state0 = vec![0.0f32; (d_conv-1)*conv_dim];
        let yc = conv1d_causal_step(d, &up(xbc, vec![conv_dim]), &up(&cw, vec![d_conv, conv_dim]), &up(&cb, vec![conv_dim]), &up(&state0, vec![(d_conv-1)*conv_dim]), conv_dim as u32, d_conv as u32).unwrap();
        let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
        let x_ssm = &xbc_act[0..d_ssm]; let bmat = &xbc_act[d_ssm..d_ssm+n_groups*d_state]; let cmat = &xbc_act[d_ssm+n_groups*d_state..conv_dim];
        let dt_bias = g(&format!("{p}.mamba.dt_bias"));
        let dt: Vec<f32> = (0..m_nh).map(|i| softplus(dt_raw[i]+dt_bias[i])).collect();
        let a_log = g(&format!("{p}.mamba.A_log")); let dsk = g(&format!("{p}.mamba.D"));
        let state_in = vec![0.0f32; m_nh*m_dh*d_state];
        let (_so, y_t) = ssm_step(d, &up(x_ssm, vec![d_ssm]), &up(&a_log, vec![m_nh]), &up(bmat, vec![n_groups*d_state]), &up(cmat, vec![n_groups*d_state]), &up(&dsk, vec![m_nh]), &up(&dt, vec![m_nh]), &up(&state_in, vec![m_nh*m_dh*d_state]), m_dh as u32, d_state as u32, m_nh as u32, (m_nh/n_groups) as u32).unwrap();
        let y = dl(&y_t, d_ssm);
        let sz = dl(&silu(d, &up(z, vec![d_ssm])).unwrap(), d_ssm);
        let scan: Vec<f32> = (0..d_ssm).map(|i| y[i]*sz[i]).collect();
        let mamba_out = dl(&gemv(d, &up(&g(&format!("{p}.mamba.out_proj.weight")), vec![hid, d_ssm]), &up(&scan, vec![d_ssm])).unwrap(), hid);
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![nq*ahd, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![nkv*ahd, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![nkv*ahd, hid]), &h).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, ahd]), &k.reshaped(vec![nkv, ahd]), &v.reshaped(vec![nkv, ahd]), ahd, 1, 1, (nq/nkv) as u32, ascale).unwrap();
        let attn_out_v = dl(&gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, nq*ahd]), &attn.reshaped(vec![nq*ahd])).unwrap(), hid);
        for i in 0..hid { x[i] += mamba_out[i]*ssm_out + attn_out_v[i]*attn_out; }
        let h2 = rms_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.pre_ff_layernorm.weight")), vec![hid]), eps).unwrap();
        let gate_w: Vec<f32> = g(&format!("{p}.feed_forward.gate_proj.weight")).iter().map(|w| w*gate_mult).collect();
        let gate = silu(d, &gemv(d, &up(&gate_w, vec![inter, hid]), &h2).unwrap()).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.feed_forward.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = dl(&gate, inter).iter().zip(dl(&upp, inter)).map(|(gg, uu)| gg*uu).collect::<Vec<f32>>();
        let ff = dl(&gemv(d, &up(&g(&format!("{p}.feed_forward.down_proj.weight")), vec![hid, inter]), &up(&act, vec![inter])).unwrap(), hid);
        for i in 0..hid { x[i] += ff[i]*down_mult; }
    }
    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g("model.final_layernorm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("Falcon-H1-0.5B on {plat}: top3 = {:?} (HF = [593,531,587])", &idx[..3]);
    assert_eq!(&idx[..3], &[593usize, 531, 587], "Falcon-H1 top-3 != HF");
    eprintln!("✅ Falcon-H1-0.5B (hybrid Mamba2+attn) matches HF ({plat}).");
}

/// NemotronH-Nano-Omni-30B-A3B (text backbone): 52-layer M/E/* hybrid.
/// pattern MEMEM*EMEMEM*… (M=Mamba2 mixer, E=128-expert MoE relu², *=GQA attn).
/// Single mixer per layer, pre-norm residual. BF16 weights → F32 on-device.
/// Verified against an HF-transformers CPU oracle (set NEMOTRON_ARGMAX).
pub fn verify_nemotron(d: &dyn Device, plat: &str) {
    use ffai_ops::{conv1d_causal_step, rms_norm, rope_llama, silu, ssm_step};
    const PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
    let dir = std::env::var("NEMOTRON_DIR")
        .unwrap_or_else(|_| "/home/pidtom/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16".into());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, vocab, eps) = (2688usize, 131072usize, 1e-5f32);
    // Mamba2: d_inner 4096 (64h×64), n_groups 8, d_state 128, conv_kernel 4.
    let (di, m_nh, m_dh, ds, ng, kc) = (4096usize, 64usize, 64usize, 128usize, 8usize, 4usize);
    let conv_dim = di + 2 * ng * ds;            // 6144
    let in_proj_out = 2 * di + 2 * ng * ds + m_nh; // 10304
    // MoE: 128 experts top-6, relu² ungated, shared 3712, sigmoid+bias router, ×2.5.
    let (n_exp, top_k, inter, shared_inter, scale_f) = (128usize, 6usize, 1856usize, 3712usize, 2.5f32);
    // Attn: GQA 32q/2kv hd128, rope θ1e4.
    let (nq, nkv, hd, rope_theta) = (32usize, 2usize, 128usize, 10000f32);
    let (qdim, kvdim) = (nq * hd, nkv * hd); // 4096, 256
    let ascale = 1.0 / (hd as f32).sqrt();

    // Coarse perf accounting: how much of a token is weight dequant (CPU BF16→F32)
    // vs host↔device transfer vs everything else (GPU op dispatch). This baseline
    // quantifies the resident-weights + quant headroom.
    use std::cell::Cell;
    use std::time::Instant;
    let t_deq = Cell::new(0f64); // BF16→F32 dequant + mmap read
    let t_xfer = Cell::new(0f64); // upload + download bytes
    let g = |name: &str| -> Vec<f32> { let t = Instant::now(); let r = st.tensor_f32(name).unwrap().0; t_deq.set(t_deq.get() + t.elapsed().as_secs_f64()); r };
    let up = |v: &[f32]| -> Tensor { let t = Instant::now(); let b = d.upload(&tb(v)).unwrap(); t_xfer.set(t_xfer.get() + t.elapsed().as_secs_f64()); Tensor::new(b, vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { let t = Instant::now(); let b = d.upload(&tb(v)).unwrap(); t_xfer.set(t_xfer.get() + t.elapsed().as_secs_f64()); Tensor::new(b, sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let s = Instant::now(); let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); let r = fb(&b); t_xfer.set(t_xfer.get() + s.elapsed().as_secs_f64()); r };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    let relu2 = |v: &mut [f32]| for x in v.iter_mut() { let r = x.max(0.0); *x = r * r; };
    let t_total = Instant::now();

    let token: usize = std::env::var("NEMOTRON_TOKEN").ok().and_then(|s| s.parse().ok()).unwrap_or(1234);
    let embed = g("language_model.backbone.embeddings.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();
    let enorm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!("Nemotron: token {token} embed‖·‖={enorm:.3}");
    let dump = std::env::var("NEMOTRON_DUMP").is_ok();
    let fp = |tag: &str, v: &[f32]| {
        let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        eprintln!("{tag} norm={n:.4} head={:?}", v[..4].iter().map(|a| (a * 10000.0).round() / 10000.0).collect::<Vec<_>>());
    };
    if dump { fp("L00", &x); }

    for (l, mix) in PATTERN.chars().enumerate() {
        let p = format!("language_model.backbone.layers.{l}");
        let xn = rms_norm(d, &up(&x), &up(&g(&format!("{p}.norm.weight"))), eps).unwrap();
        match mix {
            'M' => {
                let in_proj = upm(&g(&format!("{p}.mixer.in_proj.weight")), vec![in_proj_out, hid]);
                let proj = dl(&gemv(d, &in_proj, &xn).unwrap(), in_proj_out);
                let z = &proj[0..di];
                let xbc = &proj[di..di + conv_dim];
                let dt_raw = &proj[di + conv_dim..di + conv_dim + m_nh];
                let cw_hf = g(&format!("{p}.mixer.conv1d.weight"));
                let mut cw = vec![0.0f32; kc * conv_dim];
                for ch in 0..conv_dim { for k in 0..kc { cw[k * conv_dim + ch] = cw_hf[ch * kc + k]; } }
                let cb = g(&format!("{p}.mixer.conv1d.bias"));
                let state0 = vec![0.0f32; (kc - 1) * conv_dim];
                let yc = conv1d_causal_step(d, &up(xbc), &up(&cw), &up(&cb), &up(&state0), conv_dim as u32, kc as u32).unwrap();
                let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
                let x_ssm = &xbc_act[0..di];
                let bmat = &xbc_act[di..di + ng * ds];
                let cmat = &xbc_act[di + ng * ds..di + 2 * ng * ds];
                let dt_bias = g(&format!("{p}.mixer.dt_bias"));
                let dt: Vec<f32> = (0..m_nh).map(|i| softplus(dt_raw[i] + dt_bias[i])).collect();
                let a_log = g(&format!("{p}.mixer.A_log"));
                let dsk = g(&format!("{p}.mixer.D"));
                let state_in = vec![0.0f32; m_nh * m_dh * ds];
                let (_so, y_t) = ssm_step(d, &up(x_ssm), &up(&a_log), &up(bmat), &up(cmat), &up(&dsk), &up(&dt), &up(&state_in), m_dh as u32, ds as u32, m_nh as u32, (m_nh / ng) as u32).unwrap();
                let y = dl(&y_t, di);
                let sz = dl(&silu(d, &up(z)).unwrap(), di);
                let y_gated: Vec<f32> = (0..di).map(|i| y[i] * sz[i]).collect();
                // Zamba2RMSNormGated: group-wise RMSNorm (group_size = d_inner/n_groups),
                // gate applied before, weight after. NOT a full-vector RMSNorm.
                let nw = g(&format!("{p}.mixer.norm.weight"));
                let gs = di / ng; // 512
                let mut yn = vec![0.0f32; di];
                for grp in 0..ng {
                    let s = grp * gs;
                    let seg = dl(&rms_norm(d, &up(&y_gated[s..s + gs]), &up(&nw[s..s + gs]), eps).unwrap(), gs);
                    yn[s..s + gs].copy_from_slice(&seg);
                }
                let out = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.out_proj.weight")), vec![hid, di]), &up(&yn)).unwrap(), hid);
                for i in 0..hid { x[i] += out[i]; }
            }
            'E' => {
                let rl = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.gate.weight")), vec![n_exp, hid]), &xn).unwrap(), n_exp);
                let sig: Vec<f32> = rl.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
                let bias = g(&format!("{p}.mixer.gate.e_score_correction_bias"));
                let choice: Vec<f32> = (0..n_exp).map(|i| sig[i] + bias[i]).collect();
                let eidx = ffai_runtime::topk(&choice, top_k);
                let mut w: Vec<f32> = eidx.iter().map(|&e| sig[e]).collect();
                let wsum: f32 = w.iter().sum::<f32>() + 1e-20;
                for v in w.iter_mut() { *v = *v / wsum * scale_f; }
                let mut acc = vec![0.0f32; hid];
                for (j, &e) in eidx.iter().enumerate() {
                    let ep = format!("{p}.mixer.experts.{e}");
                    let mut a = dl(&gemv(d, &upm(&g(&format!("{ep}.up_proj.weight")), vec![inter, hid]), &xn).unwrap(), inter);
                    relu2(&mut a);
                    let de = dl(&gemv(d, &upm(&g(&format!("{ep}.down_proj.weight")), vec![hid, inter]), &up(&a)).unwrap(), hid);
                    for i in 0..hid { acc[i] += w[j] * de[i]; }
                }
                // shared expert (relu², inter 3712)
                let mut sa = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.shared_experts.up_proj.weight")), vec![shared_inter, hid]), &xn).unwrap(), shared_inter);
                relu2(&mut sa);
                let sde = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.shared_experts.down_proj.weight")), vec![hid, shared_inter]), &up(&sa)).unwrap(), hid);
                for i in 0..hid { x[i] += acc[i] + sde[i]; }
            }
            '*' => {
                let q = gemv(d, &upm(&g(&format!("{p}.mixer.q_proj.weight")), vec![qdim, hid]), &xn).unwrap();
                let k = gemv(d, &upm(&g(&format!("{p}.mixer.k_proj.weight")), vec![kvdim, hid]), &xn).unwrap();
                let v = gemv(d, &upm(&g(&format!("{p}.mixer.v_proj.weight")), vec![kvdim, hid]), &xn).unwrap();
                let q = rope_llama(d, &q.reshaped(vec![nq, hd]), 0, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap();
                let k = rope_llama(d, &k.reshaped(vec![nkv, hd]), 0, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap();
                let attn = sdpa_decode(d, &q, &k, &v.reshaped(vec![nkv, hd]), hd, 1, 1, (nq / nkv) as u32, ascale).unwrap();
                let o = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.o_proj.weight")), vec![hid, qdim]), &attn.reshaped(vec![qdim])).unwrap(), hid);
                for i in 0..hid { x[i] += o[i]; }
            }
            _ => unreachable!("bad pattern char"),
        }
        if dump { fp(&format!("L{:02}[{mix}]", l + 1), &x); }
    }
    let xf = rms_norm(d, &up(&x), &up(&g("language_model.backbone.norm_f.weight")), eps).unwrap();
    let logits = dl(&gemv(d, &upm(&g("language_model.lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    let idx = ffai_runtime::topk(&logits, 5);
    let total = t_total.elapsed().as_secs_f64();
    let (deq, xfer) = (t_deq.get(), t_xfer.get());
    let compute = (total - deq - xfer).max(0.0);
    eprintln!("──────── NemotronH-Nano BASELINE (1 token, F32, naive re-load) ────────");
    eprintln!("  total      {total:7.1}s   ({:.3} tok/s)", 1.0 / total);
    eprintln!("  dequant    {deq:7.1}s   {:.0}%  (CPU BF16→F32; gone once weights are quantized+resident)", 100.0 * deq / total);
    eprintln!("  transfer   {xfer:7.1}s   {:.0}%  (host↔device; gone once weights resident)", 100.0 * xfer / total);
    eprintln!("  compute    {compute:7.1}s   {:.0}%  (GPU op dispatch — the resident-decode floor in F32)", 100.0 * compute / total);
    eprintln!("  ⇒ resident-weights ceiling ≈ {:.2} tok/s (F32 compute only); quant+fusion cuts further", 1.0 / compute.max(1e-3));
    eprintln!("───────────────────────────────────────────────────────────────────────");
    eprintln!("NemotronH-Nano text forward on {plat}: argmax={argmax} top5={:?}", &idx[..5]);
    if let Ok(exp) = std::env::var("NEMOTRON_ARGMAX") {
        assert_eq!(argmax, exp.parse::<usize>().unwrap(), "Nemotron argmax != HF");
    } else if token == 1234 {
        // HF-transformers CPU oracle (NemotronHForCausalLM, bf16, naive Mamba2 path).
        assert_eq!(&idx[..5], &[1234usize, 99493, 99391, 67501, 49418], "Nemotron top5 != HF oracle");
        eprintln!("✅ NemotronH-Nano-30B-A3B text backbone matches HF ({plat}).");
    }
}

/// NemotronH-Nano-30B-A3B **resident Q8 decode benchmark**: quantize every big
/// matrix to Q8_0 and upload it ONCE (resident), then run a real decode loop —
/// per-token forward reuses the resident weights via [`gemv_q8`], carries an
/// attention KV cache (6 layers) + Mamba2 conv/SSM state (23 layers), and times
/// steady-state tok/s. This is the path off the 0.003 tok/s naive-reload floor.
/// Env: NEMOTRON_DECODE (steps, default 32), NEMOTRON_PREFILL (warm the cache to
/// this context length before timing, default 0).
pub fn bench_nemotron(d: &dyn Device, plat: &str) {
    use ffai_ops::{add, cast_f16_f32, cast_f32_f16, conv1d_causal_step, conv_roll, gated_group_rmsnorm, gemv, gemv_q4, gemv_q4_accum, gemv_q4_relu2, kv_append, moe_gather_down, moe_gather_up_relu2, moe_router_device, moe_weighted_sum, mul, quantize_q4, rms_norm, rope_llama, silu, slice, sdpa_decode, sdpa_decode_2pass, sdpa_decode_2pass_bc4, sdpa_decode_2pass_tiled, softplus_add, ssm_step};
    use std::collections::HashMap;
    use std::time::Instant;
    const PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
    let dir = std::env::var("NEMOTRON_DIR")
        .unwrap_or_else(|_| "/home/pidtom/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16".into());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, vocab, eps) = (2688usize, 131072usize, 1e-5f32);
    let (di, m_nh, m_dh, ds, ng, kc) = (4096usize, 64usize, 64usize, 128usize, 8usize, 4usize);
    let conv_dim = di + 2 * ng * ds;
    let in_proj_out = 2 * di + 2 * ng * ds + m_nh;
    let (n_exp, top_k, inter, shared_inter, scale_f) = (128usize, 6usize, 1856usize, 3712usize, 2.5f32);
    let (nq, nkv, hd, rope_theta) = (32usize, 2usize, 128usize, 10000f32);
    let (qdim, kvdim) = (nq * hd, nkv * hd);
    let ascale = 1.0 / (hd as f32).sqrt();
    let gs = di / ng;

    let tbu = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    // f32 → f16 (round-to-nearest-even, positive normals; Q4 scales = amax/7 are in range).
    fn f32_to_f16(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112; // 127 - 15
        if e <= 0 { return sign; }
        if e >= 0x1f { return sign | 0x7c00; }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    let tb_f16 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| f32_to_f16(f).to_le_bytes()).collect() };
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32]| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };

    // ── Q4 weight cache ─────────────────────────────────────────────────────────
    // Gated by NEMOTRON_Q4CACHE env (default ON). On first run, writes Q4+scales to
    // ~/.cache/nemo_q4/<sanitized_name>.q4b. Subsequent runs skip BF16→F32+quantize
    // and read the cached bytes directly → load time ~15-25s vs ~120s.
    // Cache file format (little-endian):
    //   [8B: m as u64][8B: k as u64][1B: f16 flag (0=f32, 1=f16)]
    //   [N*4 bytes: qs as u32 LE][M*2 or M*4 bytes: scales as f16 or f32 LE]
    let use_q4cache = std::env::var("NEMOTRON_Q4CACHE").map(|v| v != "0" && v != "false").unwrap_or(true);
    let cache_dir = std::env::var("NEMOTRON_Q4CACHE_DIR")
        .unwrap_or_else(|_| format!("{}/.cache/nemo_q4", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
    if use_q4cache { let _ = std::fs::create_dir_all(&cache_dir); }

    let cache_path = |name: &str| -> std::path::PathBuf {
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect();
        std::path::PathBuf::from(&cache_dir).join(format!("{safe}.q4b"))
    };

    // Read from cache: returns (qs_bytes, sc_bytes, m, k, f16) or None if miss/disabled.
    let cache_read = |name: &str| -> Option<(Vec<u8>, Vec<u8>, usize, usize, bool)> {
        if !use_q4cache { return None; }
        let bytes = std::fs::read(cache_path(name)).ok()?;
        if bytes.len() < 17 { return None; }
        let m = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let k = u64::from_le_bytes(bytes[8..16].try_into().ok()?) as usize;
        let f16 = bytes[16] != 0;
        let bpr = k / 32;
        let qs_len_u32 = m * bpr * 4;
        let sc_len = m * bpr;
        let sc_bytes = if f16 { sc_len * 2 } else { sc_len * 4 };
        let expected = 17 + qs_len_u32 * 4 + sc_bytes;
        if bytes.len() != expected { return None; } // stale/corrupt
        let qs_bytes = bytes[17..17 + qs_len_u32 * 4].to_vec();
        let sc_bytes = bytes[17 + qs_len_u32 * 4..].to_vec();
        Some((qs_bytes, sc_bytes, m, k, f16))
    };

    // Write to cache.
    let cache_write = |name: &str, qs: &[u32], sc: &[f32], m: usize, k: usize, f16: bool| {
        if !use_q4cache { return; }
        let mut out = Vec::with_capacity(17 + qs.len() * 4 + sc.len() * if f16 { 2 } else { 4 });
        out.extend_from_slice(&(m as u64).to_le_bytes());
        out.extend_from_slice(&(k as u64).to_le_bytes());
        out.push(f16 as u8);
        for &q in qs { out.extend_from_slice(&q.to_le_bytes()); }
        if f16 {
            for &s in sc { out.extend_from_slice(&f32_to_f16(s).to_le_bytes()); }
        } else {
            for &s in sc { out.extend_from_slice(&s.to_le_bytes()); }
        }
        let _ = std::fs::write(cache_path(name), &out);
    };

    // ── SETUP: quantize + upload all big matrices to Q4 resident (once) ──
    let t_load = Instant::now();
    let mut qw: HashMap<String, (Tensor, Tensor, usize, usize)> = HashMap::new(); // name → (qs, scales, m, k)
    let mut fw: HashMap<String, Vec<f32>> = HashMap::new(); // f32 weights used HOST-side
    let mut fwd: HashMap<String, Tensor> = HashMap::new(); // f32 weights RESIDENT on device
    let fd = |fwd: &mut HashMap<String, Tensor>, name: &str, v: &[f32], shape: Vec<usize>| {
        fwd.insert(name.to_string(), Tensor::new(d.upload(&tb(v)).unwrap(), shape, DType::F32));
    };
    // f16: true for weights read by the PLAIN gemv kernel (qmv) — its scale param
    // is f16. false for shared-expert weights (relu2/accum kernels, f32 scale).
    let qload = |qw: &mut HashMap<String, (Tensor, Tensor, usize, usize)>, name: &str, m: usize, k: usize, f16: bool| {
        // Try cache hit first: skip BF16→F32 + quantize if available.
        if let Some((qs_bytes, sc_bytes, cm, ck, cf16)) = cache_read(name) {
            if cm == m && ck == k && cf16 == f16 {
                let qt = Tensor::new(d.upload(&qs_bytes).unwrap(), vec![qs_bytes.len() / 4], DType::U32);
                let sct = Tensor::new(d.upload(&sc_bytes).unwrap(), vec![sc_bytes.len() / if f16 { 2 } else { 4 }], if f16 { DType::F16 } else { DType::F32 });
                qw.insert(name.to_string(), (qt, sct, m, k));
                return;
            }
        }
        // Cache miss: full BF16→F32 + Q4 quantize path.
        let w = g(name);
        let (qs, sc) = quantize_q4(&w, m, k);
        cache_write(name, &qs, &sc, m, k, f16);
        let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
        let sct = if f16 {
            Tensor::new(d.upload(&tb_f16(&sc)).unwrap(), vec![sc.len()], DType::F16)
        } else {
            Tensor::new(d.upload(&tb(&sc)).unwrap(), vec![sc.len()], DType::F32)
        };
        qw.insert(name.to_string(), (qt, sct, m, k));
    };
    let embed = g("language_model.backbone.embeddings.weight"); // host lookup table
    fd(&mut fwd, "norm_f", &g("language_model.backbone.norm_f.weight"), vec![hid]);
    qload(&mut qw, "language_model.lm_head.weight", vocab, hid, true);
    for (l, mix) in PATTERN.chars().enumerate() {
        let p = format!("language_model.backbone.layers.{l}");
        fd(&mut fwd, &format!("{p}.norm.weight"), &g(&format!("{p}.norm.weight")), vec![hid]);
        match mix {
            'M' => {
                qload(&mut qw, &format!("{p}.mixer.in_proj.weight"), in_proj_out, hid, true);
                qload(&mut qw, &format!("{p}.mixer.out_proj.weight"), hid, di, true);
                // conv weight pre-reorganized [kc, conv_dim] ONCE (was redone per step).
                let cw_hf = g(&format!("{p}.mixer.conv1d.weight"));
                let mut cw = vec![0.0f32; kc * conv_dim];
                for ch in 0..conv_dim { for kk in 0..kc { cw[kk * conv_dim + ch] = cw_hf[ch * kc + kk]; } }
                fd(&mut fwd, &format!("{p}.mixer.conv1d.weight"), &cw, vec![kc * conv_dim]);
                fd(&mut fwd, &format!("{p}.mixer.conv1d.bias"), &g(&format!("{p}.mixer.conv1d.bias")), vec![conv_dim]);
                fd(&mut fwd, &format!("{p}.mixer.A_log"), &g(&format!("{p}.mixer.A_log")), vec![m_nh]);
                fd(&mut fwd, &format!("{p}.mixer.D"), &g(&format!("{p}.mixer.D")), vec![m_nh]);
                fw.insert(format!("{p}.mixer.dt_bias"), g(&format!("{p}.mixer.dt_bias")));     // host (softplus, tiny)
                fw.insert(format!("{p}.mixer.norm.weight"), g(&format!("{p}.mixer.norm.weight")));
            }
            'E' => {
                fd(&mut fwd, &format!("{p}.mixer.gate.weight"), &g(&format!("{p}.mixer.gate.weight")), vec![n_exp, hid]);
                fw.insert(format!("{p}.mixer.gate.e_score_correction_bias"), g(&format!("{p}.mixer.gate.e_score_correction_bias")));
                // Experts stored CONTIGUOUS ([n_exp*inter, hid] up, [n_exp*hid, inter] down)
                // so the batched gather kernel runs them as one big efficient GEMV.
                // Cache the combined expert packs under "{p}.moe_up_all" / ".moe_down_all".
                let moe_up_name = format!("{p}.moe_up_all");
                let moe_down_name = format!("{p}.moe_down_all");
                let (mup_hit, mdown_hit) = (cache_read(&moe_up_name), cache_read(&moe_down_name));
                if let (Some((uqb, usb, cm, ck, _)), Some((dqb, dsb, dm, dk, _))) = (mup_hit, mdown_hit) {
                    if cm == n_exp * inter && ck == hid && dm == n_exp * hid && dk == inter {
                        let ut = Tensor::new(d.upload(&uqb).unwrap(), vec![uqb.len() / 4], DType::U32);
                        let ust = Tensor::new(d.upload(&usb).unwrap(), vec![usb.len() / 2], DType::F16);
                        let dt2 = Tensor::new(d.upload(&dqb).unwrap(), vec![dqb.len() / 4], DType::U32);
                        let dst2 = Tensor::new(d.upload(&dsb).unwrap(), vec![dsb.len() / 2], DType::F16);
                        qw.insert(moe_up_name, (ut, ust, n_exp * inter, hid));
                        qw.insert(moe_down_name, (dt2, dst2, n_exp * hid, inter));
                    } else {
                        // Dimension mismatch — fall through to rebuild.
                        let (mut uqs, mut usc, mut dqs, mut dsc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                        for e in 0..n_exp {
                            let (q, s) = quantize_q4(&g(&format!("{p}.mixer.experts.{e}.up_proj.weight")), inter, hid);
                            uqs.extend(q); usc.extend(s);
                            let (q, s) = quantize_q4(&g(&format!("{p}.mixer.experts.{e}.down_proj.weight")), hid, inter);
                            dqs.extend(q); dsc.extend(s);
                        }
                        cache_write(&moe_up_name, &uqs, &usc, n_exp * inter, hid, true);
                        cache_write(&moe_down_name, &dqs, &dsc, n_exp * hid, inter, true);
                        qw.insert(moe_up_name, (Tensor::new(d.upload(&tbu(&uqs)).unwrap(), vec![uqs.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&usc)).unwrap(), vec![usc.len()], DType::F16), n_exp * inter, hid));
                        qw.insert(moe_down_name, (Tensor::new(d.upload(&tbu(&dqs)).unwrap(), vec![dqs.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&dsc)).unwrap(), vec![dsc.len()], DType::F16), n_exp * hid, inter));
                    }
                } else {
                    // Cache miss: build + cache.
                    let (mut uqs, mut usc, mut dqs, mut dsc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    for e in 0..n_exp {
                        let (q, s) = quantize_q4(&g(&format!("{p}.mixer.experts.{e}.up_proj.weight")), inter, hid);
                        uqs.extend(q); usc.extend(s);
                        let (q, s) = quantize_q4(&g(&format!("{p}.mixer.experts.{e}.down_proj.weight")), hid, inter);
                        dqs.extend(q); dsc.extend(s);
                    }
                    cache_write(&moe_up_name, &uqs, &usc, n_exp * inter, hid, true);
                    cache_write(&moe_down_name, &dqs, &dsc, n_exp * hid, inter, true);
                    qw.insert(moe_up_name, (Tensor::new(d.upload(&tbu(&uqs)).unwrap(), vec![uqs.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&usc)).unwrap(), vec![usc.len()], DType::F16), n_exp * inter, hid));
                    qw.insert(moe_down_name, (Tensor::new(d.upload(&tbu(&dqs)).unwrap(), vec![dqs.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&dsc)).unwrap(), vec![dsc.len()], DType::F16), n_exp * hid, inter));
                }
                qload(&mut qw, &format!("{p}.mixer.shared_experts.up_proj.weight"), shared_inter, hid, true);
                qload(&mut qw, &format!("{p}.mixer.shared_experts.down_proj.weight"), hid, shared_inter, true);
            }
            '*' => {
                qload(&mut qw, &format!("{p}.mixer.q_proj.weight"), qdim, hid, true);
                qload(&mut qw, &format!("{p}.mixer.k_proj.weight"), kvdim, hid, true);
                qload(&mut qw, &format!("{p}.mixer.v_proj.weight"), kvdim, hid, true);
                qload(&mut qw, &format!("{p}.mixer.o_proj.weight"), hid, qdim, true);
            }
            _ => unreachable!(),
        }
    }
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!("Nemotron resident-Q8 setup: {:.1}s ({} Q8 matrices, ~{:.1}GB)", load_s, qw.len(), qw.values().map(|(q, s, _, _)| (q.elem_count() * 4 + s.elem_count() * 4) as f64).sum::<f64>() / 1e9);

    // resident-weight quantized matvec
    let qmv = |x: &Tensor, name: &str| -> Tensor {
        let (qs, sc, m, k) = &qw[name];
        gemv_q4(d, qs, sc, x, *m, *k, *m).unwrap()
    };
    // resident-weight matvec that scales + accumulates into `acc` in one kernel.
    let qacc = |x: &Tensor, name: &str, acc: &Tensor, sb: &Tensor| {
        let (qs, sc, m, k) = &qw[name];
        gemv_q4_accum(d, qs, sc, x, acc, sb, *m, *k, *m).unwrap();
    };
    // resident-weight matvec with fused ReLU² (MoE expert up-projection).
    let qrelu2 = |x: &Tensor, name: &str| -> Tensor {
        let (qs, sc, m, k) = &qw[name];
        gemv_q4_relu2(d, qs, sc, x, *m, *k, *m).unwrap()
    };

    // ── DECODE: per-token forward reusing resident weights, KV + Mamba state ──
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let prefill = env("NEMOTRON_PREFILL", 0);
    let n_decode = env("NEMOTRON_DECODE", 32);
    let fakectx = env("NEMOTRON_FAKECTX", 0);
    // GQA split-K flash-decode sdpa: default ON for long context (≈+15-25% @ 32K
    // where the single-pass re-reads each shared KV head gqa_factor×); single-pass
    // is marginally better at tiny ctx (2-pass partial overhead). Opt out w/ NEMOTRON_NO_2PASS.
    let no_2pass = std::env::var("NEMOTRON_NO_2PASS").is_ok();
    // Device MoE router REGRESSED (-17-24%, clock-locked): the 1-simdgroup serial
    // top-k bubble underutilizes the GPU worse than host top-k + a cheap sync drain
    // (same lesson as device-Mamba). Host router is DEFAULT; device is opt-in only.
    let no_devrouter = !std::env::var("NEMOTRON_DEVROUTER").is_ok();
    // CUDA-graph capture/replay: collapse ~390 per-token kernel launches into ONE
    // cuGraphLaunch to remove host-launch-gap overhead. Requires NEMOTRON_DEVROUTER=1
    // (device MoE router, no host sync) and gates the final argmax download so the
    // captured region is fully sync-free. Set NEMOTRON_GRAPH=1 to enable.
    let use_graph = std::env::var("NEMOTRON_GRAPH").is_ok();
    // skip_dl: Cell<bool> threaded into the step closure. Set true during capture and
    // replay so the dl(logits) host-sync is skipped (forbidden during capture; not
    // needed for the graph throughput measurement).
    let skip_dl = std::cell::Cell::new(false);
    // F16 KV cache: the clock-locked "+11-27%" was a thermal/order artifact (the
    // internal A/B shows it neutral-to-negative — casts cost ≥ halved-read saves).
    // DEFAULT OFF (f32, no casts); opt in with NEMOTRON_F16KV.
    let f16kv = std::env::var("NEMOTRON_F16KV").is_ok();
    let cap = fakectx.max(prefill) + n_decode + 8; // KV-cache capacity (positions)
    // per-layer state, indexed by absolute layer id. KV cache is now ON-DEVICE
    // ([nkv,cap,hd] per attn layer), so the growing context never round-trips
    // through the host — the 32K decode fix.
    let mut conv_state: Vec<Vec<f32>> = vec![Vec::new(); 52];
    let conv_dev: std::cell::RefCell<Vec<Option<Tensor>>> = std::cell::RefCell::new((0..52).map(|_| None).collect()); // device conv state (NEMOTRON_DEVMAMBA)
    let mut ssm_state: Vec<Option<Tensor>> = (0..52).map(|_| None).collect(); // recurrent SSM state ON-DEVICE
    let mut kvcache: Vec<Option<(Tensor, Tensor)>> = (0..52).map(|_| None).collect();
    let u32buf = |v: u32| Tensor::new(d.upload(&v.to_le_bytes()).unwrap(), vec![1], DType::U32);
    let ones_gs = up(&vec![1.0f32; gs]); // grouped-norm: normalize each 512-group weightless, then scale by the real weight
    // Optional per-section profiling (NEMOTRON_PROFILE=1): synchronize around each
    // mixer type to attribute the per-token time. Adds sync overhead — proportions only.
    let prof = std::env::var("NEMOTRON_PROFILE").is_ok();
    let (tm, te, ta, th) = (std::cell::Cell::new(0f64), std::cell::Cell::new(0f64), std::cell::Cell::new(0f64), std::cell::Cell::new(0f64));

    // one decode step at absolute position `pos`; returns next-token logits' argmax
    let step = |token: usize, pos: usize,
                    conv_state: &mut Vec<Vec<f32>>, ssm_state: &mut Vec<Option<Tensor>>,
                    kvcache: &mut Vec<Option<(Tensor, Tensor)>>| -> usize {
        // residual stream stays ON-DEVICE the whole forward — no per-layer up(x)/dl(out).
        let mut xt = up(&embed[token * hid..(token + 1) * hid]);
        for (l, mix) in PATTERN.chars().enumerate() {
            let p = format!("language_model.backbone.layers.{l}");
            let pt = if prof { d.synchronize().ok(); Some(Instant::now()) } else { None };
            let xn = rms_norm(d, &xt, &fwd[&format!("{p}.norm.weight")], eps).unwrap();
            match mix {
                'M' => {
                    if std::env::var("NEMOTRON_SKIPMAMBA").is_ok() { continue; }
                    if !std::env::var("NEMOTRON_HOSTMAMBA").is_ok() {
                        // ALL-DEVICE Mamba (DEFAULT): no dl/host round-trips. +3.7% clean
                        // internal A/B; argmax 1234. Opt out: NEMOTRON_HOSTMAMBA.
                        let proj = qmv(&xn, &format!("{p}.mixer.in_proj.weight"));
                        let zt = slice(d, &proj, 0, di).unwrap();
                        let xbc_t = slice(d, &proj, di, conv_dim).unwrap();
                        let dt_raw_t = slice(d, &proj, di + conv_dim, m_nh).unwrap();
                        { let mut cd = conv_dev.borrow_mut(); if cd[l].is_none() { cd[l] = Some(up(&vec![0.0f32; (kc - 1) * conv_dim])); } }
                        let yc = { let cd = conv_dev.borrow(); conv1d_causal_step(d, &xbc_t, &fwd[&format!("{p}.mixer.conv1d.weight")], &fwd[&format!("{p}.mixer.conv1d.bias")], cd[l].as_ref().unwrap(), conv_dim as u32, kc as u32).unwrap() };
                        let xbc_act = silu(d, &yc).unwrap();
                        { let mut cd = conv_dev.borrow_mut(); let rolled = conv_roll(d, cd[l].as_ref().unwrap(), &xbc_t, conv_dim, kc).unwrap(); cd[l] = Some(rolled); }
                        let x_ssm = slice(d, &xbc_act, 0, di).unwrap();
                        let bmat = slice(d, &xbc_act, di, ng * ds).unwrap();
                        let cmat = slice(d, &xbc_act, di + ng * ds, ng * ds).unwrap();
                        let dt = softplus_add(d, &dt_raw_t, &up(&fw[&format!("{p}.mixer.dt_bias")])).unwrap();
                        if ssm_state[l].is_none() { ssm_state[l] = Some(up(&vec![0.0f32; m_nh * m_dh * ds])); }
                        let (so, y_t) = ssm_step(d, &x_ssm, &fwd[&format!("{p}.mixer.A_log")], &bmat, &cmat, &fwd[&format!("{p}.mixer.D")], &dt, ssm_state[l].as_ref().unwrap(), m_dh as u32, ds as u32, m_nh as u32, (m_nh / ng) as u32).unwrap();
                        ssm_state[l] = Some(so);
                        let yn = gated_group_rmsnorm(d, &y_t, &zt, &up(&fw[&format!("{p}.mixer.norm.weight")]), eps, di, gs).unwrap();
                        let out = qmv(&yn, &format!("{p}.mixer.out_proj.weight"));
                        xt = add(d, &xt, &out).unwrap();
                        continue;
                    }
                    let proj = dl(&qmv(&xn, &format!("{p}.mixer.in_proj.weight")), in_proj_out);
                    let z = &proj[0..di];
                    let xbc = &proj[di..di + conv_dim];
                    let dt_raw = &proj[di + conv_dim..di + conv_dim + m_nh];
                    if conv_state[l].is_empty() { conv_state[l] = vec![0.0f32; (kc - 1) * conv_dim]; }
                    let yc = conv1d_causal_step(d, &up(xbc), &fwd[&format!("{p}.mixer.conv1d.weight")], &fwd[&format!("{p}.mixer.conv1d.bias")], &up(&conv_state[l]), conv_dim as u32, kc as u32).unwrap();
                    let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
                    { let s = &mut conv_state[l]; s.drain(0..conv_dim); s.extend_from_slice(xbc); }
                    let x_ssm = &xbc_act[0..di];
                    let bmat = &xbc_act[di..di + ng * ds];
                    let cmat = &xbc_act[di + ng * ds..di + 2 * ng * ds];
                    let dt_bias = &fw[&format!("{p}.mixer.dt_bias")];
                    let dt: Vec<f32> = (0..m_nh).map(|i| softplus(dt_raw[i] + dt_bias[i])).collect();
                    if ssm_state[l].is_none() { ssm_state[l] = Some(up(&vec![0.0f32; m_nh * m_dh * ds])); }
                    let (so, y_t) = ssm_step(d, &up(x_ssm), &fwd[&format!("{p}.mixer.A_log")], &up(bmat), &up(cmat), &fwd[&format!("{p}.mixer.D")], &up(&dt), ssm_state[l].as_ref().unwrap(), m_dh as u32, ds as u32, m_nh as u32, (m_nh / ng) as u32).unwrap();
                    ssm_state[l] = Some(so);
                    let y = dl(&y_t, di);
                    let nw = &fw[&format!("{p}.mixer.norm.weight")];
                    let mut yn = vec![0.0f32; di];
                    for grp in 0..ng {
                        let s = grp * gs;
                        let mut ss = 0.0f32;
                        for i in 0..gs { let g = y[s + i] * (z[s + i] / (1.0 + (-z[s + i]).exp())); yn[s + i] = g; ss += g * g; }
                        let r = 1.0 / ((ss / gs as f32) + eps).sqrt();
                        for i in 0..gs { yn[s + i] = yn[s + i] * r * nw[s + i]; }
                    }
                    let out = qmv(&up(&yn), &format!("{p}.mixer.out_proj.weight"));
                    xt = add(d, &xt, &out).unwrap();
                }
                'E' => {
                    if std::env::var("NEMOTRON_SKIPMOE").is_ok() { continue; }
                    // Router: ON-DEVICE (sigmoid+bias+top-k+norm+scale, no host sync) by
                    // default; host path kept for A/B via NEMOTRON_HOSTROUTER.
                    let (idx_buf, wts_buf) = if !std::env::var("NEMOTRON_DEVROUTER").is_ok() {
                        let rl = dl(&gemv(d, &fwd[&format!("{p}.mixer.gate.weight")], &xn).unwrap(), n_exp);
                        let sig: Vec<f32> = rl.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
                        let bias = &fw[&format!("{p}.mixer.gate.e_score_correction_bias")];
                        let choice: Vec<f32> = (0..n_exp).map(|i| sig[i] + bias[i]).collect();
                        let eidx = ffai_runtime::topk(&choice, top_k);
                        let mut w: Vec<f32> = eidx.iter().map(|&e| sig[e]).collect();
                        let wsum: f32 = w.iter().sum::<f32>() + 1e-20;
                        for v in w.iter_mut() { *v = *v / wsum * scale_f; }
                        (Tensor::new(d.upload(&tbu(&eidx.iter().map(|&e| e as u32).collect::<Vec<_>>())).unwrap(), vec![top_k], DType::U32), up(&w))
                    } else {
                        let logits = gemv(d, &fwd[&format!("{p}.mixer.gate.weight")], &xn).unwrap();
                        let bias_dev = up(&fw[&format!("{p}.mixer.gate.e_score_correction_bias")]);
                        moe_router_device(d, &logits, &bias_dev, n_exp, top_k, scale_f).unwrap()
                    };
                    let acc_dev = up(&vec![0.0f32; hid]);
                    let (uqs, usc, _, _) = &qw[&format!("{p}.moe_up_all")];
                    let a = moe_gather_up_relu2(d, uqs, usc, &xn, &idx_buf, top_k, inter, hid).unwrap();
                    let (dqs, dsc, _, _) = &qw[&format!("{p}.moe_down_all")];
                    let downs = moe_gather_down(d, dqs, dsc, &a, &idx_buf, top_k, inter, hid).unwrap();
                    moe_weighted_sum(d, &downs, &wts_buf, &acc_dev, hid, top_k).unwrap();
                    // shared expert (single, not gathered)
                    let sa = qrelu2(&xn, &format!("{p}.mixer.shared_experts.up_proj.weight"));
                    qacc(&sa, &format!("{p}.mixer.shared_experts.down_proj.weight"), &acc_dev, &up(&[1.0f32]));
                    xt = add(d, &xt, &acc_dev).unwrap();
                }
                '*' => {
                    // q/k/v + RoPE stay ON-DEVICE; append k,v straight into the
                    // device KV cache; sdpa reads the cache. No host KV traffic.
                    let q = rope_llama(d, &qmv(&xn, &format!("{p}.mixer.q_proj.weight")).reshaped(vec![nq, hd]), pos as u32, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap();
                    let k = rope_llama(d, &qmv(&xn, &format!("{p}.mixer.k_proj.weight")).reshaped(vec![nkv, hd]), pos as u32, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap();
                    let v = qmv(&xn, &format!("{p}.mixer.v_proj.weight"));
                    // F16 KV: halve the 32K KV read (sdpa = 34% of GPU). Cache + q/k/v in
                    // f16, 2pass reads them natively (registry-tested f16), attn→f32 for o_proj.
                    let (q, k, v) = if f16kv {
                        (cast_f32_f16(d, &q).unwrap(), cast_f32_f16(d, &k).unwrap(), cast_f32_f16(d, &v).unwrap())
                    } else { (q, k, v) };
                    if kvcache[l].is_none() {
                        kvcache[l] = if f16kv {
                            let zf16 = || Tensor::new(d.upload(&vec![0u8; nkv * cap * hd * 2]).unwrap(), vec![nkv * cap * hd], DType::F16);
                            Some((zf16(), zf16()))
                        } else {
                            Some((up(&vec![0.0f32; nkv * cap * hd]), up(&vec![0.0f32; nkv * cap * hd])))
                        };
                    }
                    let (kcache, vcache) = kvcache[l].as_ref().unwrap();
                    let pb = u32buf(pos as u32);
                    kv_append(d, &k, kcache, &pb, hd, cap, nkv * hd).unwrap();
                    kv_append(d, &v, vcache, &pb, hd, cap, nkv * hd).unwrap();
                    let len = (pos + 1) as u32;
                    // Diagnostic: skip the sdpa (placeholder output, right shape) to measure
                    // its wall-time contribution via the internal A/B (NEMOTRON_AB=NEMOTRON_SKIPSDPA).
                    let attn = if std::env::var("NEMOTRON_SKIPSDPA").is_ok() {
                        q.clone()
                    } else if !no_2pass && len > 1024 {
                        // NEMOTRON_BLOCKS overrides the split-K block count.
                        // GB10 sweep: 128 ≈ optimal at all ctx (256 within noise, both
                        // beat 512/1024). Default 128; blocks MUST be a multiple of 32.
                        let blocks: u32 = env("NEMOTRON_BLOCKS", 128usize) as u32;
                        // NEMOTRON_BC4=1: BC=4 pass-1 (batched 4 positions/iter).
                        // NEMOTRON_TILED=1: tiled pass-1 (contiguous chunks, L2-friendly).
                        let use_bc4 = std::env::var("NEMOTRON_BC4").is_ok();
                        let use_tiled = std::env::var("NEMOTRON_TILED").is_ok();
                        let a = if use_bc4 {
                            sdpa_decode_2pass_bc4(d, &q, &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, len, cap as u32, (nq / nkv) as u32, ascale, blocks).unwrap()
                        } else if use_tiled {
                            sdpa_decode_2pass_tiled(d, &q, &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, len, cap as u32, (nq / nkv) as u32, ascale, blocks).unwrap()
                        } else {
                            sdpa_decode_2pass(d, &q, &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, len, cap as u32, (nq / nkv) as u32, ascale, blocks).unwrap()
                        };
                        if f16kv { cast_f16_f32(d, &a).unwrap() } else { a }
                    } else {
                        let a = sdpa_decode(d, &q, &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, len, cap as u32, (nq / nkv) as u32, ascale).unwrap();
                        if f16kv { cast_f16_f32(d, &a).unwrap() } else { a }
                    };
                    let o = qmv(&attn.reshaped(vec![qdim]), &format!("{p}.mixer.o_proj.weight"));
                    xt = add(d, &xt, &o).unwrap();
                }
                _ => unreachable!(),
            }
            if let Some(pt) = pt { d.synchronize().ok(); let e = pt.elapsed().as_secs_f64(); let c = match mix { 'M' => &tm, 'E' => &te, _ => &ta }; c.set(c.get() + e); }
        }
        let ht = if prof { d.synchronize().ok(); Some(Instant::now()) } else { None };
        let xf = rms_norm(d, &xt, &fwd["norm_f"], eps).unwrap();
        // Gate the final argmax download: forbidden during CUDA-graph capture (host-sync).
        // During capture and graph replay we return a dummy 0 — the bench only measures
        // throughput, not token correctness.
        if skip_dl.get() {
            let _ = qmv(&xf, "language_model.lm_head.weight"); // keep the lm_head kernel in graph
            if let Some(ht) = ht { d.synchronize().ok(); th.set(th.get() + ht.elapsed().as_secs_f64()); }
            return 0;
        }
        let logits = dl(&qmv(&xf, "language_model.lm_head.weight"), vocab);
        if let Some(ht) = ht { d.synchronize().ok(); th.set(th.get() + ht.elapsed().as_secs_f64()); }
        ffai_runtime::argmax(&logits)
    };

    let mut tok = 1234usize;
    let mut pos = 0usize;
    // For the 32K measurement, start at `fakectx` (the device KV cache is alloc'd
    // to `cap` and zero-filled on first use, so the sdpa genuinely reads `fakectx`
    // cached positions — the real 32K work — without a 38-min token prefill).
    if fakectx > 0 { pos = fakectx; }
    // warm the cache to `prefill` positions (untimed), then time `n_decode` steps
    for _ in 0..prefill { let nxt = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); tok = nxt; pos += 1; }
    let first = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); // 1 warm step (JIT) + correctness peek
    tok = first; pos += 1;
    if prefill == 0 && fakectx == 0 { eprintln!("Nemotron decode: token1234 → next argmax {first} (F32 ref argmax=1234; Q8 may perturb the near-tie)"); }
    // Fast internal A/B: one model load, toggle an env flag (e.g. MT_GEMV_2ROW)
    // between alternating decode batches IN-PROCESS — same thermal/clock state,
    // order-alternated to cancel drift. Resets pos each batch so the KV cap holds.
    if let Ok(ab) = std::env::var("NEMOTRON_AB") {
        // "on" sets the flag to NEMOTRON_AB_VAL (default "1"); "off" unsets it.
        // For value-flags like MT_MOE_RPT/MT_GEMV_RPT use NEMOTRON_AB_VAL=2 so
        // ON=rpt2 vs OFF=rpt1 (unset default), not the no-op "1" vs unset(=1).
        let ab_val = std::env::var("NEMOTRON_AB_VAL").unwrap_or_else(|_| "1".to_string());
        let rounds = 6usize;
        let base_pos = pos;
        let (mut t_off, mut t_on) = (0f64, 0f64);
        for r in 0..rounds {
            // alternate which config runs first each round to cancel position bias
            let first_on = r % 2 == 1;
            for &on in &[first_on, !first_on] {
                unsafe { if on { std::env::set_var(&ab, &ab_val); } else { std::env::remove_var(&ab); } }
                pos = base_pos;
                let s = Instant::now();
                for _ in 0..n_decode { let nxt = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); tok = nxt; pos += 1; }
                let e = s.elapsed().as_secs_f64();
                if on { t_on += e; } else { t_off += e; }
            }
        }
        let off_tps = (rounds * n_decode) as f64 / t_off;
        let on_tps = (rounds * n_decode) as f64 / t_on;
        eprintln!("──── AB[{ab}] internal ({rounds} rounds × {n_decode} tok, ctx {}) ────", fakectx.max(prefill));
        eprintln!("  OFF {off_tps:.2} tok/s | ON {on_tps:.2} tok/s | delta {:+.1}%", (on_tps / off_tps - 1.0) * 100.0);
        let _ = tok;
        return;
    }
    let t0 = Instant::now();
    for _ in 0..n_decode { let nxt = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); tok = nxt; pos += 1; }
    let dt = t0.elapsed().as_secs_f64();
    let eager_tps = n_decode as f64 / dt;
    eprintln!("──────── NemotronH-Nano RESIDENT-Q8 DECODE on {plat} ────────");
    eprintln!("  context  start {} + {n_decode} timed (pos→{pos})", fakectx.max(prefill));
    eprintln!("  decode   {n_decode} tok in {dt:.2}s = {eager_tps:.2} tok/s ({:.1} ms/tok)", dt * 1000.0 / n_decode as f64);
    if prof {
        let n = (n_decode + 1) as f64; // includes the warm step
        eprintln!("  profile/tok: M(mamba×23) {:.1}ms · E(moe×23) {:.1}ms · *(attn×6) {:.1}ms · head {:.1}ms",
            tm.get() * 1e3 / n, te.get() * 1e3 / n, ta.get() * 1e3 / n, th.get() * 1e3 / n);
    }
    eprintln!("  (vs naive-reload baseline 0.003 tok/s; resident weights uploaded once in {load_s:.0}s setup)");
    eprintln!("──────────────────────────────────────────────────────────────");

    // ── CUDA GRAPH CAPTURE/REPLAY measurement ──────────────────────────────────
    // Activated by NEMOTRON_GRAPH=1. Requires NEMOTRON_DEVROUTER=1 (device router,
    // no host sync in the step). Collapses ~390 per-token launches into one
    // cuGraphLaunch to remove host-launch-gap overhead; measures vs eager at the
    // SAME thermal/clock state in one process.
    //
    // Protocol:
    //   a. Run 5 more warm eager steps (pool stays fully populated, clocks steady).
    //   b. Time M eager steps (skip_dl=false so argmax fires for comparison baseline).
    //   c. One more eager step (pool free-list in exact capture-step state).
    //   d. begin_capture → step (skip_dl=true, no devrouter host-sync) → end_capture.
    //   e. synchronize; time M graph_launches.
    //   f. Print eager_tps_graph_run, graph_tps, ratio.
    if use_graph {
        if no_devrouter {
            eprintln!("NEMOTRON_GRAPH: WARNING — NEMOTRON_DEVROUTER not set. Host router mid-token sync will break capture. Re-run with NEMOTRON_DEVROUTER=1.");
        }
        let gm = n_decode; // use same step count for apples-to-apples
        // Use `fakectx` as the fixed position for all graph-section measurements.
        // This keeps `pos` within bounds (KV cap = fakectx + n_decode + 8) AND gives
        // the correct 32K SDPA length. We don't advance pos in the graph section.
        let gpos = fakectx.max(prefill); // fixed position for graph-section steps
        // (a) 5 warm steps at gpos (untimed) — populates pool, reaches steady clocks.
        for _ in 0..5 { step(tok, gpos, &mut conv_state, &mut ssm_state, &mut kvcache); }
        // (b) Eager baseline at fixed gpos (skip_dl=false — argmax fires for comparison).
        let t_eager = Instant::now();
        for _ in 0..gm { step(tok, gpos, &mut conv_state, &mut ssm_state, &mut kvcache); }
        d.synchronize().ok();
        let eager_dt = t_eager.elapsed().as_secs_f64();
        let eager_tps_hot = gm as f64 / eager_dt;
        // (c) One pool-alignment step (pool free-list in exact same state the capture step sees).
        step(tok, gpos, &mut conv_state, &mut ssm_state, &mut kvcache);
        // (d) Capture: skip_dl = true so no host-sync in step.
        skip_dl.set(true);
        d.begin_capture().expect("NEMOTRON_GRAPH: begin_capture failed — check DEVROUTER and that no cuMemAlloc slipped into the step");
        step(tok, gpos, &mut conv_state, &mut ssm_state, &mut kvcache);
        let graph_handle = d.end_capture().expect("NEMOTRON_GRAPH: end_capture failed — check for stray sync/alloc in the captured region");
        // (e) Synchronize then time M graph launches — two modes:
        //   serial:  graph_launch (sync-per-token) measures single-token latency
        //   batched: graph_launch_batch (N enqueues, one sync) eliminates per-token
        //            host-GPU handoff, giving maximum pipeline throughput
        d.synchronize().ok();
        // Serial: measures true per-token wall time (GPU + host round-trip).
        let t_graph_serial = Instant::now();
        for _ in 0..gm { d.graph_launch(graph_handle).expect("graph_launch failed"); }
        let graph_serial_dt = t_graph_serial.elapsed().as_secs_f64();
        let graph_serial_tps = gm as f64 / graph_serial_dt;
        // Batched: N cuGraphLaunch enqueues, one cuStreamSynchronize — removes
        // per-token host-GPU handoff to reveal the true GPU throughput ceiling.
        d.synchronize().ok();
        let t_graph_batch = Instant::now();
        d.graph_launch_batch(graph_handle, gm).expect("graph_launch_batch failed");
        let graph_batch_dt = t_graph_batch.elapsed().as_secs_f64();
        let graph_batch_tps = gm as f64 / graph_batch_dt;
        let ratio_serial = graph_serial_tps / eager_tps_hot;
        let ratio_batch  = graph_batch_tps  / eager_tps_hot;
        eprintln!("──────── CUDA GRAPH CAPTURE/REPLAY (ctx {}, {} tok each) ────────", fakectx.max(prefill), gm);
        eprintln!("  eager (hot)       {eager_tps_hot:.2} tok/s  ({:.2} ms/tok)", eager_dt * 1000.0 / gm as f64);
        eprintln!("  graph serial      {graph_serial_tps:.2} tok/s  ({:.2} ms/tok)  ratio {ratio_serial:.3}x", graph_serial_dt * 1000.0 / gm as f64);
        eprintln!("  graph batched     {graph_batch_tps:.2} tok/s  ({:.2} ms/tok)  ratio {ratio_batch:.3}x", graph_batch_dt * 1000.0 / gm as f64);
        if graph_batch_tps >= 75.0 {
            eprintln!("  ✓ TARGET MET: graph_batch_tps {graph_batch_tps:.1} >= 75 tok/s");
        } else {
            eprintln!("  ✗ target 75 tok/s not yet met (gap {:.1} tok/s)", 75.0 - graph_batch_tps);
        }
        eprintln!("──────────────────────────────────────────────────────────────────────");
    }
}
