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
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
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
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
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
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
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
    let mut idx: Vec<usize> = (0..vocab).collect(); idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
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
    let mut idx: Vec<usize> = (0..vocab).collect(); idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
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
    let mut idx: Vec<usize> = (0..vocab).collect(); idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
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
    let mut idx: Vec<usize> = (0..vocab).collect(); idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
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
    let mut idx: Vec<usize> = (0..vocab).collect(); idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
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
        let mut eidx: Vec<usize> = (0..n_exp).collect(); eidx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let mut acc = vec![0.0f32; hid];
        for &e in &eidx[..top_k] {
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
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
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
    let mut idx: Vec<usize> = (0..vocab).collect(); idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    eprintln!("Falcon-H1-0.5B on {plat}: top3 = {:?} (HF = [593,531,587])", &idx[..3]);
    assert_eq!(&idx[..3], &[593usize, 531, 587], "Falcon-H1 top-3 != HF");
    eprintln!("✅ Falcon-H1-0.5B (hybrid Mamba2+attn) matches HF ({plat}).");
}
