// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Transformer-LLM builder (Llama / Qwen / Mistral / Yi / Phi / SmolLM …).
//! One decode layer assembled entirely from [`ffai_ops`] — so it runs on any
//! backend that implements [`ffai_core::Device`]. Config-parameterized: the
//! same code serves every model in this family by varying [`LlamaConfig`].

use ffai_core::{Device, Error, Result, Tensor};
use ffai_ops as ops;

/// Architecture config shared by the transformer-LLM family.
#[derive(Debug, Clone, Copy)]
pub struct LlamaConfig {
    pub hidden: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// Qwen3-style per-head RMSNorm on Q and K before RoPE.
    pub qk_norm: bool,
    /// QKV projection bias (Qwen2/Qwen2.5).
    pub attn_bias: bool,
}

impl LlamaConfig {
    pub fn heads_per_group(&self) -> u32 {
        (self.n_q_heads / self.n_kv_heads) as u32
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Per-layer weights (dense). Row-major: a projection `[out, in]` is applied
/// as `gemv(W, x)`.
pub struct LayerWeights {
    pub attn_norm: Tensor, // [hidden]
    pub wq: Tensor,        // [n_q_heads*head_dim, hidden]
    pub wk: Tensor,        // [n_kv_heads*head_dim, hidden]
    pub wv: Tensor,        // [n_kv_heads*head_dim, hidden]
    pub wo: Tensor,        // [hidden, n_q_heads*head_dim]
    /// QKV biases `[*_heads*head_dim]` (None unless attn_bias).
    pub bias_q: Option<Tensor>,
    pub bias_k: Option<Tensor>,
    pub bias_v: Option<Tensor>,
    /// Qwen3 per-head Q/K RMSNorm weights `[head_dim]` (None unless qk_norm).
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub mlp_norm: Tensor,  // [hidden]
    pub w_gate: Tensor,    // [intermediate, hidden]
    pub w_up: Tensor,      // [intermediate, hidden]
    pub w_down: Tensor,    // [hidden, intermediate]
}

/// One transformer decode step for a single token, attending only to itself
/// (single-position KV — `n_kv = 1`). This exercises the full layer pipeline
/// — RMSNorm → QKV proj → RoPE → SDPA → O proj → residual → RMSNorm →
/// SwiGLU MLP → residual — through the shared op layer. The multi-position
/// KV cache (write new k/v at `pos`, attend over `[0, pos]`) is the model
/// loop's responsibility; this is the per-layer compute.
///
/// `x` is the `[hidden]` residual stream for the current token.
pub fn decode_layer_self(
    dev: &dyn Device,
    cfg: &LlamaConfig,
    w: &LayerWeights,
    x: &Tensor,
    pos: u32,
) -> Result<Tensor> {
    let hd = cfg.head_dim;
    let theta = cfg.rope_theta;

    // ── attention ────────────────────────────────────────────────────
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;
    let mut q = ops::gemv(dev, &w.wq, &h)?;
    let mut k = ops::gemv(dev, &w.wk, &h)?;
    let mut v = ops::gemv(dev, &w.wv, &h)?;
    if cfg.attn_bias {
        q = ops::add(dev, &q, w.bias_q.as_ref().ok_or_else(|| Error::Msg("attn_bias set but bias_q missing".into()))?)?;
        k = ops::add(dev, &k, w.bias_k.as_ref().ok_or_else(|| Error::Msg("attn_bias set but bias_k missing".into()))?)?;
        v = ops::add(dev, &v, w.bias_v.as_ref().ok_or_else(|| Error::Msg("attn_bias set but bias_v missing".into()))?)?;
    }

    // Reshape to heads; optional Qwen3 per-head Q/K RMSNorm before RoPE.
    let q = q.reshaped(vec![cfg.n_q_heads, hd]);
    let k = k.reshaped(vec![cfg.n_kv_heads, hd]);
    let (q, k) = if cfg.qk_norm {
        let qn = w.q_norm.as_ref().ok_or_else(|| Error::Msg("qk_norm set but q_norm missing".into()))?;
        let kn = w.k_norm.as_ref().ok_or_else(|| Error::Msg("qk_norm set but k_norm missing".into()))?;
        (ops::rms_norm(dev, &q, qn, cfg.eps)?, ops::rms_norm(dev, &k, kn, cfg.eps)?)
    } else {
        (q, k)
    };

    // RoPE on Q and K (vanilla — freq-band scaling disabled).
    let q = ops::rope_llama(dev, &q, pos, theta, 1.0, 1.0, 1.0, 1e9)?;
    let k = ops::rope_llama(dev, &k, pos, theta, 1.0, 1.0, 1.0, 1e9)?;

    // Single-position attention: KV cache is just this token (n_kv=1, stride=1).
    let attn = ops::sdpa_decode(
        dev,
        &q,
        &k,
        &v.reshaped(vec![cfg.n_kv_heads, hd]),
        hd,
        1,
        1,
        cfg.heads_per_group(),
        cfg.attn_scale(),
    )?;

    let o = ops::gemv(dev, &w.wo, &attn.reshaped(vec![cfg.n_q_heads * hd]))?;
    let x1 = ops::add(dev, x, &o)?;

    // ── MLP (SwiGLU) ─────────────────────────────────────────────────
    let h2 = ops::rms_norm(dev, &x1, &w.mlp_norm, cfg.eps)?;
    let gate = ops::gemv(dev, &w.w_gate, &h2)?;
    let up = ops::gemv(dev, &w.w_up, &h2)?;
    let act = ops::swiglu(dev, &gate, &up)?;
    let down = ops::gemv(dev, &w.w_down, &act)?;
    let x2 = ops::add(dev, &x1, &down)?;

    Ok(x2)
}

/// Full model weights for the transformer-LLM family.
pub struct ModelWeights {
    pub embed: Tensor,           // [vocab, hidden]
    pub layers: Vec<LayerWeights>,
    pub final_norm: Tensor,      // [hidden]
    pub lm_head: Tensor,         // [vocab, hidden]  (untied; tie = embed)
}

/// Single-token forward: embed → every decode layer (self-attention) →
/// final RMSNorm → lm_head → next-token logits `[vocab]`. The whole model
/// graph on the shared op layer — embedding, all layers, and the head. (The
/// multi-position KV cache for sequences ≥ 2 is the decode loop's job; this
/// is the per-token compute that proves the full graph.)
pub fn forward_single(
    dev: &dyn Device,
    cfg: &LlamaConfig,
    w: &ModelWeights,
    token_id: u32,
) -> Result<Tensor> {
    let ids = Tensor::new(
        dev.upload(&token_id.to_le_bytes())?,
        vec![1],
        ffai_core::DType::U32,
    );
    let mut x = ops::gather(dev, &w.embed, &ids)?.reshaped(vec![cfg.hidden]);
    for layer in &w.layers {
        x = decode_layer_self(dev, cfg, layer, &x, 0)?;
    }
    let xn = ops::rms_norm(dev, &x, &w.final_norm, cfg.eps)?;
    ops::gemv(dev, &w.lm_head, &xn)
}

/// Load a Qwen3 (or any HF-named transformer-LLM) checkpoint from a parsed
/// SafeTensors blob, uploading every weight to `dev`. Maps the HF names
/// (`model.layers.N.self_attn.q_proj.weight`, …) to [`ModelWeights`]. Falls
/// back to tied embeddings when `lm_head.weight` is absent.
pub fn load_qwen3(
    dev: &dyn Device,
    st: &ffai_loader::SafeTensors,
    cfg: &LlamaConfig,
    n_layers: usize,
) -> Result<ModelWeights> {
    let up = |name: &str| -> Result<Tensor> {
        let (bytes, dt, shape) = st.tensor(name)?;
        Ok(Tensor::new(dev.upload(bytes)?, shape.to_vec(), dt))
    };

    let embed = up("model.embed_tokens.weight")?;
    let final_norm = up("model.norm.weight")?;
    let lm_head = match up("lm_head.weight") {
        Ok(t) => t,
        Err(_) => up("model.embed_tokens.weight")?, // tied
    };

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        layers.push(LayerWeights {
            attn_norm: up(&format!("{p}.input_layernorm.weight"))?,
            wq: up(&format!("{p}.self_attn.q_proj.weight"))?,
            wk: up(&format!("{p}.self_attn.k_proj.weight"))?,
            wv: up(&format!("{p}.self_attn.v_proj.weight"))?,
            wo: up(&format!("{p}.self_attn.o_proj.weight"))?,
            bias_q: if cfg.attn_bias { Some(up(&format!("{p}.self_attn.q_proj.bias"))?) } else { None },
            bias_k: if cfg.attn_bias { Some(up(&format!("{p}.self_attn.k_proj.bias"))?) } else { None },
            bias_v: if cfg.attn_bias { Some(up(&format!("{p}.self_attn.v_proj.bias"))?) } else { None },
            q_norm: if cfg.qk_norm {
                Some(up(&format!("{p}.self_attn.q_norm.weight"))?)
            } else {
                None
            },
            k_norm: if cfg.qk_norm {
                Some(up(&format!("{p}.self_attn.k_norm.weight"))?)
            } else {
                None
            },
            mlp_norm: up(&format!("{p}.post_attention_layernorm.weight"))?,
            w_gate: up(&format!("{p}.mlp.gate_proj.weight"))?,
            w_up: up(&format!("{p}.mlp.up_proj.weight"))?,
            w_down: up(&format!("{p}.mlp.down_proj.weight"))?,
        });
    }

    Ok(ModelWeights { embed, layers, final_norm, lm_head })
}

/// A model loaded from an HF directory, with its derived config.
pub struct LoadedModel {
    pub cfg: LlamaConfig,
    pub weights: ModelWeights,
    pub n_layers: usize,
    pub vocab: usize,
}

/// Load any dense transformer-LLM straight from an HF directory: parse
/// `config.json` for the geometry, detect arch flags from the tensor names
/// (qk-norm by `q_norm.weight`, QKV bias by `q_proj.bias`), and upload the
/// weights. This is what makes the whole Llama/Qwen/Mistral/Yi/Phi/SmolLM
/// family load with one code path — no per-model hardcoding.
pub fn load_hf(dev: &dyn Device, dir: &str) -> Result<LoadedModel> {
    let cfg_txt = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| Error::Msg(format!("read config.json: {e}")))?;
    let j: serde_json::Value =
        serde_json::from_str(&cfg_txt).map_err(|e| Error::Msg(format!("config.json: {e}")))?;
    let u = |k: &str| j[k].as_u64().map(|x| x as usize);
    let hidden = u("hidden_size").ok_or_else(|| Error::Msg("config: hidden_size".into()))?;
    let n_q_heads = u("num_attention_heads").ok_or_else(|| Error::Msg("config: n_heads".into()))?;
    let n_kv_heads = u("num_key_value_heads").unwrap_or(n_q_heads);
    let head_dim = u("head_dim").unwrap_or(hidden / n_q_heads);
    let intermediate = u("intermediate_size").ok_or_else(|| Error::Msg("config: inter".into()))?;
    let n_layers = u("num_hidden_layers").ok_or_else(|| Error::Msg("config: n_layers".into()))?;
    let vocab = u("vocab_size").ok_or_else(|| Error::Msg("config: vocab".into()))?;
    let rope_theta = j["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let eps = j["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;

    let st = ffai_loader::SafeTensors::open(&format!("{dir}/model.safetensors"))?;
    let qk_norm = st.info("model.layers.0.self_attn.q_norm.weight").is_some();
    let attn_bias = st.info("model.layers.0.self_attn.q_proj.bias").is_some();

    let cfg = LlamaConfig {
        hidden,
        n_q_heads,
        n_kv_heads,
        head_dim,
        intermediate,
        rope_theta,
        eps,
        qk_norm,
        attn_bias,
    };
    let weights = load_qwen3(dev, &st, &cfg, n_layers)?;
    Ok(LoadedModel { cfg, weights, n_layers, vocab })
}

// ════════════════════════════════════════════════════════════════════════
// GGUF loader + KV-cache decode (Qwen2.5 / llama.cpp tensor naming)
// ════════════════════════════════════════════════════════════════════════
//
// The SafeTensors path above expects HF-named, float-on-disk tensors. GGUF
// ships quantized (Q8_0 here) under llama.cpp names (`blk.N.attn_q.weight`…).
// `load_qwen_gguf` dequantizes every weight to f32 on the host and uploads it,
// so the whole forward graph runs in f32 — backend-agnostic (the f32 gemv /
// rms_norm / sdpa kernels are registry-tested on Metal, CUDA, and Vulkan).
//
// Geometry is read from GGUF metadata (`qwen2.*` / `general.*`), so the same
// builder serves any GGUF dense LLM in the Llama/Qwen family with this layout.

/// Read geometry from a GGUF's metadata and derive the [`LlamaConfig`].
/// Probes for the `qwen2`/`llama` arch prefix and detects the QKV-bias flag
/// from the presence of `blk.0.attn_q.bias`.
pub fn gguf_config(g: &ffai_loader::gguf::Gguf) -> Result<(LlamaConfig, usize, usize)> {
    // Architecture prefix for the geometry keys (e.g. "qwen2", "llama").
    let arch = g.meta_str("general.architecture").unwrap_or("qwen2").to_string();
    let mu = |suffix: &str| -> Option<u32> { g.meta_u32(&format!("{arch}.{suffix}")) };
    let need = |suffix: &str| -> Result<u32> {
        mu(suffix).ok_or_else(|| Error::Msg(format!("gguf: missing metadata {arch}.{suffix}")))
    };

    let hidden = need("embedding_length")? as usize;
    let n_layers = need("block_count")? as usize;
    let n_q_heads = need("attention.head_count")? as usize;
    let n_kv_heads = mu("attention.head_count_kv").unwrap_or(n_q_heads as u32) as usize;
    let intermediate = need("feed_forward_length")? as usize;
    let head_dim = mu("attention.key_length").map(|v| v as usize).unwrap_or(hidden / n_q_heads);
    let rope_theta = g.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(10000.0);
    let eps = g
        .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    // GGUF stores vocab as the token-list length; fall back to the embedding rows.
    let vocab = g
        .metadata_arr_str
        .get("tokenizer.ggml.tokens")
        .map(|v| v.len())
        .or_else(|| g.tensor("token_embd.weight").map(|t| t.dims[t.dims.len() - 1] as usize))
        .ok_or_else(|| Error::Msg("gguf: cannot determine vocab size".into()))?;

    // Qwen2.5 has QKV projection bias and no q/k-norm (vs Qwen3).
    let attn_bias = g.tensor("blk.0.attn_q.bias").is_some();
    let qk_norm = g.tensor("blk.0.attn_q_norm.weight").is_some();

    let cfg = LlamaConfig {
        hidden,
        n_q_heads,
        n_kv_heads,
        head_dim,
        intermediate,
        rope_theta,
        eps,
        qk_norm,
        attn_bias,
    };
    Ok((cfg, n_layers, vocab))
}

/// Build [`ModelWeights`] from a GGUF file (Q8_0/F16/F32 dequantized to f32 and
/// uploaded). Maps the llama.cpp tensor names to the builder's slots and wires
/// the Qwen2.5 QKV biases when `cfg.attn_bias`. Parallel to [`load_qwen3`].
pub fn load_qwen_gguf(
    dev: &dyn Device,
    g: &ffai_loader::gguf::Gguf,
    cfg: &LlamaConfig,
    n_layers: usize,
) -> Result<ModelWeights> {
    // Dequant a GGUF tensor → f32, upload, tag with its on-disk dims.
    let up = |name: &str| -> Result<Tensor> {
        let t = g
            .tensor(name)
            .ok_or_else(|| Error::Msg(format!("gguf: tensor '{name}' not found")))?;
        let data = g.dequant_f32(name)?;
        // GGUF dims are stored fastest-first (col-major-ish): a [out,in] matrix
        // is listed as dims=[in, out]. The f32 buffer is already row-major
        // [out, in] (block r·in .. (r+1)·in), so report shape [out, in].
        let shape: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        let bytes: &[u8] = bytemuck_cast(&data);
        Ok(Tensor::new(dev.upload(bytes)?, shape, ffai_core::DType::F32))
    };

    let embed = up("token_embd.weight")?;
    let final_norm = up("output_norm.weight")?;
    let lm_head = match up("output.weight") {
        Ok(t) => t,
        Err(_) => up("token_embd.weight")?, // tied embeddings
    };

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("blk.{l}");
        layers.push(LayerWeights {
            attn_norm: up(&format!("{p}.attn_norm.weight"))?,
            wq: up(&format!("{p}.attn_q.weight"))?,
            wk: up(&format!("{p}.attn_k.weight"))?,
            wv: up(&format!("{p}.attn_v.weight"))?,
            wo: up(&format!("{p}.attn_output.weight"))?,
            bias_q: if cfg.attn_bias { Some(up(&format!("{p}.attn_q.bias"))?) } else { None },
            bias_k: if cfg.attn_bias { Some(up(&format!("{p}.attn_k.bias"))?) } else { None },
            bias_v: if cfg.attn_bias { Some(up(&format!("{p}.attn_v.bias"))?) } else { None },
            q_norm: if cfg.qk_norm { Some(up(&format!("{p}.attn_q_norm.weight"))?) } else { None },
            k_norm: if cfg.qk_norm { Some(up(&format!("{p}.attn_k_norm.weight"))?) } else { None },
            mlp_norm: up(&format!("{p}.ffn_norm.weight"))?,
            w_gate: up(&format!("{p}.ffn_gate.weight"))?,
            w_up: up(&format!("{p}.ffn_up.weight"))?,
            w_down: up(&format!("{p}.ffn_down.weight"))?,
        });
    }

    Ok(ModelWeights { embed, layers, final_norm, lm_head })
}

/// `&[f32]` → `&[u8]` for upload (little-endian host; the device consumes raw
/// f32 bytes). Plain reinterpret — no external dep.
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid bit patterns; we expose its bytes
    // read-only for the duration of the upload call.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A GGUF-loaded model with a resident multi-position KV cache, exposing a
/// `step(token, pos) -> logits` suitable for `ffai_runtime::generate`. The KV
/// cache is `[n_kv_heads, cap, head_dim]` per layer; each step writes k/v at
/// `pos` and attends over `[0, pos]`. Everything runs in f32 on `dev`.
pub struct GgufModel {
    pub cfg: LlamaConfig,
    pub weights: ModelWeights,
    pub n_layers: usize,
    pub vocab: usize,
    cap: usize,
    // Per-layer (k_cache, v_cache), each [n_kv_heads * cap * head_dim] f32.
    kcache: Vec<Tensor>,
    vcache: Vec<Tensor>,
}

impl GgufModel {
    /// Load from a GGUF path, deriving config from metadata and allocating a
    /// KV cache of `cap` positions.
    pub fn open(dev: &dyn Device, path: &str, cap: usize) -> Result<Self> {
        let g = ffai_loader::gguf::Gguf::open(path)?;
        let (cfg, n_layers, vocab) = gguf_config(&g)?;
        let weights = load_qwen_gguf(dev, &g, &cfg, n_layers)?;
        Self::with_weights(dev, cfg, weights, n_layers, vocab, cap)
    }

    /// Construct from already-loaded weights (lets a caller reuse a parsed Gguf
    /// for the tokenizer without re-loading).
    pub fn with_weights(
        dev: &dyn Device,
        cfg: LlamaConfig,
        weights: ModelWeights,
        n_layers: usize,
        vocab: usize,
        cap: usize,
    ) -> Result<Self> {
        let nkv_hd = cfg.n_kv_heads * cfg.head_dim;
        let zero = vec![0.0f32; nkv_hd * cap];
        let zb = bytemuck_cast(&zero);
        let mut kcache = Vec::with_capacity(n_layers);
        let mut vcache = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            kcache.push(Tensor::new(dev.upload(zb)?, vec![nkv_hd * cap], ffai_core::DType::F32));
            vcache.push(Tensor::new(dev.upload(zb)?, vec![nkv_hd * cap], ffai_core::DType::F32));
        }
        Ok(GgufModel { cfg, weights, n_layers, vocab, cap, kcache, vcache })
    }

    /// One decode step: embed `token`, run every layer with KV-cache attention
    /// over `[0, pos]`, final-norm + lm_head, and download the `[vocab]` f32
    /// logits to the host (so the pure-Rust sampler can pick the next token).
    pub fn step(&self, dev: &dyn Device, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let hd = cfg.head_dim;
        let cap = self.cap;
        let nkv = cfg.n_kv_heads;
        let nq = cfg.n_q_heads;

        let ids = Tensor::new(
            dev.upload(&token.to_le_bytes())?,
            vec![1],
            ffai_core::DType::U32,
        );
        let mut x = ops::gather(dev, &self.weights.embed, &ids)?.reshaped(vec![cfg.hidden]);

        let pb = Tensor::new(dev.upload(&(pos as u32).to_le_bytes())?, vec![1], ffai_core::DType::U32);

        for l in 0..self.n_layers {
            let w = &self.weights.layers[l];
            // ── attention ──────────────────────────────────────────────
            let h = ops::rms_norm(dev, &x, &w.attn_norm, cfg.eps)?;
            let mut q = ops::gemv(dev, &w.wq, &h)?;
            let mut k = ops::gemv(dev, &w.wk, &h)?;
            let mut v = ops::gemv(dev, &w.wv, &h)?;
            if cfg.attn_bias {
                q = ops::add(dev, &q, w.bias_q.as_ref().unwrap())?;
                k = ops::add(dev, &k, w.bias_k.as_ref().unwrap())?;
                v = ops::add(dev, &v, w.bias_v.as_ref().unwrap())?;
            }
            let q = q.reshaped(vec![nq, hd]);
            let k = k.reshaped(vec![nkv, hd]);
            let (q, k) = if cfg.qk_norm {
                (
                    ops::rms_norm(dev, &q, w.q_norm.as_ref().unwrap(), cfg.eps)?,
                    ops::rms_norm(dev, &k, w.k_norm.as_ref().unwrap(), cfg.eps)?,
                )
            } else {
                (q, k)
            };
            let q = ops::rope_llama(dev, &q, pos as u32, cfg.rope_theta, 1.0, 1.0, 1.0, 1e9)?;
            let k = ops::rope_llama(dev, &k, pos as u32, cfg.rope_theta, 1.0, 1.0, 1.0, 1e9)?;
            let v = v.reshaped(vec![nkv, hd]);

            // Write this token's K/V into the cache at `pos`, then attend [0,pos].
            ops::kv_append(dev, &k, &self.kcache[l], &pb, hd, cap, nkv * hd)?;
            ops::kv_append(dev, &v, &self.vcache[l], &pb, hd, cap, nkv * hd)?;
            let len = (pos + 1) as u32;
            let attn = ops::sdpa_decode(
                dev,
                &q,
                &self.kcache[l].reshaped(vec![nkv, cap, hd]),
                &self.vcache[l].reshaped(vec![nkv, cap, hd]),
                hd,
                len,
                cap as u32,
                cfg.heads_per_group(),
                cfg.attn_scale(),
            )?;
            let o = ops::gemv(dev, &w.wo, &attn.reshaped(vec![nq * hd]))?;
            x = ops::add(dev, &x, &o)?;

            // ── MLP (SwiGLU) ───────────────────────────────────────────
            let h2 = ops::rms_norm(dev, &x, &w.mlp_norm, cfg.eps)?;
            let gate = ops::gemv(dev, &w.w_gate, &h2)?;
            let up = ops::gemv(dev, &w.w_up, &h2)?;
            let act = ops::swiglu(dev, &gate, &up)?;
            let down = ops::gemv(dev, &w.w_down, &act)?;
            x = ops::add(dev, &x, &down)?;
        }

        let xn = ops::rms_norm(dev, &x, &self.weights.final_norm, cfg.eps)?;
        let logits = ops::gemv(dev, &self.weights.lm_head, &xn)?;
        dev.synchronize()?;
        let mut bytes = vec![0u8; self.vocab * 4];
        dev.download(logits.buffer.as_ref(), &mut bytes)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
