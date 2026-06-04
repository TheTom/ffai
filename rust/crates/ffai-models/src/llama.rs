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
