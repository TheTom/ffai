// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! End-to-end **Phi-3** GGUF load + generate on the Apple GPU. Phi-3 ships
//! arch="phi3" with FUSED projection tensors — `blk.N.attn_qkv.weight`
//! (Q‖K‖V stacked) and `blk.N.ffn_up.weight` (gate‖up stacked) — and the
//! Llama/SentencePiece tokenizer (no merges). This exercises the new
//! fused-tensor split (`load_phi3_gguf` / `load_phi3_gguf_q8`) plus the SPM
//! tokenizer path. Once the fused tensors are split, the same Qwen f32/Q8
//! forward graph serves Phi-3 — backend-agnostic, so this runs on CUDA/Vulkan
//! too. macOS + Metal here; skips elsewhere or if the model file is absent.

use ffai_loader::gguf::Gguf;
use ffai_metal::MetalDevice;
use ffai_models::gguf_tokenizer::GgufTokenizer;
use ffai_models::llama::GgufModel;
use ffai_runtime::{generate, Sampling, StopOn};
use std::time::Instant;

fn model_path() -> String {
    std::env::var("PHI3_GGUF")
        .unwrap_or_else(|_| "/Users/tom/models/Phi-3.1-mini-128k-instruct-Q4_K_M_2.gguf".to_string())
}

#[test]
fn phi3_mini_gguf_generate_on_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found at {path} — skipping");
        return;
    }
    eprintln!("device: {} — loading Phi-3 GGUF {path}", dev.name());

    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    eprintln!(
        "arch={:?} tokenizer={:?} vocab={}",
        g.meta_str("general.architecture"),
        g.meta_str("tokenizer.ggml.model"),
        tok.vocab_size(),
    );

    let t_load = Instant::now();
    let model = GgufModel::open(dev.as_ref(), &path, 256).expect("load phi3 gguf (fused-split)");
    eprintln!(
        "model loaded in {:.2}s — layers={} hidden={} q_heads={} kv_heads={} head_dim={} bias={} theta={} eps={}",
        t_load.elapsed().as_secs_f32(),
        model.n_layers,
        model.cfg.hidden,
        model.cfg.n_q_heads,
        model.cfg.n_kv_heads,
        model.cfg.head_dim,
        model.cfg.attn_bias,
        model.cfg.rope_theta,
        model.cfg.eps,
    );

    // Plain completion (base-style continuation, no chat template).
    let prompt_text = "The capital of France is";
    let prompt = tok.encode(prompt_text);
    eprintln!("prompt {prompt_text:?} → {} ids: {prompt:?}", prompt.len());
    assert!(!prompt.is_empty(), "tokenizer produced no ids");

    let eos = tok.token_id("<|endoftext|>").or_else(|| tok.token_id("<|end|>"));
    let stop = StopOn { max_new: 16, eos };

    let t_gen = Instant::now();
    let mut decode_steps = 0usize;
    let out = generate(&prompt, &stop, &Sampling::Greedy, 0, |token, pos| {
        if pos + 1 > prompt.len() {
            decode_steps += 1;
        }
        model.step(dev.as_ref(), token, pos).expect("step")
    });
    let secs = t_gen.elapsed().as_secs_f32();
    let total_steps = prompt.len() + out.len();
    let decode_tps = decode_steps as f32 / secs.max(1e-6);

    let text = tok.decode(&out);
    eprintln!(
        "generated {} tokens in {:.2}s ({:.1} tok/s incl. prefill, {:.1} tok/s decode-only)",
        out.len(),
        secs,
        total_steps as f32 / secs,
        decode_tps,
    );
    eprintln!("OUTPUT: {text:?}");
    eprintln!("FULL:   {prompt_text:?}{text}");

    assert!(
        text.contains("Paris"),
        "expected 'Paris' in continuation, got {text:?}"
    );
    eprintln!("✅ Phi-3-mini Q4_K_M fused-tensor GGUF loaded + generated coherent output on Apple GPU.");
}

/// Resident-Q8 decode of Phi-3 — the fused tensors are split AND re-quantized to
/// Q8, then decode runs through `gemv_q8` (no f32 big-matmul upload). Asserts the
/// Q8 path stays coherent ("Paris") and reports decode tok/s.
#[test]
fn phi3_mini_gguf_q8_resident_decode_on_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found at {path} — skipping");
        return;
    }
    let d = dev.as_ref();
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");

    let prompt_text = "The capital of France is";
    let prompt = tok.encode(prompt_text);
    let eos = tok.token_id("<|endoftext|>").or_else(|| tok.token_id("<|end|>"));
    let stop = StopOn { max_new: 16, eos };

    let t_load = Instant::now();
    let m_q8 = GgufModel::open_q8(d, &path, 256).expect("load phi3 q8 (fused-split + repack)");
    eprintln!("phi3 q8 loaded in {:.2}s", t_load.elapsed().as_secs_f32());
    let _ = m_q8.step(d, prompt[0], 0).expect("q8 warmup");

    let t = Instant::now();
    let mut decode_steps = 0usize;
    let out = generate(&prompt, &stop, &Sampling::Greedy, 0, |token, pos| {
        if pos + 1 > prompt.len() {
            decode_steps += 1;
        }
        m_q8.step(d, token, pos).expect("q8 step")
    });
    let secs = t.elapsed().as_secs_f32();
    let tps = decode_steps as f32 / secs.max(1e-6);
    let text = tok.decode(&out);
    eprintln!("Q8 OUTPUT: {text:?} ({tps:.1} tok/s decode-only)");
    eprintln!("Q8 FULL:   {prompt_text:?}{text}");

    assert!(
        text.contains("Paris"),
        "expected 'Paris' in Q8 continuation, got {text:?}"
    );
    eprintln!("✅ Phi-3-mini resident-Q8 fused-tensor GGUF decode coherent on Apple GPU.");
}
