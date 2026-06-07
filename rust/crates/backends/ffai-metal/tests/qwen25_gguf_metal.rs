// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! End-to-end GGUF small-model load + generate on the Apple GPU. Loads
//! Qwen2.5-1.5B-Instruct **Q8_0 GGUF** (geometry + Q8_0 weights + BPE
//! tokenizer all read from the one file), runs the KV-cache decode loop, and
//! checks the continuation of "The capital of France is" contains "Paris".
//!
//! This is the proof that the Rust stack loads a standard GGUF end-to-end —
//! backend-agnostic, so the same path runs on CUDA / Vulkan. Runs on macOS
//! with a Metal GPU; skips elsewhere or if the model file is absent.

use ffai_loader::gguf::Gguf;
use ffai_metal::MetalDevice;
use ffai_models::gguf_tokenizer::GgufTokenizer;
use ffai_models::llama::GgufModel;
use ffai_runtime::{generate, Sampling, StopOn};
use std::time::Instant;

#[test]
fn qwen25_1_5b_gguf_generate_on_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let path = std::env::var("QWEN25_GGUF")
        .unwrap_or_else(|_| "/Users/tom/models/qwen2.5-1.5b-instruct-q8_0.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found at {path} — skipping");
        return;
    }
    eprintln!("device: {} — loading GGUF {path}", dev.name());

    // Parse once: tokenizer reads the same Gguf; model re-parses for weights.
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    eprintln!("arch={:?} vocab={}", g.meta_str("general.architecture"), tok.vocab_size());

    let t_load = Instant::now();
    let model = GgufModel::open(dev.as_ref(), &path, 256).expect("load gguf model");
    eprintln!(
        "model loaded in {:.2}s — layers={} hidden={} q_heads={} kv_heads={} head_dim={} bias={} theta={}",
        t_load.elapsed().as_secs_f32(),
        model.n_layers,
        model.cfg.hidden,
        model.cfg.n_q_heads,
        model.cfg.n_kv_heads,
        model.cfg.head_dim,
        model.cfg.attn_bias,
        model.cfg.rope_theta,
    );

    // Plain completion prompt (no chat template) — base-model style continuation.
    let prompt_text = "The capital of France is";
    let prompt = tok.encode(prompt_text);
    eprintln!("prompt {:?} → {} ids: {:?}", prompt_text, prompt.len(), prompt);
    assert!(!prompt.is_empty(), "tokenizer produced no ids");

    let eos = tok.token_id("<|endoftext|>");
    let stop = StopOn { max_new: 12, eos };

    let t_gen = Instant::now();
    let mut n_steps = 0usize;
    let out = generate(&prompt, &stop, &Sampling::Greedy, 0, |token, pos| {
        n_steps += 1;
        model.step(dev.as_ref(), token, pos).expect("step")
    });
    let secs = t_gen.elapsed().as_secs_f32();
    let total_steps = prompt.len() + out.len(); // prefill steps + decode steps
    let _ = n_steps;

    let text = tok.decode(&out);
    eprintln!("generated {} tokens in {:.2}s ({:.1} tok/s incl. prefill)", out.len(), secs, total_steps as f32 / secs);
    eprintln!("OUTPUT: {:?}", text);
    eprintln!("FULL:   {:?}{}", prompt_text, text);

    assert!(
        text.contains("Paris"),
        "expected 'Paris' in continuation, got {text:?}"
    );
    eprintln!("✅ Qwen2.5-1.5B-Q8 GGUF loaded + generated coherent output on Apple GPU.");
}

/// End-to-end load + generate of **Qwen2.5-7B-Instruct Q4_K_M** — the common
/// k-quant format (Q4_K/Q5_K/Q6_K super-blocks) and a 2-part split GGUF. This
/// exercises the new k-quant dequant + split-file handling in the loader. Uses
/// the f32 dequant-to-upload weight path. Set `QWEN25_7B_GGUF` to part 1 of the
/// split (`*-00001-of-00002.gguf`); the loader opens both parts automatically.
#[test]
fn qwen25_7b_q4km_gguf_generate_on_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let path = std::env::var("QWEN25_7B_GGUF").unwrap_or_else(|_| {
        "/Users/tom/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf".to_string()
    });
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found at {path} — skipping");
        return;
    }
    eprintln!("device: {} — loading split Q4_K_M GGUF {path}", dev.name());

    let g = Gguf::open(&path).expect("open split gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    eprintln!(
        "arch={:?} vocab={} tensors={}",
        g.meta_str("general.architecture"),
        tok.vocab_size(),
        g.tensor_names().count(),
    );

    let t_load = Instant::now();
    let model = GgufModel::open(dev.as_ref(), &path, 256).expect("load q4_k_m model");
    eprintln!(
        "model loaded in {:.2}s — layers={} hidden={} q_heads={} kv_heads={} head_dim={} bias={} theta={}",
        t_load.elapsed().as_secs_f32(),
        model.n_layers,
        model.cfg.hidden,
        model.cfg.n_q_heads,
        model.cfg.n_kv_heads,
        model.cfg.head_dim,
        model.cfg.attn_bias,
        model.cfg.rope_theta,
    );

    let prompt_text = "The capital of France is";
    let prompt = tok.encode(prompt_text);
    eprintln!("prompt {:?} → {} ids: {:?}", prompt_text, prompt.len(), prompt);
    assert!(!prompt.is_empty(), "tokenizer produced no ids");

    let eos = tok.token_id("<|endoftext|>");
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
    eprintln!("✅ Qwen2.5-7B-Q4_K_M split GGUF loaded + generated coherent output on Apple GPU.");
}

/// Resident-Q8 decode path: weights stay quantized (Q8) and decode runs through
/// `gemv_q8` (no f32 weight upload for the big matmuls). Runs the SAME prompt
/// through both the f32 (`open`) and Q8 (`open_q8`) paths, reports decode tok/s
/// for each, and asserts the Q8 path stays coherent ("Paris"). The Q8 token
/// stream should match or near-match the f32 path (Q8 matmul is a slight
/// numerical change). Backend-agnostic: gemv_q8 codegens to CUDA/Vulkan too.
#[test]
fn qwen25_1_5b_gguf_q8_resident_decode_on_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let path = std::env::var("QWEN25_GGUF")
        .unwrap_or_else(|_| "/Users/tom/models/qwen2.5-1.5b-instruct-q8_0.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found at {path} — skipping");
        return;
    }
    let d = dev.as_ref();
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");

    let prompt_text = "The capital of France is";
    let prompt = tok.encode(prompt_text);
    let eos = tok.token_id("<|endoftext|>");
    let stop = StopOn { max_new: 12, eos };

    // ── f32 path (baseline) ───────────────────────────────────────────────
    let m_f32 = GgufModel::open(d, &path, 256).expect("load f32");
    // Warm the JIT (first dispatch of each kernel compiles) so tok/s is steady-state.
    let _ = m_f32.step(d, prompt[0], 0).expect("f32 warmup");
    let t = Instant::now();
    let mut decode_steps_f32 = 0usize;
    let out_f32 = generate(&prompt, &stop, &Sampling::Greedy, 0, |token, pos| {
        if pos + 1 > prompt.len() { decode_steps_f32 += 1; }
        m_f32.step(d, token, pos).expect("f32 step")
    });
    let f32_secs = t.elapsed().as_secs_f32();
    let f32_tps = decode_steps_f32 as f32 / f32_secs.max(1e-6);
    let f32_text = tok.decode(&out_f32);
    drop(m_f32);

    // ── Q8 path (resident-quantized weights) ──────────────────────────────
    let m_q8 = GgufModel::open_q8(d, &path, 256).expect("load q8");
    let _ = m_q8.step(d, prompt[0], 0).expect("q8 warmup");
    let t = Instant::now();
    let mut decode_steps_q8 = 0usize;
    let out_q8 = generate(&prompt, &stop, &Sampling::Greedy, 0, |token, pos| {
        if pos + 1 > prompt.len() { decode_steps_q8 += 1; }
        m_q8.step(d, token, pos).expect("q8 step")
    });
    let q8_secs = t.elapsed().as_secs_f32();
    let q8_tps = decode_steps_q8 as f32 / q8_secs.max(1e-6);
    let q8_text = tok.decode(&out_q8);

    eprintln!("─── decode tok/s (decode-only, greedy) ───");
    eprintln!("  f32 gemv : {f32_tps:6.2} tok/s  ({decode_steps_f32} steps / {f32_secs:.2}s)");
    eprintln!("  Q8  gemv : {q8_tps:6.2} tok/s  ({decode_steps_q8} steps / {q8_secs:.2}s)");
    eprintln!("  speedup  : {:.2}×", q8_tps / f32_tps.max(1e-6));
    eprintln!("f32 OUTPUT: {f32_text:?}");
    eprintln!("Q8  OUTPUT: {q8_text:?}");
    let matched = out_f32 == out_q8;
    eprintln!("token streams identical: {matched}");

    assert!(q8_text.contains("Paris"), "Q8 path lost coherence, got {q8_text:?}");
    eprintln!("✅ Resident-Q8 decode coherent on Apple GPU (weights kept Q8, gemv_q8 path).");
}
