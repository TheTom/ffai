// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! GGUF v3 parser + Q8_0 dequant vs gguf-py reference (DSv4-Flash checkpoint).
use ffai_loader::gguf::Gguf;

#[test]
fn gguf_parse_and_q8_0_dequant_match_ggufpy() {
    let path = std::env::var("GGUF_PATH").unwrap_or_else(|_| {
        "/Users/tom/models/ds4-model/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf".to_string()
    });
    let Ok(g) = Gguf::open(&path) else {
        eprintln!("no GGUF at {path} — skipping");
        return;
    };
    let bc = g.metadata_u32.get("deepseek4.block_count").copied();
    eprintln!("deepseek4.block_count = {bc:?}, tensors = {}", g.tensor_names().count());
    assert_eq!(bc, Some(43), "block_count metadata");

    let d = g.dequant_f32("blk.0.attn_kv.weight").expect("dequant attn_kv");
    let want = [-0.002768f32, 0.039214, 0.010611, 0.004613, -0.000923, 0.05859, 0.014763, 0.042905];
    eprintln!("rust first8 = {:?}", &d[..8]);
    let mut e = 0.0f32;
    for i in 0..8 {
        e = e.max((d[i] - want[i]).abs());
    }
    assert!(e <= 1e-4, "Q8_0 dequant mismatch vs gguf-py: max|Δ|={e:.3e}");
    eprintln!("✅ GGUF v3 parse + Q8_0 dequant match gguf-py (max|Δ|={e:.1e})");

    // Q2_K dequant vs gguf-py.
    let q2 = g.dequant_f32("blk.0.ffn_down_exps.weight").expect("dequant down_exps");
    let want2 = [-0.017595f32, 0.015516, 0.032072, -0.017595, 0.015516, -0.017595, 0.015516, -0.00104];
    eprintln!("rust Q2_K first8 = {:?}", &q2[..8]);
    let mut e2 = 0.0f32;
    for i in 0..8 {
        e2 = e2.max((q2[i] - want2[i]).abs());
    }
    assert!(e2 <= 1e-4, "Q2_K dequant mismatch vs gguf-py: max|Δ|={e2:.3e}");
    eprintln!("✅ GGUF Q2_K dequant matches gguf-py (max|Δ|={e2:.1e})");
}
