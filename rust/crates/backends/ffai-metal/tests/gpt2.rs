// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! GPT-2 vs HF — thin wrapper; the forward lives once in ffai-modeltests.
use ffai_metal::MetalDevice;
#[test]
fn gpt2_124m_full_forward_vs_hf() {
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    ffai_modeltests::verify_gpt2(dev.as_ref(), "Apple GPU");
}
