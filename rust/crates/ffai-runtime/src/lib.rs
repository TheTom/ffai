// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-runtime
//!
//! Generation orchestration: KV cache, sampler, the prefill/decode loop.
//! Backend-neutral logic over [`ffai_core::Device`] + [`ffai_models`].
//! Skeleton.

#![allow(dead_code)]

/// Decoding parameters for a generation request.
#[derive(Debug, Clone)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for SampleParams {
    fn default() -> Self {
        SampleParams { temperature: 0.7, top_p: 0.95, max_tokens: 256 }
    }
}
