// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-models
//!
//! Model forward passes as plain Rust over [`ffai_ops`]. This is the big
//! surface ported from FFAI-Swift `Models/` (~35 families). Each model is
//! validated bit-for-bit against the working Swift implementation (the
//! oracle) before it is trusted. Skeleton: the [`Model`] trait only.

use ffai_core::{Device, Result, Tensor};

/// A loaded model: weights resident on some [`Device`], able to run a
/// forward pass. Backend-neutral — the same impl runs on Metal/CUDA/Vulkan.
pub trait Model: Send + Sync {
    fn name(&self) -> &str;

    /// Run one forward pass over `tokens`, returning next-token logits.
    fn forward(&self, dev: &dyn Device, tokens: &[u32]) -> Result<Tensor>;
}
