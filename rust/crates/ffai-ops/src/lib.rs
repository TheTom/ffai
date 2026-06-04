// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-ops
//!
//! The **seam** between model code and kernels — the Rust analog of
//! FFAI-Swift's `Ops/`. Each function takes Tensors, builds or looks up the
//! corresponding metaltile [`Kernel`](ffai_core::Kernel), and dispatches it
//! through the [`Device`](ffai_core::Device) trait. Model code calls these;
//! it never touches a GPU API or a kernel directly.
//!
//! Re-implementing this layer once (here) is what lets the entire model
//! surface above it run on every backend. Skeleton: signatures are defined;
//! bodies land alongside the kernel-IR builders.

use ffai_core::{Device, Error, Result, Tensor};

/// `out = x * rsqrt(mean(x²) + eps) * weight`, row-wise.
pub fn rms_norm(_dev: &dyn Device, _x: &Tensor, _weight: &Tensor, _eps: f32) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::rms_norm"))
}

/// Dense matmul `a @ b`.
pub fn matmul(_dev: &dyn Device, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::matmul"))
}

/// SiLU activation, elementwise.
pub fn silu(_dev: &dyn Device, _x: &Tensor) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::silu"))
}

/// Softmax over the last dim.
pub fn softmax(_dev: &dyn Device, _x: &Tensor) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::softmax"))
}
