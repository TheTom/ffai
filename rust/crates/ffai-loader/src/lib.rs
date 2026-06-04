// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-loader
//!
//! Weight loaders (GGUF / SafeTensors / HF). Pure CPU byte-parsing +
//! upload through the [`Device`](ffai_core::Device) trait — no GPU API,
//! fully shared across backends. Skeleton.

#![allow(dead_code)]

/// Supported on-disk weight formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Gguf,
    SafeTensors,
    HuggingFace,
}
