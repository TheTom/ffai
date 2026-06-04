// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-metal
//!
//! Rust Metal backend (via `metal-rs`, future). NOTE: on Apple hardware the
//! **Swift FFAI engine is the primary, shipping path** (native, fast, runs
//! on every iPhone/iPad/Mac). This Rust Metal backend exists so the
//! *cross-platform Rust engine* can also run on a Mac — chiefly to
//! cross-validate Rust model ports against Swift without a CUDA box. It is
//! lower priority than CUDA/Vulkan/ROCm. Skeleton: stub `Device` impl.

use std::sync::Arc;
use ffai_core::{Backend, Binding, Device, DeviceBuffer, Error, Grid, Kernel, Result};

pub struct MetalDevice {
    name: String,
}

impl MetalDevice {
    /// Probe for a Metal device. Returns `Ok(None)` until the metal-rs impl
    /// lands (so the engine reports the backend as present-but-unbuilt).
    pub fn create() -> Result<Option<Arc<dyn Device>>> {
        Ok(None)
    }
}

impl Device for MetalDevice {
    fn backend(&self) -> Backend { Backend::Metal }
    fn name(&self) -> &str { &self.name }
    fn alloc(&self, _len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        Err(Error::Unimplemented("ffai-metal::alloc (metal-rs pending)"))
    }
    fn upload(&self, _bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>> {
        Err(Error::Unimplemented("ffai-metal::upload"))
    }
    fn download(&self, _buf: &dyn DeviceBuffer, _out: &mut [u8]) -> Result<()> {
        Err(Error::Unimplemented("ffai-metal::download"))
    }
    fn dispatch(&self, _k: &Kernel, _b: &[Binding], _g: Grid) -> Result<()> {
        Err(Error::Unimplemented("ffai-metal::dispatch"))
    }
    fn synchronize(&self) -> Result<()> {
        Err(Error::Unimplemented("ffai-metal::synchronize"))
    }
}
