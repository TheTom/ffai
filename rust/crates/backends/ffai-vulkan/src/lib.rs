// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-vulkan
//!
//! Vulkan backend — the portable GPU path (AMD/Intel/Android/anything with
//! a Vulkan driver) once metaltile gains a SPIR-V codegen target. Proof
//! that the [`ffai_core::Device`] seam is genuinely backend-agnostic:
//! adding a backend is one crate, not a fork. Skeleton: stub `Device` impl.

use std::sync::Arc;
use ffai_core::{Backend, Binding, Device, DeviceBuffer, Error, Grid, Kernel, Result};

pub struct VulkanDevice {
    name: String,
}

impl VulkanDevice {
    pub fn create() -> Result<Option<Arc<dyn Device>>> {
        Ok(None)
    }
}

impl Device for VulkanDevice {
    fn backend(&self) -> Backend { Backend::Vulkan }
    fn name(&self) -> &str { &self.name }
    fn alloc(&self, _len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        Err(Error::Unimplemented("ffai-vulkan::alloc (SPIR-V codegen pending)"))
    }
    fn upload(&self, _bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>> {
        Err(Error::Unimplemented("ffai-vulkan::upload"))
    }
    fn download(&self, _buf: &dyn DeviceBuffer, _out: &mut [u8]) -> Result<()> {
        Err(Error::Unimplemented("ffai-vulkan::download"))
    }
    fn dispatch(&self, _k: &Kernel, _b: &[Binding], _g: Grid) -> Result<()> {
        Err(Error::Unimplemented("ffai-vulkan::dispatch"))
    }
    fn synchronize(&self) -> Result<()> {
        Err(Error::Unimplemented("ffai-vulkan::synchronize"))
    }
}
