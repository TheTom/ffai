// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-cuda
//!
//! CUDA backend. The real impl (under `--features cuda`) wraps
//! `metaltile_runtime::CudaDevice` — the NVRTC -> PTX -> driver path that
//! already runs the full kernel corpus bit-accurately on GB10 — behind the
//! [`ffai_core::Device`] trait. Without the feature the crate is a stub so
//! non-CUDA hosts still build the workspace. Skeleton: stub `Device` impl;
//! the metaltile delegation lands next.

use std::sync::Arc;
use ffai_core::{Backend, Binding, Device, DeviceBuffer, Error, Grid, Kernel, Result};

pub struct CudaDevice {
    name: String,
}

impl CudaDevice {
    /// Probe for a CUDA device. Returns `Ok(None)` until the
    /// metaltile-runtime delegation lands.
    pub fn create() -> Result<Option<Arc<dyn Device>>> {
        Ok(None)
    }
}

impl Device for CudaDevice {
    fn backend(&self) -> Backend { Backend::Cuda }
    fn name(&self) -> &str { &self.name }
    fn alloc(&self, _len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        Err(Error::Unimplemented("ffai-cuda::alloc (metaltile-runtime delegation pending)"))
    }
    fn upload(&self, _bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>> {
        Err(Error::Unimplemented("ffai-cuda::upload"))
    }
    fn download(&self, _buf: &dyn DeviceBuffer, _out: &mut [u8]) -> Result<()> {
        Err(Error::Unimplemented("ffai-cuda::download"))
    }
    fn dispatch(&self, _k: &Kernel, _b: &[Binding], _g: Grid) -> Result<()> {
        Err(Error::Unimplemented("ffai-cuda::dispatch"))
    }
    fn synchronize(&self) -> Result<()> {
        Err(Error::Unimplemented("ffai-cuda::synchronize"))
    }
}
