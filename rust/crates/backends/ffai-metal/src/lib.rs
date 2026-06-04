// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-metal
//!
//! Metal backend for the FFAI engine. Wraps `metaltile_runtime::Context`
//! (the public Metal face — MSL JIT, PSO cache, dispatch) behind the shared
//! [`ffai_core::Device`] trait, so the **same Rust models run on Apple GPUs
//! (and iOS)** as on CUDA. This is the half that makes "models shared across
//! CUDA *and* Metal" real.
//!
//! Buffers are host-shadowed (`Vec<u8>`): the Metal `Context` dispatch model
//! is host-bytes-in / host-bytes-out, so a tensor carries its bytes and each
//! dispatch round-trips through the GPU. Correctness-first; a resident-buffer
//! fast path is a later optimization.

use ffai_core::{Backend, Binding, Device, DeviceBuffer, Error, Grid, Kernel, Result};
use metaltile_runtime::{Context, MetalTileError};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

fn err(e: MetalTileError) -> Error {
    Error::Dispatch(e.to_string())
}

/// Host-shadowed buffer. The bytes are the source of truth; dispatch uploads
/// them and writes outputs back here.
pub struct MetalBuffer {
    data: RwLock<Vec<u8>>,
}
impl DeviceBuffer for MetalBuffer {
    fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Metal device: a metaltile `Context` behind the shared `Device` trait.
pub struct MetalDevice {
    ctx: Mutex<Context>,
    name: String,
}
// Context holds objc2 (`!Send`) types on macOS; we serialize all access
// through the Mutex and only ever submit from one logical owner, so the
// manual Send/Sync are sound for this usage.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    /// Probe for a Metal GPU; `Ok(None)` if none (e.g. off Apple silicon).
    pub fn create() -> Result<Option<Arc<dyn Device>>> {
        let ctx = Context::new().map_err(err)?;
        if !ctx.has_gpu() {
            return Ok(None);
        }
        let name = format!("Apple GPU (family {:?})", ctx.gpu_family());
        Ok(Some(Arc::new(MetalDevice { ctx: Mutex::new(ctx), name })))
    }

    fn shadow(b: &Arc<dyn DeviceBuffer>) -> Result<&MetalBuffer> {
        b.as_any()
            .downcast_ref::<MetalBuffer>()
            .ok_or_else(|| Error::Msg("metal: binding is not a MetalBuffer".into()))
    }
}

impl Device for MetalDevice {
    fn backend(&self) -> Backend {
        Backend::Metal
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn alloc(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        Ok(Arc::new(MetalBuffer { data: RwLock::new(vec![0u8; len]) }))
    }

    fn upload(&self, bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>> {
        Ok(Arc::new(MetalBuffer { data: RwLock::new(bytes.to_vec()) }))
    }

    fn download(&self, buf: &dyn DeviceBuffer, out: &mut [u8]) -> Result<()> {
        let mb = buf
            .as_any()
            .downcast_ref::<MetalBuffer>()
            .ok_or_else(|| Error::Msg("metal: download buffer is not a MetalBuffer".into()))?;
        let src = mb.data.read().unwrap();
        let n = out.len().min(src.len());
        out[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    fn dispatch(&self, kernel: &Kernel, bindings: &[Binding], grid: Grid) -> Result<()> {
        let n_params = kernel.params.len();
        // Map every binding to its parameter/constexpr NAME (Context keys by
        // name): tensor params first, then constexprs — same contract as the
        // CUDA backend and the kernel-test corpus.
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (i, b) in bindings.iter().enumerate() {
            let name = if i < n_params {
                kernel.params[i].name.clone()
            } else {
                kernel.constexprs[i - n_params].name.name().to_string()
            };
            let bytes = match b {
                Binding::Buffer(buf) => Self::shadow(buf)?.data.read().unwrap().clone(),
                Binding::Scalar(s) => s.clone(),
            };
            buffers.insert(name, bytes);
        }

        let g = [grid.grid[0] as usize, grid.grid[1] as usize, grid.grid[2] as usize];
        let t = [grid.block[0] as usize, grid.block[1] as usize, grid.block[2] as usize];

        let result = self
            .ctx
            .lock()
            .unwrap()
            .dispatch_with_grid(kernel, &buffers, &BTreeMap::new(), g, t)
            .map_err(err)?;

        // Copy each output param's result back into its binding's shadow.
        for (i, p) in kernel.params.iter().enumerate() {
            if p.is_output {
                if let Some(out_bytes) = result.output(&p.name) {
                    if let Binding::Buffer(buf) = &bindings[i] {
                        let mb = Self::shadow(buf)?;
                        let mut w = mb.data.write().unwrap();
                        let n = w.len().min(out_bytes.len());
                        w[..n].copy_from_slice(&out_bytes[..n]);
                    }
                }
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // Context dispatch is synchronous (waits for completion), so there is
        // nothing outstanding to wait on.
        Ok(())
    }
}
