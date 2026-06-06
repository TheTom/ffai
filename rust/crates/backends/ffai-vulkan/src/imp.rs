// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Real Vulkan backend (compiled under `--features vulkan`). Wraps
//! `metaltile_runtime::VulkanDevice` — the GlslGenerator → shaderc → SPIR-V →
//! VkPipeline path that runs the kernel corpus bit-accurately on any Vulkan
//! 1.2+ device (validated 4176/4176 on AMD RX 9070 XT / RDNA4) — behind the
//! shared [`ffai_core::Device`] trait.
//!
//! ## Why a host-shadow buffer model (and not CUDA's persistent-device model)
//!
//! The CUDA backend (`ffai-cuda`) keeps tensors resident in VRAM: its
//! `alloc`/`upload` return a raw `CUdeviceptr` handle, and `dispatch` binds
//! those pre-uploaded device buffers to a cached module via
//! `launch_async(func, grid, block, args)`. That requires three things from
//! the runtime that metaltile's **public** Vulkan API does not (yet) expose:
//!
//!   1. a raw-handle allocator (`alloc_raw` / free-by-handle),
//!   2. publicly readable buffer handles (metaltile's `VulkanBuffer` fields
//!      `buffer`/`memory` are `pub(crate)`, and the struct is borrow-bound to
//!      the device as `VulkanBuffer<'d>`, so it can't live in a `'static`
//!      `Arc<dyn DeviceBuffer>`), and
//!   3. a "bind pre-uploaded buffers + dispatch" seam.
//!
//! metaltile's only public dispatch entry point is the monolithic
//! `VulkanDevice::run_kernel(kernel, BTreeMap<name, host_bytes>, grid, block)`,
//! which uploads every param from host bytes, dispatches, reads the outputs
//! back, and frees the device buffers — all in one call.
//!
//! So this backend models `DeviceBuffer` as a **host-side shadow** (`Vec<u8>`
//! in RAM). `alloc`/`upload`/`download` are pure host ops; `dispatch` collects
//! the bound shadows into the name→bytes map, calls `run_kernel` (which does
//! the actual H2D → compute → D2H round trip on the GPU), then copies the
//! returned outputs back into the corresponding shadow buffers so a later
//! `download` observes them.
//!
//! This is correct and runs real kernels on the GPU, but it re-uploads every
//! bound buffer to VRAM on each `dispatch` (no cross-dispatch residency) and
//! recompiles the pipeline per call. It is the right shape for op-level
//! validation and a portable correctness path; a perf-grade path needs
//! metaltile to expose a persistent VkBuffer + cached-pipeline dispatch seam
//! (TODO: upstream a `VulkanDevice::{alloc_raw, run_pipeline_bound}` API and
//! switch this backend to the CUDA-style resident-buffer model).

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use ffai_core::{Backend, Binding, Device, DeviceBuffer, Error, Grid, Kernel, Result};
use metaltile_core::ir::ParamKind;
use metaltile_runtime::{MetalTileError, VulkanDevice as MtVulkanDevice};

fn dispatch_err(e: MetalTileError) -> Error {
    Error::Dispatch(e.to_string())
}

/// Vulkan implementation of the shared [`Device`] trait. Holds the metaltile
/// runtime device in an `Arc` so it outlives every buffer/dispatch.
pub struct VulkanDevice {
    dev: Arc<MtVulkanDevice>,
    name: String,
}

impl VulkanDevice {
    /// Probe for a Vulkan device; `Ok(None)` if no loader / device is present.
    pub fn create() -> Result<Option<Arc<dyn Device>>> {
        match MtVulkanDevice::create().map_err(dispatch_err)? {
            Some(d) => {
                let name = format!("Vulkan device (qfam={})", d.queue_family());
                Ok(Some(Arc::new(VulkanDevice { dev: Arc::new(d), name })))
            }
            None => Ok(None),
        }
    }
}

/// A host-shadow allocation: the bytes live in RAM and are streamed to VRAM
/// per-dispatch by `run_kernel`. `Mutex` gives interior mutability so a
/// dispatch can write outputs back through a shared `&dyn DeviceBuffer`.
pub struct VulkanBuffer {
    bytes: Mutex<Vec<u8>>,
    len: usize,
}

impl VulkanBuffer {
    fn new(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        VulkanBuffer { bytes: Mutex::new(bytes), len }
    }
    /// Snapshot the current bytes (used to build the run_kernel input map).
    fn snapshot(&self) -> Vec<u8> {
        self.bytes.lock().unwrap().clone()
    }
    /// Overwrite the shadow with output bytes from a dispatch.
    fn store(&self, data: &[u8]) {
        let mut g = self.bytes.lock().unwrap();
        let n = g.len().min(data.len());
        g[..n].copy_from_slice(&data[..n]);
    }
}

impl DeviceBuffer for VulkanBuffer {
    fn len(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Device for VulkanDevice {
    fn backend(&self) -> Backend {
        Backend::Vulkan
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn alloc(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        Ok(Arc::new(VulkanBuffer::new(vec![0u8; len])))
    }

    fn upload(&self, bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>> {
        Ok(Arc::new(VulkanBuffer::new(bytes.to_vec())))
    }

    fn download(&self, buf: &dyn DeviceBuffer, out: &mut [u8]) -> Result<()> {
        let vb = buf
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| Error::Msg("download: buffer is not a VulkanBuffer".into()))?;
        let g = vb.bytes.lock().unwrap();
        let n = out.len().min(g.len());
        out[..n].copy_from_slice(&g[..n]);
        Ok(())
    }

    fn dispatch(&self, kernel: &Kernel, bindings: &[Binding], grid: Grid) -> Result<()> {
        // Map positional bindings (signature order) onto the named buffer map
        // `run_kernel` expects: Buffer bindings → `kernel.params` in order,
        // Scalar bindings → `kernel.constexprs` in order. `run_kernel` itself
        // synthesizes the `_n_elems` push-constant and any strided
        // `_shape`/`_strides` companion buffers, so we don't add those here.
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();

        // Keep handles to output shadow buffers so we can write results back.
        let mut output_shadows: Vec<(String, &VulkanBuffer)> = Vec::new();

        let mut param_i = 0usize;
        let mut const_i = 0usize;
        for b in bindings {
            match b {
                Binding::Buffer(buf) => {
                    let p = kernel.params.get(param_i).ok_or_else(|| {
                        Error::Msg(format!(
                            "dispatch: more buffer bindings than kernel params ({})",
                            kernel.params.len()
                        ))
                    })?;
                    let vb = buf
                        .as_any()
                        .downcast_ref::<VulkanBuffer>()
                        .ok_or_else(|| {
                            Error::Msg("dispatch: binding is not a VulkanBuffer".into())
                        })?;
                    buffers.insert(p.name.clone(), vb.snapshot());
                    if p.is_output {
                        output_shadows.push((p.name.clone(), vb));
                    }
                    // `Strided` params carry companion `_shape`/`_strides`
                    // buffers; metaltile synthesizes them from `p.shape` when
                    // absent, so positional binding stays 1:1 with params here.
                    debug_assert!(
                        p.kind != ParamKind::Scalar,
                        "scalar param bound as a buffer"
                    );
                    param_i += 1;
                }
                Binding::Scalar(scalar_bytes) => {
                    let ce = kernel.constexprs.get(const_i).ok_or_else(|| {
                        Error::Msg(format!(
                            "dispatch: more scalar bindings than kernel constexprs ({})",
                            kernel.constexprs.len()
                        ))
                    })?;
                    buffers.insert(ce.name.name().to_string(), scalar_bytes.clone());
                    const_i += 1;
                }
            }
        }

        let out = self
            .dev
            .run_kernel(kernel, &buffers, grid.grid, grid.block)
            .map_err(dispatch_err)?;

        // Copy outputs back into the host shadows so `download` sees them.
        for (name, shadow) in output_shadows {
            if let Some(result) = out.get(&name) {
                shadow.store(result);
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // `run_kernel` already submits + `vkQueueWaitIdle`s synchronously, so
        // by the time any dispatch returns the work is complete and the host
        // shadows are up to date. Nothing outstanding to wait on.
        Ok(())
    }
}
