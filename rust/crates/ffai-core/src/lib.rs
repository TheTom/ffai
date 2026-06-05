// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-core
//!
//! Backend-neutral primitives for the FFAI inference engine. This crate
//! defines the single seam every backend implements — the [`Device`]
//! trait — plus the [`Tensor`] handle and the dtype/buffer/binding types
//! that flow through it unchanged on Metal, CUDA, Vulkan, or ROCm.
//!
//! ## How code is shared
//!
//! Kernels are shared via the **metaltile IR** ([`Kernel`], re-exported
//! from `metaltile-core`). A model op builds or looks up a `Kernel` and
//! hands it to [`Device::dispatch`]; the backend lowers that IR to its
//! target language (MSL / CUDA C++ / SPIR-V) and launches it. So:
//!
//! - **Above** this trait (models, ops, loaders, KV cache, sampler) is
//!   plain Rust that never names a GPU API — written once, runs everywhere.
//! - **Below** it, each backend is a thin [`Device`] impl.
//! - The kernels themselves are generated once by metaltile.

use std::any::Any;
use std::sync::Arc;

mod error;
pub use error::{Error, Result};

// Re-export the shared kernel IR + dtype so the whole engine speaks one
// vocabulary and nothing downstream depends on metaltile-core directly.
pub use metaltile_core::dtype::DType;
pub use metaltile_core::ir::Kernel;

/// Which accelerator family a [`Device`] targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Metal,
    Cuda,
    Vulkan,
    Rocm,
    Cpu,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Rocm => "rocm",
            Backend::Cpu => "cpu",
        }
    }
}

/// An opaque device-side allocation. Each backend returns its own concrete
/// type (an `MTLBuffer` wrapper, a `CUdeviceptr` wrapper, a `VkBuffer`
/// wrapper) behind this trait, so [`Tensor`] stays backend-agnostic.
pub trait DeviceBuffer: Send + Sync {
    /// Length in bytes.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Escape hatch so backend code can downcast to its concrete buffer
    /// type when it needs the native handle for a launch.
    fn as_any(&self) -> &dyn Any;
}

/// A single kernel argument in signature order: either a device buffer or a
/// small by-value scalar/constexpr (little-endian bytes).
#[derive(Clone)]
pub enum Binding {
    Buffer(Arc<dyn DeviceBuffer>),
    Scalar(Vec<u8>),
}

/// Launch geometry: grid (blocks) × block (threads-per-block), 3-D. Maps
/// onto both CUDA's grid/block and Metal's threadgroups / threads-per-tg.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

impl Grid {
    /// 1-D launch: `blocks` threadgroups of `threads` lanes each.
    pub fn d1(blocks: u32, threads: u32) -> Self {
        Grid { grid: [blocks, 1, 1], block: [threads, 1, 1] }
    }
}

/// The one interface every backend implements. Object-safe so the engine
/// holds `Arc<dyn Device>` and dispatches without knowing the hardware.
pub trait Device: Send + Sync {
    fn backend(&self) -> Backend;
    /// Human-readable device name (e.g. `"Apple M5 Max"`, `"NVIDIA GB10"`).
    fn name(&self) -> &str;

    /// Allocate `len` bytes of uninitialized device memory.
    fn alloc(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>>;

    /// Allocate + upload host bytes in one shot.
    fn upload(&self, bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>>;

    /// Copy device memory back into a host slice.
    fn download(&self, buf: &dyn DeviceBuffer, out: &mut [u8]) -> Result<()>;

    /// Lower `kernel` (shared metaltile IR) for this backend and launch it
    /// over `grid` with `bindings` in signature order.
    fn dispatch(&self, kernel: &Kernel, bindings: &[Binding], grid: Grid) -> Result<()>;

    /// Block until all submitted work has completed.
    fn synchronize(&self) -> Result<()>;

    /// CUDA-graph capture (megakernel). `begin_capture` starts recording stream
    /// work; run an all-device (no host-sync) sequence; `end_capture` returns an
    /// opaque executable-graph handle; `graph_launch` replays it as ONE launch.
    /// Default impls error (backend without graph support). Returns handle as u64.
    fn begin_capture(&self) -> Result<()> {
        Err(Error::Msg("graph capture unsupported on this backend".into()))
    }
    fn end_capture(&self) -> Result<u64> {
        Err(Error::Msg("graph capture unsupported on this backend".into()))
    }
    fn graph_launch(&self, _exec: u64) -> Result<()> {
        Err(Error::Msg("graph capture unsupported on this backend".into()))
    }
}

/// A handle to a region of device memory + shape + dtype. Backend-neutral:
/// the buffer is an `Arc<dyn DeviceBuffer>`, so one `Tensor` type flows
/// through every backend's code path unchanged.
#[derive(Clone)]
pub struct Tensor {
    pub buffer: Arc<dyn DeviceBuffer>,
    /// Byte offset into `buffer` where this tensor begins (slices share the
    /// parent allocation).
    pub offset: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(buffer: Arc<dyn DeviceBuffer>, shape: Vec<usize>, dtype: DType) -> Self {
        Tensor { buffer, offset: 0, shape, dtype }
    }

    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_count(&self) -> usize {
        self.elem_count() * self.dtype.size_bytes()
    }

    /// Allocate a fresh contiguous tensor on `dev`.
    pub fn empty(dev: &dyn Device, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let bytes = shape.iter().product::<usize>() * dtype.size_bytes();
        Ok(Tensor::new(dev.alloc(bytes)?, shape, dtype))
    }

    /// Reshape without copying. Element count must be preserved.
    pub fn reshaped(&self, new_shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            new_shape.iter().product::<usize>(),
            self.elem_count(),
            "reshape changes element count"
        );
        Tensor {
            buffer: self.buffer.clone(),
            offset: self.offset,
            shape: new_shape,
            dtype: self.dtype,
        }
    }
}
