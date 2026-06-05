// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-ops
//!
//! The **seam** between model code and kernels — the Rust analog of
//! FFAI-Swift's `Ops/`. Each function takes Tensors, builds (or looks up)
//! the corresponding metaltile [`Kernel`](ffai_core::Kernel), and dispatches
//! it through the [`Device`](ffai_core::Device) trait. Model code calls
//! these; it never touches a GPU API or a kernel directly. Re-implementing
//! this layer once is what lets the whole model surface above it run on
//! every backend.
//!
//! Every op here is real and runs on any backend implementing [`Device`]
//! (verified vs HF on CUDA + Metal). Elementwise ops hand-build their IR;
//! the heavier ones (matmul / rms_norm / layer_norm / sdpa / rope / conv1d /
//! ssm …) resolve a registered metaltile kernel via `lookup`, which carries
//! the correct dispatch mode.

use ffai_core::{Binding, DType, Device, Error, Grid, Kernel, Result, Tensor};
use metaltile_core::ir::{ActKind, BinOpKind, IndexExpr, Op, Param, ParamKind, ValueId};
use metaltile_core::shape::Shape;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::sync::OnceLock;

/// Cache of resolved kernel IR keyed by (name, dtype). Resolving walks the
/// whole test registry building setups, which is expensive — a forward pass
/// dispatches the same handful of kernels hundreds of times, so cache them.
/// FxHashMap (single-cycle hash) + parking_lot Mutex (unpoisoned, inlinable).
fn kernel_cache() -> &'static Mutex<FxHashMap<(String, DType), Kernel>> {
    static C: OnceLock<Mutex<FxHashMap<(String, DType), Kernel>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(FxHashMap::default()))
}

/// Cache an inline-built kernel IR by (name, dtype). `kernel_ir_for` rebuilds the
/// full IR (~30+ ops) on every call — on the decode hot path that's ~250
/// rebuilds/token. Build once, clone thereafter.
fn cached_ir(name: &str, dtype: DType, build: impl FnOnce() -> Kernel) -> Kernel {
    let key = (name.to_string(), dtype);
    if let Some(k) = kernel_cache().lock().get(&key) {
        return k.clone();
    }
    let k = build();
    kernel_cache().lock().insert(key, k.clone());
    k
}

/// Look up a registered metaltile kernel by name and instantiate its IR for
/// `dtype`, **with the correct dispatch mode already set**. We pull it from
/// the test registry (`metaltile_std::all_tests`) rather than the raw kernel
/// registry: the test/bench setup is where each kernel's `KernelMode`
/// (Elementwise / Reduction / Grid3D …) is configured, and that mode drives
/// codegen (e.g. whether `tid` vs `gid` is defined, the `_n_elems` arg). This
/// is the same path the CUDA corpus dispatches, so we inherit its correctness.
fn lookup(name: &str, dtype: DType) -> Result<Kernel> {
    let key = (name.to_string(), dtype);
    if let Some(k) = kernel_cache().lock().get(&key) {
        return Ok(k.clone());
    }
    for entry in metaltile_std::all_tests() {
        let t = entry.test();
        if !t.dtypes().contains(&dtype) {
            continue;
        }
        let setup = t.setup(dtype);
        let k = setup.kernel();
        if k.name == name {
            kernel_cache().lock().insert(key, k.clone());
            return Ok(k.clone());
        }
    }
    Err(Error::Msg(format!(
        "kernel '{name}' [{dtype}] not found in the metaltile test registry"
    )))
}

/// Build an Elementwise `out[i] = a[i] <op> b[i]` kernel for `dtype`.
fn binop_kernel(name: &str, dtype: DType, op: BinOpKind) -> Kernel {
    let mut k = Kernel::new(name);
    for (pname, is_out) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: pname.into(),
            dtype,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load {
            src: "a".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            mask: None,
            other: None,
        },
        ValueId::new(1),
    );
    k.body.push_op(
        Op::Load {
            src: "b".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            mask: None,
            other: None,
        },
        ValueId::new(2),
    );
    k.body.push_op(
        Op::BinOp { op, lhs: ValueId::new(1), rhs: ValueId::new(2) },
        ValueId::new(3),
    );
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(3),
        mask: None,
    });
    k
}

/// Shared implementation for elementwise binary ops over matching-shape
/// tensors. Allocates a fresh output on `dev` and dispatches through the
/// backend-neutral [`Device`] trait.
fn elementwise(
    dev: &dyn Device,
    a: &Tensor,
    b: &Tensor,
    op: BinOpKind,
    name: &str,
) -> Result<Tensor> {
    if a.shape != b.shape {
        return Err(Error::Msg(format!(
            "{name}: shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    if a.dtype != b.dtype {
        return Err(Error::Msg(format!("{name}: dtype mismatch")));
    }
    let out = Tensor::empty(dev, a.shape.clone(), a.dtype)?;
    let k = binop_kernel(name, a.dtype, op);
    let n = a.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(a.buffer.clone()),
            Binding::Buffer(b.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Elementwise sum `a + b` (e.g. residual connections).
pub fn add(dev: &dyn Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    elementwise(dev, a, b, BinOpKind::Add, "ffai_add")
}

/// Elementwise product `a * b` (e.g. gating).
pub fn mul(dev: &dyn Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    elementwise(dev, a, b, BinOpKind::Mul, "ffai_mul")
}

// ── DeepSeek-V4 MoE ops ─────────────────────────────────────────────────

/// DSv4 clamped SwiGLU: `out = silu(min(gate, limit)) * clip(up, ±limit)`.
/// Dispatches `ffai_dsv4_swiglu_limit` (limit = 10 for DSv4).
pub fn swiglu_limit(dev: &dyn Device, gate: &Tensor, up: &Tensor, limit: f32) -> Result<Tensor> {
    if gate.shape != up.shape {
        return Err(Error::Msg("swiglu_limit: gate/up shape mismatch".into()));
    }
    let k = lookup("ffai_dsv4_swiglu_limit", gate.dtype)?;
    let out = Tensor::empty(dev, gate.shape.clone(), gate.dtype)?;
    let n = gate.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(gate.buffer.clone()),
            Binding::Buffer(up.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar(limit.to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// DSv4 MoE router scoring: `score_unbiased[e] = sqrt(softplus(logit[e]))`,
/// `score_biased[e] = score_unbiased[e] + bias[e]`. Computed **host-side** —
/// the router operates on the tiny `[n_experts]` logit vector that the MoE
/// already downloads for top-k selection, so a GPU kernel buys nothing and
/// avoids the multi-output dispatch path. `biased` selects the top-k experts;
/// `unbiased` weights the combine.
pub fn sqrtsoftplus_route(logits: &[f32], bias: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let unbiased: Vec<f32> = logits
        .iter()
        .map(|&x| (x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()).sqrt())
        .collect();
    let biased: Vec<f32> = unbiased.iter().zip(bias).map(|(u, b)| u + b).collect();
    (unbiased, biased)
}

// ── Causal conv1d step (Mamba2 short-conv / audio front-ends) ───────────

/// Causal depthwise conv1d decode step (one thread per channel):
/// `y[d] = b[d] + w[K-1,d]·x[d] + Σ_{k<K-1} w[k,d]·state[k,d]`, then shift the
/// ring `state` and insert `x`. `w` is `[kernel_size·n_channels]`, `state` is
/// `[(kernel_size-1)·n_channels]` (updated in place). Dispatches
/// `conv1d_causal_step`. Returns `y [n_channels]`; `state` buffer is mutated.
pub fn conv1d_causal_step(
    dev: &dyn Device,
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    state: &Tensor,
    n_channels: u32,
    kernel_size: u32,
) -> Result<Tensor> {
    let k = lookup("conv1d_causal_step", x.dtype)?;
    let y = Tensor::empty(dev, vec![n_channels as usize], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let bindings = vec![
        Binding::Buffer(x.buffer.clone()),
        Binding::Buffer(w.buffer.clone()),
        Binding::Buffer(b.buffer.clone()),
        Binding::Buffer(state.buffer.clone()), // in-place ring state
        Binding::Buffer(y.buffer.clone()),
        u(n_channels),
        u(kernel_size),
    ];
    let grid = Grid { grid: [n_channels, 1, 1], block: [1, 1, 1] };
    dev.dispatch(&k, &bindings, grid)?;
    Ok(y)
}

// ── SSM (Mamba2 SSD selective scan) ─────────────────────────────────────

/// Mamba2 SSD selective-scan **decode step** (single token, batch=1):
/// `da = exp(-exp(a_log)·dt)`; `state' = da·state + x·dt·B`;
/// `out = Σ_s C·state' + x·D`. Dispatches `mt_ssm_step`. Returns
/// `(state_out [n_heads·dh·ds], out [n_heads·dh])`. `ds` must be a multiple
/// of 32. Shapes: x `[n_heads·dh]`, a_log/d_skip/dt `[n_heads]`,
/// b_mat/c_mat `[n_groups·ds]`, state_in `[n_heads·dh·ds]`.
#[allow(clippy::too_many_arguments)]
pub fn ssm_step(
    dev: &dyn Device,
    x: &Tensor,
    a_log: &Tensor,
    b_mat: &Tensor,
    c_mat: &Tensor,
    d_skip: &Tensor,
    dt: &Tensor,
    state_in: &Tensor,
    dh: u32,
    ds: u32,
    n_heads: u32,
    heads_per_group: u32,
) -> Result<(Tensor, Tensor)> {
    let k = lookup("mt_ssm_step", x.dtype)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), x.dtype)?;
    let out = Tensor::empty(dev, vec![(n_heads * dh) as usize], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let bindings = vec![
        Binding::Buffer(x.buffer.clone()),
        Binding::Buffer(a_log.buffer.clone()),
        Binding::Buffer(b_mat.buffer.clone()),
        Binding::Buffer(c_mat.buffer.clone()),
        Binding::Buffer(d_skip.buffer.clone()),
        Binding::Buffer(dt.buffer.clone()),
        Binding::Buffer(state_in.buffer.clone()),
        Binding::Buffer(state_out.buffer.clone()),
        Binding::Buffer(out.buffer.clone()),
        u(dh),
        u(ds),
        u(n_heads),
        u(heads_per_group),
    ];
    let grid = Grid { grid: [dh, n_heads, 1], block: [32, 1, 1] };
    dev.dispatch(&k, &bindings, grid)?;
    Ok((state_out, out))
}

// ── DeepSeek-V4 mHC (hyper-connection 4-channel residual) ───────────────

/// mHC sinkhorn split (single token, host-side): the 24-value `mixes`
/// vector → `pre[4]`, `post[4]`, `comb[4×4]`. Faithful transcription of
/// `ffai_dsv4_mhc_sinkhorn_split` (a 3-output kernel; trivial at one token,
/// so it runs on the host). `scale` is `[pre, post, comb]`, `base` is `[24]`.
/// - pre[c]  = sigmoid(mixes[c]·pre_scale + base[c]) + eps
/// - post[c] = 2·sigmoid(mixes[4+c]·post_scale + base[4+c])
/// - comb    = per-row softmax of (mixes·comb_scale + base), then
///   `iters` Sinkhorn steps (column-normalize, then row-normalize).
pub fn dsv4_mhc_sinkhorn_split(
    mixes: &[f32],
    scale: [f32; 3],
    base: &[f32],
    eps: f32,
    iters: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let pre: Vec<f32> = (0..4).map(|c| sig(mixes[c] * scale[0] + base[c]) + eps).collect();
    let post: Vec<f32> = (0..4).map(|c| 2.0 * sig(mixes[4 + c] * scale[1] + base[4 + c])).collect();

    let mut c = [[0.0f32; 4]; 4];
    for i in 0..4 {
        let r: [f32; 4] = std::array::from_fn(|j| mixes[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]);
        let m = r.iter().cloned().fold(f32::MIN, f32::max);
        let e: [f32; 4] = std::array::from_fn(|j| (r[j] - m).exp());
        let s: f32 = e.iter().sum();
        for j in 0..4 {
            c[i][j] = e[j] / s + eps;
        }
    }
    for _ in 0..iters {
        for j in 0..4 {
            let cs: f32 = (0..4).map(|i| c[i][j]).sum::<f32>() + eps;
            for row in c.iter_mut() {
                row[j] /= cs;
            }
        }
        for row in c.iter_mut() {
            let rs: f32 = row.iter().sum::<f32>() + eps;
            for v in row.iter_mut() {
                *v /= rs;
            }
        }
    }
    let comb: Vec<f32> = (0..4).flat_map(|i| (0..4).map(move |j| c[i][j])).collect();
    (pre, post, comb)
}


/// mHC collapse: `out[d] = Σ_c pre[c] · state[c, d]` — mix the 4-channel
/// residual state `[n_hc, hidden]` down to `[hidden]` using the per-channel
/// `pre` weights. Single token. Dispatches `ffai_dsv4_mhc_collapse`.
pub fn dsv4_mhc_collapse(
    dev: &dyn Device,
    state: &Tensor,
    pre: &Tensor,
    hidden_dim: u32,
    n_hc: u32,
) -> Result<Tensor> {
    let k = lookup("ffai_dsv4_mhc_collapse", state.dtype)?;
    let out = Tensor::empty(dev, vec![hidden_dim as usize], state.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid::d1((hidden_dim).div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(state.buffer.clone()),
            Binding::Buffer(pre.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(hidden_dim),
            u(n_hc),
            u(1),
        ],
        grid,
    )?;
    Ok(out)
}

/// mHC expand: write the new 4-channel residual state
/// `state[dst, d] = block_out[d]·post[dst] + Σ_src comb[dst, src]·residual[src, d]`.
/// Single token. Returns `[n_hc, hidden]`. Dispatches `ffai_dsv4_mhc_expand`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_mhc_expand(
    dev: &dyn Device,
    block_out: &Tensor,
    post: &Tensor,
    comb: &Tensor,
    residual_state: &Tensor,
    hidden_dim: u32,
    n_hc: u32,
) -> Result<Tensor> {
    let k = lookup("ffai_dsv4_mhc_expand", block_out.dtype)?;
    let state = Tensor::empty(dev, vec![(n_hc * hidden_dim) as usize], block_out.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid::d1((hidden_dim).div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(block_out.buffer.clone()),
            Binding::Buffer(post.buffer.clone()),
            Binding::Buffer(comb.buffer.clone()),
            Binding::Buffer(residual_state.buffer.clone()),
            Binding::Buffer(state.buffer.clone()),
            u(hidden_dim),
            u(n_hc),
            u(1),
        ],
        grid,
    )?;
    Ok(state)
}

// ── Heavier ops — dispatch the registered metaltile kernels ─────────────

/// Row-wise RMS norm: `out[r] = x[r] * rsqrt(mean(x[r]²) + eps) * weight`.
/// Dispatches the registered `mt_rms_norm` reduction kernel — the same one
/// the Swift side runs. The last dim is the row width `n`; the kernel owns 4
/// elements per thread, so `n` must be a multiple of 128 and ≤ 4096 (the
/// `mt_rms_norm_wide` variant lifts this — wired later).
pub fn rms_norm(dev: &dyn Device, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let n = *x.shape.last().ok_or_else(|| Error::Msg("rms_norm: scalar input".into()))?;
    let rows = x.elem_count() / n;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let eps_buf = dev.upload(&eps.to_le_bytes())?;

    // Fast path: 4 elems/thread, needs n a multiple of 128 and ≤ 4096.
    // Otherwise the strided wide variant handles any row width.
    let (kname, block) = if n % 128 == 0 && n <= 4096 {
        ("mt_rms_norm", (n / 4) as u32)
    } else {
        ("mt_rms_norm_wide", 256u32)
    };
    let k = lookup(kname, x.dtype)?;
    let grid = Grid { grid: [rows as u32, 1, 1], block: [block, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Buffer(eps_buf),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Matrix-vector product `mat @ vec`: `mat` is `[m, k]` row-major, `vec` is
/// `[k]`, result is `[m]`. Dispatches the registered `mt_gemv` kernel (one
/// threadgroup per output row). This is the decode-time projection path; the
/// batched/prefill cooperative matmul is a separate kernel, wired later.
pub fn gemv(dev: &dyn Device, mat: &Tensor, vec: &Tensor) -> Result<Tensor> {
    if mat.shape.len() != 2 {
        return Err(Error::Msg(format!("gemv: mat must be 2-D, got {:?}", mat.shape)));
    }
    let (m, kdim) = (mat.shape[0], mat.shape[1]);
    if vec.elem_count() != kdim {
        return Err(Error::Msg(format!(
            "gemv: vec len {} != mat K {kdim}",
            vec.elem_count()
        )));
    }
    let k = lookup("mt_gemv", mat.dtype)?;
    let out = Tensor::empty(dev, vec![m], mat.dtype)?;

    let grid = Grid { grid: [m as u32, 1, 1], block: [256, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(mat.buffer.clone()),
            Binding::Buffer(vec.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((kdim as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Q8_0 grouped matvec — weights stay QUANTIZED resident (int8, block 32, one
/// f32 scale/block): `out[r] = Σ_k dequant(qs[r,k], scales[r, k/32]) ·
/// x[(r/rows_per_group)·k_in + k]`. `qs` is u32-packed (4 int8/u32, 8 u32/block),
/// `scales` is `[m_out · k_in/32]` f32. Dense gemv ⇒ `rows_per_group = m_out`
/// (n_groups=1, `x = [k_in]`); MoE ⇒ per-expert grouping. Dispatches the
/// Reduction kernel `ffai_grouped_gemv_q8_rows` (8-bit ⇒ ~4× less weight DRAM
/// than F32, the resident-decode bandwidth win). `k_in` must be a multiple of 32.
pub fn gemv_q8(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    x: &Tensor,
    m_out: usize,
    k_in: usize,
    rows_per_group: usize,
) -> Result<Tensor> {
    if k_in % 32 != 0 {
        return Err(Error::Msg(format!("gemv_q8: k_in {k_in} must be a multiple of 32")));
    }
    // Not in the test registry, so build the kernel IR directly (constexpr dims
    // are JIT-specialized by the runtime from the scalar bindings below).
    // Coalesced variant: consecutive lanes read consecutive qs words (~2× the
    // strided original's DRAM bandwidth on GB10).
    let k = cached_ir("ffai_gemv_q8_coalesced", x.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_gemv_q8_coalesced::kernel_ir_for(x.dtype); k.mode = metaltile_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![m_out], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(k_in as u32),
            u(m_out as u32),
            u(rows_per_group as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Q8 gemv with fused ReLU²: `out[r] = max(0, (Wq·x)[r])²` — a MoE expert's
/// `up` projection + activation in one dispatch. Dispatches `ffai_gemv_q8_coalesced_relu2`.
pub fn gemv_q8_relu2(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, m_out: usize, k_in: usize, rows_per_group: usize) -> Result<Tensor> {
    let k = cached_ir("ffai_gemv_q8_coalesced_relu2", x.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_gemv_q8_coalesced_relu2::kernel_ir_for(x.dtype); k.mode = metaltile_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![m_out], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] };
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(k_in as u32), u(m_out as u32), u(rows_per_group as u32)], grid)?;
    Ok(out)
}

/// Q8 gemv that scales + accumulates in place: `acc[r] += scale · (Wq · x)[r]`.
/// Fuses an MoE expert's `down` projection with its router-weighted sum into the
/// layer accumulator — no per-expert scalar-broadcast upload or separate add.
/// `scale_buf` is a 1-element device buffer. Dispatches `ffai_gemv_q8_coalesced_accum`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q8_accum(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    x: &Tensor,
    acc: &Tensor,
    scale_buf: &Tensor,
    m_out: usize,
    k_in: usize,
    rows_per_group: usize,
) -> Result<()> {
    if k_in % 32 != 0 {
        return Err(Error::Msg(format!("gemv_q8_accum: k_in {k_in} must be a multiple of 32")));
    }
    let k = cached_ir("ffai_gemv_q8_coalesced_accum", acc.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_gemv_q8_coalesced_accum::kernel_ir_for(acc.dtype); k.mode = metaltile_core::ir::KernelMode::Reduction; k });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(acc.buffer.clone()),
            Binding::Buffer(scale_buf.buffer.clone()),
            u(k_in as u32),
            u(m_out as u32),
            u(rows_per_group as u32),
        ],
        grid,
    )?;
    Ok(())
}

/// Append one decode step's K (or V) into an in-place device KV cache:
/// `dst[h, pos, :] = src[h, :]`, where `dst` is `[nkv, cap, hd]`, `src` is
/// `[nkv*hd]`, and `posbuf` is a 1-element u32 device buffer. Keeps the growing
/// context entirely on-device — no host reorg/reupload per step (the 32K-context
/// fix). Dispatches `ffai_kv_append` (runtime `pos` ⇒ compiled once, not per step).
pub fn kv_append(dev: &dyn Device, src: &Tensor, dst: &Tensor, posbuf: &Tensor, hd: usize, cap: usize, n: usize) -> Result<()> {
    let k = cached_ir("ffai_kv_append", src.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_kv_append::kernel_ir_for(src.dtype); k.mode = metaltile_core::ir::KernelMode::Grid3D; k });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid { grid: [(n as u32).div_ceil(64), 1, 1], block: [64, 1, 1] };
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(dst.buffer.clone()), Binding::Buffer(posbuf.buffer.clone()), u(hd as u32), u(cap as u32)], grid)?;
    Ok(())
}

/// Device slice `out[i] = src[off + i]` for `len` elements (no host round-trip).
pub fn slice(dev: &dyn Device, src: &Tensor, off: usize, len: usize) -> Result<Tensor> {
    let k = cached_ir("ffai_slice", src.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_slice::kernel_ir_for(src.dtype); k.mode = metaltile_core::ir::KernelMode::Grid3D; k });
    let out = Tensor::empty(dev, vec![len], src.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(off as u32), u(len as u32)], Grid { grid: [(len as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Cast f32 → f16 (KV-cache compaction: store attention K/V at half precision
/// to halve the 32K-context sdpa read). Returns a fresh F16 tensor.
pub fn cast_f32_f16(dev: &dyn Device, src: &Tensor) -> Result<Tensor> {
    let k = cached_ir("ffai_cast_f32_f16", DType::F16, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_cast_f32_f16::kernel_ir(); k.mode = metaltile_core::ir::KernelMode::Grid3D; k });
    let n = src.elem_count();
    let out = Tensor::empty(dev, src.shape.clone(), DType::F16)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Cast f16 → f32 (reverse: widen the sdpa f16 output back to f32 for the
/// downstream o_proj Q4 GEMV, which consumes f32 activations). Fresh F32 tensor.
pub fn cast_f16_f32(dev: &dyn Device, src: &Tensor) -> Result<Tensor> {
    let k = cached_ir("ffai_cast_f16_f32", DType::F32, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_cast_f16_f32::kernel_ir(); k.mode = metaltile_core::ir::KernelMode::Grid3D; k });
    let n = src.elem_count();
    let out = Tensor::empty(dev, src.shape.clone(), DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Device Mamba dt: `softplus(dt_raw + dt_bias)` — no host round-trip.
pub fn softplus_add(dev: &dyn Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let k = cached_ir("ffai_softplus_add", DType::F32, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_softplus_add::kernel_ir(); k.mode = metaltile_core::ir::KernelMode::Grid3D; k });
    let n = a.elem_count();
    let out = Tensor::empty(dev, a.shape.clone(), DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(a.buffer.clone()), Binding::Buffer(b.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// NemotronH/Zamba2 gated grouped RMSNorm ON-DEVICE: out = (y·silu(z)) normalized
/// per `gs`-group, ×w. Removes the per-Mamba-layer dl→host-norm→up sync. `y` is f32.
pub fn gated_group_rmsnorm(dev: &dyn Device, y: &Tensor, z: &Tensor, w: &Tensor, eps: f32, di: usize, gs: usize) -> Result<Tensor> {
    let k = cached_ir("ffai_gated_group_rmsnorm", z.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_gated_group_rmsnorm::kernel_ir_for(z.dtype); k.mode = metaltile_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![di], z.dtype)?;
    let eps_buf = Tensor::new(dev.upload(&eps.to_le_bytes()).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![1], DType::F32);
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let ng = (di / gs) as u32;
    dev.dispatch(&k, &[Binding::Buffer(y.buffer.clone()), Binding::Buffer(z.buffer.clone()), Binding::Buffer(w.buffer.clone()), Binding::Buffer(out.buffer.clone()), Binding::Buffer(eps_buf.buffer.clone()), u(gs as u32)], Grid { grid: [ng, 1, 1], block: [(gs / 4) as u32, 1, 1] })?;
    Ok(out)
}

/// Roll a causal-conv state on-device: `new = [old[conv_dim..], xbc]`.
pub fn conv_roll(dev: &dyn Device, old: &Tensor, xbc: &Tensor, conv_dim: usize, kc: usize) -> Result<Tensor> {
    let n = (kc - 1) * conv_dim;
    let keep = (kc - 2) * conv_dim;
    let k = cached_ir("ffai_conv_roll", old.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_conv_roll::kernel_ir_for(old.dtype); k.mode = metaltile_core::ir::KernelMode::Grid3D; k });
    let out = Tensor::empty(dev, vec![n], old.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(old.buffer.clone()), Binding::Buffer(xbc.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(conv_dim as u32), u(keep as u32), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Batched MoE up+ReLU²: gather `top_k` experts (indices `idx`) from the
/// contiguous `[n_exp*inter, hid]` Q4 weight into one `[top_k*inter]` GEMV.
pub fn moe_gather_up_relu2(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, idx: &Tensor, top_k: usize, inter: usize, hid: usize) -> Result<Tensor> {
    let k = cached_ir("ffai_moe_gather_q4_relu2", x.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_moe_gather_q4_relu2::kernel_ir_for(x.dtype); k.mode = metaltile_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![top_k * inter], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let rpt: u32 = std::env::var("MT_MOE_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(1);
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(hid as u32), u(inter as u32), u(rpt)], Grid { grid: [(inter as u32).div_ceil(rpt), top_k as u32, 1], block: [32 * rpt, 1, 1] })?;
    Ok(out)
}
/// Batched MoE down + router-weighted accumulate into `acc[hid]`. `qs` is the
/// contiguous `[n_exp*hid, inter]` Q4 weight; `x` is the `[top_k*inter]` up output.
#[allow(clippy::too_many_arguments)]
pub fn moe_gather_down_accum(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, idx: &Tensor, wts: &Tensor, acc: &Tensor, top_k: usize, inter: usize, hid: usize) -> Result<()> {
    let k = cached_ir("ffai_moe_gather_q4_down_accum", x.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_moe_gather_q4_down_accum::kernel_ir_for(x.dtype); k.mode = metaltile_core::ir::KernelMode::Reduction; k });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(wts.buffer.clone()), Binding::Buffer(acc.buffer.clone()), u(inter as u32), u(hid as u32), u(top_k as u32)], Grid { grid: [hid as u32, 1, 1], block: [32, 1, 1] })?;
    Ok(())
}

/// Batched MoE down gather → `[top_k*hid]` (one big GEMV, no accumulate).
pub fn moe_gather_down(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, idx: &Tensor, top_k: usize, inter: usize, hid: usize) -> Result<Tensor> {
    let k = cached_ir("ffai_moe_gather_q4_down", x.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_moe_gather_q4_down::kernel_ir_for(x.dtype); k.mode = metaltile_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![top_k * hid], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let rpt: u32 = std::env::var("MT_MOE_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(1);
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(inter as u32), u(hid as u32), u(rpt)], Grid { grid: [(hid as u32).div_ceil(rpt), top_k as u32, 1], block: [32 * rpt, 1, 1] })?;
    Ok(out)
}
/// Router-weighted sum of per-expert down outputs into `acc`: `acc[h] += Σ wts·downs[·,h]`.
/// Fully ON-DEVICE MoE router (NemotronH sigmoid + e_score_correction_bias, top-k by
/// biased score, weights from unbiased renormalized to sum-1, ×routed_scaling_factor).
/// Returns (idx[top_k] U32, wts[top_k] F32) WITHOUT a host round-trip — replaces the
/// per-layer dl(gate)+host-topk+up(idx) sync that drains the async pipeline 23×/token.
pub fn moe_router_device(dev: &dyn Device, gate_logits: &Tensor, bias: &Tensor, n_exp: usize, top_k: usize, scale: f32) -> Result<(Tensor, Tensor)> {
    use metaltile_core::ir::KernelMode;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let f = |v: f32| Binding::Scalar(v.to_le_bytes().to_vec());
    let unbiased = Tensor::empty(dev, vec![n_exp], DType::F32)?;
    let biased = Tensor::empty(dev, vec![n_exp], DType::F32)?;
    let ks = cached_ir("ffai_moe_sigmoid_bias", DType::F32, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_moe_sigmoid_bias::kernel_ir(); k.mode = KernelMode::Grid3D; k });
    dev.dispatch(&ks, &[Binding::Buffer(gate_logits.buffer.clone()), Binding::Buffer(bias.buffer.clone()), Binding::Buffer(unbiased.buffer.clone()), Binding::Buffer(biased.buffer.clone()), u(n_exp as u32)], Grid { grid: [(n_exp as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    let idx = Tensor::empty(dev, vec![top_k], DType::U32)?;
    let wts = Tensor::empty(dev, vec![top_k], DType::F32)?;
    let kr = cached_ir("mt_dsv4_router_topk", DType::F32, || { let mut k = metaltile_std::ffai::dsv4_router_topk::mt_dsv4_router_topk::kernel_ir_for(DType::F32); k.mode = KernelMode::Reduction; k });
    dev.dispatch(&kr, &[Binding::Buffer(biased.buffer.clone()), Binding::Buffer(unbiased.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(wts.buffer.clone()), u(n_exp as u32), u(top_k as u32)], Grid { grid: [1, 1, 1], block: [32, 1, 1] })?;
    let kv = cached_ir("ffai_vscale", DType::F32, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_vscale::kernel_ir(); k.mode = KernelMode::Grid3D; k });
    dev.dispatch(&kv, &[Binding::Buffer(wts.buffer.clone()), f(scale), u(top_k as u32)], Grid { grid: [(top_k as u32).div_ceil(64), 1, 1], block: [64, 1, 1] })?;
    Ok((idx, wts))
}

pub fn moe_weighted_sum(dev: &dyn Device, downs: &Tensor, wts: &Tensor, acc: &Tensor, hid: usize, top_k: usize) -> Result<()> {
    let k = cached_ir("ffai_moe_weighted_sum", acc.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_moe_weighted_sum::kernel_ir_for(acc.dtype); k.mode = metaltile_core::ir::KernelMode::Grid3D; k });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(downs.buffer.clone()), Binding::Buffer(wts.buffer.clone()), Binding::Buffer(acc.buffer.clone()), u(hid as u32), u(top_k as u32)], Grid { grid: [(hid as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(())
}

// ── Q4 (4-bit) family — half the weight DRAM of Q8 (the decode lever). ──
fn gemv_q4_dispatch(dev: &dyn Device, kernel: &str, qs: &Tensor, scales: &Tensor, x: &Tensor, acc: Option<&Tensor>, scale_buf: Option<&Tensor>, m_out: usize, k_in: usize, rows_per_group: usize) -> Result<Tensor> {
    // 2-row tiling: 2 output rows per warp, shared activation read, 2 weight
    // streams in flight (more memory-level-parallelism on the latency-bound Q4 read).
    let vec = kernel == "plain" && std::env::var("MT_GEMV_VEC").is_ok();
    let two_row = !vec && kernel == "plain" && std::env::var("MT_GEMV_2ROW").is_ok();
    let name = if vec { "ffai_gemv_q4_vec" } else if two_row { "ffai_gemv_q4_coalesced_2row" } else { match kernel { "plain" => "ffai_gemv_q4_coalesced", "relu2" => "ffai_gemv_q4_coalesced_relu2", _ => "ffai_gemv_q4_coalesced_accum" } };
    let k = cached_ir(name, x.dtype, || {
        let mut k = if vec {
            metaltile_std::ffai::gemv_q8::ffai_gemv_q4_vec::kernel_ir_for(x.dtype)
        } else if two_row {
            metaltile_std::ffai::gemv_q8::ffai_gemv_q4_coalesced_2row::kernel_ir_for(x.dtype)
        } else { match kernel {
            "plain" => metaltile_std::ffai::gemv_q8::ffai_gemv_q4_coalesced::kernel_ir_for(x.dtype),
            "relu2" => metaltile_std::ffai::gemv_q8::ffai_gemv_q4_coalesced_relu2::kernel_ir_for(x.dtype),
            _ => metaltile_std::ffai::gemv_q8::ffai_gemv_q4_coalesced_accum::kernel_ir_for(x.dtype),
        }};
        k.mode = metaltile_core::ir::KernelMode::Reduction;
        k
    });
    let out = acc.cloned().unwrap_or(Tensor::empty(dev, vec![m_out], x.dtype)?);
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let mut b = vec![Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(out.buffer.clone())];
    if let Some(sb) = scale_buf { b.push(Binding::Buffer(sb.buffer.clone())); }
    b.extend([u(k_in as u32), u(m_out as u32), u(rows_per_group as u32)]);
    // Plain gemv is multi-warp capable (rows_per_tg warps/TG to hide the
    // ncu-measured global-load latency). MT_GEMV_RPT=1 → original single-warp.
    let grid = if vec {
        // single-warp/row, vectorized loads (no rpt binding)
        Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] }
    } else if kernel == "plain" {
        let rpt: u32 = std::env::var("MT_GEMV_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(1);
        b.push(u(rpt));
        let rows_per_block = if two_row { 2 * rpt } else { rpt };
        Grid { grid: [(m_out as u32).div_ceil(rows_per_block), 1, 1], block: [32 * rpt, 1, 1] }
    } else {
        Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] }
    };
    dev.dispatch(&k, &b, grid)?;
    Ok(out)
}
/// Q4 matvec `out = Wq4·x`. (block 32, int4 [-7,7], f32 scale/block.)
pub fn gemv_q4(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, m_out: usize, k_in: usize, rpg: usize) -> Result<Tensor> {
    gemv_q4_dispatch(dev, "plain", qs, scales, x, None, None, m_out, k_in, rpg)
}
/// Q4 matvec with fused ReLU² (MoE expert up).
pub fn gemv_q4_relu2(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, m_out: usize, k_in: usize, rpg: usize) -> Result<Tensor> {
    gemv_q4_dispatch(dev, "relu2", qs, scales, x, None, None, m_out, k_in, rpg)
}
/// Q4 matvec, scale + accumulate into `acc` in place (MoE expert down).
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_accum(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, acc: &Tensor, scale_buf: &Tensor, m_out: usize, k_in: usize, rpg: usize) -> Result<()> {
    gemv_q4_dispatch(dev, "accum", qs, scales, x, Some(acc), Some(scale_buf), m_out, k_in, rpg)?;
    Ok(())
}
/// Quantize a row-major `[m,k]` F32 weight to Q4 blocks (block 32, symmetric int4
/// in [-7,7], f32 scale `amax/7`). Returns `(qs_u32, scales)`: qs `[m·k/32·4]`
/// (8 nibbles/u32, 4 u32/block), scales `[m·k/32]`. `k` must be a multiple of 32.
pub fn quantize_q4(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % 32, 0, "quantize_q4: k must be a multiple of 32");
    let bpr = k / 32;
    let mut qs = vec![0u32; m * bpr * 4];
    let mut scales = vec![0f32; m * bpr];
    for r in 0..m {
        for b in 0..bpr {
            let base = r * k + b * 32;
            let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
            let d = amax / 7.0;
            scales[r * bpr + b] = d;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for word in 0..4 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let q = (w[base + word * 8 + i] * inv).round().clamp(-7.0, 7.0) as i32;
                    packed |= ((q as u32) & 0xf) << (i * 4);
                }
                qs[r * bpr * 4 + b * 4 + word] = packed;
            }
        }
    }
    (qs, scales)
}

/// Quantize a row-major `[m, k]` F32 weight to Q8_0 blocks (block 32, symmetric
/// int8, per-block f32 scale `amax/127`). Returns `(qs_u32_packed, scales_f32)`
/// laid out exactly as [`gemv_q8`] expects: qs `[m · k/32 · 8]` u32, scales
/// `[m · k/32]` f32. CPU, one-time at load. `k` must be a multiple of 32.
pub fn quantize_q8(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % 32, 0, "quantize_q8: k must be a multiple of 32");
    let bpr = k / 32;
    let mut qs = vec![0u32; m * bpr * 8];
    let mut scales = vec![0f32; m * bpr];
    for r in 0..m {
        for b in 0..bpr {
            let base = r * k + b * 32;
            let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
            let d = amax / 127.0;
            scales[r * bpr + b] = d;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for w_i in 0..8 {
                let mut packed = 0u32;
                for i in 0..4 {
                    let q = (w[base + w_i * 4 + i] * inv).round().clamp(-127.0, 127.0) as i32;
                    packed |= ((q as u8) as u32) << (i * 8);
                }
                qs[r * bpr * 8 + b * 8 + w_i] = packed;
            }
        }
    }
    (qs, scales)
}

/// Prefill linear: `out[r, :] = weight · input[r, :]` for a block of rows in
/// one dispatch — `weight` is `[out_dim, in_dim]`, `input` is `[n_rows, in_dim]`,
/// result `[n_rows, out_dim]`. Dispatches the 32×32-tiled Reduction kernel
/// `ffai_gemm` (weight read once per tile, reused across the row block — the
/// many-token path that a `gemv`-per-row would re-stream). Requires
/// `in_dim % 16 == 0` (the K-tile contract); row/out-dim edges are handled
/// in-kernel. This is the matmul the prefill + VLM/ViT towers run on.
pub fn matmul(dev: &dyn Device, weight: &Tensor, input: &Tensor) -> Result<Tensor> {
    if weight.shape.len() != 2 {
        return Err(Error::Msg(format!("matmul: weight must be 2-D, got {:?}", weight.shape)));
    }
    let (out_dim, in_dim) = (weight.shape[0], weight.shape[1]);
    let rows = input.elem_count() / in_dim;
    if input.elem_count() != rows * in_dim {
        return Err(Error::Msg(format!("matmul: input {:?} not a multiple of in_dim {in_dim}", input.shape)));
    }
    if in_dim % 16 != 0 {
        return Err(Error::Msg(format!("matmul: in_dim {in_dim} must be a multiple of 16")));
    }
    let k = lookup("ffai_gemm", weight.dtype)?;
    let out = Tensor::empty(dev, vec![rows, out_dim], weight.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid {
        grid: [(out_dim as u32).div_ceil(32), (rows as u32).div_ceil(32), 1],
        block: [1024, 1, 1],
    };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(input.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(in_dim as u32),
            u(out_dim as u32),
            u(rows as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// DeepSeek-V4 MLA decode attention: d512 SDPA with a per-head learnable
/// **attention sink** (a virtual logit that extends the softmax denominator).
/// `q` is `[n_q_heads, 512]`; `k`/`v` are the latent KV cache
/// `[n_kv_heads, kv_stride, 512]`; `sink_logit` is `[n_q_heads]` f32.
/// Dispatches `ffai_sdpa_decode_d512_sink` (tg 512).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_sink(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sink_logit: &Tensor,
    n_kv: u32,
    kv_stride: u32,
    heads_per_group: u32,
    scale: f32,
) -> Result<Tensor> {
    const HD: usize = 512;
    let n_q_heads = q.elem_count() / HD;
    let out = Tensor::empty(dev, vec![n_q_heads, HD], q.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let kern = lookup("ffai_sdpa_decode_d512_sink", q.dtype)?;
    let bindings = vec![
        Binding::Buffer(q.buffer.clone()),
        Binding::Buffer(k.buffer.clone()),
        Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(sink_logit.buffer.clone()),
        Binding::Buffer(out.buffer.clone()),
        u(HD as u32),
        u(n_kv),
        u(kv_stride),
        u(heads_per_group),
        Binding::Scalar(scale.to_le_bytes().to_vec()),
    ];
    let grid = Grid { grid: [n_q_heads as u32, 1, 1], block: [512, 1, 1] };
    dev.dispatch(&kern, &bindings, grid)?;
    Ok(out)
}

/// Decode-time scaled-dot-product attention for a single query token.
///
/// Layout (matching the kernel): `q` is `[n_q_heads, head_dim]`; `k`/`v` are
/// `[n_kv_heads, kv_stride, head_dim]` (kv cache, `kv_stride` = capacity,
/// `n_kv` = filled length); `kv_head = q_head / heads_per_group` (GQA). No
/// attention sink / sliding window (dense causal). Picks the head-dim
/// specialized kernel variant. Output is `[n_q_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_kv: u32,
    kv_stride: u32,
    heads_per_group: u32,
    scale: f32,
) -> Result<Tensor> {
    let n_q_heads = q.elem_count() / head_dim;
    let out = Tensor::empty(dev, vec![n_q_heads, head_dim], q.dtype)?;

    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());

    // Per-variant trailing constexprs (everything after head_dim, n_kv,
    // kv_stride, heads_per_group). Dense path → sink/window disabled.
    let (name, block, trailing): (&str, u32, Vec<Binding>) = match head_dim {
        64 => ("ffai_sdpa_decode_d64", 1024, vec![u(0), f(0.0), f(scale)]), // has_sink, sink_logit, scale
        96 => ("ffai_sdpa_decode_d96", 1024, vec![f(scale)]),
        128 => ("ffai_sdpa_decode", 1024, vec![u(0), u(0), u(0), f(0.0), f(scale)]), // sink_end, window_start, has_sink, sink_logit, scale
        256 => ("ffai_sdpa_decode_d256", 1024, vec![u(0), f(0.0), f(scale)]),
        512 => ("ffai_sdpa_decode_d512", 512, vec![f(scale)]),
        _ => return Err(Error::Msg(format!("sdpa_decode: unsupported head_dim {head_dim}"))),
    };

    let kern = lookup(name, q.dtype)?;
    let mut bindings = vec![
        Binding::Buffer(q.buffer.clone()),
        Binding::Buffer(k.buffer.clone()),
        Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(out.buffer.clone()),
        u(head_dim as u32),
        u(n_kv),
        u(kv_stride),
        u(heads_per_group),
    ];
    bindings.extend(trailing);
    let grid = Grid { grid: [n_q_heads as u32, 1, 1], block: [block, 1, 1] };
    dev.dispatch(&kern, &bindings, grid)?;
    Ok(out)
}

/// GQA-aware split-K flash-decode SDPA (MLX `sdpa_vector_2pass` port, head_dim
/// 128). Pass 1: grid `(n_kv_heads, blocks)`, block `(32, gqa_factor)` — one
/// simdgroup per Q-head of the group, so the `gqa_factor` heads SHARE each K/V
/// load (the single-pass kernel re-reads the shared KV head `gqa_factor`× — at
/// 32K that's the dominant cost). Each block-row strides a `1/blocks` slice of
/// the KV positions, emitting per-block (max, sum, partial_o). Pass 2: grid
/// `(n_q_heads)`, block 1024 — online-softmax merge of the `blocks` partials.
/// `blocks` MUST be a multiple of 32 (pass-2 reducer constraint). Kernel math
/// is registry-validated (`test_ffai_sdpa_decode_2pass_combined`, f32/f16/bf16).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_2pass(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_kv: u32,
    kv_stride: u32,
    gqa_factor: u32,
    scale: f32,
    blocks: u32,
) -> Result<Tensor> {
    use metaltile_core::ir::KernelMode;
    let n_q_heads = q.elem_count() / head_dim;
    let n_kv_heads = n_q_heads / gqa_factor as usize;
    let nb = blocks as usize;
    let partial_o = Tensor::empty(dev, vec![n_q_heads * nb * head_dim], q.dtype)?;
    let partial_m = Tensor::empty(dev, vec![n_q_heads * nb], DType::F32)?;
    let partial_l = Tensor::empty(dev, vec![n_q_heads * nb], DType::F32)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());

    let k1 = cached_ir("sdpa_decode_2pass_pass1", q.dtype, || {
        let mut k = metaltile_std::ffai::sdpa_decode_2pass::sdpa_decode_2pass_pass1::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k1, &[
        Binding::Buffer(q.buffer.clone()), Binding::Buffer(k.buffer.clone()), Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        u(head_dim as u32), u(n_kv), u(kv_stride), u(gqa_factor), u(blocks), f(scale),
    ], Grid { grid: [n_kv_heads as u32, blocks, 1], block: [32 * gqa_factor, 1, 1] })?;

    let out = Tensor::empty(dev, vec![n_q_heads, head_dim], q.dtype)?;
    let k2 = cached_ir("sdpa_decode_2pass_pass2", q.dtype, || {
        let mut k = metaltile_std::ffai::sdpa_decode_2pass::sdpa_decode_2pass_pass2::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k2, &[
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        Binding::Buffer(out.buffer.clone()), u(head_dim as u32), u(blocks),
    ], Grid { grid: [n_q_heads as u32, 1, 1], block: [1024, 1, 1] })?;
    Ok(out)
}

/// Run a registered elementwise kernel (Grid3D, one thread per element) with
/// the given ordered bindings, producing a fresh `out`-shaped output.
fn elementwise_kernel(
    dev: &dyn Device,
    name: &str,
    dtype: DType,
    out_shape: Vec<usize>,
    mut bindings: Vec<Binding>,
) -> Result<Tensor> {
    let k = lookup(name, dtype)?;
    let out = Tensor::empty(dev, out_shape, dtype)?;
    bindings.push(Binding::Buffer(out.buffer.clone()));
    let n = out.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    // output binding goes in its signature position — callers pass inputs,
    // we append the single output last (matches a,…,out param order).
    dev.dispatch(&k, &bindings, grid)?;
    Ok(out)
}

/// SiLU activation `out = x * sigmoid(x)`, elementwise. Dispatches `mt_silu`.
pub fn silu(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    elementwise_kernel(dev, "mt_silu", x.dtype, x.shape.clone(), vec![Binding::Buffer(x.buffer.clone())])
}

/// ReLU² activation `out = max(x,0)²`, elementwise — NemotronH MoE expert act.
/// Built inline (Relu → square) so the expert `up → relu² → down` chain stays
/// ON-DEVICE (no host round-trip between the two Q8 GEMVs). Dispatches `mt_relu2`.
pub fn relu2(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    let mut k = Kernel::new("mt_relu2");
    for (pname, is_out) in [("a", false), ("c", true)] {
        k.params.push(Param { name: pname.into(), dtype: x.dtype, shape: Shape::scalar(), is_output: is_out, kind: ParamKind::Tensor });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None }, ValueId::new(1));
    k.body.push_op(Op::Activation { kind: ActKind::Relu, value: ValueId::new(1) }, ValueId::new(2));
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(2) }, ValueId::new(3));
    k.body.push_op_no_result(Op::Store { dst: "c".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], value: ValueId::new(3), mask: None });
    k.mode = metaltile_core::ir::KernelMode::Elementwise;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let n = out.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(&k, &[Binding::Buffer(x.buffer.clone()), Binding::Buffer(out.buffer.clone())], grid)?;
    Ok(out)
}

/// Build an Elementwise `out[i] = act(a[i])` kernel for `dtype`. (`mt_gelu`
/// is bench-only in the metaltile corpus — no registered correctness test — so
/// we hand-build the IR the same way `binop_kernel` does for add/mul.)
fn unary_act_kernel(name: &str, dtype: DType, kind: ActKind) -> Kernel {
    let mut k = Kernel::new(name);
    for (pname, is_out) in [("a", false), ("c", true)] {
        k.params.push(Param {
            name: pname.into(),
            dtype,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
        ValueId::new(1),
    );
    k.body.push_op(Op::Activation { kind, value: ValueId::new(1) }, ValueId::new(2));
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(2),
        mask: None,
    });
    k
}

/// Fused multiply-add into an accumulator, elementwise IN-PLACE:
/// `acc[i] += x[i] · s[i]`. Lets the MoE expert sum stay ON-DEVICE — each
/// expert's `down` output is folded into `acc` on the GPU (one final download
/// per layer instead of one per expert). `s` is the per-expert weight broadcast
/// to `[len]`. Dispatches `mt_fma_inplace`.
pub fn fma_inplace(dev: &dyn Device, acc: &Tensor, x: &Tensor, s: &Tensor) -> Result<()> {
    let mut k = Kernel::new("mt_fma_inplace");
    for (pname, is_out) in [("acc", true), ("x", false), ("s", false)] {
        k.params.push(Param { name: pname.into(), dtype: acc.dtype, shape: Shape::scalar(), is_output: is_out, kind: ParamKind::Tensor });
    }
    let idx = || vec![IndexExpr::Value(ValueId::new(0))];
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(Op::Load { src: "acc".into(), indices: idx(), mask: None, other: None }, ValueId::new(1));
    k.body.push_op(Op::Load { src: "x".into(), indices: idx(), mask: None, other: None }, ValueId::new(2));
    k.body.push_op(Op::Load { src: "s".into(), indices: idx(), mask: None, other: None }, ValueId::new(3));
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(3) }, ValueId::new(4));
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(4) }, ValueId::new(5));
    k.body.push_op_no_result(Op::Store { dst: "acc".into(), indices: idx(), value: ValueId::new(5), mask: None });
    k.mode = metaltile_core::ir::KernelMode::Elementwise;
    let n = acc.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(&k, &[Binding::Buffer(acc.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(s.buffer.clone())], grid)?;
    Ok(())
}

/// GELU activation (PyTorch tanh approximation, = `gelu_pytorch_tanh`),
/// elementwise. Hand-built `Activation(Gelu)` kernel. Used by ViT / SigLIP /
/// CLIP towers and the GELU-MLP LLM families.
pub fn gelu(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let k = unary_act_kernel("mt_gelu", x.dtype, ActKind::Gelu);
    let n = x.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[Binding::Buffer(x.buffer.clone()), Binding::Buffer(out.buffer.clone())],
        grid,
    )?;
    Ok(out)
}

/// LayerNorm `out = (x - mean) / sqrt(var + eps) * w + b`, normalized over the
/// last dim. Mean-subtracting + bias (unlike RMSNorm). Dispatches the
/// Reduction-mode `mt_layer_norm` (one threadgroup per row, block = n/4 — needs
/// the row width `n` divisible by `4·lsize`; ViT/SigLIP widths are). Used by
/// every transformer with LayerNorm (vision towers, BERT-style, GPT-2, …).
pub fn layer_norm(dev: &dyn Device, x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    let n = *x.shape.last().ok_or_else(|| Error::Msg("layer_norm: scalar input".into()))?;
    let rows = x.elem_count() / n;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let eps_buf = dev.upload(&eps.to_le_bytes())?;
    let k = lookup("mt_layer_norm", x.dtype)?;
    let grid = Grid { grid: [rows as u32, 1, 1], block: [(n / 4) as u32, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(bias.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Buffer(eps_buf),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Fused SwiGLU `out = silu(gate) * up`, elementwise. Dispatches `mt_swiglu`.
pub fn swiglu(dev: &dyn Device, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.shape != up.shape {
        return Err(Error::Msg("swiglu: gate/up shape mismatch".into()));
    }
    elementwise_kernel(
        dev,
        "mt_swiglu",
        gate.dtype,
        gate.shape.clone(),
        vec![Binding::Buffer(gate.buffer.clone()), Binding::Buffer(up.buffer.clone())],
    )
}

/// Embedding gather: `out[t, :] = table[indices[t], :]`. `table` is
/// `[vocab, dim]`, `indices` is `[n_tokens]` (u32). Dispatches `ffai_gather`.
pub fn gather(dev: &dyn Device, table: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if table.shape.len() != 2 {
        return Err(Error::Msg(format!("gather: table must be 2-D, got {:?}", table.shape)));
    }
    let dim = table.shape[1];
    let n_tokens = indices.elem_count();
    let k = lookup("ffai_gather", table.dtype)?;
    let out = Tensor::empty(dev, vec![n_tokens, dim], table.dtype)?;
    let n = (n_tokens * dim) as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(table.buffer.clone()),
            Binding::Buffer(indices.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((dim as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// DeepSeek-V4 partial RoPE: rotate only the rope tail of each head
/// (`[n_nope .. head_dim]`, adjacent-pair / GPT-J convention) in place; the
/// nope dims pass through. `qk` is `[n_heads, head_dim]`. Full-attention
/// layers: YaRN disabled (freq_scale=1, ext_factor=0). Dispatches
/// `ffai_dsv4_partial_rope` (grid `[n_heads, half_rot, 1]`). Mutates and
/// returns a view of `qk`'s buffer (matches the Swift in-place RoPE).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_partial_rope(
    dev: &dyn Device,
    qk: &Tensor,
    n_heads: u32,
    head_dim: u32,
    n_nope: u32,
    half_rot: u32,
    position: u32,
    theta_base: f32,
    inverse: bool,
) -> Result<Tensor> {
    let k = lookup("ffai_dsv4_partial_rope", qk.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());
    // Bind qk as both input and output → in-place (nope dims preserved).
    let bindings = vec![
        Binding::Buffer(qk.buffer.clone()),
        Binding::Buffer(qk.buffer.clone()),
        u(head_dim),
        u(n_nope),
        u(half_rot),
        u(position),
        f(theta_base),
        u(if inverse { 1 } else { 0 }),
        f(1.0), // freq_scale (YaRN off)
        f(0.0), // ext_factor
        f(0.0), // corr_low
        f(1.0), // corr_high
    ];
    let grid = Grid { grid: [n_heads, half_rot, 1], block: [1, 1, 1] };
    dev.dispatch(&k, &bindings, grid)?;
    Ok(Tensor::new(qk.buffer.clone(), qk.shape.clone(), qk.dtype))
}

/// Softmax over the last dim, row-wise. Dispatches `mt_softmax`. Row width
/// `n` must be a multiple of 1024 (the kernel's 4-elems/thread loop).
pub fn softmax(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    let n = *x.shape.last().ok_or_else(|| Error::Msg("softmax: scalar input".into()))?;
    if n % 1024 != 0 {
        return Err(Error::Msg(format!("softmax: row width {n} must be a multiple of 1024")));
    }
    let rows = x.elem_count() / n;
    let k = lookup("mt_softmax", x.dtype)?;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let grid = Grid { grid: [rows as u32, 1, 1], block: [256, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Llama-style rotary position embedding applied to a `[n_heads, head_dim]`
/// Q or K tensor at sequence `position`. Dispatches `ffai_rope_llama`. Pass
/// the freq-band knobs disabled (`scale_factor=1`, `low=1`, `high=1`,
/// `orig_max=1e9`) for vanilla RoPE.
#[allow(clippy::too_many_arguments)]
pub fn rope_llama(
    dev: &dyn Device,
    qk: &Tensor,
    position: u32,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Result<Tensor> {
    let head_dim = *qk.shape.last().ok_or_else(|| Error::Msg("rope: scalar input".into()))?;
    let n_heads = qk.elem_count() / head_dim;
    let half = head_dim / 2;
    let k = lookup("ffai_rope_llama", qk.dtype)?;
    let out = Tensor::empty(dev, qk.shape.clone(), qk.dtype)?;
    let grid = Grid { grid: [n_heads as u32, half as u32, 1], block: [1, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qk.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((head_dim as u32).to_le_bytes().to_vec()),
            Binding::Scalar((half as u32).to_le_bytes().to_vec()),
            Binding::Scalar(position.to_le_bytes().to_vec()),
            Binding::Scalar(theta_base.to_le_bytes().to_vec()),
            Binding::Scalar(scale_factor.to_le_bytes().to_vec()),
            Binding::Scalar(low_freq_factor.to_le_bytes().to_vec()),
            Binding::Scalar(high_freq_factor.to_le_bytes().to_vec()),
            Binding::Scalar(original_max_position.to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::dsv4_mhc_sinkhorn_split;

    #[test]
    fn sinkhorn_comb_is_doubly_stochastic() {
        // mixes/base arbitrary; after enough Sinkhorn iters the 4×4 comb
        // matrix must be (near) doubly-stochastic: every row and column ≈ 1.
        let mixes: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.2).collect();
        let base: Vec<f32> = (0..24).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let (pre, post, comb) = dsv4_mhc_sinkhorn_split(&mixes, [1.0, 1.0, 1.0], &base, 1e-6, 20);
        assert_eq!((pre.len(), post.len(), comb.len()), (4, 4, 16));
        for i in 0..4 {
            let row: f32 = (0..4).map(|j| comb[i * 4 + j]).sum();
            let col: f32 = (0..4).map(|j| comb[j * 4 + i]).sum();
            assert!((row - 1.0).abs() < 1e-2, "row {i} sum {row}");
            assert!((col - 1.0).abs() < 1e-2, "col {i} sum {col}");
        }
        // pre is sigmoid+eps ∈ (0,1+eps); post is 2·sigmoid ∈ (0,2).
        assert!(pre.iter().all(|&x| x > 0.0 && x < 1.01));
        assert!(post.iter().all(|&x| x > 0.0 && x < 2.0));
    }
}
