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

/// Mamba2 batched-prefill causal depthwise conv1d with inline SiLU (NEMOTRON_CONV_DEVICE path).
///
/// Processes all S prompt tokens in one dispatch with zero initial state (prefill
/// starts from scratch). Each thread computes one element of `y[s, conv_dim]` =
/// silu(bias[ch] + Σ_{k<kc} w[k,ch] * xbc_in[ti-(kc-1-k), ch]) where out-of-bounds
/// reads are zero. No host round-trip.
///
/// - `xbc_in`: `[s * conv_dim]` flat row-major
/// - `w`: `[kc * conv_dim]` reorganized (same layout as decode step: w[k, ch])
/// - `bias`: `[conv_dim]`
/// - returns `y [s * conv_dim]` with SiLU applied
pub fn conv1d_causal_prefill(
    dev: &dyn Device,
    xbc_in: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    s: usize,
    conv_dim: usize,
    kc: usize,
) -> Result<Tensor> {
    let kernel = lookup("conv1d_causal_prefill", DType::F32)?;
    let y = Tensor::empty(dev, vec![s * conv_dim], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(xbc_in.buffer.clone()),
            Binding::Buffer(w.buffer.clone()),
            Binding::Buffer(bias.buffer.clone()),
            Binding::Buffer(y.buffer.clone()),
            u(conv_dim as u32),
            u(kc as u32),
        ],
        Grid { grid: [(s * conv_dim) as u32, 1, 1], block: [1, 1, 1] },
    )?;
    Ok(y)
}

/// Extract `[s, width]` columns from row-major `src [s, stride]` at column offset `col_off`.
/// Returns `dst [s * width]` — a contiguous slab. Used to carve z, xbc, dt_raw out of
/// the [s, in_proj_out] projection matrix on device (NEMOTRON_CONV_DEVICE path).
pub fn strided_col_copy(
    dev: &dyn Device,
    src: &Tensor,
    s: usize,
    stride: usize,
    col_off: usize,
    width: usize,
) -> Result<Tensor> {
    let kernel = lookup("strided_col_copy", DType::F32)?;
    let dst = Tensor::empty(dev, vec![s * width], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(src.buffer.clone()),
            Binding::Buffer(dst.buffer.clone()),
            u(stride as u32),
            u(col_off as u32),
            u(width as u32),
        ],
        Grid { grid: [(s * width) as u32, 1, 1], block: [1, 1, 1] },
    )?;
    Ok(dst)
}

/// Batched softplus + tiled bias: `dst[ti*n + hi] = softplus(src[ti*n + hi] + bias[hi])`.
/// Converts the raw `[s, m_nh]` dt_raw tensor + dt_bias to `dt_all [s, m_nh]` on device.
/// (NEMOTRON_CONV_DEVICE path — replaces the CPU softplus loop.)
pub fn softplus_add_rows(
    dev: &dyn Device,
    src: &Tensor,
    bias: &Tensor,
    s: usize,
    n: usize,
) -> Result<Tensor> {
    let kernel = lookup("softplus_add_rows", DType::F32)?;
    let dst = Tensor::empty(dev, vec![s * n], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(src.buffer.clone()),
            Binding::Buffer(bias.buffer.clone()),
            Binding::Buffer(dst.buffer.clone()),
            u(n as u32),
        ],
        Grid { grid: [(s * n) as u32, 1, 1], block: [1, 1, 1] },
    )?;
    Ok(dst)
}

/// Batched MoE up+ReLU²: gather `top_k` experts (indices `idx`) from the
/// contiguous `[n_exp*inter, hid]` Q4 weight into one `[top_k*inter]` GEMV.
pub fn moe_gather_up_relu2(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, idx: &Tensor, top_k: usize, inter: usize, hid: usize) -> Result<Tensor> {
    let k = cached_ir("ffai_moe_gather_q4_relu2", x.dtype, || { let mut k = metaltile_std::ffai::gemv_q8::ffai_moe_gather_q4_relu2::kernel_ir_for(x.dtype); k.mode = metaltile_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![top_k * inter], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    // Default rpt=2: MoE-gather kernels are big + latency-bound; 2 warps/row hides
    // global-load latency (clean internal-A/B: +1.8% @ctx4096, MoE cost is ctx-independent).
    let rpt: u32 = std::env::var("MT_MOE_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(2);
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
    // Default rpt=2: MoE-gather kernels are big + latency-bound; 2 warps/row hides
    // global-load latency (clean internal-A/B: +1.8% @ctx4096, MoE cost is ctx-independent).
    let rpt: u32 = std::env::var("MT_MOE_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(2);
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(inter as u32), u(hid as u32), u(rpt)], Grid { grid: [(hid as u32).div_ceil(rpt), top_k as u32, 1], block: [32 * rpt, 1, 1] })?;
    Ok(out)
}
// ── Fused MoE FFN (NEMOTRON_MOE_FUSED=1) ─────────────────────────────────────
// Raw CUDA C++ source for the cooperative-groups two-phase fused kernel.
//
// Grid design for cooperative launch feasibility on GB10 (148 SMs, 2048 threads/SM):
//   total_warps = hid = 2688  (one warp per acc[h] output element in phase 2)
//   block = 256 (8 warps), grid = ceil(hid/8) = 336 blocks
//   336 blocks / 148 SMs ≈ 2.3 blocks/SM < 8 max → cooperative launch WORKS
//
// Phase 1: each warp serializes over its assigned slice of top_k*inter up-proj rows.
//   rows_per_warp ≈ (top_k * inter) / hid ≈ 4.14 (integer: 4 or 5 rows per warp)
//   → warp `w` computes scratch[row] for row in [w*n_up/hid .. (w+1)*n_up/hid)
// Phase 2: after grid.sync(), warp `w` computes acc[w] (one output element) by
//   reading ALL of scratch and the down Q4 weights (down-proj GEMV over inter).
//
// The 44KB scratch (6×1856×4B) lives in L2 (~2TB/s GB10) throughout, not HBM.
const MOE_FUSED_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Q4 (signed 4-bit, nibble-packed) + f16 scale per 32 → f32 warp dot product.
// lane 0 holds the result after return.
__device__ __forceinline__ float q4f16_dot_warp(
    const unsigned* __restrict__ qs,
    const __half*   __restrict__ sc,
    const float*    __restrict__ x,
    int row, int k_in, int lane)
{
    int bpr = k_in >> 5;
    int nw  = bpr * 4;
    const unsigned* qrow = qs + (size_t)row * nw;
    const __half*   drow = sc + (size_t)row * bpr;
    float dot = 0.f;
    for (int j = lane; j < nw; j += 32) {
        int blk = j >> 2, sub = j & 3;
        unsigned p = qrow[j];
        float sc_f = __half2float(drow[blk]);
        const float* xb = x + (blk << 5) + (sub << 3);
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int nb = (p >> (i * 4)) & 0xf;
            a += (float)(nb > 7 ? nb - 16 : nb) * xb[i];
        }
        dot += sc_f * a;
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
    return dot;
}

__device__ __forceinline__ float relu2(float v) {
    float r = v > 0.f ? v : 0.f;
    return r * r;
}

extern "C" __global__ void moe_fused_ffn(
    const unsigned* __restrict__ up_qs,   // [n_exp*inter, hid] Q4 up weights
    const __half*   __restrict__ up_sc,   // [n_exp*inter, hid/32] f16 scales
    const unsigned* __restrict__ dn_qs,   // [n_exp*hid, inter] Q4 down weights
    const __half*   __restrict__ dn_sc,   // [n_exp*hid, inter/32] f16 scales
    const float*    __restrict__ x,       // [hid] input activation
    const unsigned* __restrict__ idx,     // [top_k] expert indices (u32)
    const float*    __restrict__ wts,     // [top_k] router weights (f32)
    float*          __restrict__ acc,     // [hid] output (pre-zeroed by caller)
    float*          __restrict__ scratch, // [top_k * inter] temp (pre-allocated)
    int hid, int inter, int top_k)
{
    cg::grid_group gg = cg::this_grid();

    // Global warp id — one warp per acc[h] for phase 2.
    int tid     = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane    = threadIdx.x & 31;
    int warp_id = tid >> 5;
    int n_warps = (int)gridDim.x * ((int)blockDim.x >> 5);

    // ── Phase 1: each warp serializes over its slice of up-proj rows ──────
    // Linearised row space: row_lin = k * inter + i  (k=expert index 0..top_k-1)
    int n_up = top_k * inter;
    // Use 64-bit division to avoid overflow; compiler optimises for small top_k.
    int row_start = (int)((long long)warp_id * n_up / n_warps);
    int row_end   = (int)((long long)(warp_id + 1) * n_up / n_warps);
    for (int row_lin = row_start; row_lin < row_end; row_lin++) {
        int k   = row_lin / inter;
        int i   = row_lin % inter;
        int exp = (int)idx[k];
        int row = exp * inter + i;
        float v = q4f16_dot_warp(up_qs, up_sc, x, row, hid, lane);
        if (lane == 0) scratch[k * inter + i] = relu2(v);
    }

    // ── Global grid barrier: all up-proj results are in scratch ──────────
    gg.sync();

    // ── Phase 2: each warp computes one acc[h] via down-proj ─────────────
    if (warp_id < hid) {
        int h = warp_id;
        float acc_h = 0.f;
        for (int k = 0; k < top_k; k++) {
            int exp = (int)idx[k];
            int row = exp * hid + h;
            int bpr = inter >> 5;
            int nw  = bpr * 4;
            const unsigned* qrow  = dn_qs + (size_t)row * nw;
            const __half*   drow  = dn_sc + (size_t)row * bpr;
            const float*    xb_base = scratch + k * inter;
            float dot = 0.f;
            for (int j = lane; j < nw; j += 32) {
                int blk = j >> 2, sub = j & 3;
                unsigned p = qrow[j];
                float sc_f = __half2float(drow[blk]);
                const float* xb = xb_base + (blk << 5) + (sub << 3);
                float a = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int nb = (p >> (i * 4)) & 0xf;
                    a += (float)(nb > 7 ? nb - 16 : nb) * xb[i];
                }
                dot += sc_f * a;
            }
            #pragma unroll
            for (int o = 16; o; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
            if (lane == 0) acc_h += dot * wts[k];
        }
        if (lane == 0) acc[h] += acc_h;
    }
}
"#;

/// Fused MoE FFN: up-proj + relu² + down-proj + router-weighted accumulate,
/// all in ONE cooperative-groups kernel. The `[top_k×inter]` intermediate lives
/// in L2 cache instead of HBM (scratch pre-allocated once by caller).
///
/// Replaces `moe_gather_up_relu2` + `moe_gather_down` + `moe_weighted_sum` with
/// a single launch. Enabled by `NEMOTRON_MOE_FUSED=1`.
///
/// `scratch` must be a device buffer of at least `top_k * inter * 4` bytes.
/// `acc` must be zero-initialised by the caller before the first expert's
/// contribution (matches `moe_gather_down_accum` contract).
#[allow(clippy::too_many_arguments)]
pub fn moe_fused_ffn(
    dev: &dyn Device,
    up_qs: &Tensor, up_sc: &Tensor,
    dn_qs: &Tensor, dn_sc: &Tensor,
    x: &Tensor,
    idx: &Tensor,
    wts: &Tensor,
    acc: &Tensor,
    scratch: &Tensor,
    hid: usize, inter: usize, top_k: usize,
) -> Result<()> {
    // n_warps = hid: one warp per acc[h] output element (phase 2).
    // Phase 1 distributes top_k*inter up-proj rows across the same n_warps warps
    // (each warp handles ≈4 rows = top_k*inter/hid ≈ 4.14 for NemotronH dims).
    // Grid: ceil(hid/16) = 168 blocks. GB10: 48 SMs × 4 blocks/SM (at 512 threads)
    // = 192 max cooperative blocks → 168 ≤ 192 ✓
    let block = 512u32;  // 16 warps/block
    let warps_per_block = block / 32;
    let grid = ((hid as u32) + warps_per_block - 1) / warps_per_block;
    let scalars: &[Vec<u8>] = &[
        (hid   as i32).to_le_bytes().to_vec(),
        (inter as i32).to_le_bytes().to_vec(),
        (top_k as i32).to_le_bytes().to_vec(),
    ];
    dev.dispatch_raw_cuda(
        MOE_FUSED_SRC,
        "moe_fused_ffn.cu",
        "moe_fused_ffn",
        &[
            (up_qs.buffer.as_ref(),   0),
            (up_sc.buffer.as_ref(),   0),
            (dn_qs.buffer.as_ref(),   0),
            (dn_sc.buffer.as_ref(),   0),
            (x.buffer.as_ref(),       0),
            (idx.buffer.as_ref(),     0),
            (wts.buffer.as_ref(),     0),
            (acc.buffer.as_ref(),     0),
            (scratch.buffer.as_ref(), 0),
        ],
        scalars,
        [grid, 1, 1],
        [block, 1, 1],
        0,
        true,  // cooperative = use cuLaunchCooperativeKernel for grid.sync()
    )
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
        // relu2 / accum: multi-warp capable (rows_per_tg warps/TG). The shared-
        // expert matrices (3712×2688, 2688×3712) are small; rpt>1 gives no
        // measured gain (these kernels are NOT latency-bound — the ~50% BW the
        // sync-bracketed profiler showed is a measurement artifact, the kernel
        // time is small). Default rpt=1 (bit-identical to the original single-
        // warp form); MT_GEMV_SHARED_RPT overrides for experimentation.
        let rpt: u32 = std::env::var("MT_GEMV_SHARED_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(1);
        b.push(u(rpt));
        Grid { grid: [(m_out as u32).div_ceil(rpt), 1, 1], block: [32 * rpt, 1, 1] }
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

/// Tensor-core GEMM via cuBLAS (Path A escape hatch). Computes the row-major
/// `out[m,n] = x[m,k] · w[n,k]ᵀ` — the projection `out[r,o] = Σ_k w[o,k]·x[r,k]`
/// where `w` is the DENSE `[n,k]` weight (f16/bf16, NOT quantized) and `x` is
/// `[m,k]`. Runs on the GB10 tensor cores (f32 accumulate). The caller dequants
/// the Q4 weight to f16/bf16 once, then this hits real TFLOP/s (vs the
/// coop_tile software emulation at ~0.1% of peak).
pub fn gemm_cublas(
    dev: &dyn Device,
    x: &Tensor,
    w: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor> {
    if x.dtype != w.dtype {
        return Err(Error::Msg(format!("gemm_cublas: x/w dtype mismatch {:?} vs {:?}", x.dtype, w.dtype)));
    }
    let out = Tensor::empty(dev, vec![m, n], x.dtype)?;
    dev.gemm_tc(x.buffer.as_ref(), w.buffer.as_ref(), out.buffer.as_ref(), m, n, k, x.dtype)?;
    Ok(out)
}

/// Dense Q4 GEMM via cooperative-tensor MMA (`ffai_gemm_q4_mpp`). Weight
/// `[out_dim, k_in]` is the bench's Q4 layout (qs u32 4-words/block, scales
/// f16 amax/7); `x` is `[n_rows, k_in]` of dtype T; out is `[n_rows, out_dim]`
/// of T. The tensor-core projection GEMM for prefill (replaces the f32 scalar
/// `matmul` that sat at ~0.1% of peak). Reduction-mode, grid
/// `[ceil(out/64), ceil(rows/64), 1]`, block 128. `k_in % 32 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_mpp(
    dev: &dyn Device,
    x: &Tensor,
    qs: &Tensor,
    scales: &Tensor,
    n_rows: usize,
    out_dim: usize,
    k_in: usize,
) -> Result<Tensor> {
    use metaltile_core::ir::KernelMode;
    if k_in % 32 != 0 {
        return Err(Error::Msg(format!("gemm_q4_mpp: k_in {k_in} must be a multiple of 32")));
    }
    let out = Tensor::empty(dev, vec![n_rows, out_dim], x.dtype)?;
    let kern = cached_ir("ffai_gemm_q4_mpp", x.dtype, || {
        let mut kk = metaltile_std::ffai::gemm_q4_mpp::ffai_gemm_q4_mpp::kernel_ir_for(x.dtype);
        kk.mode = KernelMode::Reduction;
        kk
    });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid {
        grid: [(out_dim as u32).div_ceil(64), (n_rows as u32).div_ceil(64), 1],
        block: [128, 1, 1],
    };
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(n_rows as u32),
            u(out_dim as u32),
            u(k_in as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Routed-expert MoE Q4 grouped BGEMM via MMA (`ffai_moe_bgemm_q4_bm64`). The
/// token rows of `x` `[m_total, k_in]` MUST be pre-sorted by expert, with
/// `indices` `[m_total]` giving each row's expert id. Expert weights are the
/// contiguous Q4 pool `[n_experts*n_out, k_in]` (qs u32 + scales f16). Output
/// `[m_total, n_out]` (sorted order; host scatters back to token order). The
/// batched-over-S replacement for the per-token MoE gather loop. Reduction-mode,
/// grid `[n_out/64, ceil(m_total/64), 1]`, block 128. `n_out % 64 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn moe_bgemm_q4_bm64(
    dev: &dyn Device,
    x: &Tensor,
    qs: &Tensor,
    scales: &Tensor,
    indices: &Tensor,
    m_total: usize,
    n_out: usize,
    k_in: usize,
) -> Result<Tensor> {
    use metaltile_core::ir::KernelMode;
    if n_out % 64 != 0 {
        return Err(Error::Msg(format!("moe_bgemm_q4_bm64: n_out {n_out} must be a multiple of 64")));
    }
    let out = Tensor::empty(dev, vec![m_total, n_out], x.dtype)?;
    let kern = cached_ir("ffai_moe_bgemm_q4_bm64", x.dtype, || {
        let mut kk = metaltile_std::ffai::moe_bgemm_q4_bm64::ffai_moe_bgemm_q4_bm64::kernel_ir_for(x.dtype);
        kk.mode = KernelMode::Reduction;
        kk
    });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid {
        grid: [(n_out as u32) / 64, (m_total as u32).div_ceil(64), 1],
        block: [128, 1, 1],
    };
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(indices.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(m_total as u32),
            u(n_out as u32),
            u(k_in as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Q4 → dense `[m, k]` dequant. Expands a resident Q4 weight (the bench's
/// `quantize_q4` layout: `qs [m·(k/32)·4]` u32, signed nibbles; `scales
/// [m·(k/32)]` f32, amax/7) into a dense slab of `out_dtype` for the
/// compute-bound prefill GEMM (dequant-once → tensor-core `ffai_gemm`).
/// 1-D grid, one thread per output value.
pub fn dequant_q4(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    m: usize,
    k: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    dequant_q4_off(dev, qs, scales, m, k, out_dtype, 0)
}

/// Like [`dequant_q4`] but dequants the `[m,k]` slab starting at block offset
/// `blk_off` (in 32-value Q4 blocks) inside the qs/scales pool — used to peel
/// one MoE expert's rows out of the contiguous `[n_exp*out, k]` Q4 pool without
/// a tensor view. `blk_off = expert * m * (k/32)`.
#[allow(clippy::too_many_arguments)]
pub fn dequant_q4_off(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    m: usize,
    k: usize,
    out_dtype: DType,
    blk_off: usize,
) -> Result<Tensor> {
    use metaltile_core::ir::KernelMode;
    let n = m * k;
    let out = Tensor::empty(dev, vec![m, k], out_dtype)?;
    let kern = cached_ir("ffai_dequant_q4", out_dtype, || {
        let mut kk = metaltile_std::ffai::ffai_dequant_q4::ffai_dequant_q4::kernel_ir_for(out_dtype);
        kk.mode = KernelMode::Grid3D;
        kk
    });
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid::d1((n as u32).div_ceil(256), 256);
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(k as u32),
            u(n as u32),
            u(blk_off as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Multi-query (prefill) SDPA — attends `n_query` query rows against a shared
/// K/V cache in one dispatch. `causal=1` → query `r` attends `[0, base_kv+r+1)`
/// (causal within the block, prefix fully visible); `causal=0` → bidirectional
/// over `[0, base_kv+n_query)`. Q/out `[n_query, n_q_heads, head_dim]`, K/V
/// `[n_kv_heads, kv_stride, head_dim]`. head_dim hard-128. This is the prefill
/// flash-attn (`ffai_sdpa_multi`), Reduction-mode, grid `[n_q_heads*n_query]`,
/// block 1024.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_multi(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_q_heads: u32,
    base_kv: u32,
    n_query: u32,
    kv_stride: u32,
    heads_per_group: u32,
    causal: bool,
    scale: f32,
) -> Result<Tensor> {
    use metaltile_core::ir::KernelMode;
    if head_dim != 128 {
        return Err(Error::Msg(format!("sdpa_multi: head_dim must be 128, got {head_dim}")));
    }
    let out = Tensor::empty(dev, vec![n_query as usize, n_q_heads as usize, head_dim], q.dtype)?;
    let kern = cached_ir("ffai_sdpa_multi", q.dtype, || {
        let mut kk = metaltile_std::ffai::sdpa_multi::ffai_sdpa_multi::kernel_ir_for(q.dtype);
        kk.mode = KernelMode::Reduction;
        kk
    });
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid { grid: [n_q_heads * n_query, 1, 1], block: [1024, 1, 1] };
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(q.buffer.clone()),
            Binding::Buffer(k.buffer.clone()),
            Binding::Buffer(v.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(head_dim as u32),
            u(n_q_heads),
            u(base_kv),
            u(n_query),
            u(kv_stride),
            u(heads_per_group),
            u(u32::from(causal)),
            f(scale),
        ],
        grid,
    )?;
    Ok(out)
}

/// Raw CUDA source for the **chunked parallel SSD prefill scan** for the
/// NemotronH-Nano-30B Mamba2 cell (Dh=64, Ds=128, H=64, G=8, n_per_t=4).
///
/// The existing `ssm_step_record_d64_128_64_8` runs a serial loop over T
/// positions per (h, dh) cell — the serial chain of length T is the bottleneck.
/// This kernel replaces it with a **two-pass parallel segment scan**:
///
/// **Pass 1 (parallel)**: `N_SEG` warp-segments each process `T/N_SEG` positions
///   independently (assuming initial state = 0), computing a "partial state"
///   contribution and the segment's decay product `alpha_seg`.
///
/// **Pass 2 (sequential, warp 0 only)**: Stitch the true initial state for each
///   segment by walking the T/N_SEG chain: `s_in[k+1] = alpha_seg[k] * s_in[k] + partial[k]`.
///
/// **Pass 3 (parallel)**: Each segment re-computes its positions with the true
///   `s_in[seg]` and emits the correct `y[t]` values.
///
/// **Complexity**: 2T work (passes 1+3) + T/N_SEG serial (pass 2). At N_SEG=32,
/// depth drops from T=2048 to T/32=64. Speedup ≈ 32× for pure serial bottleneck.
///
/// **Memory**: Shared: `N_SEG × n_per_t × WARP` = 32×4×32 = 4096 f32 per block ≈ 16KB.
/// Block size: `[WARP × N_SEG, 1, 1]`; Grid3D: `[1, Dh, H]` (same as sequential).
///
/// **Dimensions fixed for NemotronH**: Dh=64, Ds=128, H=64, G=8, n_per_t=4.
/// The Ds/warp = 128/32 = 4 state values per lane (= n_per_t). G/H ratio = 8/64 = 1/8.
const SSM_CHUNKED_SCAN_SRC: &str = r#"
#include <cuda_fp16.h>
// Note: math.h / cmath not needed for NVRTC device code.
// expf, __shfl_xor_sync etc. are CUDA intrinsics, available without includes.

// NemotronH-Nano Mamba2 cell constants (hardcoded for performance).
#define DH      64u   // head_dim
#define DS      128u  // state_dim
#define NH      64u   // n_heads
#define NG      8u    // n_groups (G = H/ratio, ratio = NH/NG = 8)
#define NPT     4u    // n_per_t = DS / WARP (128/32 = 4 state slots per lane)
#define WARP    32u

// N_SEG segments per (h, dh) cell. Must divide T cleanly.
// 32 segments → at T=2048 each segment handles 64 steps; T=512 → 16 steps.
// Shared mem = N_SEG * NPT * WARP * 4 = 32*4*32*4 = 16KB + alpha (128B) + s_in (512B) per block.
// Block size = WARP * N_SEG = 32*32 = 1024 (CUDA max threads/block limit).
#define N_SEG   32u

// Total threads per block: WARP * N_SEG = 32 * 32 = 1024.
// Grid: [1, DH, NH] = [1, 64, 64] = 4096 blocks (same as sequential kernel).

extern "C" __global__ __launch_bounds__(WARP * N_SEG)
void ssm_chunked_prefill(
    const float*  __restrict__ x,        // [T, NH, DH]
    const float*  __restrict__ a_log,    // [NH]
    const float*  __restrict__ b_mat,    // [T, NG, DS]
    const float*  __restrict__ c_mat,    // [T, NG, DS]
    const float*  __restrict__ d_skip,   // [NH]
    const float*  __restrict__ dt,       // [T, NH]
    const float*  __restrict__ state_in, // [NH, DH, DS]
    float*        __restrict__ y,        // [T, NH, DH]
    float*        __restrict__ state_out,// [NH, DH, DS]
    unsigned int  t_total)
{
    // Each block owns one (d_idx, h_idx) cell.
    const unsigned int d_idx  = blockIdx.y;           // ∈ [0, DH)
    const unsigned int h_idx  = blockIdx.z;           // ∈ [0, NH)
    const unsigned int g_idx  = h_idx / (NH / NG);   // group index ∈ [0, NG)
    const unsigned int lane   = threadIdx.x & (WARP - 1u);  // ∈ [0, 32)
    const unsigned int seg_id = threadIdx.x / WARP;         // ∈ [0, N_SEG)

    // Shared memory: partial state contribution for each segment × state slots.
    // Layout: [N_SEG, NPT] where NPT lanes × each 1 float = 128 floats/segment.
    __shared__ float sh_partial[N_SEG * NPT * WARP];  // seg → [NPT, WARP] colmajor
    __shared__ float sh_alpha[N_SEG];                  // seg decay product
    __shared__ float sh_s_in[NPT * WARP];              // true initial state for each seg (broadcast)

    // Chunk parameters.
    const unsigned int T = t_total;
    const unsigned int seg_len = (T + N_SEG - 1u) / N_SEG;  // positions per segment (~T/N_SEG)
    const unsigned int t_start = seg_id * seg_len;
    const unsigned int t_end   = (t_start + seg_len < T) ? (t_start + seg_len) : T;

    // A = -exp(A_log[h]) (constant for this cell).
    const float a_neg = -expf(a_log[h_idx]);

    // State-slice offsets for this (h, d_idx) cell.
    const unsigned int state_base = (h_idx * DH + d_idx) * DS;

    // ── Pass 1: Compute partial state and alpha for this segment ─────────────
    // Assume state_in = 0. Walk t_start..t_end updating local state in registers.
    // Each lane owns NPT=4 consecutive Ds slots: lane*NPT .. lane*NPT+3.
    float local_s[NPT] = {0.f, 0.f, 0.f, 0.f};  // partial state (assuming s_in=0)
    float seg_alpha = 1.f;   // product of dA over [t_start, t_end)

    for (unsigned int t = t_start; t < t_end; ++t) {
        const unsigned int bt_h = t * NH + h_idx;
        const unsigned int bt_g = t * NG + g_idx;
        const float dt_v   = dt[bt_h];
        const float d_a    = expf(a_neg * dt_v);
        const float x_v    = x[bt_h * DH + d_idx];

        seg_alpha *= d_a;

        // Update local state and accumulate y later (will redo in pass 3).
        for (unsigned int i = 0u; i < NPT; ++i) {
            const unsigned int s_idx = lane * NPT + i;
            const float b_v = b_mat[bt_g * DS + s_idx];
            local_s[i] = d_a * local_s[i] + dt_v * b_v * x_v;
        }
    }

    // Write partial state to shared memory: sh_partial[seg_id, lane, i].
    for (unsigned int i = 0u; i < NPT; ++i) {
        sh_partial[(seg_id * NPT + i) * WARP + lane] = local_s[i];
    }
    if (lane == 0u) sh_alpha[seg_id] = seg_alpha;

    __syncthreads();

    // ── Pass 2: Sequential stitch (only seg 0 of each block = warp 0) ─────
    // Walk segments 0..N_SEG sequentially to compute true s_in for each segment.
    // Seg 0's s_in is the model's real state_in (loaded from global memory).
    if (seg_id == 0u) {
        // Load true initial state into sh_s_in (seg 0's start = state_in).
        for (unsigned int i = 0u; i < NPT; ++i) {
            sh_s_in[i * WARP + lane] = state_in[state_base + (lane * NPT + i)];
        }
        // Seg 0: its partial result already assumed s_in=0; add the real s_in×alpha.
        // (We store the TRUE s_in for seg k, then update to seg k+1's s_in.)
        for (unsigned int seg = 0u; seg < N_SEG; ++seg) {
            const float alpha_s = sh_alpha[seg];
            // True s_in for next seg = alpha_s * true_s_in[seg] + partial[seg].
            for (unsigned int i = 0u; i < NPT; ++i) {
                float s_in_val = sh_s_in[i * WARP + lane];
                float partial  = sh_partial[(seg * NPT + i) * WARP + lane];
                // The corrected partial for this seg: partial + alpha_s * s_in_val
                // We store the NEXT seg's s_in (= alpha_s * s_in_val + partial).
                sh_s_in[i * WARP + lane] = alpha_s * s_in_val + partial;
                // Overwrite partial with corrected carry for pass 3 to read.
                sh_partial[(seg * NPT + i) * WARP + lane] = s_in_val;
            }
        }
        // sh_s_in now holds the true final state after all T positions.
    }

    __syncthreads();

    // ── Pass 3: Recompute with true s_in, emit y ────────────────────────
    // sh_partial[seg_id * NPT + i, lane] now holds the TRUE s_in for this seg.
    for (unsigned int i = 0u; i < NPT; ++i) {
        local_s[i] = sh_partial[(seg_id * NPT + i) * WARP + lane];
    }

    for (unsigned int t = t_start; t < t_end; ++t) {
        const unsigned int bt_h = t * NH + h_idx;
        const unsigned int bt_g = t * NG + g_idx;
        const float dt_v   = dt[bt_h];
        const float d_a    = expf(a_neg * dt_v);
        const float x_v    = x[bt_h * DH + d_idx];

        float y_acc = 0.f;
        for (unsigned int i = 0u; i < NPT; ++i) {
            const unsigned int s_idx = lane * NPT + i;
            const float b_v = b_mat[bt_g * DS + s_idx];
            const float c_v = c_mat[bt_g * DS + s_idx];
            local_s[i] = d_a * local_s[i] + dt_v * b_v * x_v;
            y_acc += c_v * local_s[i];
        }
        // Warp-reduce y_acc across all 32 lanes.
        for (unsigned int mask = WARP >> 1u; mask > 0u; mask >>= 1u)
            y_acc += __shfl_xor_sync(0xffffffffu, y_acc, mask);

        if (lane == 0u) {
            const float d_v = d_skip[h_idx];
            y[bt_h * DH + d_idx] = y_acc + x_v * d_v;
        }
    }

    // ── Write state_out ───────────────────────────────────────────────────
    // Only warp 0 (seg_id=0, lane=all) holds the FINAL state in sh_s_in.
    if (seg_id == 0u) {
        for (unsigned int i = 0u; i < NPT; ++i) {
            state_out[state_base + (lane * NPT + i)] = sh_s_in[i * WARP + lane];
        }
    }
}
"#;

/// Mamba2 SSD **chunked parallel prefill scan** for the NemotronH cell
/// (Dh=64, Ds=128, H=64, G=8). Replaces the sequential-in-T `ssm_prefill_scan`
/// with a two-pass parallel segment scan that drops the serial depth from T to
/// T/N_SEG=T/64. Gated by `NEMOTRON_CHUNKED_SCAN=1`; correctness gate is the
/// same `NEMOTRON_PREFILL_CHECK` comparison against sequential.
///
/// Grid3D: `[1, Dh=64, H=64]`, block `[WARP × N_SEG]` = `[32 × 32 = 1024]`.
/// Shared mem per block: ~17KB (partial states + alpha + s_in scratch).
///
/// Returns `(state_out, y)` — same signature as `ssm_prefill_scan`.
#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_scan_chunked(
    dev: &dyn Device,
    x: &Tensor,
    a_log: &Tensor,
    b_mat: &Tensor,
    c_mat: &Tensor,
    d_skip: &Tensor,
    dt: &Tensor,
    state_in: &Tensor,
    t_total: u32,
    dh: u32,
    ds: u32,
    n_heads: u32,
    n_groups: u32,
) -> Result<(Tensor, Tensor)> {
    if (dh, ds, n_heads, n_groups) != (64, 128, 64, 8) {
        return Err(Error::Msg(format!(
            "ssm_prefill_scan_chunked: only Nemotron cell (64,128,64,8) wired, got ({dh},{ds},{n_heads},{n_groups})"
        )));
    }
    let t = t_total as usize;
    let (nh, dhu, _dsu) = (n_heads as usize, dh as usize, ds as usize);
    let y = Tensor::empty(dev, vec![t * nh * dhu], x.dtype)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), x.dtype)?;
    // N_SEG must divide t_total (or kernel pads). 64 divides 512,2048,8192.
    // For t_total not divisible by 64, pass as-is — the kernel uses a guard.
    let t_bytes = t_total.to_le_bytes().to_vec();
    // Grid: [1, Dh, H]. Block: [WARP * N_SEG] = [32 * 64] = [2048].
    let grid = [1u32, dh, n_heads];
    let block = [32u32 * 32, 1, 1]; // WARP * N_SEG = 32*32 = 1024 (CUDA max threads/block)
    // Shared mem: N_SEG * NPT * WARP * 4 + N_SEG * 4 + NPT * WARP * 4
    //           = 32*4*32*4 + 32*4 + 4*32*4 = 16384 + 128 + 512 = 17024 bytes
    let shared_bytes: u32 = 32 * 4 * 32 * 4 + 32 * 4 + 4 * 32 * 4;
    // Cast inputs to f32 for the raw CUDA kernel (which reads f32).
    // The existing sequential kernel also accumulates in f32.
    // NOTE: if x/state are f16/bf16 we need casts. For now, only f32 is wired.
    if x.dtype != DType::F32 {
        return Err(Error::Msg("ssm_prefill_scan_chunked: only F32 input wired (cast first)".into()));
    }
    dev.dispatch_raw_cuda(
        SSM_CHUNKED_SCAN_SRC,
        "ssm_chunked_prefill.cu",
        "ssm_chunked_prefill",
        &[
            (x.buffer.as_ref(), 0),
            (a_log.buffer.as_ref(), 0),
            (b_mat.buffer.as_ref(), 0),
            (c_mat.buffer.as_ref(), 0),
            (d_skip.buffer.as_ref(), 0),
            (dt.buffer.as_ref(), 0),
            (state_in.buffer.as_ref(), 0),
            (y.buffer.as_ref(), 0),
            (state_out.buffer.as_ref(), 0),
        ],
        &[t_bytes],
        grid,
        block,
        shared_bytes,
        false,
    )?;
    Ok((state_out, y))
}

/// Mamba2 SSD **batched-prefill scan** — runs the sequential SSD forward over
/// all `t_total` prompt tokens in ONE dispatch, emitting every per-token
/// `y[t]` plus the final recurrent `state_out` (for decode continuity).
/// Dispatches `ssm_step_record_d64_128_64_8` (Nemotron cell: Dh=64, Ds=128,
/// H=64, G=8). The `(da_log, dbx_log)` tapes are written but ignored here
/// (they exist for the speculative-rollback replay path). Layouts:
/// `x [t·H·Dh]`, `a_log/d [H]`, `dt [t·H]`, `b/c [t·G·Ds]`,
/// `state_in/out [H·Dh·Ds]`, `y [t·H·Dh]`. Grid3D `[1, Dh, H]`, tg `[32,1,1]`.
#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_scan(
    dev: &dyn Device,
    x: &Tensor,
    a_log: &Tensor,
    b_mat: &Tensor,
    c_mat: &Tensor,
    d_skip: &Tensor,
    dt: &Tensor,
    state_in: &Tensor,
    t_total: u32,
    dh: u32,
    ds: u32,
    n_heads: u32,
    n_groups: u32,
) -> Result<(Tensor, Tensor)> {
    use metaltile_core::ir::KernelMode;
    if (dh, ds, n_heads, n_groups) != (64, 128, 64, 8) {
        return Err(Error::Msg(format!(
            "ssm_prefill_scan: only Nemotron cell (64,128,64,8) wired, got ({dh},{ds},{n_heads},{n_groups})"
        )));
    }
    let t = t_total as usize;
    let (nh, dhu, dsu) = (n_heads as usize, dh as usize, ds as usize);
    let y = Tensor::empty(dev, vec![t * nh * dhu], x.dtype)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), x.dtype)?;
    // Tape outputs — written by the kernel, unused by prefill.
    let da_log = Tensor::empty(dev, vec![t * nh * dsu], x.dtype)?;
    let dbx_log = Tensor::empty(dev, vec![t * nh * dhu * dsu], x.dtype)?;
    // mask buffer: has_mask=0 → kernel ignores contents, but the binding must exist.
    let mask = Tensor::new(dev.upload(&vec![0u8; t * 4]).unwrap(), vec![t], DType::U32);
    let kern = cached_ir("ssm_step_record_d64_128_64_8", x.dtype, || {
        let mut kk = metaltile_std::ffai::ssm_replay::ssm_step_record_d64_128_64_8::kernel_ir_for(x.dtype);
        kk.mode = KernelMode::Grid3D;
        kk
    });
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid { grid: [1, dh, n_heads], block: [32, 1, 1] };
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(a_log.buffer.clone()),
            Binding::Buffer(b_mat.buffer.clone()),
            Binding::Buffer(c_mat.buffer.clone()),
            Binding::Buffer(d_skip.buffer.clone()),
            Binding::Buffer(dt.buffer.clone()),
            Binding::Buffer(state_in.buffer.clone()),
            Binding::Buffer(mask.buffer.clone()),
            Binding::Buffer(y.buffer.clone()),
            Binding::Buffer(state_out.buffer.clone()),
            Binding::Buffer(da_log.buffer.clone()),
            Binding::Buffer(dbx_log.buffer.clone()),
            u(t_total),
            u(0), // has_mask
        ],
        grid,
    )?;
    Ok((state_out, y))
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

/// Like `sdpa_decode_2pass` but uses the TILED pass-1 variant: assigns each
/// block a contiguous chunk of KV positions for L2-cache-friendly sequential
/// access (vs the strided pattern which thrashes L2). Activated by NEMOTRON_TILED=1.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_2pass_tiled(
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

    let k1 = cached_ir("sdpa_decode_2pass_pass1_tiled", q.dtype, || {
        let mut k = metaltile_std::ffai::sdpa_decode_2pass::sdpa_decode_2pass_pass1_tiled::kernel_ir_for(q.dtype);
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

/// Like `sdpa_decode_2pass` but uses the BC=4 pass-1 variant: processes 4 KV
/// positions per loop iteration to expose memory-level parallelism and hide
/// load latency. Same pass-2 reduction. Same correctness guarantees.
/// Activated by NEMOTRON_BC4=1 in the bench harness.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_2pass_bc4(
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

    // BC=4 pass 1: 4 positions per loop iter for MLP / load-latency hiding.
    let k1 = cached_ir("sdpa_decode_2pass_pass1_bc4", q.dtype, || {
        let mut k = metaltile_std::ffai::sdpa_decode_2pass::sdpa_decode_2pass_pass1_bc4::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k1, &[
        Binding::Buffer(q.buffer.clone()), Binding::Buffer(k.buffer.clone()), Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        u(head_dim as u32), u(n_kv), u(kv_stride), u(gqa_factor), u(blocks), f(scale),
    ], Grid { grid: [n_kv_heads as u32, blocks, 1], block: [32 * gqa_factor, 1, 1] })?;

    // Pass 2 unchanged: same partial buffer layout, same reduction.
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

/// Raw CUDA source for the **MoE scatter-add** kernel.
///
/// After the batched down-GEMM produces `dn_out [mt, hid]` (sorted by expert),
/// this kernel scatters each row back to the corresponding token slot in the
/// output accumulator, weighted by the router weight:
///
///   `acc[token_indices[r], h] += dn_out[r, h] * weights[r] * unscale`
///
/// Uses `atomicAdd(float*, float)` (available since CUDA 2.0/GPGPU) so multiple
/// experts for the same token add correctly. Block: `[BLOCK_H, 1]`; Grid:
/// `[mt, ceil(hid/BLOCK_H)]`. For NemotronH hid=2688, BLOCK_H=128: 21 col-blocks.
const MOE_SCATTER_ADD_SRC: &str = r#"
#include <cuda_fp16.h>
#define BLOCK_H 128

extern "C" __global__ void moe_scatter_add_f32(
    const float*   __restrict__ dn,      // [mt, hid] sorted-row expert outputs
    const unsigned* __restrict__ tidx,   // [mt] token index per row
    const float*   __restrict__ wts,     // [mt] router weights
    float*                      acc,     // [s, hid] output accumulator (pre-zeroed)
    int mt, int hid, float unscale)
{
    int r = (int)blockIdx.x;  // sorted row ∈ [0, mt)
    int h = (int)(blockIdx.y * BLOCK_H + threadIdx.x);  // hidden dim
    if (r < mt && h < hid) {
        float w = wts[r] * unscale;
        atomicAdd(&acc[(int)tidx[r] * hid + h], dn[r * hid + h] * w);
    }
}
"#;

/// Device-side MoE scatter-add: write `dn_out[r, h] * weights[r] * unscale`
/// into `acc[token_indices[r], h]` for each row `r` in the sorted batch.
/// `acc` must be pre-zeroed (use `Tensor::zeros`). Returns nothing (in-place).
///
/// Replaces the host scatter loop in the prefill MoE path, enabling fully
/// on-device processing when combined with device gather + batched bm64 GEMMs.
pub fn moe_scatter_add(
    dev: &dyn Device,
    dn: &Tensor,   // [mt, hid] f32 down-GEMM output (sorted)
    tidx: &Tensor, // [mt] u32 token indices
    wts: &Tensor,  // [mt] f32 router weights
    acc: &Tensor,  // [s, hid] f32 accumulator
    mt: usize,
    hid: usize,
    unscale: f32,
) -> Result<()> {
    let mt_i = (mt as i32).to_le_bytes().to_vec();
    let hid_i = (hid as i32).to_le_bytes().to_vec();
    let uscale_f = unscale.to_le_bytes().to_vec();
    const BLOCK_H: u32 = 128;
    let grid = [mt as u32, (hid as u32).div_ceil(BLOCK_H), 1];
    let block = [BLOCK_H, 1, 1];
    dev.dispatch_raw_cuda(
        MOE_SCATTER_ADD_SRC,
        "moe_scatter_add.cu",
        "moe_scatter_add_f32",
        &[
            (dn.buffer.as_ref(),   0),
            (tidx.buffer.as_ref(), 0),
            (wts.buffer.as_ref(),  0),
            (acc.buffer.as_ref(),  0),
        ],
        &[mt_i, hid_i, uscale_f],
        grid,
        block,
        0,
        false,
    )
}

/// Raw CUDA source for `relu2_scale_f16`: `out[i] = max(0, (float)in[i])^2 * scale`
/// where `in` and `out` are __half (f16). Fuses the cast-to-f32, relu, square,
/// scale, and cast-back-to-f16 that the MoE prefill loop previously did by
/// downloading to host, computing on CPU, and re-uploading. One kernel replaces
/// the full host-round-trip between the up and down GEMMs.
const RELU2_SCALE_F16_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void relu2_scale_f16(
    const __half* __restrict__ inp,
    __half*       __restrict__ out,
    int n, float scale)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float v = __half2float(inp[i]);
        if (v < 0.f) v = 0.f;
        out[i] = __float2half(v * v * scale);
    }
}
"#;

/// Fused relu² + scalar scale for f16 tensors, all on device.
/// Computes `out[i] = max(0, x[i])^2 * scale` (x and out are f16).
/// Used to replace the host round-trip between the MoE expert up and down GEMMs:
///   old: dl(a_f16) → relu2+scale on host → ul(a2_f16)  [forces GPU sync]
///   new: relu2_scale_f16(a_f16, scale)                  [stays on device]
pub fn relu2_scale_f16(dev: &dyn Device, x: &Tensor, scale: f32) -> Result<Tensor> {
    let n = x.elem_count();
    let out = Tensor::empty(dev, x.shape.clone(), DType::F16)?;
    let scale_bytes = scale.to_le_bytes().to_vec();
    let n_bytes = (n as i32).to_le_bytes().to_vec();
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    dev.dispatch_raw_cuda(
        RELU2_SCALE_F16_SRC,
        "relu2_scale_f16.cu",
        "relu2_scale_f16",
        &[(x.buffer.as_ref(), 0), (out.buffer.as_ref(), 0)],
        &[n_bytes, scale_bytes],
        [grid, 1, 1],
        [block, 1, 1],
        0,
        false,
    )?;
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

/// Batched Llama-style RoPE: apply position-dependent rotation to ALL T rows in
/// ONE dispatch. Replaces the T-loop of per-token `rope_llama` calls in the
/// prefill attention path, eliminating the dl(q_all)/dl(k_all)/dl(v_all) + S×dl
/// overhead for the attention layers.
///
/// `qk`: [T, n_heads, head_dim] float  
/// `positions`: [T] u32 positions (one per token row)  
/// returns: [T, n_heads, head_dim] rotated (same dtype as qk)
pub fn rope_llama_many(
    dev: &dyn Device,
    qk: &Tensor,
    positions: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Result<Tensor> {
    use metaltile_core::ir::KernelMode;
    let t = qk.elem_count() / (n_heads * head_dim);
    let half = head_dim / 2;
    let k = cached_ir("ffai_rope_llama_many", qk.dtype, || {
        let mut kk = metaltile_std::ffai::rope_llama_many::ffai_rope_llama_many::kernel_ir_for(qk.dtype);
        kk.mode = KernelMode::Grid3D;
        kk
    });
    let out = Tensor::empty(dev, qk.shape.clone(), qk.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let f = |v: f32| Binding::Scalar(v.to_le_bytes().to_vec());
    let row_stride = (n_heads * head_dim) as u32;
    let grid = Grid { grid: [t as u32, n_heads as u32, half as u32], block: [1, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qk.buffer.clone()),
            Binding::Buffer(positions.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(head_dim as u32),
            u(half as u32),
            u(row_stride),
            f(theta_base),
            f(scale_factor),
            f(low_freq_factor),
            f(high_freq_factor),
            f(original_max_position),
        ],
        grid,
    )?;
    Ok(out)
}

/// Batched KV-cache append: write T tokens' K (or V) rows into the per-head
/// cache in ONE dispatch. Replaces the T-loop of per-token `kv_append` calls.
///
/// `src`: [T, n_kv_heads, head_dim]  
/// `positions`: [T] u32 positions  
/// `cache`: [n_kv_heads, max_seq, head_dim] (pre-allocated)
pub fn kv_append_many(
    dev: &dyn Device,
    src: &Tensor,
    positions: &Tensor,
    cache: &Tensor,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> Result<()> {
    use metaltile_core::ir::KernelMode;
    let t = src.elem_count() / (n_kv_heads * head_dim);
    let k = cached_ir("kv_cache_update_many", src.dtype, || {
        let mut kk = metaltile_std::ffai::kv_cache_update_many::kv_cache_update_many::kernel_ir_for(src.dtype);
        kk.mode = KernelMode::Grid3D;
        kk
    });
    let total = t * n_kv_heads * head_dim;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(src.buffer.clone()),
            Binding::Buffer(positions.buffer.clone()),
            Binding::Buffer(cache.buffer.clone()),
            u(head_dim as u32),
            u(max_seq as u32),
            u((n_kv_heads * head_dim) as u32),
        ],
        Grid { grid: [total as u32, 1, 1], block: [1, 1, 1] },
    )?;
    Ok(())
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

// ── Fused Mamba projection split (Part-B: 3 strided_col_copy → 1 dispatch) ─
//
// Splits proj [s, in_proj_out] into z [s, di], xbc [s, conv_dim], dt_raw [s, m_nh]
// in ONE device dispatch. Replaces the 3 sequential `strided_col_copy` launches
// that were 20-37% of prefill time at S=512-2048 (138 dispatches/forward).
pub fn mamba_split_proj(
    dev: &dyn Device,
    proj: &Tensor,
    s: usize,
    in_proj_out: usize,
    di: usize,
    conv_dim: usize,
    m_nh: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let kernel = cached_ir("mamba_split_proj", DType::F32, || {
        use metaltile_core::ir::KernelMode;
        let mut k = metaltile_std::ffai::ssm::mamba_split_proj::kernel_ir_for();
        k.mode = KernelMode::Grid3D;
        k
    });
    let z_out  = Tensor::empty(dev, vec![s * di], DType::F32)?;
    let xbc_out = Tensor::empty(dev, vec![s * conv_dim], DType::F32)?;
    let dt_out  = Tensor::empty(dev, vec![s * m_nh], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(proj.buffer.clone()),
            Binding::Buffer(z_out.buffer.clone()),
            Binding::Buffer(xbc_out.buffer.clone()),
            Binding::Buffer(dt_out.buffer.clone()),
            u(in_proj_out as u32),
            u(di as u32),
            u(conv_dim as u32),
            u(m_nh as u32),
        ],
        Grid { grid: [(s * in_proj_out) as u32, 1, 1], block: [1, 1, 1] },
    )?;
    Ok((z_out, xbc_out, dt_out))
}

// ── Fused Mamba conv-output split (Part-B: 3 strided_col_copy → 1 dispatch) ─
//
// Splits yc_silu [s, conv_dim] into x [s, di], b [s, ng*ds], c [s, ng*ds]
// in ONE device dispatch.
pub fn mamba_split_conv(
    dev: &dyn Device,
    yc_silu: &Tensor,
    s: usize,
    conv_dim: usize,
    di: usize,
    ng_ds: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let kernel = cached_ir("mamba_split_conv", DType::F32, || {
        use metaltile_core::ir::KernelMode;
        let mut k = metaltile_std::ffai::ssm::mamba_split_conv::kernel_ir_for();
        k.mode = KernelMode::Grid3D;
        k
    });
    let x_out  = Tensor::empty(dev, vec![s * di], DType::F32)?;
    let b_out  = Tensor::empty(dev, vec![s * ng_ds], DType::F32)?;
    let c_out  = Tensor::empty(dev, vec![s * ng_ds], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(yc_silu.buffer.clone()),
            Binding::Buffer(x_out.buffer.clone()),
            Binding::Buffer(b_out.buffer.clone()),
            Binding::Buffer(c_out.buffer.clone()),
            u(conv_dim as u32),
            u(di as u32),
            u(ng_ds as u32),
        ],
        Grid { grid: [(s * conv_dim) as u32, 1, 1], block: [1, 1, 1] },
    )?;
    Ok((x_out, b_out, c_out))
}

// ── Batched gated group RMSNorm (Mamba2, prefill) ─────────────────────────
//
// Processes S tokens in one dispatch.  Each thread-group handles one (token,
// norm-group) pair.  Replaces the host dl+compute+upload loop in the
// CONV_DEVICE prefill path (TODO comment at line ~1543 of modeltests/lib.rs).
pub fn gated_group_rmsnorm_batched(
    dev: &dyn Device,
    y: &Tensor,
    z: &Tensor,
    w: &Tensor,
    eps: f32,
    s: usize,
    di: usize,
    gs: usize,
) -> Result<Tensor> {
    let ng = di / gs;
    let kernel = cached_ir("gated_group_rmsnorm_batched", DType::F32, || {
        use metaltile_core::ir::KernelMode;
        let mut k = metaltile_std::ffai::ssm::gated_group_rmsnorm_batched::kernel_ir_for();
        k.mode = KernelMode::Reduction;
        k
    });
    let out = Tensor::empty(dev, vec![s * di], DType::F32)?;
    let eps_buf = Tensor::new(dev.upload(&eps.to_le_bytes()).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![1], DType::F32);
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(y.buffer.clone()),
            Binding::Buffer(z.buffer.clone()),
            Binding::Buffer(w.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Buffer(eps_buf.buffer.clone()),
            u(gs as u32),
            u(ng as u32),
        ],
        Grid { grid: [(s * ng) as u32, 1, 1], block: [(gs / 4) as u32, 1, 1] },
    )?;
    Ok(out)
}

const BATCHED_DEQUANT_Q4_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void moe_batched_dequant_q4(
    const unsigned int* __restrict__ qs,
    const __half*       __restrict__ sc,
    const unsigned int* __restrict__ eid,
    __half*             __restrict__ out,
    int n_active, int n_out, int k_in, int bpr_in
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_active * n_out * k_in) return;
    int slot   = i / (n_out * k_in);
    int local  = i - slot * n_out * k_in;
    int r      = local / k_in;
    int c      = local - r * k_in;
    int expert = (int)eid[slot];
    int b      = c / 32;
    int j      = c % 32;
    int w_idx  = j / 8;
    int nib_idx= j % 8;
    int global_blk = expert * n_out * bpr_in + r * bpr_in + b;
    unsigned int word = qs[global_blk * 4 + w_idx];
    unsigned int nib  = (word >> (nib_idx * 4)) & 0xFu;
    int q_signed  = (nib >= 8u) ? (int)nib - 16 : (int)nib;
    float scale   = __half2float(sc[global_blk]);
    __half val    = __float2half((float)q_signed * scale);
    out[i] = val;
}
"#;

/// Fused grouped-GEMM MoE expert pass (CUDA 13+, `NEMOTRON_GROUPED_GEMM=1`).
///
/// Per-expert dequant → GEMM pipelined (interleaved): dequant expert i into its slot
/// in the pre-allocated scratch buffer, immediately GEMM expert i, continue. No host
/// sync between experts. Device relu2_scale_f16 eliminates the host relu2 round-trip.
///
/// Uses the pre-allocated scratch buffers (passed via `up_scratch`/`dn_scratch`) to
/// avoid per-expert `cuMemAlloc`. Each expert's weight slot in the scratch is
/// `slot * n_out * k_in * 2` bytes. The scratch must be pre-allocated to hold
/// `n_exp * n_out * k_in * 2` bytes (at least `n_active` slots needed).
///
/// Returns `[mt, hid]` f16 output (relu² applied, 1/256 scaled). Caller scatter-adds.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm(
    dev: &dyn Device,
    qs_up: &Tensor,
    sc_up: &Tensor,
    qs_dn: &Tensor,
    sc_dn: &Tensor,
    xs_f16: &Tensor,        // [mt, hid] f16 — rows sorted by expert id
    g_starts: &[usize],     // length = n_active + 1
    expert_ids: &[usize],   // expert id for each group
    hid: usize,
    inter: usize,
    bpr_up: usize,          // hid / 32
    bpr_dn: usize,          // inter / 32
    up_scratch: Option<&Tensor>, // Pre-allocated [n_exp*inter, hid] f16
    dn_scratch: Option<&Tensor>, // Pre-allocated [n_exp*hid, inter] f16
) -> Result<Tensor> {
    use ffai_core::DeviceBuffer;
    let n_active = expert_ids.len();
    let mt = g_starts.last().copied().unwrap_or(0);
    if n_active == 0 || mt == 0 {
        return Tensor::empty(dev, vec![mt.max(1), hid], DType::F16);
    }

    let eid_bytes: Vec<u8> = expert_ids.iter().flat_map(|&e| (e as u32).to_le_bytes()).collect();
    let eid_dev = dev.upload(&eid_bytes)
        .map_err(|e| Error::Msg(format!("moe_grouped_gemm: upload eid: {e:?}")))?;

    // Allocate UP scratch (or use pre-allocated).
    let up_w_owned: Option<Tensor>;
    let up_w = if let Some(scratch) = up_scratch {
        Tensor { buffer: scratch.buffer.clone(), offset: 0, shape: vec![n_active * inter, hid], dtype: DType::F16 }
    } else {
        up_w_owned = Some(Tensor::empty(dev, vec![n_active * inter, hid], DType::F16)?);
        up_w_owned.as_ref().unwrap().clone()
    };

    // [mt, inter] f16 output for UP pass.
    let up_out = Tensor::empty(dev, vec![mt, inter], DType::F16)?;

    // ── UP pass: per-expert dequant (into scratch slot) → GEMM (immediately).
    // All async on the same stream — the GPU pipelines dequant[i] with GEMM[i-1].
    for (slot, &eid) in expert_ids.iter().enumerate() {
        let rows = g_starts[slot+1] - g_starts[slot];
        if rows == 0 { continue; }

        // Dequant expert eid's UP weight into slot `slot` of the scratch buffer.
        // Slot offset: slot * inter * hid * 2 bytes.
        let slot_off = slot * inter * hid * 2;
        let blk_off = eid * inter * bpr_up;
        let n_elem = inter * hid;
        let kern = cached_ir("ffai_dequant_q4", DType::F16, || {
            use metaltile_core::ir::KernelMode;
            let mut kk = metaltile_std::ffai::ffai_dequant_q4::ffai_dequant_q4::kernel_ir_for(DType::F16);
            kk.mode = KernelMode::Grid3D;
            kk
        });
        let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
        dev.dispatch(
            &kern,
            &[
                Binding::Buffer(qs_up.buffer.clone()),
                Binding::Buffer(sc_up.buffer.clone()),
                Binding::Buffer(up_w.buffer.clone()),
                u(hid as u32),
                u(n_elem as u32),
                u(blk_off as u32),
            ],
            Grid::d1((n_elem as u32).div_ceil(256), 256),
        ).map_err(|e| Error::Msg(format!("moe_grouped_gemm: dequant UP eid={eid}: {e:?}")))?;
        // NOTE: ^^ dequant_q4_off writes to `out[0..n_elem]` but our scratch has offset `slot_off`.
        // We use gemm_tc_off to read from the correct offset.

        // GEMM UP: xs_f16[g_starts[slot]:g_starts[slot+1], :] × up_w[slot*inter:..] → up_out[...]
        dev.gemm_tc_off(
            xs_f16.buffer.as_ref(), g_starts[slot] * hid * 2,
            up_w.buffer.as_ref(), slot_off,
            up_out.buffer.as_ref(), g_starts[slot] * inter * 2,
            rows, inter, hid, DType::F16,
        ).map_err(|e| Error::Msg(format!("moe_grouped_gemm: UP gemm eid={eid}: {e:?}")))?;
    }
    if up_scratch.is_none() { let _ = up_w_owned; }

    // Device relu² + scale 1/256.
    let up_relu2 = relu2_scale_f16(dev, &up_out, 1.0f32 / 256.0)?;
    drop(up_out);

    // DN scratch.
    let dn_w_owned: Option<Tensor>;
    let dn_w = if let Some(scratch) = dn_scratch {
        Tensor { buffer: scratch.buffer.clone(), offset: 0, shape: vec![n_active * hid, inter], dtype: DType::F16 }
    } else {
        dn_w_owned = Some(Tensor::empty(dev, vec![n_active * hid, inter], DType::F16)?);
        dn_w_owned.as_ref().unwrap().clone()
    };

    let dn_out = Tensor::empty(dev, vec![mt, hid], DType::F16)?;

    // ── DN pass: same pipelined dequant → GEMM pattern.
    for (slot, &eid) in expert_ids.iter().enumerate() {
        let rows = g_starts[slot+1] - g_starts[slot];
        if rows == 0 { continue; }

        let slot_off = slot * hid * inter * 2;
        let blk_off = eid * hid * bpr_dn;
        let n_elem = hid * inter;
        let kern = cached_ir("ffai_dequant_q4", DType::F16, || {
            use metaltile_core::ir::KernelMode;
            let mut kk = metaltile_std::ffai::ffai_dequant_q4::ffai_dequant_q4::kernel_ir_for(DType::F16);
            kk.mode = KernelMode::Grid3D;
            kk
        });
        let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
        dev.dispatch(
            &kern,
            &[
                Binding::Buffer(qs_dn.buffer.clone()),
                Binding::Buffer(sc_dn.buffer.clone()),
                Binding::Buffer(dn_w.buffer.clone()),
                u(inter as u32),
                u(n_elem as u32),
                u(blk_off as u32),
            ],
            Grid::d1((n_elem as u32).div_ceil(256), 256),
        ).map_err(|e| Error::Msg(format!("moe_grouped_gemm: dequant DN eid={eid}: {e:?}")))?;

        dev.gemm_tc_off(
            up_relu2.buffer.as_ref(), g_starts[slot] * inter * 2,
            dn_w.buffer.as_ref(), slot_off,
            dn_out.buffer.as_ref(), g_starts[slot] * hid * 2,
            rows, hid, inter, DType::F16,
        ).map_err(|e| Error::Msg(format!("moe_grouped_gemm: DN gemm eid={eid}: {e:?}")))?;
    }
    if dn_scratch.is_none() { let _ = dn_w_owned; }

    Ok(dn_out)
}
