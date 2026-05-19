// OpsValidation — pure, testable wrapper precondition logic.
//
// Background: every `Ops.*` wrapper around a reduction-mode metaltile
// kernel has dispatch-shape invariants that, if violated, range from
// silent miscompute to a non-preemptive GPU pin (system freeze).
// Before this file existed, the invariants lived inline in each
// wrapper as `precondition` calls. That was load-bearing but
// untestable in CI — a `precondition` failure traps the entire test
// process, so the only "test" of a precondition was producing it in
// production.
//
// Each validation function below returns:
//   * `nil` — dispatch shape is valid, wrapper proceeds.
//   * `String` — human-readable reason. Wrapper calls
//     `preconditionFailure("Ops.<fn>: \(reason)")` to halt.
//
// Tests in `Tests/FFAITests/OpsValidationTests.swift` exercise good +
// bad inputs without producing a trap. New reduction-mode wrappers
// should add a `validate*` function here in the same commit as the
// wrapper.
//
// See `papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md`
// for the full story behind why this file exists.

import Foundation

public enum OpsValidation {

    // ─── rmsNorm + rmsNormRows ─────────────────────────────────────
    //
    // Kernel invariants (from `crates/metaltile-std/src/mlx/rms_norm.rs`
    // §"DISPATCH INVARIANTS"):
    //   1. `N = TPG * 4` — each thread owns 4 consecutive elements.
    //      The wrapper computes `TPG = n / 4`.
    //   2. `TPG` must be a multiple of 32 (cross-simdgroup reduction).
    //      Combined with (1): `n` must be a multiple of 128.
    //   3. `TPG ≤ 1024` (Apple's max-threads-per-threadgroup cap).
    //      Combined with (1): `n ≤ 4096`. Larger rows need
    //      `rmsNormRows` chunked dispatch (forthcoming).

    /// Validate the row-width parameter for `Ops.rmsNorm` and
    /// `Ops.rmsNormRows`. The single-row and multi-row dispatches
    /// share the per-row invariant.
    public static func validateRmsNorm(n: Int) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if !n.isMultiple(of: 128) {
            return "n=\(n) must be a multiple of 128 (32-lane simdgroup × 4 elements/thread)"
        }
        if n / 4 > 1024 {
            return "n=\(n) > 4096 — exceeds the 1024-thread cap of this kernel; use rmsNormRows or a chunked variant for larger rows"
        }
        return nil
    }

    // ─── sdpaDecode ────────────────────────────────────────────────
    //
    // Kernel invariants (from `crates/metaltile-std/src/ffai/sdpa_decode.rs`
    // §"DISPATCH INVARIANTS"):
    //   1. `head_dim == 128`. Each lane owns 4 consecutive Q/K/V
    //      elements; loads are unconditional. The 2026-05-19 GPU
    //      freeze hit this — wrong head_dim → wrong TPG → infinite
    //      loop in `for _t in range(sg, n_kv, ns=TPG/32)` when
    //      `TPG < 32` makes `ns = 0`.
    //   2. `nQHeads % nKVHeads == 0` so GQA fan-out is integer.
    //   3. `n_kv ≤ kv_stride`. The kernel walks `[0, n_kv)` only;
    //      `kv_stride` is the pre-allocated maxSeq capacity.

    public static func validateSdpaDecode(
        headDim: Int, nQHeads: Int, nKVHeads: Int,
        nKV: Int, kvStride: Int
    ) -> String? {
        if headDim != 128 {
            return "head_dim must be 128 (got \(headDim)); other specializations (64, 256) not yet emitted"
        }
        if nQHeads <= 0 {
            return "nQHeads must be positive (got \(nQHeads))"
        }
        if nKVHeads <= 0 {
            return "nKVHeads must be positive (got \(nKVHeads))"
        }
        if !nQHeads.isMultiple(of: nKVHeads) {
            return "nQHeads (\(nQHeads)) must be a multiple of nKVHeads (\(nKVHeads))"
        }
        if nKV < 0 {
            return "nKV must be non-negative (got \(nKV))"
        }
        if nKV > kvStride {
            return "nKV (\(nKV)) must not exceed kvStride (\(kvStride)) — kernel would read past cache"
        }
        return nil
    }

    // ─── gemv ──────────────────────────────────────────────────────
    //
    // `mt_gemv` (MLX-derived). Adaptive `lsize` reduction → no GPU-pin
    // risk regardless of TPG. Caller-controllable shape (outDim, inDim);
    // the wrapper already checks shape consistency via the `weight`
    // tensor. Validation here just pins the basic positivity contract.

    public static func validateGemv(outDim: Int, inDim: Int) -> String? {
        if outDim <= 0 {
            return "outDim must be positive (got \(outDim))"
        }
        if inDim <= 0 {
            return "inDim must be positive (got \(inDim))"
        }
        return nil
    }

    // ─── dequantGemv ───────────────────────────────────────────────
    //
    // Affine-dequant + matvec for int{3,4,5,6,8} weights. Three silent-
    // miscompute footguns the wrapper didn't previously catch:
    //
    //   1. `inDim % groupSize != 0` → kernel's `n_groups = inDim/groupSize`
    //      rounds down, silently dropping the partial trailing group.
    //   2. For pack-strided bit-widths (int4, int8), `inDim % vals_per_pack`
    //      must be 0 — kernel's `n_packs_per_row = inDim/vals_per_pack`
    //      rounds down, silently dropping unaligned tail elements.
    //   3. `scales`/`biases` element counts must be `outDim × n_groups`.
    //      Smaller → OOB reads → garbage output. Larger → no harm.
    //
    // Element-strided bit-widths {3, 5, 6} don't have the pack-alignment
    // constraint — the kernel walks individual elements.

    public static func validateDequantGemv(
        outDim: Int, inDim: Int, bits: Int, groupSize: Int,
        scalesCount: Int, biasesCount: Int
    ) -> String? {
        // Supported bit-widths (mirrors the wrapper's switch arms).
        if !(bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8) {
            return "bits=\(bits) unsupported — must be one of 3, 4, 5, 6, or 8"
        }
        if outDim <= 0 {
            return "outDim must be positive (got \(outDim))"
        }
        if inDim <= 0 {
            return "inDim must be positive (got \(inDim))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        // Footgun 1: partial trailing group.
        if !inDim.isMultiple(of: groupSize) {
            return "inDim=\(inDim) must be a multiple of groupSize=\(groupSize) — partial trailing group would be silently dropped"
        }
        // Footgun 2: pack-alignment for pack-strided variants.
        if bits == 4 || bits == 8 {
            let valsPerPack = 32 / bits  // 8 for int4, 4 for int8
            if !inDim.isMultiple(of: valsPerPack) {
                return "inDim=\(inDim) must be a multiple of \(valsPerPack) for bits=\(bits) (pack-strided kernel — unaligned tail elements silently dropped)"
            }
            if !groupSize.isMultiple(of: valsPerPack) {
                return "groupSize=\(groupSize) must be a multiple of \(valsPerPack) for bits=\(bits) (packs_per_group must be exact)"
            }
        }
        // Footgun 3: scales/biases sizing.
        let nGroups = inDim / groupSize
        let expected = outDim * nGroups
        if scalesCount != expected {
            return "scales must have outDim × n_groups = \(outDim) × \(nGroups) = \(expected) elements, got \(scalesCount)"
        }
        if biasesCount != expected {
            return "biases must have outDim × n_groups = \(outDim) × \(nGroups) = \(expected) elements, got \(biasesCount)"
        }
        return nil
    }

    // ─── auraEncode ────────────────────────────────────────────────
    //
    // Kernel invariants (from `crates/metaltile-std/src/ffai/aura_encode.rs`
    // §"DISPATCH INVARIANTS"):
    //   1. `TPG = dim` — one thread per rotated coordinate.
    //   2. `dim` must be a multiple of 32 (`simd_sum` reduction).
    //   3. `dim ≤ 1024` (`threadgroup_alloc("shared_unit", 1024)`).
    //   4. `bits ∈ {2, 3, 4, 8}` — encode kernel only emits these.

    public static func validateAuraEncode(
        rows: Int, dim: Int, bits: Int
    ) -> String? {
        if rows <= 0 {
            return "rows must be positive (got \(rows))"
        }
        if dim <= 0 {
            return "dim must be positive (got \(dim))"
        }
        if !dim.isMultiple(of: 32) {
            return "dim=\(dim) must be a multiple of 32 (one Apple simdgroup); simd_sum reduction is undefined otherwise"
        }
        if dim > 1024 {
            return "dim=\(dim) > 1024 — exceeds the kernel's TPG cap (shared_unit allocates 1024 slots)"
        }
        if bits != 2 && bits != 3 && bits != 4 && bits != 8 {
            return "bits=\(bits) unsupported — encode kernel emits only int2/int3/int4/int8 variants"
        }
        return nil
    }
}
