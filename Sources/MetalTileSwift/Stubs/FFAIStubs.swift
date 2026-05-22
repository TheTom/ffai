import Metal

// Stub dispatch functions for kernels that FFAI references but the
// `tile build --emit all` auto-discovery (via `inventory::iter` over
// `crates/metaltile-std/`) does not currently produce. They let the
// FFAI package compile; calling any of them at runtime traps.
//
// To remove a stub: (a) implement the kernel in
// `crates/metaltile-std/src/...` with an `inventory::submit!` block,
// (b) re-run `tile build --emit all --out
// /Users/tom/dev/ffai/Sources/MetalTileSwift`, (c) drop the matching
// stub func below.
//
// The `aura_dequant_rotated_*` and `aura_encode_*` stub families that
// used to live here were dropped once Eric's auto-discovery emit picked
// up the full `int{2,3,4,8} × {f32,f16,bf16}` matrix on
// metaltile-ffai `feat/ffai-kernel-pack` (commit `e219d05`+). The
// surviving stubs below (`ffai_gemm`, `ffai_rope_yarn`,
// `ffai_sdpa_multi`, `mt_rms_norm_wide`) still lack canonical
// `metaltile-std` sources + inventory blocks.

extension MetalTileKernels {
    @inline(never) private static func unimplemented(_ name: String) -> Never {
        fatalError("\(name): kernel not currently emitted. Add a Rust source under `crates/metaltile-std/` with an `inventory::submit!` block; `tile build --emit all` picks them up via inventory::iter.")
    }

    // MARK: - ffai_gemm (naive matmul fallback)

    public static func ffai_gemm_f32(
        weight: MTLBuffer, weightOffset: Int = 0,
        input: MTLBuffer, inputOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        in_dim: UInt32, out_dim: UInt32, n_rows: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func ffai_gemm_f16(
        weight: MTLBuffer, weightOffset: Int = 0,
        input: MTLBuffer, inputOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        in_dim: UInt32, out_dim: UInt32, n_rows: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func ffai_gemm_bf16(
        weight: MTLBuffer, weightOffset: Int = 0,
        input: MTLBuffer, inputOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        in_dim: UInt32, out_dim: UInt32, n_rows: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    // MARK: - ffai_rope_yarn (YaRN-scaled RoPE)

    public static func ffai_rope_yarn_f32(
        qk: MTLBuffer, qkOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        head_dim: UInt32, half_dim: UInt32, position: UInt32,
        theta_base: Float, factor: Float, low: Float, high: Float, attn_factor: Float,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func ffai_rope_yarn_f16(
        qk: MTLBuffer, qkOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        head_dim: UInt32, half_dim: UInt32, position: UInt32,
        theta_base: Float, factor: Float, low: Float, high: Float, attn_factor: Float,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func ffai_rope_yarn_bf16(
        qk: MTLBuffer, qkOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        head_dim: UInt32, half_dim: UInt32, position: UInt32,
        theta_base: Float, factor: Float, low: Float, high: Float, attn_factor: Float,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    // MARK: - ffai_sdpa_multi (sparse-batched SDPA decode)

    public static func ffai_sdpa_multi_f32(
        q: MTLBuffer, qOffset: Int = 0,
        k: MTLBuffer, kOffset: Int = 0,
        v: MTLBuffer, vOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        head_dim: UInt32, n_q_heads: UInt32, base_kv: UInt32, n_query: UInt32,
        kv_stride: UInt32, heads_per_group: UInt32,
        causal: UInt32, scale: Float,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func ffai_sdpa_multi_f16(
        q: MTLBuffer, qOffset: Int = 0,
        k: MTLBuffer, kOffset: Int = 0,
        v: MTLBuffer, vOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        head_dim: UInt32, n_q_heads: UInt32, base_kv: UInt32, n_query: UInt32,
        kv_stride: UInt32, heads_per_group: UInt32,
        causal: UInt32, scale: Float,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func ffai_sdpa_multi_bf16(
        q: MTLBuffer, qOffset: Int = 0,
        k: MTLBuffer, kOffset: Int = 0,
        v: MTLBuffer, vOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        head_dim: UInt32, n_q_heads: UInt32, base_kv: UInt32, n_query: UInt32,
        kv_stride: UInt32, heads_per_group: UInt32,
        causal: UInt32, scale: Float,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    // MARK: - dequant_gemv_int4_*_indirect (Day-1 GPU-router plumbing)
    //
    // Indirect-dispatch variants of dequant_gemv_int4 — same PSO + args
    // as the direct kernels but dispatch shape comes from an
    // `MTLBuffer` instead of `MTLSize`. The old `metaltile-emit/main.rs`
    // produced these via a custom Swift wrapper generator (commit
    // `b2eadca` on `feat/ffai-kernel-pack`); the new `tile build --emit
    // all` auto-discovery path doesn't carry that custom logic.
    // `Ops.dequantGemvIndirect` references these but is **not** wired
    // into the production decode / prefill path today (the GPU router
    // is still host-side). These stubs unblock compile; restoring the
    // indirect path needs the wrapper logic ported into `tile build`
    // or a hand-written wrapper here.

    public static func dequant_gemv_int4_f16_indirect(
        weight: MTLBuffer, weightOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        input: MTLBuffer, inputOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        in_dim: UInt32, group_size: UInt32,
        indirectBuffer: MTLBuffer, indirectBufferOffset: Int,
        threadgroupSize: MTLSize, on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func dequant_gemv_int4_bf16_indirect(
        weight: MTLBuffer, weightOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        input: MTLBuffer, inputOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        in_dim: UInt32, group_size: UInt32,
        indirectBuffer: MTLBuffer, indirectBufferOffset: Int,
        threadgroupSize: MTLSize, on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    // MARK: - mt_rms_norm_wide (wide-hidden RMSNorm)

    public static func mt_rms_norm_wide_f32(
        x: MTLBuffer, xOffset: Int = 0,
        w: MTLBuffer, wOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        eps_buf: MTLBuffer, eps_bufOffset: Int = 0,
        n: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func mt_rms_norm_wide_f16(
        x: MTLBuffer, xOffset: Int = 0,
        w: MTLBuffer, wOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        eps_buf: MTLBuffer, eps_bufOffset: Int = 0,
        n: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func mt_rms_norm_wide_bf16(
        x: MTLBuffer, xOffset: Int = 0,
        w: MTLBuffer, wOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        eps_buf: MTLBuffer, eps_bufOffset: Int = 0,
        n: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }
}
