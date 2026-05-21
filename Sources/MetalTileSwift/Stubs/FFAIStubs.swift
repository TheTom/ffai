import Metal

// Stub dispatch functions for kernels that FFAI references but
// metaltile-emit does not currently emit (no Rust kernel source in
// metaltile-ffai). These let the FFAI package compile; calling any
// of them at runtime traps with a clear message pointing back to
// task #93 / the metaltile-ffai emitter.
//
// To remove a stub: (a) implement the kernel in
// `crates/metaltile-std/src/...`, (b) register it in
// `crates/metaltile-emit/src/main.rs::register_kernels`, (c)
// re-emit, then drop the matching stub func below.

extension MetalTileKernels {
    // MARK: - aura_dequant_rotated (codebook product-quantized dequant)

    @inline(never) private static func unimplemented(_ name: String) -> Never {
        fatalError("\(name): kernel not currently emitted by metaltile-emit. See task #93 — needs Rust kernel source in crates/metaltile-std/ + registration in crates/metaltile-emit/src/main.rs.")
    }

    public static func aura_dequant_rotated_int2_f32(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int2_f16(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int2_bf16(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int3_f32(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int3_f16(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int3_bf16(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int4_f32(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int4_f16(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int4_bf16(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int8_f32(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int8_f16(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_dequant_rotated_int8_bf16(
        packed: MTLBuffer, packedOffset: Int = 0,
        norms: MTLBuffer, normsOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32, tokens: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    // MARK: - aura_encode (codebook product-quantized encode)

    public static func aura_encode_int2_f32(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int2_f16(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int2_bf16(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int3_f32(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int3_f16(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int3_bf16(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int4_f32(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int4_f16(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int4_bf16(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int8_f32(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int8_f16(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func aura_encode_int8_bf16(
        input: MTLBuffer, inputOffset: Int = 0,
        rotation: MTLBuffer, rotationOffset: Int = 0,
        boundaries: MTLBuffer, boundariesOffset: Int = 0,
        codebook: MTLBuffer, codebookOffset: Int = 0,
        packed_out: MTLBuffer, packed_outOffset: Int = 0,
        norms_out: MTLBuffer, norms_outOffset: Int = 0,
        dim: UInt32, packed_width: UInt32,
        gridSize: MTLSize, threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

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
