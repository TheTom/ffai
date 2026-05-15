// Standard transformer layer building blocks: Linear, Embedding,
// RMSNorm. Each holds its weight tensors as fields and exposes
// `parameters()` for SafeTensors weight binding.

import Foundation
import Metal

// ─── Linear (no bias for now — Llama doesn't use biases) ─────────────

public final class Linear: Module {
    /// weight shape [out_features, in_features], row-major.
    public let weight: Tensor

    public init(weight: Tensor) {
        precondition(weight.shape.count == 2, "Linear: weight must be 2D")
        self.weight = weight
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.gemv(weight: weight, input: x, on: cmd)
    }
}

// ─── QuantizedLinear (mlx int4 format) ────────────────────────────────

/// Linear layer backed by mlx-format quantized weights (int4 or int8).
/// Storage is the (weight, scales, biases) triplet plus (bits, group_size).
///
///   weight   [out_features, in_features / pack_factor]  uint32
///            pack_factor = 32 / bits (8 for int4, 4 for int8)
///   scales   [out_features, in_features / group_size]
///   biases   [out_features, in_features / group_size]
///
/// callAsFunction dispatches Ops.dequantGemv — fused dequant + gemv.
public final class QuantizedLinear: Module {
    public let weight: Tensor
    public let scales: Tensor
    public let biases: Tensor
    public let bits: Int
    public let groupSize: Int

    public init(weight: Tensor, scales: Tensor, biases: Tensor,
                bits: Int, groupSize: Int) {
        precondition(weight.dtype == .u32, "QuantizedLinear: weight must be u32 packed")
        precondition(weight.shape.count == 2, "QuantizedLinear: weight must be 2D")
        precondition(bits == 4 || bits == 8, "QuantizedLinear: bits must be 4 or 8")
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.bits = bits
        self.groupSize = groupSize
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales), ("biases", biases)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.dequantGemv(
            weight: weight, scales: scales, biases: biases,
            input: x, bits: bits, groupSize: groupSize, on: cmd
        )
    }
}

/// Type-erasing wrapper so layers can hold either a regular Linear or a
/// QuantizedLinear without templating every call site.
public final class AnyLinear: Module {
    public let inner: any Module
    private let forward: (Tensor, MTLCommandBuffer) -> Tensor

    public init(_ linear: Linear) {
        self.inner = linear
        self.forward = { linear($0, on: $1) }
    }

    public init(_ linear: QuantizedLinear) {
        self.inner = linear
        self.forward = { linear($0, on: $1) }
    }

    public func parameters() -> [(String, Tensor)] { inner.parameters() }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        forward(x, cmd)
    }
}

/// Build the right Linear variant for a weight at `<base>.weight` —
/// QuantizedLinear if the bundle has matching `.scales`/`.biases` and
/// the config quantization block specifies a supported bit-width
/// (4 or 8), regular Linear otherwise.
public func loadLinear(
    base: String, in bundle: SafeTensorsBundle,
    quantization: ModelConfig.QuantizationConfig?
) throws -> AnyLinear {
    if let q = quantization, (q.bits == 4 || q.bits == 8), bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        return AnyLinear(QuantizedLinear(
            weight: t.weight, scales: t.scales, biases: t.biases,
            bits: q.bits, groupSize: q.groupSize
        ))
    }
    return AnyLinear(Linear(weight: try bundle.tensor(named: "\(base).weight")))
}

// ─── Embedding ───────────────────────────────────────────────────────

public final class Embedding: Module {
    /// weight shape [vocab_size, hidden_size]
    public let weight: Tensor

    public init(weight: Tensor) {
        precondition(weight.shape.count == 2, "Embedding: weight must be 2D")
        self.weight = weight
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    /// Look up `tokenIds` (one-element u32 tensor for decode) and return
    /// [n_tokens, hidden] in the table's dtype.
    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.gather(table: weight, tokenIds: tokenIds, on: cmd)
    }
}

// ─── QuantizedEmbedding (mlx int4 format) ─────────────────────────────

public final class QuantizedEmbedding: Module {
    public let weight: Tensor   // [vocab, hidden/pack_factor] uint32
    public let scales: Tensor
    public let biases: Tensor
    public let hidden: Int
    public let bits: Int
    public let groupSize: Int

    public init(weight: Tensor, scales: Tensor, biases: Tensor,
                hidden: Int, bits: Int, groupSize: Int) {
        precondition(bits == 4 || bits == 8, "QuantizedEmbedding: bits must be 4 or 8")
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.hidden = hidden
        self.bits = bits
        self.groupSize = groupSize
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales), ("biases", biases)]
    }

    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.dequantGather(
            weight: weight, scales: scales, biases: biases,
            tokenIds: tokenIds, hidden: hidden, bits: bits, groupSize: groupSize,
            on: cmd
        )
    }
}

/// Type-erasing wrapper over Embedding / QuantizedEmbedding so model
/// loaders don't need to template every call site.
public final class AnyEmbedding: Module {
    public let inner: any Module
    public let weight: Tensor   // expose for tying with lm_head
    private let forward: (Tensor, MTLCommandBuffer) -> Tensor

    public init(_ embed: Embedding) {
        self.inner = embed
        self.weight = embed.weight
        self.forward = { embed($0, on: $1) }
    }

    public init(_ embed: QuantizedEmbedding) {
        self.inner = embed
        self.weight = embed.weight
        self.forward = { embed($0, on: $1) }
    }

    public func parameters() -> [(String, Tensor)] { inner.parameters() }

    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        forward(tokenIds, cmd)
    }
}

/// Build the right Embedding variant depending on quantization presence.
public func loadEmbedding(
    base: String, in bundle: SafeTensorsBundle,
    hidden: Int, quantization: ModelConfig.QuantizationConfig?
) throws -> AnyEmbedding {
    if let q = quantization, (q.bits == 4 || q.bits == 8), bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        return AnyEmbedding(QuantizedEmbedding(
            weight: t.weight, scales: t.scales, biases: t.biases,
            hidden: hidden, bits: q.bits, groupSize: q.groupSize
        ))
    }
    return AnyEmbedding(Embedding(weight: try bundle.tensor(named: "\(base).weight")))
}

// ─── RMSNorm ─────────────────────────────────────────────────────────

public final class RMSNorm: Module {
    /// weight shape [n] — per-channel scale.
    public let weight: Tensor
    public let eps: Float

    public init(weight: Tensor, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.rmsNorm(x, weight: weight, eps: eps, on: cmd)
    }
}
