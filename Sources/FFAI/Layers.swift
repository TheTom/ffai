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

/// Linear layer backed by mlx-format int4-quantized weights.
/// Storage is the (weight, scales, biases) triplet plus group_size:
///
///   weight   [out_features, in_features / 8]   uint32 (packed)
///   scales   [out_features, in_features / group_size]
///   biases   [out_features, in_features / group_size]
///
/// callAsFunction dispatches Ops.dequantGemvInt4 — fused dequant + gemv.
public final class QuantizedLinear: Module {
    public let weight: Tensor
    public let scales: Tensor
    public let biases: Tensor
    public let groupSize: Int

    public init(weight: Tensor, scales: Tensor, biases: Tensor, groupSize: Int) {
        precondition(weight.dtype == .u32, "QuantizedLinear: weight must be u32 packed")
        precondition(weight.shape.count == 2, "QuantizedLinear: weight must be 2D")
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales), ("biases", biases)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.dequantGemvInt4(
            weight: weight, scales: scales, biases: biases,
            input: x, groupSize: groupSize, on: cmd
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
/// QuantizedLinear if the bundle has matching `.scales`/`.biases`,
/// regular Linear otherwise.
public func loadLinear(
    base: String, in bundle: SafeTensorsBundle,
    quantization: ModelConfig.QuantizationConfig?
) throws -> AnyLinear {
    if let q = quantization, q.bits == 4, bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        return AnyLinear(QuantizedLinear(
            weight: t.weight, scales: t.scales, biases: t.biases,
            groupSize: q.groupSize
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
    public let weight: Tensor   // [vocab, hidden/8] uint32
    public let scales: Tensor
    public let biases: Tensor
    public let hidden: Int
    public let groupSize: Int

    public init(weight: Tensor, scales: Tensor, biases: Tensor,
                hidden: Int, groupSize: Int) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.hidden = hidden
        self.groupSize = groupSize
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales), ("biases", biases)]
    }

    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.dequantGatherInt4(
            weight: weight, scales: scales, biases: biases,
            tokenIds: tokenIds, hidden: hidden, groupSize: groupSize,
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
    if let q = quantization, q.bits == 4, bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        return AnyEmbedding(QuantizedEmbedding(
            weight: t.weight, scales: t.scales, biases: t.biases,
            hidden: hidden, groupSize: q.groupSize
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
