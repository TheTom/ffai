// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ConvertDriver — quantize a bf16/fp16 HuggingFace checkpoint to MLX
// 4-bit (or 8-bit / 2-bit) affine format using FFAI's own GPU kernels.
//
// Replaces `mlx_lm.convert` for the common case:
//   ConvertDriver.convert(sourceDir: src, destDir: dst, options: opts)
//
// Output is a safetensors directory with the same layout that FFAI's
// SafeTensorsBundle + quantized-weight loaders expect:
//   model.safetensors       — packed u32 weights + bf16/f16 scales/biases
//   config.json             — original config + "quantization" block added
//   tokenizer.json / *.txt  — copied unchanged
//
// ## Quantization selection
//
// A weight tensor is quantized when ALL of these hold:
//   1. It is 2D (Linear weight shape [out, in]).
//   2. `in` is divisible by `group_size` (64 — kernel invariant).
//   3. `in` is divisible by `pack_factor` (8 for int4, 4 for int8).
//   4. Its key ends in `.weight` but not `norm.weight` (LayerNorm /
//      RMSNorm weights are not quantized — they're tiny and sensitive).
//   5. It is not `embed_tokens.weight` unless `quantizeEmbeddings=true`.
//   6. It is not `lm_head.weight` unless `quantizeLmHead=true`.
//
// Everything else is copied through unchanged, preserving the original
// dtype and raw bytes.

import Foundation
import Metal

/// Options controlling quantization behaviour.
///
/// Bit-widths are per-tensor-class. The default text path (`bits`)
/// covers the attention + MLP linears (q/k/v/o, gate/up/down, MoE
/// experts). The three `*Bits` optionals override it for specific
/// roles — `nil` means "keep this tensor in its source dtype (don't
/// quantize)". Each accepts any value the affine-quantize kernels can
/// handle (today: 2 / 4 / 8 — the clean pack-factor path; 3 / 5 / 6
/// use a different odd-width packing that the convert driver does
/// not write yet).
///
/// A single conversion can mix bit-widths — FFAI's `loadLinear` /
/// `loadEmbedding` derive the per-tensor bit-width from the saved
/// shapes via `deriveAffineQuantBits`, so mixed checkpoints load
/// correctly without per-tensor config.json entries.
public struct ConvertOptions: Sendable {
    /// Bit-width for the main linear projections (attention + MLP).
    /// Supported: 2 / 4 / 8. Default 4.
    public var bits: Int = 4
    /// Group size — must be 64 (kernel invariant). Do not change.
    public var groupSize: Int = 64
    /// Dtype for non-quantized params (scales/biases inherit from the
    /// source weight's dtype; this field governs unquantized pass-
    /// throughs).
    public var dtype: DType = .bf16
    /// Bit-width for the token embedding table. `nil` (the default)
    /// keeps it full-precision (mlx-lm convention). Set to 2 / 4 / 8
    /// to quantize independently of the main `bits`.
    public var embeddingBits: Int? = nil
    /// Bit-width for the `lm_head` projection. `nil` (the default)
    /// keeps it full-precision; when `lm_head` is tied to the
    /// embedding the loader reuses the embedding triplet, so this
    /// knob only matters for untied heads (Qwen3.6, some Gemma).
    public var lmHeadBits: Int? = nil
    /// Bit-width for vision-tower weights. `nil` (the default) keeps
    /// the tower full-precision — FFAI's VL towers (Qwen3VL / Qwen3.5-VL,
    /// Pixtral, SigLIP, Idefics3, MiniCPM-V, FastVLM) use plain
    /// `Linear`, not `QuantizedLinear`, so quantizing vision weights
    /// crashes the loader. Set only when wiring a new VL tower that
    /// consumes `QuantizedLinear`.
    public var visionBits: Int? = nil

    public init() {}
}

/// Errors surfaced by ConvertDriver.
public enum ConvertDriverError: Error, CustomStringConvertible {
    case missingConfigJSON(URL)
    case configJSONMalformed(URL)
    case configJSONWriteFailed(URL, Error)
    case noSafetensorsFound(URL)
    case unsupportedBits(Int)
    case mkdirFailed(URL, Error)

    public var description: String {
        switch self {
        case .missingConfigJSON(let d):
            return "config.json missing in source dir: \(d.path)"
        case .configJSONMalformed(let u):
            return "config.json is not a JSON object: \(u.path)"
        case .configJSONWriteFailed(let u, let e):
            return "failed to write config.json to \(u.path): \(e)"
        case .noSafetensorsFound(let d):
            return "no .safetensors files found in: \(d.path)"
        case .unsupportedBits(let b):
            return "unsupported bits=\(b) — must be 2, 4, or 8"
        case .mkdirFailed(let u, let e):
            return "failed to create output directory \(u.path): \(e)"
        }
    }
}

public enum ConvertDriver {

    // ─── Public entry point ──────────────────────────────────────────

    /// Quantize the checkpoint at `sourceDir` and write the result to
    /// `destDir`. `progress` is called with a human-readable status line
    /// for each tensor processed (useful for a CLI progress display).
    public static func convert(
        sourceDir: URL,
        destDir: URL,
        options: ConvertOptions = ConvertOptions(),
        progress: (@Sendable (String) -> Void)? = nil
    ) throws {
        // Validate every requested bit-width up front so a bad override
        // surfaces before we touch the GPU or open the writer.
        for (label, bits) in [
            ("--bits", Optional.some(options.bits)),
            ("--embedding-bits", options.embeddingBits),
            ("--lm-head-bits", options.lmHeadBits),
            ("--vision-bits", options.visionBits),
        ] {
            guard let b = bits else { continue }
            if QuantizedOpsValidation.packFactor(forBits: b) == nil {
                throw ConvertDriverError.unsupportedBits(b)
            }
            _ = label  // label is for future error messages
        }

        // Create the output directory (including parents).
        do {
            try FileManager.default.createDirectory(
                at: destDir, withIntermediateDirectories: true)
        } catch {
            throw ConvertDriverError.mkdirFailed(destDir, error)
        }

        // ─── Load source tensors ─────────────────────────────────────
        let bundle = try SafeTensorsBundle(directory: sourceDir, device: .shared)
        guard !bundle.files.isEmpty else {
            throw ConvertDriverError.noSafetensorsFound(sourceDir)
        }

        // ─── Prepare GPU command infrastructure ──────────────────────
        let device = Device.shared

        // ─── Build output writer ─────────────────────────────────────
        let outputURL = destDir.appendingPathComponent("model.safetensors")
        let writer = SafeTensorsWriter(url: outputURL)

        // Process tensors in sorted key order (deterministic output).
        let allKeys = bundle.allKeys.sorted()

        for key in allKeys {
            let entry = try bundle.tensor(named: key)

            if let bits = effectiveBits(key: key, tensor: entry, options: options) {
                // ─── Quantize this weight ────────────────────────────
                // Per-tensor bit-width — the `bits` value here may differ
                // from `options.bits` when an `embeddingBits` /
                // `lmHeadBits` / `visionBits` override applied.
                let pf = QuantizedOpsValidation.packFactor(forBits: bits)!
                progress?("quantizing \(key) \(entry.shape) @ \(bits)bit")
                let (packedBytes, scalesBytes, biasesBytes) = try quantizeTensor(
                    entry, key: key, options: options,
                    bits: bits, packFactor: pf, device: device)

                // The weight key in the output is unchanged (e.g.
                // "model.layers.0.self_attn.q_proj.weight"). The triplet
                // naming matches what SafeTensorsBundle.quantizedTriplet
                // expects: base.weight / base.scales / base.biases.
                //
                // Derive the base from the key by dropping ".weight" suffix.
                let base = String(key.dropLast(".weight".count))
                let weightShape = packedShape(original: entry.shape, packFactor: pf)
                let groupShape = scalesShape(original: entry.shape, groupSize: options.groupSize)

                try writer.append(
                    name: "\(base).weight", dtype: .u32,
                    shape: weightShape, bytes: packedBytes)
                try writer.append(
                    name: "\(base).scales", dtype: entry.dtype,
                    shape: groupShape, bytes: scalesBytes)
                try writer.append(
                    name: "\(base).biases", dtype: entry.dtype,
                    shape: groupShape, bytes: biasesBytes)
            } else {
                // ─── Pass through unchanged ──────────────────────────
                progress?("copying   \(key) \(entry.shape)")
                let bytes = rawBytes(from: entry)
                try writer.append(
                    name: key, dtype: entry.dtype,
                    shape: entry.shape, bytes: bytes)
            }
        }

        try writer.finalize()
        progress?("wrote \(outputURL.lastPathComponent)")

        // ─── Write updated config.json ───────────────────────────────
        try writeConfig(sourceDir: sourceDir, destDir: destDir, options: options)

        // ─── Copy tokenizer + auxiliary files ────────────────────────
        copyAuxiliaryFiles(from: sourceDir, to: destDir, progress: progress)
    }

    // ─── Quantization eligibility ────────────────────────────────────

    /// Pick the bit-width to use for `key`. `nil` means "skip
    /// quantization, pass this tensor through unchanged" (norms, 1D
    /// weights, tensors whose role opts out via a `nil` override).
    ///
    /// Routing:
    ///   * `embed_tokens.weight` / `embeddings.weight` → `embeddingBits`
    ///   * `lm_head.weight`                           → `lmHeadBits`
    ///   * vision-tower keys (matches `.visual.` / `visual.` /
    ///     `vision_tower.` / `vision_model.` prefixes used across
    ///     Qwen3VL / Qwen3.5-VL / Pixtral / SigLIP / Idefics3 /
    ///     MiniCPM-V / FastVLM)              → `visionBits`
    ///   * everything else (attention + MLP)          → `bits`
    ///
    /// After role routing the shape is rechecked: a tensor only quantizes
    /// if it is 2D, ends in `.weight`, and its inner dim is divisible by
    /// both `group_size` and the chosen bit-width's `packFactor`. Norm
    /// weights short-circuit here (1D) so they always pass through.
    private static func effectiveBits(
        key: String, tensor: Tensor, options: ConvertOptions
    ) -> Int? {
        // Must be a 2D `.weight` tensor to be eligible at all. Norms,
        // biases, conv1d kernels, etc. fall through unquantized.
        guard tensor.shape.count == 2, key.hasSuffix(".weight") else { return nil }

        // Norm layers (1D usually, but guard explicitly): never quantize.
        if key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") {
            return nil
        }

        // Pick the role-specific bit-width.
        let roleBits: Int?
        if key.contains("embed_tokens.weight") || key.contains("embeddings.weight") {
            roleBits = options.embeddingBits
        } else if key.contains("lm_head.weight") {
            roleBits = options.lmHeadBits
        } else if key.contains(".visual.") || key.hasPrefix("visual.")
            || key.contains("vision_tower.") || key.contains("vision_model.")
        {
            roleBits = options.visionBits
        } else {
            roleBits = options.bits
        }
        guard let bits = roleBits else { return nil }

        // Shape feasibility for the chosen bit-width.
        guard let pf = QuantizedOpsValidation.packFactor(forBits: bits) else { return nil }
        let inDim = tensor.shape[1]
        guard inDim.isMultiple(of: options.groupSize), inDim.isMultiple(of: pf) else {
            return nil
        }
        return bits
    }

    // ─── Per-tensor quantization ─────────────────────────────────────

    /// GPU-quantize a single 2D weight tensor to `bits` bits per code.
    /// Returns raw bytes for the packed weight, scales, and biases
    /// buffers — ready for the writer.
    ///
    /// The kernel operates on a flat `[numel]` view of the weight; the
    /// original `[out, in]` shape is preserved at the caller level via
    /// `packedShape` / `scalesShape`. `bits` here may differ from
    /// `options.bits` when an embedding / lm_head / vision override
    /// applied — every per-tensor knob runs through this same path,
    /// just with a different `bits` + `packFactor`.
    private static func quantizeTensor(
        _ src: Tensor, key: String,
        options: ConvertOptions, bits: Int, packFactor: Int,
        device: Device
    ) throws -> (packed: Data, scales: Data, biases: Data) {
        let numel = src.elementCount
        let nGroups = numel / options.groupSize
        let nPacks = numel / packFactor

        // Allocate output buffers.
        let packed = Tensor.empty(shape: [nPacks], dtype: .u32, device: device)
        let scales = Tensor.empty(shape: [nGroups], dtype: src.dtype, device: device)
        let biases = Tensor.empty(shape: [nGroups], dtype: src.dtype, device: device)
        packed.zero()
        scales.zero()
        biases.zero()

        // Reshape src to flat [numel] so the kernel sees one contiguous row.
        let flat = src.reshaped(to: [numel])

        // Encode + commit + wait for the GPU to finish.
        let cb = device.makeCommandBuffer()
        QuantizedOps.quantizeAffine(
            weight: flat,
            packed: packed, scales: scales, biases: biases,
            bits: bits, groupSize: options.groupSize,
            on: cb)
        cb.commit()
        cb.waitUntilCompleted()

        return (
            rawBytes(from: packed),
            rawBytes(from: scales),
            rawBytes(from: biases)
        )
    }

    // ─── Shape helpers ───────────────────────────────────────────────

    /// Shape of the packed u32 tensor given the original [out, in] shape.
    /// The pack collapses the innermost dim: [out, in / pack_factor].
    private static func packedShape(original: [Int], packFactor: Int) -> [Int] {
        var s = original
        s[s.count - 1] = s[s.count - 1] / packFactor
        return s
    }

    /// Shape of the scales / biases tensor: [out, in / group_size].
    private static func scalesShape(original: [Int], groupSize: Int) -> [Int] {
        var s = original
        s[s.count - 1] = s[s.count - 1] / groupSize
        return s
    }

    // ─── Raw byte extraction ─────────────────────────────────────────

    /// Copy a tensor's bytes out of its MTLBuffer into a Data value.
    /// The buffer uses shared storage, so this is a plain memcpy — no GPU
    /// sync needed (we've already waited on the command buffer above).
    private static func rawBytes(from tensor: Tensor) -> Data {
        let ptr = tensor.buffer.contents().advanced(by: tensor.offset)
        return Data(bytes: ptr, count: tensor.byteCount)
    }

    // ─── config.json ─────────────────────────────────────────────────

    /// Read config.json from source, inject quantization blocks, write to
    /// dest. Both `quantization` and `quantization_config` are written for
    /// compatibility: mlx-lm checks `quantization`, Transformers checks
    /// `quantization_config`.
    private static func writeConfig(
        sourceDir: URL, destDir: URL, options: ConvertOptions
    ) throws {
        let srcURL = sourceDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: srcURL.path) else {
            throw ConvertDriverError.missingConfigJSON(sourceDir)
        }

        let data = try Data(contentsOf: srcURL)
        let parsed: Any
        if #available(macOS 12.0, iOS 15.0, *) {
            parsed = try JSONSerialization.jsonObject(with: data, options: [.json5Allowed])
        } else {
            parsed = try JSONSerialization.jsonObject(with: data)
        }
        guard var obj = parsed as? [String: Any] else {
            throw ConvertDriverError.configJSONMalformed(srcURL)
        }

        let quantBlock: [String: Any] = [
            "bits": options.bits,
            "group_size": options.groupSize,
            "mode": "affine",
        ]
        obj["quantization"] = quantBlock
        obj["quantization_config"] = quantBlock

        // Sanitize before NSJSONSerialization: Python's `json` module emits
        // `Infinity` / `-Infinity` / `NaN` literals when `allow_nan=True`
        // (the default), and several upstream configs use them
        // (e.g. NemotronH's `time_step_limit: [0.0, Infinity]`).
        // `JSONSerialization` is strict JSON — it throws on non-finite
        // doubles. Replace each with a JSON-legal sentinel that the
        // model code won't trip on (`1e308` ≈ DBL_MAX; NaN → null).
        let sanitized = sanitizeForJSON(obj)

        let outData = try JSONSerialization.data(
            withJSONObject: sanitized, options: [.sortedKeys, .prettyPrinted])
        let destURL = destDir.appendingPathComponent("config.json")
        do {
            try outData.write(to: destURL, options: .atomic)
        } catch {
            throw ConvertDriverError.configJSONWriteFailed(destURL, error)
        }
    }

    /// Walk a JSON-shaped tree and replace non-finite `Double` / `Float`
    /// values with JSON-legal stand-ins. `+Inf` → `1e308`, `-Inf` →
    /// `-1e308`, `NaN` → `NSNull`. NSJSONSerialization rejects non-finite
    /// numbers; Python's `json` module writes them as `Infinity` / `NaN`
    /// literals (a non-spec extension), which is what upstream configs
    /// like NemotronH ship.
    private static func sanitizeForJSON(_ value: Any) -> Any {
        if let n = value as? Double {
            if n.isNaN { return NSNull() }
            if n == .infinity { return 1e308 }
            if n == -.infinity { return -1e308 }
            return n
        }
        if let n = value as? Float {
            if n.isNaN { return NSNull() }
            if n == .infinity { return Float(1e30) }
            if n == -.infinity { return Float(-1e30) }
            return n
        }
        if let dict = value as? [String: Any] {
            return dict.mapValues(sanitizeForJSON)
        }
        if let arr = value as? [Any] {
            return arr.map(sanitizeForJSON)
        }
        return value
    }

    // ─── Auxiliary file copy ─────────────────────────────────────────

    /// Copy tokenizer and other non-weight files that must travel with the
    /// checkpoint. Missing files are silently skipped (not every model has
    /// every optional file).
    private static func copyAuxiliaryFiles(
        from sourceDir: URL, to destDir: URL,
        progress: (@Sendable (String) -> Void)?
    ) {
        // Explicit-name files to try copying.
        let namedFiles = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "tokenizer.model",
            "vocab.txt",
            "merges.txt",
        ]

        let fm = FileManager.default
        for name in namedFiles {
            let src = sourceDir.appendingPathComponent(name)
            // fileExists follows symlinks, so this correctly handles HF
            // cache blobs that are symlinked from the snapshot directory.
            guard fm.fileExists(atPath: src.path) else { continue }
            let dst = destDir.appendingPathComponent(name)
            // Overwrite if already exists (e.g. re-running the convert).
            try? fm.removeItem(at: dst)
            do {
                // HF snapshot directories present files as relative
                // symlinks pointing into the blobs store
                // (e.g. `../../blobs/<sha>`). `copyItem` would copy the
                // symlink, not the target, so the destination would be a
                // broken relative symlink. Resolve to the real blob path
                // first so we copy the actual bytes.
                let resolved = src.resolvingSymlinksInPath()
                try fm.copyItem(at: resolved, to: dst)
                progress?("copied    \(name)")
            } catch {
                // Non-fatal — log but continue so the tokenizer-less
                // model weight file is still written.
                progress?("warning: could not copy \(name): \(error)")
            }
        }

        // Also copy any remaining *.txt and *.json files that are not
        // safetensors-adjacent (e.g. generation_config.json).
        let copyExtensions: Set<String> = ["txt", "json"]
        if let contents = try? fm.contentsOfDirectory(
            at: sourceDir, includingPropertiesForKeys: nil)
        {
            for file in contents where copyExtensions.contains(file.pathExtension.lowercased()) {
                let name = file.lastPathComponent
                // Skip files we already handled above, and config.json
                // (written separately with quantization blocks added).
                if namedFiles.contains(name) || name == "config.json" { continue }
                // Skip safetensors index files.
                if name.hasPrefix("model.safetensors") { continue }
                let dst = destDir.appendingPathComponent(name)
                try? fm.removeItem(at: dst)
                do {
                    // Resolve symlinks for the same reason as above.
                    let resolved = file.resolvingSymlinksInPath()
                    try fm.copyItem(at: resolved, to: dst)
                    progress?("copied    \(name)")
                } catch {
                    progress?("warning: could not copy \(name): \(error)")
                }
            }
        }
    }
}
