// KV-cache-scheme integration: load the same Qwen3 1.7B bf16 model
// under every KV cache scheme FFAI exposes and assert each one
// produces coherent generated text. Pins the contract that every
// KVCacheKind round-trips real K/V values well enough that the
// downstream SDPA produces good tokens.
//
// One @Test per scheme so Swift Testing's `.serialized` trait lets
// the previous Model go out of scope before the next one loads.
// Each scheme loads + decodes ~64 tokens; total wall-clock is the
// load time × N schemes (no KV write/decode amortization across
// schemes). Skipped if network/checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("KV cache schemes — coherent output", .serialized)
struct KVCacheSchemeIntegrationTests {

    /// Common test fixture — load Qwen3 1.7B bf16 under `kvCache`
    /// and run `Once upon a time…` → 64 greedy-decoded tokens.
    /// Returns the GenerationResult; caller asserts coherence on
    /// the token stream and prints the decoded text.
    private func decode(_ kvCache: KVCacheKind) async throws -> GenerationResult? {
        let modelId = "mlx-community/Qwen3-1.7B-bf16"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 64

        let m: Model
        do {
            let opts = LoadOptions(kvCache: kvCache)
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(modelId, options: opts)
            }
        } catch {
            print("KV-cache-scheme test (\(kvCache)) skipped: \(error)")
            return nil
        }

        return try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
    }

    @Test("raw bf16 KV cache produces coherent output")
    func rawScheme() async throws {
        guard let result = try await decode(.raw) else { return }
        print("[KV=raw] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "raw")
    }

    @Test("affineQuantized int8 KV cache produces coherent output")
    func affineInt8Scheme() async throws {
        guard let result = try await decode(.affineQuantized(bits: 8, groupSize: 64)) else { return }
        print("[KV=affine8] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "affine8")
    }

    @Test("affineQuantized int4 KV cache produces coherent output")
    func affineInt4Scheme() async throws {
        guard let result = try await decode(.affineQuantized(bits: 4, groupSize: 64)) else { return }
        print("[KV=affine4] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "affine4")
    }

    @Test("auraQuantized aura4v4 (symmetric, identity rotation) produces coherent output")
    func auraSymmetric4v4() async throws {
        guard let result = try await decode(.auraQuantized(scheme: .default)) else { return }
        print("[KV=aura4v4] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "aura4v4")
    }

    @Test("auraQuantized aura4v2 (asymmetric K/V, identity rotation) produces coherent output")
    func auraAsymmetric4v2() async throws {
        guard let result = try await decode(.auraQuantized(scheme: .aura4v2)) else { return }
        print("[KV=aura4v2] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "aura4v2")
    }
}
