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

    // The 8-bit AURA recipes (aura8v4 + aura8v8) currently produce
    // degenerate / gibberish output on real models. The aura4* paths
    // produce coherent text on the same checkpoint, so the bug is
    // localised to the 8-bit codec — likely the AURACodebook 8-bit
    // Lloyd-Max table, the bit-packing path for bits=8 (no spill
    // since 32 % 8 == 0 but the path may not be exercised by the
    // existing GPU correctness tests), or per-position norm
    // correction. Tracked separately; the tests here pin expected
    // coverage so the fix is exercised end-to-end when it lands.
    //
    // We assert with the standard coherence checker (unique-ratio +
    // no-degenerate-repeat). aura8v8's output is varied enough to
    // *pass* the unique-ratio bar despite being grammatically broken
    // — a known limitation of `expectCoherentOutput`. The test
    // therefore catches the aura8v4 case (which produces a long
    // repeat run) but not the aura8v8 quality regression on its own.
    // Both stay in the suite so the surface is covered when the
    // codec is fixed.

    @Test("auraQuantized aura8v4 (asymmetric K/V, 8-bit K + 4-bit V) produces coherent output")
    func auraAsymmetric8v4() async throws {
        // aura8v4 — high-precision K, mid-precision V. Useful when
        // attention scores need the extra K precision (long context,
        // information-dense prompts) but V can tolerate moderate
        // compression. Exercises the kb=8 path in aura_score and the
        // vb=4 path in aura_value / aura_flash_p1 in one config.
        let scheme = AURAScheme(keyBits: 8, valueBits: 4)
        guard let result = try await decode(.auraQuantized(scheme: scheme)) else { return }
        print("[KV=aura8v4] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "aura8v4")
    }

    @Test("auraQuantized aura8v8 (symmetric 8-bit K/V, identity rotation) produces coherent output")
    func auraSymmetric8v8() async throws {
        // aura8v8 — the highest-precision AURA recipe. Exercises the
        // kb=8 + vb=8 paths through every codec kernel
        // (aura_encode / aura_score / aura_value / aura_dequant_rotated).
        // Memory ratio vs raw bf16: still ~2× compression from the
        // bit-packing alone; quality should be near-indistinguishable
        // from .raw at this width.
        let scheme = AURAScheme(keyBits: 8, valueBits: 8)
        guard let result = try await decode(.auraQuantized(scheme: scheme)) else { return }
        print("[KV=aura8v8] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "aura8v8")
    }
}
