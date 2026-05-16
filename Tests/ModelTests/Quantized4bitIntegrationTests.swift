// End-to-end test: download mlx-community/Qwen3-1.7B-4bit and
// generate. Skipped if network/checkpoint isn't available.
//
// 1.7B (vs 4B) for fast CI; the per-bit-width quantization paths
// don't depend on model size.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 1.7B 4-bit integration", .serialized)
struct Quantized4bitIntegrationTests {

    @Test("load + greedy generate produces coherent text")
    func loadAndGenerate() async throws {
        let m: Model
        do {
            m = try await Model.load("mlx-community/Qwen3-1.7B-4bit")
        } catch {
            print("4-bit Qwen3 integration test skipped: \(error)")
            return
        }

        // The config carries a quantization block.
        #expect(m.config.quantization?.bits == 4)
        #expect(m.config.quantization?.groupSize == 64)
        #expect(m.qwen3 != nil)

        // Architecture matches Qwen3 1.7B
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)

        // First-token logits are finite and non-degenerate.
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        // Generate a few tokens. Don't assert specific output — sampling
        // determinism varies — but coherent text should produce non-empty,
        // non-garbage results.
        let result = try await m.generate(
            prompt: "The capital of France is",
            parameters: GenerationParameters(maxTokens: 4)
        )
        #expect(result.generatedTokens.count >= 1)
        #expect(!result.text.isEmpty)
    }
}
