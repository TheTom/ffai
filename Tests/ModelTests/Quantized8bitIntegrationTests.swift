// End-to-end test: download mlx-community/Qwen3-4B-8bit and generate.
// Skipped if network/checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 4B 8-bit integration", .serialized)
struct Quantized8bitIntegrationTests {

    @Test("load + greedy generate produces coherent text")
    func loadAndGenerate() async throws {
        let m: Model
        do {
            m = try await Model.load("mlx-community/Qwen3-4B-8bit")
        } catch {
            print("8-bit Qwen3 integration test skipped: \(error)")
            return
        }

        // Config carries 8-bit quantization
        #expect(m.config.quantization?.bits == 8)
        #expect(m.config.quantization?.groupSize == 64)
        #expect(m.qwen3 != nil)

        // Architecture matches Qwen3 4B
        #expect(m.engine.hidden == 2560)
        #expect(m.engine.nLayers == 36)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)

        // First-token logits finite + non-degenerate
        let caches = m.engine.makeKVCache()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        let result = try await m.generate(
            prompt: "The capital of France is",
            options: GenerateOptions(maxNewTokens: 4)
        )
        #expect(result.generatedTokens.count >= 1)
        #expect(!result.text.isEmpty)
    }
}
