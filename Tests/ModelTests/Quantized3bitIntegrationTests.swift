// End-to-end test: download mlx-community/Qwen3-1.7B-3bit and assert
// FFAI's greedy decode matches the mlx-lm reference within the
// per-fixture tolerance (see GoldenFixture.expectGoldenMatch).
//
// Skipped if network/checkpoint isn't available. Exercises the
// dequant_gemv_int3 kernel end-to-end on Qwen3 1.7B.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 1.7B 3-bit integration", .serialized)
struct Quantized3bitIntegrationTests {

    @Test("load + greedy generate matches mlx-lm golden")
    func loadAndGenerate() async throws {
        let golden = try GoldenFixture.load("Qwen3-1.7B-3bit")

        let m: Model
        do {
            m = try await Model.load(golden.model)
        } catch {
            print("3-bit Qwen3 integration test skipped: \(error)")
            return
        }

        #expect(m.config.quantization?.bits == 3)
        #expect(m.config.quantization?.groupSize == 64)
        #expect(m.qwen3 != nil)

        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)

        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        let result = try await m.generate(
            prompt: golden.prompt,
            parameters: GenerationParameters(maxTokens: golden.maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectGoldenMatch(result.generatedTokens, against: golden)
    }
}
