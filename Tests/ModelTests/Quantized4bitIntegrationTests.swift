// End-to-end test: download mlx-community/Qwen3-1.7B-4bit and assert
// FFAI's greedy decode matches the mlx-lm reference token-by-token.
//
// Skipped if network/checkpoint isn't available. The golden fixture was
// captured via `Tools/capture-fixtures.py --model
// mlx-community/Qwen3-1.7B-4bit` and exercises the
// `dequant_gemv_int4_*` Metal kernel end-to-end on a real model.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 1.7B 4-bit integration", .serialized)
struct Quantized4bitIntegrationTests {

    @Test("load + greedy generate matches mlx-lm golden")
    func loadAndGenerate() async throws {
        let golden = try GoldenFixture.load("Qwen3-1.7B-4bit")

        let m: Model
        do {
            m = try await Model.load(golden.model)
        } catch {
            print("4-bit Qwen3 integration test skipped: \(error)")
            return
        }

        // Config carries the quantization block.
        #expect(m.config.quantization?.bits == 4)
        #expect(m.config.quantization?.groupSize == 64)
        #expect(m.qwen3 != nil)

        // Architecture matches Qwen3 1.7B.
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)

        // First-token forward: finite, non-degenerate logits.
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        // Greedy decode of the fixture prompt — exercises dequant_gemv_int4
        // through the per-token forward pass.
        let result = try await m.generate(
            prompt: golden.prompt,
            parameters: GenerationParameters(maxTokens: golden.maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectGoldenMatch(result.generatedTokens, against: golden)
    }
}
