// Slow integration test for Llama 3.2 1B. Asserts:
//   1. Model loads + shapes match the published config.
//   2. A single-token forward pass yields finite, non-zero logits.
//   3. Greedy decode of `Tests/Fixtures/Llama-3.2-1B/golden.json`'s prompt
//      produces the same token IDs the mlx-lm reference produced.
//
// The golden was captured with `Tools/capture-fixtures.py --model
// unsloth/Llama-3.2-1B`. Regenerate after intentional model / kernel
// changes; an unexplained mismatch indicates FFAI has drifted from MLX.
//
// Skipped automatically if the network/checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("Llama 3.2 1B integration", .serialized)
struct LlamaIntegrationTests {

    @Test("load + greedy generate matches mlx-lm golden")
    func loadAndGenerate() async throws {
        let golden = try GoldenFixture.load("Llama-3.2-1B")

        let m: Model
        do {
            m = try await Model.load(golden.model)
        } catch {
            print("Llama integration test skipped: \(error)")
            return
        }

        // Sanity: shapes match the published config.
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 16)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 128_256)
        #expect(m.llama != nil, "expected engine to be a LlamaModel")

        // Single-token forward: BOS → finite, non-zero logits.
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 128_000, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 != 0)

        // Greedy decode of the fixture prompt and compare token-by-token.
        let result = try await m.generate(
            prompt: golden.prompt,
            parameters: GenerationParameters(maxTokens: golden.maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectGoldenMatch(result.generatedTokens, against: golden)
    }
}
