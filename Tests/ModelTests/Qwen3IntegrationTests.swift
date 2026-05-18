// Slow integration test: downloads Qwen3 1.7B bf16 and asserts FFAI's
// greedy decode matches the mlx-lm reference within the per-fixture
// tolerance (see GoldenFixture.expectGoldenMatch).
//
// 1.7B (vs 4B) keeps the integration suite fast — same architecture,
// smaller weights. Skipped automatically if checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 1.7B integration", .serialized)
struct Qwen3IntegrationTests {

    @Test("load + greedy generate matches mlx-lm golden")
    func loadAndGenerate() async throws {
        let golden = try GoldenFixture.load("Qwen3-1.7B-bf16")

        let m: Model
        do {
            m = try await Model.load(golden.model)
        } catch {
            print("Qwen3 integration test skipped: \(error)")
            return
        }

        // Engine should be Qwen3 (not Llama).
        #expect(m.qwen3 != nil)
        #expect(m.llama == nil)

        // Shapes match the published config (Qwen3 1.7B).
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 151_936)

        // First-token forward: finite logits.
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        // Greedy decode of the fixture prompt — exercises Qwen3's
        // q_norm/k_norm path through the full per-token forward.
        let result = try await m.generate(
            prompt: golden.prompt,
            parameters: GenerationParameters(maxTokens: golden.maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectGoldenMatch(result.generatedTokens, against: golden)
    }
}
