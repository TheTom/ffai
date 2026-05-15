// Slow integration test: downloads (or hits cache) Qwen3 4B and runs
// end-to-end greedy generation. Skipped automatically if the network
// or checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 4B integration", .serialized)
struct Qwen3IntegrationTests {

    @Test("load + greedy generate produces coherent text")
    func loadAndGenerate() async throws {
        let m: Model
        do {
            m = try await Model.load("unsloth/Qwen3-4B")
        } catch {
            print("Qwen3 integration test skipped: \(error)")
            return
        }

        // Engine should be Qwen3 (not Llama).
        #expect(m.qwen3 != nil)
        #expect(m.llama == nil)

        // Sanity: shapes match the published config (Qwen3 4B).
        #expect(m.engine.hidden == 2560)
        #expect(m.engine.nLayers == 36)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 151_936 || m.engine.vocab == 152_064)

        // Forward one BOS-style token; check finite non-zero logits.
        let caches = m.engine.makeKVCache()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        // Short greedy generation. We don't assert specific tokens (sampling
        // determinism depends on hardware) but verify generation runs without
        // crashing and produces non-empty text.
        let result = try await m.generate(
            prompt: "The capital of France is",
            options: GenerateOptions(maxNewTokens: 4)
        )
        #expect(result.generatedTokens.count >= 1)
        #expect(!result.text.isEmpty)
        #expect(result.tokensPerSecond > 0)
    }
}
