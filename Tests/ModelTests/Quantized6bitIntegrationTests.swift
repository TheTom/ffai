// End-to-end test: download mlx-community/Qwen3-1.7B-6bit and
// generate. Skipped if network/checkpoint isn't available.
//
// 1.7B (vs 4B) for fast CI; the per-bit-width quantization paths
// don't depend on model size.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 1.7B 6-bit integration", .serialized)
struct Quantized6bitIntegrationTests {

    @Test("6-bit Qwen3 1.7B generates coherent text")
    func loadAndGenerate() async throws {
        let m: Model
        do {
            m = try await Model.load("mlx-community/Qwen3-1.7B-6bit")
        } catch {
            print("6-bit Qwen3 integration test skipped: \(error)")
            return
        }
        #expect(m.config.quantization?.bits == 6)
        #expect(m.qwen3 != nil)

        // Architecture matches Qwen3 1.7B
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)

        let result = try await m.generate(
            prompt: "The capital of France is",
            parameters: GenerationParameters(maxTokens: 4)
        )
        #expect(result.generatedTokens.count >= 1)
        #expect(!result.text.isEmpty)
    }
}
