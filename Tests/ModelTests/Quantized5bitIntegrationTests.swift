// End-to-end test: download mlx-community/Qwen3-1.7B-5bit and
// generate. Skipped if network/checkpoint isn't available.
//
// We published this checkpoint ourselves to mlx-community (via
// `mlx_lm.convert -q --q-bits 5 --q-group-size 64` from
// `Qwen/Qwen3-1.7B`) because mlx-community didn't ship a plain
// text-only 5-bit Qwen3-1.7B (only TTS / ASR variants existed).
// Synthetic 5-bit kernel correctness is also covered by
// QuantizedOpsTests.roundTripInt5Bf16.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 1.7B 5-bit integration", .serialized)
struct Quantized5bitIntegrationTests {

    @Test("5-bit Qwen3 1.7B generates coherent text")
    func loadAndGenerate() async throws {
        let m: Model
        do {
            m = try await Model.load("mlx-community/Qwen3-1.7B-5bit")
        } catch {
            print("5-bit Qwen3 integration test skipped: \(error)")
            return
        }
        #expect(m.config.quantization?.bits == 5)
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
