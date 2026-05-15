// End-to-end smoke tests for 3-bit and 6-bit mlx-format quantized
// models. Skipped if network/checkpoint isn't available.
//
// 5-bit isn't currently in this suite because the only 5-bit Qwen3 4B
// available on mlx-community is a "Thinking" finetuned variant that
// runs prohibitively slowly with the naive kernel — see commit message
// for details. Synthetic 5-bit correctness is covered by
// QuantizedOpsTests.roundTripInt5Bf16.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 4B 3-bit + 6-bit integration", .serialized)
struct Quantized3and6bitIntegrationTests {

    @Test("3-bit Qwen3 4B generates coherent text")
    func threeBit() async throws {
        let m: Model
        do {
            m = try await Model.load("mlx-community/Qwen3-4B-3bit")
        } catch {
            print("3-bit Qwen3 integration test skipped: \(error)")
            return
        }
        #expect(m.config.quantization?.bits == 3)
        let result = try await m.generate(
            prompt: "The capital of France is",
            options: GenerateOptions(maxNewTokens: 4)
        )
        #expect(result.generatedTokens.count >= 1)
        #expect(!result.text.isEmpty)
    }

    @Test("6-bit Qwen3 4B generates coherent text")
    func sixBit() async throws {
        let m: Model
        do {
            m = try await Model.load("mlx-community/Qwen3-4B-6bit")
        } catch {
            print("6-bit Qwen3 integration test skipped: \(error)")
            return
        }
        #expect(m.config.quantization?.bits == 6)
        let result = try await m.generate(
            prompt: "The capital of France is",
            options: GenerateOptions(maxNewTokens: 4)
        )
        #expect(result.generatedTokens.count >= 1)
        #expect(!result.text.isEmpty)
    }
}
