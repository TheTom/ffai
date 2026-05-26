// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// End-to-end test: download a published 2-bit mlx-community checkpoint
// (Qwen2 1.5B dolphin fine-tune, the smallest pure-2-bit Qwen variant
// the registry currently publishes — see the HF API search referenced
// in PR notes) and assert FFAI's greedy decode produces coherent
// output. Exercises the dequant_gemv_int2 + mt_qmm_mma_int2 kernels
// end-to-end through the LlamaDense Qwen2 routing.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Qwen2 2-bit Integration", .serialized)
struct Quantized2bitIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        // mlx-community didn't publish a Qwen3 1.7B 2-bit. Of the small
        // Qwen2/2.5 2-bit conversions in the registry, the dolphin
        // Qwen2 line writes `"bits": "2"` (string) in config.json — a
        // checkpoint-side bug that the FFAI parser rejects. The
        // Josiefied Qwen2.5 1.5B 2-bit checkpoint writes `"bits": 2`
        // correctly and loads through the standard Qwen2.5 (LlamaDense)
        // path; it's the smallest valid 2-bit target we can hit today.
        let modelId = "mlx-community/Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v1-2bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        #expect(m.config.quantization?.bits == 2)
        #expect(m.config.quantization?.groupSize == 64)

        // Qwen2 1.5B: hidden=1536, nLayers=28, nHeads=12, nKVHeads=2,
        // headDim=128. Verifies the loader picked the right
        // architecture/sizing for the dolphin fine-tune.
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 12)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 128)

        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Qwen2.5 1.5B 2-bit")
    }
}
