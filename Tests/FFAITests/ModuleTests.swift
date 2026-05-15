import Testing
@testable import FFAI

@Suite("Module")
struct ModuleTests {
    final class StubModule: Module {
        let params: [(String, Tensor)]
        init(_ params: [(String, Tensor)]) { self.params = params }
        func parameters() -> [(String, Tensor)] { params }
    }

    @Test("parameterSummary lines up name : shape dtype")
    func summaryShape() {
        let t1 = Tensor.empty(shape: [2, 3], dtype: .f32)
        let t2 = Tensor.empty(shape: [4], dtype: .f16)
        let mod = StubModule([("alpha.weight", t1), ("beta.bias", t2)])

        let summary = mod.parameterSummary()
        #expect(summary.contains("alpha.weight"))
        #expect(summary.contains("[2, 3]"))
        #expect(summary.contains("f32"))
        #expect(summary.contains("beta.bias"))
        #expect(summary.contains("[4]"))
        #expect(summary.contains("f16"))
    }

    @Test("empty parameter list summarizes to empty string")
    func emptyParams() {
        let mod = StubModule([])
        #expect(mod.parameterSummary() == "")
    }
}
