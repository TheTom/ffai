import Foundation
import Testing
@testable import FFAI

@Suite("Capability", .serialized)
struct CapabilityTests {
    @Test("textOnly contains exactly textIn + textOut")
    func textOnlySet() {
        #expect(Capability.textOnly == [.textIn, .textOut])
    }

    @Test("textWithTools adds toolCalling")
    func textWithToolsSet() {
        #expect(Capability.textWithTools == [.textIn, .textOut, .toolCalling])
    }

    @Test("all cases enumerated")
    func allCases() {
        let s = Set(Capability.allCases)
        #expect(s.contains(.textIn))
        #expect(s.contains(.textOut))
        #expect(s.contains(.visionIn))
        #expect(s.contains(.audioIn))
        #expect(s.contains(.audioOut))
        #expect(s.contains(.toolCalling))
        #expect(s.count == 6)
    }

    @Test("Codable round-trip via raw value")
    func codable() throws {
        let original: Capability = .visionIn
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Capability.self, from: data)
        #expect(decoded == original)
    }
}
