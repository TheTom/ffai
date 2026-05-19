import Foundation
import Testing
@testable import FFAI

@Suite("ModelLifecycle", .serialized)
struct ModelLifecycleTests {
    @Test("LoadProgress.fraction handles total > 0 and total == 0")
    func loadProgressFraction() {
        let p = LoadProgress(stage: "weights", completed: 42, total: 84)
        #expect(p.fraction == 0.5)

        let zero = LoadProgress(stage: "noop", completed: 0, total: 0)
        #expect(zero.fraction == 0)
    }

    @Test("LoadProgress carries stage, completed, total")
    func loadProgressFields() {
        let p = LoadProgress(stage: "config", completed: 3, total: 10)
        #expect(p.stage == "config")
        #expect(p.completed == 3)
        #expect(p.total == 10)
    }

    @Test("ModelLifecycleError wraps Error and preserves message")
    func wrapError() {
        struct Boom: Error, CustomStringConvertible {
            var description: String { "boom" }
        }
        let wrapped = ModelLifecycleError(Boom())
        #expect(wrapped.message.contains("boom"))
        #expect(String(describing: wrapped).contains("boom"))
    }

    @Test("ModelLifecycleError accepts a raw message")
    func messageInit() {
        let e = ModelLifecycleError(message: "something")
        #expect(e.message == "something")
        #expect(e.description == "something")
    }

    @Test("ModelLifecycleEvent default capability is nil")
    func eventDefaults() {
        let e = ModelLifecycleEvent(state: .ready)
        #expect(e.capability == nil)
        if case .ready = e.state { /* ok */ } else {
            Issue.record("expected .ready")
        }
    }

    @Test("ModelLifecycleEvent can target a specific capability")
    func eventCapability() {
        let e = ModelLifecycleEvent(capability: .visionIn,
                                    state: .loading(LoadProgress(stage: "vision", completed: 0, total: 1)))
        #expect(e.capability == .visionIn)
        if case .loading(let p) = e.state {
            #expect(p.stage == "vision")
        } else {
            Issue.record("expected .loading state")
        }
    }
}
