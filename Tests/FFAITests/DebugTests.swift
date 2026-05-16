// DebugTests — env-var gating + per-subsystem opt-in.
//
// Tests mutate the process env via setenv/unsetenv. Restored on exit
// of each test so the suite stays hermetic.

import Foundation
import Testing
@testable import FFAI

@Suite("Debug", .serialized)
struct DebugTests {

    // MARK: helpers

    private func clearAllDebugEnv() {
        unsetenv("FFAI_DEBUG")
        for sub in DebugSubsystem.allCases {
            unsetenv("FFAI_DEBUG_\(sub.rawValue.uppercased())")
        }
    }

    // MARK: tests

    @Test("All subsystems off by default")
    func defaultOff() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        for sub in DebugSubsystem.allCases {
            #expect(sub.isEnabled == false)
        }
        #expect(Debug.isAnyEnabled == false)
    }

    @Test("FFAI_DEBUG=1 enables every subsystem")
    func globalGate() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        setenv("FFAI_DEBUG", "1", 1)
        for sub in DebugSubsystem.allCases {
            #expect(sub.isEnabled, "\(sub.rawValue) should be enabled by FFAI_DEBUG")
        }
        #expect(Debug.isAnyEnabled)
    }

    @Test("FFAI_DEBUG_<NAME>=1 enables just that subsystem")
    func perSubsystemGate() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        setenv("FFAI_DEBUG_LOADER", "1", 1)
        #expect(DebugSubsystem.loader.isEnabled)
        #expect(DebugSubsystem.kernels.isEnabled == false)
        #expect(Debug.isAnyEnabled)
    }

    @Test("Debug.enableAll() flips the global gate")
    func enableAll() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        Debug.enableAll()
        #expect(DebugSubsystem.generate.isEnabled)
    }

    @Test("Debug.enable(_:) flips one subsystem")
    func enableOne() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        Debug.enable(.bench)
        #expect(DebugSubsystem.bench.isEnabled)
        #expect(DebugSubsystem.kvcache.isEnabled == false)
    }

    @Test("Debug.log evaluates the closure only when subsystem enabled")
    func lazyEvaluation() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        // Off — closure must NOT fire.
        var called = false
        Debug.log(.kvcache, { () -> String in called = true; return "x" }())
        #expect(called == false)

        // On — closure DOES fire.
        setenv("FFAI_DEBUG_KVCACHE", "1", 1)
        Debug.log(.kvcache, { () -> String in called = true; return "x" }())
        #expect(called == true)
    }

    @Test("DebugSubsystem rawValues stable")
    func rawValues() {
        let expected: [DebugSubsystem: String] = [
            .loader: "loader", .load: "load", .kernels: "kernels",
            .sampling: "sampling", .kvcache: "kvcache",
            .generate: "generate", .dispatch: "dispatch", .bench: "bench",
        ]
        for (sub, raw) in expected {
            #expect(sub.rawValue == raw)
        }
    }
}
