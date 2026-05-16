// Debug — gated, subsystem-tagged log helpers.
//
// FFAI subsystems opt into noisy debug output via env vars. The
// global `FFAI_DEBUG=1` enables every subsystem; per-subsystem
// `FFAI_DEBUG_<NAME>=1` enables just that one. The CLI's `--debug`
// flag sets `FFAI_DEBUG=1` early in main before any model load runs.
//
// Output goes to stderr so it doesn't pollute stdout (where the model
// text + stats live). When no subsystem is enabled, `Debug.log(...)`
// is a guarded no-op — the message string isn't even constructed
// because the parameter is `@autoclosure`.

import Foundation

public enum DebugSubsystem: String, Sendable, CaseIterable {
    case loader      // ModelLocator / ModelDownloader
    case load        // Model.load + family loaders
    case kernels     // Per-kernel dispatch (very chatty; opt-in)
    case sampling    // Sampling.swift
    case kvcache     // KVCache append + slice
    case generate    // Generate loop + per-token decisions
    case dispatch    // Per-MTLCommandBuffer commit/wait
    case bench       // Bench harness internals

    /// `true` when the global `FFAI_DEBUG=1` is set OR
    /// `FFAI_DEBUG_<RAWVALUE_UPPERCASED>=1`. Reads via `getenv(3)`
    /// rather than `ProcessInfo.processInfo.environment` so callers
    /// that mutate the env at runtime (CLI `--debug`, tests) see the
    /// change immediately — `ProcessInfo` snapshots once and never
    /// updates.
    public var isEnabled: Bool {
        if getenv("FFAI_DEBUG") != nil { return true }
        return getenv("FFAI_DEBUG_\(rawValue.uppercased())") != nil
    }
}

public enum Debug {
    /// Emit a debug line for `subsystem`. The message closure is only
    /// evaluated when the subsystem is enabled.
    public static func log(_ subsystem: DebugSubsystem,
                           _ message: @autoclosure () -> String) {
        guard subsystem.isEnabled else { return }
        let line = "[ffai:\(subsystem.rawValue)] \(message())\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    /// Programmatically enable the global gate from Swift code (e.g. a
    /// CLI `--debug` flag handler). Equivalent to setting
    /// `FFAI_DEBUG=1` in the process env.
    public static func enableAll() {
        setenv("FFAI_DEBUG", "1", 1)
    }

    /// Programmatically enable just one subsystem.
    public static func enable(_ subsystem: DebugSubsystem) {
        setenv("FFAI_DEBUG_\(subsystem.rawValue.uppercased())", "1", 1)
    }

    /// `true` when *any* subsystem (or the global gate) is enabled.
    /// Useful for callers that want to opt out of expensive
    /// instrumentation paths when nobody's listening.
    public static var isAnyEnabled: Bool {
        if getenv("FFAI_DEBUG") != nil { return true }
        return DebugSubsystem.allCases.contains(where: { $0.isEnabled })
    }
}
