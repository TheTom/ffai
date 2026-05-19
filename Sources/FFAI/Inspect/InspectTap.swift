// InspectTap — shared per-op intermediate-value dump helper that
// every model's `forward(...)` uses for first-light debugging.
//
// Replaces bespoke per-model env knobs (the original
// `GEMMA3_DEBUG_TAPS=1` pattern that found the Gemma 3 GELU NaN)
// with a single uniform surface:
//
//   FFAI_INSPECT_TAP=1                 — turn on dumps
//   FFAI_INSPECT_TAP_LAYERS=0,1,5      — optional layer filter
//
// Every new model implementation calls `tap.dumpLayerBoundary(...)`
// at the layer-input + layer-output boundaries inside
// `<Family>Model.forward(...)`. That's enough granularity to
// localise the failing layer in two `ffai inspect --layer-trace`
// runs (one to find which layer's output goes non-finite, one to
// confirm the fix). For inside-layer triage (which op produced
// the NaN) drop temporary fine-grained calls — see the Gemma 3
// 2026-05-19 post-mortem for the pattern.
//
// Wired into `ffai inspect --layer-trace` so the diagnostic
// surface is reachable from the CLI without setting env vars
// manually. See documentation/using-the-cli.md and
// documentation/developing/adding-a-model.md.

import Foundation
import Metal

/// Toggle + filter for layer-boundary intermediate dumps. Value
/// type — every `forward(...)` captures it from the environment
/// once and threads it through the layer loop.
public struct InspectTap: Sendable {
    public let active: Bool
    public let layerFilter: Set<Int>?

    public init(active: Bool, layerFilter: Set<Int>? = nil) {
        self.active = active
        self.layerFilter = layerFilter
    }

    /// Construct from `FFAI_INSPECT_TAP` + `FFAI_INSPECT_TAP_LAYERS`.
    /// Returns an inactive tap when the env var is unset — every
    /// `dumpLayerBoundary` call becomes a zero-cost no-op.
    public static var fromEnvironment: InspectTap {
        let env = ProcessInfo.processInfo.environment
        let active = env["FFAI_INSPECT_TAP"] == "1"
        let filter: Set<Int>? = env["FFAI_INSPECT_TAP_LAYERS"]
            .map { Set($0.split(separator: ",").compactMap { Int($0) }) }
        return InspectTap(active: active, layerFilter: filter)
    }

    /// True when the caller should pay the tap overhead for this
    /// layer. Inlined into the hot path so non-active taps are a
    /// single compare.
    @inline(__always)
    public func shouldDump(layer: Int) -> Bool {
        guard active else { return false }
        guard let filter = layerFilter else { return true }
        return filter.contains(layer)
    }

    /// Synchronously read a tensor's contents to fp32 and print
    /// shape + min/max/nan/inf/first-4. Commits the caller's
    /// cmdbuf, waits, and *replaces* `cmd` with a freshly-allocated
    /// cmdbuf so the caller continues queueing on a clean one.
    ///
    /// `layer` is the layer index for the printout (use `-1` for
    /// outside-layer dumps like the embed or final-norm tap).
    /// `label` describes what the tensor is — keep short for
    /// readability (`"h_in"`, `"layer_out"`, `"logits"`).
    public func dumpLayerBoundary(
        _ t: Tensor, label: String, layer: Int,
        cmd: inout MTLCommandBuffer, device: Device
    ) {
        guard shouldDump(layer: layer) else { return }
        cmd.commit()
        cmd.waitUntilCompleted()

        let n = t.elementCount
        let basePtr = t.buffer.contents().advanced(by: t.offset)
        var floats: [Float] = []
        floats.reserveCapacity(n)
        switch t.dtype {
        case .f32:
            let p = basePtr.bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { floats.append(p[i]) }
        case .f16:
            let p = basePtr.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n { floats.append(halfBitsToFloatForTest(p[i])) }
        case .bf16:
            let p = basePtr.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n { floats.append(bf16BitsToFloatForTest(p[i])) }
        default:
            print("[L\(layer) \(label)] (unsupported dtype \(t.dtype))")
            cmd = device.makeCommandBuffer()
            return
        }

        var nanCount = 0, infCount = 0
        var mn: Float = .infinity, mx: Float = -.infinity
        for v in floats {
            if v.isNaN { nanCount += 1 }
            else if !v.isFinite { infCount += 1 }
            else {
                if v < mn { mn = v }
                if v > mx { mx = v }
            }
        }
        let head = floats.prefix(4)
            .map { String(format: "%.4f", $0) }
            .joined(separator: ", ")
        let mnStr = mn.isFinite ? String(format: "%+.4f", mn) : "—"
        let mxStr = mx.isFinite ? String(format: "%+.4f", mx) : "—"
        let prefix = layer < 0 ? "[\(label)]" : "[L\(layer) \(label)]"
        print("\(prefix) n=\(n) min=\(mnStr) max=\(mxStr) nan=\(nanCount) inf=\(infCount) first=[\(head)]")

        cmd = device.makeCommandBuffer()
    }
}

public extension InspectTap {
    /// Build the cmdbuf that callers should queue work on. When the
    /// tap is active, this is a *private* cmdbuf (separate from the
    /// caller's). The tap commits + waits + replaces it at each
    /// dump; the caller's original `cmd` is never touched, so the
    /// caller's downstream commit / waitUntilCompleted stays a fast
    /// no-op when taps are active.
    ///
    /// When the tap is inactive (production path), returns the
    /// caller's cmd unchanged — zero overhead.
    func makeWorkCmd(from callerCmd: MTLCommandBuffer, device: Device) -> MTLCommandBuffer {
        active ? device.makeCommandBuffer() : callerCmd
    }
}
