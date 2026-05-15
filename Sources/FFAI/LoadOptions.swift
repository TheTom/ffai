// LoadOptions — what the user requests when calling Model.load(...).

import Foundation

public enum KVCacheKind: Sendable {
    case raw            // unquantized fp16 / bf16
    // .affineQuantized — Phase 3
    // .turbo            — Phase 4
}

public enum DispatchMode: Sendable {
    case eager
    // .argumentBuffers — Phase 5
    // .icb             — Phase 5+
}

public struct LoadOptions: Sendable {
    /// Which capabilities to load. textIn + textOut implicitly always on.
    public var capabilities: Set<Capability>
    public var kvCache: KVCacheKind
    public var dispatchMode: DispatchMode
    /// Run prewarm() before transitioning to .ready. Default true.
    public var prewarm: Bool
    /// Allow runtime enable/disable of capabilities after load.
    public var lazyCapabilities: Bool
    /// Override revision for HF download. Defaults to "main".
    public var revision: String

    public init(
        capabilities: Set<Capability> = Capability.textOnly,
        kvCache: KVCacheKind = .raw,
        dispatchMode: DispatchMode = .eager,
        prewarm: Bool = true,
        lazyCapabilities: Bool = true,
        revision: String = "main"
    ) {
        self.capabilities = capabilities.union(Capability.textOnly)
        self.kvCache = kvCache
        self.dispatchMode = dispatchMode
        self.prewarm = prewarm
        self.lazyCapabilities = lazyCapabilities
        self.revision = revision
    }
}
