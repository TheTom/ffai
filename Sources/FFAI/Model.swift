// Model — public entry point users interact with. Resolves a model
// id-or-path, downloads via HF if needed, decodes config, dispatches to
// the right family file, loads weights, and exposes a forward()/generate()
// surface.

import Foundation
import Tokenizers

public enum ModelError: Error, CustomStringConvertible {
    case unsupportedArchitecture(String)
    case unsupportedModelType(String)
    case capabilityNotAvailable(Capability)

    public var description: String {
        switch self {
        case .unsupportedArchitecture(let a): return "Unsupported architecture: \(a)"
        case .unsupportedModelType(let m): return "Unsupported model_type: \(m)"
        case .capabilityNotAvailable(let c): return "Capability not available: \(c)"
        }
    }
}

/// Routes a config to the right family file. Family files declare which
/// architecture / model_type strings they handle. Add a new family by
/// extending `dispatchAndLoad` here.
public enum ModelRegistry {
    public static func dispatchAndLoad(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> any LanguageModel {
        if let arch = config.architecture, Llama.architectures.contains(arch) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, Llama.modelTypes.contains(mt) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let arch = config.architecture, Qwen3.architectures.contains(arch) {
            return try loadQwen3(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, Qwen3.modelTypes.contains(mt) {
            return try loadQwen3(config: config, weights: weights,
                                 options: options, device: device)
        }
        throw ModelError.unsupportedArchitecture(
            config.architecture ?? config.modelType ?? "<unknown>"
        )
    }

    public static func loadLlama(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> LlamaModel {
        let variant = try Llama.variant(for: config)
        return try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
    }

    public static func loadQwen3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Qwen3Model {
        let variant = try Qwen3.variant(for: config)
        return try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
    }
}

/// High-level loaded model with tokenizer attached. The public API users
/// touch.
public final class Model: @unchecked Sendable {
    /// The concrete model engine (LlamaModel, Qwen3Model, …).
    public let engine: any LanguageModel
    public let tokenizer: any Tokenizer
    public let config: ModelConfig
    public let modelDirectory: URL
    public let availableCapabilities: Set<Capability>
    public let enabledCapabilities: Set<Capability>

    /// Convenience accessor for tests + tools that want the Llama-typed
    /// model. Returns nil if the loaded engine isn't Llama.
    public var llama: LlamaModel? { engine as? LlamaModel }

    /// Convenience accessor for the Qwen3 engine.
    public var qwen3: Qwen3Model? { engine as? Qwen3Model }

    private let stateLock = NSLock()
    private var _currentState: ModelLifecycleState = .ready

    public var currentState: ModelLifecycleState {
        stateLock.lock(); defer { stateLock.unlock() }
        return _currentState
    }

    public let events: AsyncStream<ModelLifecycleEvent>
    private let eventsContinuation: AsyncStream<ModelLifecycleEvent>.Continuation

    init(engine: any LanguageModel, tokenizer: any Tokenizer, config: ModelConfig,
         modelDirectory: URL,
         availableCapabilities: Set<Capability>,
         enabledCapabilities: Set<Capability>) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config
        self.modelDirectory = modelDirectory
        self.availableCapabilities = availableCapabilities
        self.enabledCapabilities = enabledCapabilities
        var cont: AsyncStream<ModelLifecycleEvent>.Continuation!
        self.events = AsyncStream { c in cont = c }
        self.eventsContinuation = cont
    }

    deinit {
        eventsContinuation.finish()
    }

    fileprivate func emit(_ event: ModelLifecycleEvent) {
        stateLock.lock()
        _currentState = event.state
        stateLock.unlock()
        eventsContinuation.yield(event)
    }

    // ─── Top-level loader ────────────────────────────────────────────

    /// Resolve an id-or-path, download if needed, decode config, load
    /// weights, build the family-specific model, attach tokenizer.
    public static func load(
        _ idOrPath: String,
        options: LoadOptions = LoadOptions(),
        device: Device = .shared
    ) async throws -> Model {
        let locator = ModelLocator()
        let dir = try await locator.resolve(idOrPath: idOrPath, revision: options.revision)
        let config = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir, device: device)
        let engine = try ModelRegistry.dispatchAndLoad(
            config: config, weights: bundle, options: options, device: device
        )
        let tokenizer = try await TokenizerLoader().load(from: dir)

        let model = Model(
            engine: engine, tokenizer: tokenizer, config: config,
            modelDirectory: dir,
            availableCapabilities: Capability.textOnly,
            enabledCapabilities: options.capabilities
        )

        // Phase 2: prewarm just touches the embedding lookup once so the
        // PSO is compiled before the first user-visible decode.
        if options.prewarm {
            await model.prewarm()
        }

        model.emit(ModelLifecycleEvent(state: .ready))
        return model
    }

    /// Compile PSOs for the kernels we'll need during decode by running
    /// one no-op forward step. Costs ~100ms-1s on first load.
    public func prewarm() async {
        let cache = engine.makeKVCache()
        _ = engine.forward(tokenId: 0, position: 0, caches: cache)
    }
}
