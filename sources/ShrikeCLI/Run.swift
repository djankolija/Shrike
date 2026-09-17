import Foundation
import Metal
import Shrike

private struct MessageJSON: Decodable {
    let role: String
    let content: String?

    enum CodingKeys: String, CodingKey { case role, content }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        if let value = try container.decodeIfPresent(JSONValue.self, forKey: .content) {
            content = Self.text(value)
        } else {
            content = nil
        }
    }

    private static func text(_ value: JSONValue) -> String? {
        switch value {
        case .string(let s):
            return s
        case .null:
            return nil
        case .array(let parts):
            var out = ""
            for part in parts {
                guard case .object(let dict) = part,
                      case .string(let type)? = dict["type"], type == "input_text",
                      case .string(let text)? = dict["text"] else { continue }
                out += text
            }
            return out
        default:
            return nil
        }
    }
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

private enum StageOutcome<Value> {
    case value(Value)
    case exit(RunResult)
}

private struct LoadedRuntime {
    let context: MetalContext
    let runtime: RuntimeConfiguration
    let runner: RealForwardRunner
    let scratch: RawCompletionScratch
}

/// The CLI driver: parse messages, load the model, run one completion, print the timing footer.
public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    do {
        try RuntimeConfiguration.refuseUnknownEnvironment()
        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: modelURL,
            thinkingMode: args.thinkingMode)
        // Concise mode injects a per-quantization system prompt. The routed
        // expert bit width comes from the manifest so the right prompt
        // variant is selected before the full model load.
        let expectedArch: ArchConfig
        switch resolveExpectedArch(modelURL: modelURL, stderr: stderr) {
        case .value(let arch):
            expectedArch = arch
        case .exit(let result):
            return result
        }
        let promptIds: [Int32]
        switch try buildPrompt(args: args,
                               modelURL: modelURL,
                               tokenizer: tokenizer,
                               expectedArch: expectedArch,
                               stderr: stderr) {
        case .value(let ids):
            promptIds = ids
        case .exit(let result):
            return result
        }
        if let path = args.tokenizePath {
            try writeTokenization(promptIds, tokenizer: tokenizer, to: path)
            return RunResult(exitCode: 0)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        var config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            presencePenalty: GenerationDefaults.presencePenalty,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        if let path = args.forceTokensPath {
            let forced = try readForcedTokens(path: path)
            guard forced.count <= effectiveMaxNew else {
                return errored(
                    stderr,
                    "forced tokens: \(forced.count) ids exceed the budget of \(effectiveMaxNew); raise --max-new or --max-context",
                    2)
            }
            config.forcedTokens = forced
        }
        let dump = try args.dumpLogitsPath.map {
            try FileLogitsSink(path: $0, forced: config.forcedTokens)
        }
        defer { try? dump?.finish() }
        config.logitsSink = dump
        let logitsHead = !config.isPureGreedy || args.logitsHead
            || config.forcedTokens != nil || dump != nil
        let loaded: LoadedRuntime
        switch try buildRuntime(args: args,
                                modelURL: modelURL,
                                expectedArch: expectedArch,
                                logitsHead: logitsHead,
                                promptIds: promptIds,
                                stderr: stderr) {
        case .value(let value):
            loaded = value
        case .exit(let result):
            return result
        }
        let stats = try await runRawCompletion(
            producer: loaded.runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: loaded.context,
            scratch: loaded.scratch,
            prefillConfig: loaded.runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }
        try dump?.finish()

        if !args.quiet {
            writeFooter(stats: stats, stderr: stderr)
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

private func resolveExpectedArch(modelURL: URL,
                                 stderr: FileHandle) -> StageOutcome<ArchConfig> {
    do {
        let family = try ManifestReader.peekFamily(directoryURL: modelURL)
        guard let baseline = ArchConfig.knownArchitectures[family] else {
            return .exit(errored(stderr,
                                 "no compiled baseline for family \(family.rawValue)", 1))
        }
        return .value(baseline)
    } catch {
        return .exit(errored(stderr, "cannot read model manifest: \(error)", 1))
    }
}

private func buildPrompt(args: Args,
                         modelURL: URL,
                         tokenizer: GFTokenizer,
                         expectedArch: ArchConfig,
                         stderr: FileHandle) throws -> StageOutcome<[Int32]> {
    let concisePrompt: String?
    if args.concise {
        let bits = (try? ManifestReader.load(
            directoryURL: modelURL,
            expecting: expectedArch).quant?.routedExpert.weightBits) ?? 4
        concisePrompt = ConcisePrompt.prompt(forRoutedExpertBits: bits)
    } else {
        concisePrompt = nil
    }
    let promptIds: [Int32]
    if let rawPrompt = args.prompt {
        if let concisePrompt {
            let messages = ConcisePrompt.appendingSystemPrompt(
                concisePrompt,
                to: [GFTokenizer.Message(role: .user, content: rawPrompt)])
            let rendered = try tokenizer.applyChatTemplate(messages)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else {
            promptIds = tokenizer.encode(rawPrompt, addBOS: true)
        }
    } else if let messagesFile = args.messagesFile {
        let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                            options: [.mappedIfSafe])
        let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
        var messages = try rows.map { row -> GFTokenizer.Message in
            guard let role = GFTokenizer.Role(rawValue: row.role) else {
                throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
            }
            return GFTokenizer.Message(role: role, content: row.content)
        }
        if let concisePrompt {
            messages = ConcisePrompt.appendingSystemPrompt(concisePrompt, to: messages)
        }
        let rendered = try tokenizer.applyChatTemplate(messages)
        promptIds = tokenizer.encode(rendered, addBOS: false)
    } else {
        return .exit(errored(stderr, "one of --prompt or --messages-file is required", 2))
    }
    guard !promptIds.isEmpty else { return .exit(errored(stderr, "empty prompt", 2)) }
    guard promptIds.count < args.maxContext else {
        return .exit(errored(
            stderr,
            "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
            2))
    }
    return .value(promptIds)
}

private func buildRuntime(args: Args,
                          modelURL: URL,
                          expectedArch: ArchConfig,
                          logitsHead: Bool,
                          promptIds: [Int32],
                          stderr: FileHandle) throws -> StageOutcome<LoadedRuntime> {
    let loadRuntime = try RuntimeConfiguration(
        expertCacheSlots: args.expertCacheSlots,
        forceLogitsHead: logitsHead,
        prefetchTracePath: RuntimeConfiguration.environmentPrefetchTracePath())

    guard MTLCreateSystemDefaultDevice() != nil else {
        return .exit(errored(stderr, "no Metal device", 1))
    }
    let context = try MetalContext()
    let model = try Model.load(
        directoryURL: modelURL,
        device: context.device,
        expecting: expectedArch,
        streamingMode: .pread(
            slotCount: loadRuntime.expertCacheSlots,
            perLayer: try RuntimeConfiguration.environmentExpertSlotTable(
                layers: expectedArch.numLayers, uniformSlots: loadRuntime.expertCacheSlots,
                leadingDenseLayers: expectedArch.numLeadingDenseLayers),
            policy: try RuntimeConfiguration.environmentExpertPolicy()),
        integrityPolicy: .resolved(directoryURL: modelURL))
    let prefillChunkTokens: Int
    switch args.prefillChunk {
    case .fixed(let tokens):
        prefillChunkTokens = tokens
    case .auto:
        prefillChunkTokens = RuntimeConfiguration.allowedPrefillChunkTokens
            .first(where: { $0 >= promptIds.count })
            ?? PrefillRuntimeConfig.maxChunkTokens
    case nil:
        prefillChunkTokens = model.config.family == .qwen36
            ? RuntimeConfiguration.qwenLongPrefillChunkTokens
            : loadRuntime.prefillChunkTokens
    }
    let runtime = try RuntimeConfiguration(
        expertCacheSlots: loadRuntime.expertCacheSlots,
        prefillChunkTokens: prefillChunkTokens,
        forceLogitsHead: logitsHead,
        prefetchTracePath: loadRuntime.prefetchTracePath,
        kvCachePrecision: args.kvCachePrecision,
        ropeScalingMode: args.ropeScalingMode,
        yarnContextTokens: args.ropeScalingMode == .yarn
            ? args.maxContext : RuntimeConfiguration.defaultYaRNContextTokens)
    let runner = try RealForwardRunner(
        model: model,
        context: context,
        maxContext: args.maxContext,
        runtimeConfiguration: runtime)
    let scratch = try RawCompletionScratch(context: context,
                                           vocab: model.config.vocabSize,
                                           logitSoftcap: Float(model.config.finalLogitSoftcap))
    return .value(LoadedRuntime(context: context,
                                runtime: runtime,
                                runner: runner,
                                scratch: scratch))
}

private func writeFooter(stats: RawDecodeResult, stderr: FileHandle) {
    let tokensPerSecond = stats.decodeSeconds > 0
        ? Double(stats.newTokens) / stats.decodeSeconds
        : 0
    let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok/\(String(format: "%.2f", stats.prefillSeconds))s new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
    stderr.write(Data(footer.utf8))
}

private func writeTokenization(_ ids: [Int32], tokenizer: GFTokenizer, to path: String) throws {
    let pieces = ids.map { tokenizer.decode([$0], skipSpecialTokens: false) }
    let body: [String: Any] = ["ids": ids.map { Int($0) }, "pieces": pieces]
    let data = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
    try data.write(to: URL(fileURLWithPath: path))
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
