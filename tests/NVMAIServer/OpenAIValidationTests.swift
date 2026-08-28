import Foundation
import Testing
@testable import NVMAI
@testable import NVMAIServerCore

@Suite("OpenAI request validation")
struct OpenAIValidationTests {
    @Test func omittedSamplingControlsUseProductionDefaults() throws {
        let data = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        let validated = try OpenAIRequestValidator.validate(request)

        #expect(validated.generationConfig.temperature == 0.6)
        #expect(validated.generationConfig.topK == 20)
        #expect(validated.generationConfig.topP == 0.95)
        #expect(validated.generationConfig.presencePenalty == 0)
    }

    @Test func requiredToolChoiceIsRejected() throws {
        let data = Data(#"""
        {"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":"required"}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request)
        }
    }

    @Test func acceptsLeadingSystemAndDeveloperGuidance() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"system","content":"system"},
          {"role":"developer","content":"developer"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request)
        #expect(validated.messages.map(\.role) == [.system, .developer, .user])
    }

    @Test func reasoningContentMapsToAssistantThinking() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"user","content":"hi"},
          {"role":"assistant","content":"ok","reasoning_content":"because"},
          {"role":"user","content":"more"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request)
        #expect(validated.messages[1].thinking == "because")
        #expect(validated.messages[0].thinking == nil)
    }

    @Test func reasoningContentOutsideAssistantIsRejected() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"user","content":"hi","reasoning_content":"sneaky"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request)
        }
    }

    @Test func rejectsLateDeveloperGuidance() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"user","content":"hello"},
          {"role":"developer","content":"late"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request)
        }
    }

    @Test func wideIntegerToolArgumentsRoundTripExactly() async throws {
        let expected = "9007199254740993"
        let parsed = try QwenToolCallParser().parse(
            "\n<function=lookup>\n<parameter=id>\n\(expected)\n</parameter>\n</function>\n",
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.argumentsJSON.contains(#""id":\#(expected)"#))
        let signedMinimum = String(Int64.min)
        let signedMaximum = String(Int64.max)
        let unsignedMaximum = String(UInt64.max)
        let edges = try QwenToolCallParser().parse(
            "\n<function=lookup>\n<parameter=minimum>\n\(signedMinimum)\n</parameter>\n"
                + "<parameter=maximum>\n\(signedMaximum)\n</parameter>\n"
                + "<parameter=unsigned>\n\(unsignedMaximum)\n</parameter>\n</function>\n",
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234568")
        #expect(edges.arguments.objectValue?["minimum"] == .integer(.min))
        #expect(edges.arguments.objectValue?["maximum"] == .integer(.max))
        #expect(edges.arguments.objectValue?["unsigned"] == .unsignedInteger(.max))
        let encodedEdges = try edges.arguments.encoded()
        #expect(encodedEdges.contains(signedMinimum))
        #expect(encodedEdges.contains(signedMaximum))
        #expect(encodedEdges.contains(unsignedMaximum))
        #expect(try JSONDecoder().decode(
            JSONValue.self,
            from: Data(encodedEdges.utf8)) == edges.arguments)
        // The Qwen parser keeps non-JSON parameter values as raw strings
        // (no strict numeric grammar); malformed-number rejection lives in
        // QwenToolCallParserTests via the JSONValue decode path.

        let data = Data(#"""
        {
          "model":"m",
          "messages":[
            {"role":"user","content":"lookup"},
            {"role":"assistant","tool_calls":[{
              "id":"call_0123456789abcdef01234567",
              "type":"function",
              "function":{"name":"lookup","arguments":"{\"id\":9007199254740993}"}
            }]},
            {"role":"tool","tool_call_id":"call_0123456789abcdef01234567","content":"ok"}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request)
        let call = try #require(validated.messages[1].toolCalls.first)
        #expect(call.arguments.contains(#""id":\#(expected)"#))
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let rendered = tokenizer.decode(
            try tokenizer.encodeToolChat(
                messages: validated.messages,
                tools: validated.tools),
            skipSpecialTokens: false)
        #expect(rendered.contains(expected))

        // 18446744073709551615 (UInt64.max) cannot be represented exactly as
        // an Int64 for the jinja tool renderer. The request shape is
        // otherwise valid (the tool call IS answered by a tool result), so
        // the rejection below is specifically about the unrepresentable
        // number, not the S19 unresolved-tool-call check.
        let unrepresentableHistory = Data(#"""
        {
          "model":"m",
          "messages":[
            {"role":"user","content":"lookup"},
            {"role":"assistant","tool_calls":[{
              "id":"call_0123456789abcdef01234569",
              "type":"function",
              "function":{"name":"lookup","arguments":"{\"id\":18446744073709551615}"}
            }]},
            {"role":"tool","tool_call_id":"call_0123456789abcdef01234569","content":"ok"}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
            }
          }]
        }
        """#.utf8)
        let rejected = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: unrepresentableHistory)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(rejected)
        }
    }

    @Test func acceptedNonIdentifierParameterKeysParseAndRender() async throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"lookup"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{
                "type":"object",
                "properties":{
                  "$id":{"type":"string"},
                  "file-path":{"type":"string"},
                  "nested":{"type":"object","properties":{"child-key":{"type":"integer"}}}
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request)
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        _ = try tokenizer.encodeToolChat(
            messages: validated.messages,
            tools: validated.tools)
        let parsed = try QwenToolCallParser().parse(
            #"""
            <function=lookup>
            <parameter=$id>
            item
            </parameter>
            <parameter=file-path>
            /tmp/x
            </parameter>
            <parameter=nested>
            {"child-key":7}
            </parameter>
            </function>
            """#,
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.arguments.objectValue?["$id"] == .string("item"))
        #expect(parsed.arguments.objectValue?["file-path"] == .string("/tmp/x"))
        #expect(parsed.arguments.objectValue?["nested"]
                == .object(["child-key": .integer(7)]))
    }

    @Test func freeFormParameterNamesAreAccepted() throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"lookup"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{
                "type":"object",
                "allOf":[{
                  "type":"object",
                  "properties":{"bad:key":{"type":"string"}}
                }]
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        // S-audit: tool parameter names are deliberately free-form (only the
        // schema structure is validated), even inside allOf compositions.
        let validated = try OpenAIRequestValidator.validate(request)
        #expect(validated.tools.count == 1)
    }
}

@Suite("Streaming stop matcher")
struct StreamingStopMatcherTests {
    @Test func withholdsCrossChunkStop() {
        var matcher = StreamingStopMatcher(stops: ["END"])
        #expect(matcher.push("hello E") == "hello ")
        #expect(matcher.push("N") == "")
        #expect(matcher.push("D ignored") == "")
        #expect(matcher.isStopped)
    }

    @Test func flushesUnicodeTail() {
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("hello 🌳") == "hello ")
        #expect(matcher.finish() == "🌳")
    }
}

@Suite("Server arguments")
struct ServerArgumentTests {
    @Test func defaults() throws {
        let arguments = try ServerArguments.parse(
            ["--model", "model.gturbo"], environment: [:])
        #expect(arguments.mtpModel == nil)
        #expect(arguments.mtpMemoryMiB == 384)
        #expect(arguments.port == 8080)
        #expect(arguments.maxContext == 262_144)
        #expect(arguments.queueLimit == 4)
        #expect(arguments.promptCacheMode == .multiPrefix)
        #expect(arguments.promptCacheMaximumEntries == 4)
        #expect(arguments.promptCacheMemoryMiB == 256)
        #expect(arguments.promptCacheDiskDirectory == nil)
        #expect(arguments.promptCacheDiskMiB == 8_192)
        #expect(arguments.prefillChunkTokens == nil)
        #expect(arguments.kvCachePrecision == .int8)
        #expect(arguments.ropeScalingMode == .none)
        #expect(arguments.thinkingMode == .off)
    }

    @Test func configModeIsTheDefaultAndModelIsOptional() throws {
        let arguments = try ServerArguments.parse([], environment: [:])
        #expect(arguments.model == nil)
        #expect(arguments.configPath == nil)
        #expect(arguments.modelsDir == nil)
        #expect(!arguments.preload)
    }

    @Test func parsesConfigModelsDirAndPreload() throws {
        let arguments = try ServerArguments.parse(
            ["--config", "/tmp/server.json", "--models-dir", "/models", "--preload"],
            environment: [:])
        #expect(arguments.configPath == "/tmp/server.json")
        #expect(arguments.modelsDir == "/models")
        #expect(arguments.preload)
    }

    @Test func modelExcludesConfigAndModelsDir() throws {
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.gturbo", "--config", "/tmp/c.json"])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.gturbo", "--models-dir", "/models"])
        }
    }

    @Test func modelIDAndMTPRequireModel() throws {
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model-id", "nice-name"])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--mtp-model", "mtp.gturbo"])
        }
    }

    @Test func preloadContradictsLazyLoad() throws {
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--preload", "--lazy-load"])
        }
    }

    @Test func configDefaultsMergeUnderFlagPrecedence() throws {
        let bare = try ServerArguments.parse([], environment: [:])
        let merged = try bare.merging(configDefaults: .init(
            maxContext: 32_768, ramBudget: "6G", idleUnloadSeconds: 300))
        #expect(merged.maxContext == 32_768)
        #expect(merged.expertCacheBudgetBytes == 6 << 30)
        #expect(merged.idleUnloadSeconds == 300)

        let flagged = try ServerArguments.parse(
            ["--max-context", "65536", "--ram-budget", "2G", "--idle-unload-seconds", "0"],
            environment: [:])
        let kept = try flagged.merging(configDefaults: .init(
            maxContext: 32_768, ramBudget: "6G", idleUnloadSeconds: 300))
        #expect(kept.maxContext == 65_536)
        #expect(kept.expertCacheBudgetBytes == 2 << 30)
        #expect(kept.idleUnloadSeconds == 0)
    }

    @Test func aConfigContextOutsideTheSupportedSetFails() throws {
        let bare = try ServerArguments.parse([], environment: [:])
        #expect(throws: ServerArgumentError.self) {
            _ = try bare.merging(configDefaults: .init(maxContext: 12_345))
        }
    }

    @Test func parsesOnlyBinaryThinkingModes() throws {
        let on = try ServerArguments.parse([
            "--model", "model.gturbo", "--thinking", "on",
        ])
        #expect(on.thinkingMode == .on)
        let environmentOn = try ServerArguments.parse(
            ["--model", "model.gturbo"],
            environment: ["NVMAI_THINKING_MODE": "true"])
        #expect(environmentOn.thinkingMode == .on)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo", "--thinking", "high",
            ])
        }
    }

    @Test func parsesKVPrecisionAndYaRNContexts() throws {
        let defaults = try ServerArguments.parse([
            "--model", "model.gturbo", "--kv-bits", "16",
            "--rope-scaling", "yarn",
        ])
        #expect(defaults.kvCachePrecision == .fp16)
        #expect(defaults.ropeScalingMode == .yarn)
        #expect(defaults.maxContext == 1_048_576)
        let halfMillion = try ServerArguments.parse([
            "--model", "model.gturbo", "--rope-scaling", "yarn",
            "--max-context", "524288",
        ])
        #expect(halfMillion.maxContext == 524_288)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo", "--max-context", "524288",
            ])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo", "--mtp-model", "mtp.gturbo",
                "--rope-scaling", "yarn",
            ])
        }
    }

    @Test func acceptsPublicPrefillChunksAndRejectsUnsupportedValues() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--prefill-chunk", "4096",
        ])
        #expect(arguments.prefillChunkTokens == 4_096)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo",
                "--prefill-chunk", "8192",
            ])
        }
    }

    @Test func parsesBoundedMTPOptions() throws {
        let arguments = try ServerArguments.parse([
            "--model", "qwen.gturbo",
            "--mtp-model", "qwen-mtp.gturbo",
            "--mtp-memory-mib", "512",
        ])
        #expect(arguments.mtpModel == "qwen-mtp.gturbo")
        #expect(arguments.mtpMemoryMiB == 512)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "qwen.gturbo",
                "--mtp-memory-mib", "1024",
            ])
        }
    }

    @Test func mtpForcesPromptCacheOff() {
        #expect(ServerModelSession.effectivePromptCacheMode(
            requested: .multiPrefix,
            mtpEnabled: true) == .off)
        #expect(ServerModelSession.effectivePromptCacheMode(
            requested: .multiPrefix,
            mtpEnabled: false) == .multiPrefix)
    }

    @Test func parsesSinglePrefixModeAndRejectsUnknownMode() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--prompt-cache-mode", "single-prefix",
        ])
        #expect(arguments.promptCacheMode == .singlePrefix)
        let multi = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--prompt-cache-mode", "multi-prefix",
            "--prompt-cache-entries", "8",
            "--prompt-cache-memory-mib", "512",
            "--prompt-cache-disk", "/tmp/nvmai-cache",
            "--prompt-cache-disk-mib", "16384",
        ])
        #expect(multi.promptCacheMode == .multiPrefix)
        #expect(multi.promptCacheMaximumEntries == 8)
        #expect(multi.promptCacheMemoryMiB == 512)
        #expect(multi.promptCacheDiskDirectory == "/tmp/nvmai-cache")
        #expect(multi.promptCacheDiskMiB == 16_384)
        let rollback = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--prompt-cache-mode", "off",
        ])
        #expect(rollback.promptCacheMode == .off)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo",
                "--prompt-cache-mode", "many",
            ])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo",
                "--prompt-cache-entries", "0",
            ])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo",
                "--prompt-cache-memory-mib", "4097",
            ])
        }
    }

    @Test func accepts256KContextAndRejectsUnsupportedValues() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--max-context", "262144",
        ])
        #expect(arguments.maxContext == 262_144)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo",
                "--max-context", "100000",
            ])
        }
    }
}
