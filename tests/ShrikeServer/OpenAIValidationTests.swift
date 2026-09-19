import Foundation
import Testing
@testable import Shrike
@testable import ShrikeServerCore

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

    @Test func reasoningEffortParsesAndValidates() throws {
        let data = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"low"}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request)
        #expect(validated.reasoningEffort == .low)
    }

    @Test func omittedReasoningEffortStaysNil() throws {
        let data = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(try OpenAIRequestValidator.validate(request).reasoningEffort == nil)
    }

    @Test func invalidReasoningEffortIsRejected() throws {
        let data = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"minimal"}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request)
        }
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
        let arguments = try ShrikeServerCommand.parse(
            ["--model", "model.gturbo"])
        #expect(arguments.port == 8080)
        #expect(arguments.maxContext == 262_144)
        #expect(arguments.kvCachePrecision == .int8)
        #expect(arguments.ropeScalingMode == .none)
        #expect(arguments.thinkingMode == .off)
    }

    @Test func configModeIsTheDefaultAndModelIsOptional() throws {
        let arguments = try ShrikeServerCommand.parse([])
        #expect(arguments.model == nil)
        #expect(arguments.configPath == nil)
    }

    @Test func parsesAConfigPath() throws {
        let arguments = try ShrikeServerCommand.parse(["--config", "/tmp/server.json"])
        #expect(arguments.configPath == "/tmp/server.json")
    }

    @Test func modelExcludesConfig() throws {
        #expect(throws: (any Error).self) {
            try ShrikeServerCommand.parse(["--model", "m.gturbo", "--config", "/tmp/c.json"])
        }
    }

    @Test func theTrimmedServeFlagsNoLongerParse() {
        for argv in [["--model-id", "nice-name"], ["--models-dir", "/models"],
                     ["--preload"], ["--lazy-load"], ["--queue-limit", "8"],
                     ["--idle-unload-seconds", "300"],
                     ["--prompt-cache-mode", "off"], ["--prompt-cache-entries", "8"],
                     ["--prompt-cache-memory-mib", "512"],
                     ["--prompt-cache-disk", "/tmp/c"], ["--prompt-cache-disk-mib", "16384"],
                     ["--prefill-chunk", "4096"], ["--reasoning-retention", "stripped"]] {
            #expect(throws: (any Error).self) {
                _ = try ShrikeServerCommand.parse(["--model", "m.gturbo"] + argv)
            }
        }
    }

    @Test func configDefaultsMergeUnderFlagPrecedence() throws {
        let bare = try ShrikeServerCommand.parse([])
        let merged = try bare.merging(configDefaults: .init(
            maxContext: 32_768, ramBudget: "6G"))
        #expect(merged.maxContext == 32_768)
        #expect(merged.expertCacheBudgetBytes == 6 << 30)

        let flagged = try ShrikeServerCommand.parse(
            ["--max-context", "65536", "--ram-budget", "2G"])
        let kept = try flagged.merging(configDefaults: .init(
            maxContext: 32_768, ramBudget: "6G"))
        #expect(kept.maxContext == 65_536)
        #expect(kept.expertCacheBudgetBytes == 2 << 30)
    }

    @Test func aConfigContextOutsideTheSupportedSetFails() throws {
        let bare = try ShrikeServerCommand.parse([])
        #expect(throws: (any Error).self) {
            _ = try bare.merging(configDefaults: .init(maxContext: 12_345))
        }
    }

    @Test func parsesOnlyBinaryThinkingModes() throws {
        let on = try ShrikeServerCommand.parse([
            "--model", "model.gturbo", "--thinking", "on",
        ])
        #expect(on.thinkingMode == .on)
        let environmentOn = try ShrikeServerCommand.parse(["--model", "model.gturbo"])
            .merging(configDefaults: .init(),
                     environment: ["SHRIKE_THINKING_MODE": "true"])
        #expect(environmentOn.thinkingMode == .on)
        #expect(throws: (any Error).self) {
            try ShrikeServerCommand.parse([
                "--model", "model.gturbo", "--thinking", "high",
            ])
        }
    }

    @Test func parsesReasoningEffortLevels() throws {
        let parsed = try ShrikeServerCommand.parse(
            ["--model", "m", "--reasoning-effort", "low"])
        #expect(parsed.reasoningEffort == .low)
        #expect(try ShrikeServerCommand.parse(["--model", "m"])
            .reasoningEffort == nil)
        #expect(try ShrikeServerCommand.parse(["--model", "m"])
            .merging(configDefaults: .init(),
                     environment: ["SHRIKE_REASONING_EFFORT": "HIGH"])
            .reasoningEffort == .high)
        #expect(throws: (any Error).self) {
            try ShrikeServerCommand.parse(
                ["--model", "m", "--reasoning-effort", "max"])
        }
        #expect(throws: (any Error).self) {
            try ShrikeServerCommand.parse(["--model", "m"])
                .merging(configDefaults: .init(),
                         environment: ["SHRIKE_REASONING_EFFORT": "max"])
        }
        #expect(try ShrikeServerCommand.parse(["--model", "m"])
            .merging(configDefaults: .init(),
                     environment: ["SHRIKE_REASONING_EFFORT": "low"])
            .reasoningEffort == .low)
    }

    @Test func anUnknownFlagReportsAsUnknownNotAsAMissingValue() throws {
        let error = #expect(throws: (any Error).self) {
            _ = try ShrikeServerCommand.parse(["--bogus"])
        }
        let message = ShrikeServerCommand.message(for: try #require(error))
        #expect(message.contains("Unknown option"))
        #expect(!message.contains("requires a value"))
    }

    @Test func maxContextRejectsAnOffSetValueAndNamesEveryAcceptedOne() throws {
        let error = #expect(throws: (any Error).self) {
            _ = try ShrikeServerCommand.parse(["--max-context", "50000"])
        }
        let message = ShrikeServerCommand.message(for: try #require(error))
        for accepted in RuntimeConfiguration.supportedContextTokens {
            #expect(message.contains(String(accepted)))
        }
    }

    @Test func parsingReadsNoEnvironmentAndMergingResolvesIt() throws {
        let parsed = try ShrikeServerCommand.parse(
            ["--model", "m.gturbo"])
        #expect(parsed.reasoningEffort == nil)
        #expect(parsed.thinkingMode == .off)
        #expect(throws: (any Error).self) {
            _ = try parsed.merging(configDefaults: .init(),
                                   environment: ["SHRIKE_REASONING_EFFORT": "sometimes"])
        }
        let resolved = try parsed.merging(
            configDefaults: .init(),
            environment: ["SHRIKE_THINKING_MODE": "true",
                          "SHRIKE_REASONING_EFFORT": "HIGH"])
        #expect(resolved.thinkingMode == .on)
        #expect(resolved.reasoningEffort == .high)
    }

    @Test func parsesKVPrecisionAndYaRNContexts() throws {
        let defaults = try ShrikeServerCommand.parse([
            "--model", "model.gturbo", "--kv-bits", "16",
            "--rope-scaling", "yarn",
        ])
        #expect(defaults.kvCachePrecision == .fp16)
        #expect(defaults.ropeScalingMode == .yarn)
        #expect(defaults.maxContext == 1_048_576)
        let halfMillion = try ShrikeServerCommand.parse([
            "--model", "model.gturbo", "--rope-scaling", "yarn",
            "--max-context", "524288",
        ])
        #expect(halfMillion.maxContext == 524_288)
        #expect(throws: (any Error).self) {
            try ShrikeServerCommand.parse([
                "--model", "model.gturbo", "--max-context", "524288",
            ])
        }
    }

    @Test func theSessionPlanCarriesTheSettledCacheAndPrefillValues() throws {
        let plan = ModelSessionPlan(modelDirectory: URL(fileURLWithPath: "/m.gturbo"),
                                   maxContext: 32_768,
                                   expertCacheSlots: nil)
        #expect(plan.promptCacheMode == .multiPrefix)
        #expect(plan.promptCacheMaximumEntries == 4)
        #expect(plan.promptCacheMemoryLimitBytes == 256 * 1_048_576)
        #expect(plan.promptCacheDiskDirectory == nil)
        #expect(plan.promptCacheDiskLimitBytes == 8_192 * 1_048_576)
        #expect(plan.prefillChunkTokens == nil)
        #expect(plan.reasoningRetention == nil)
    }

    @Test func accepts256KContextAndRejectsUnsupportedValues() throws {
        let arguments = try ShrikeServerCommand.parse([
            "--model", "model.gturbo",
            "--max-context", "262144",
        ])
        #expect(arguments.maxContext == 262_144)
        #expect(throws: (any Error).self) {
            try ShrikeServerCommand.parse([
                "--model", "model.gturbo",
                "--max-context", "100000",
            ])
        }
    }
}
