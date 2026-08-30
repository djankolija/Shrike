# v7 Reasoning Effort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two Harmony failure-diagnostics knobs (Phase A: tasks 1–2), then a
reasoning-effort setting for Harmony models with server-default and per-request
control (Phase B: tasks 3–6).

**Architecture:** Effort threads through render calls as a parameter — the
tokenizer stays immutable. Diagnostics mirror the existing
`SHRIKE_CACHE_DIAG` pattern. Validation is strict: unsupported knobs 400, never
silently no-op.

**Tech Stack:** Swift 6.3 SPM package, swift-testing (`@Test`, `#expect`,
`#require`) — no XCTest anywhere.

**Spec:** [v7-reasoning-effort.md](v7-reasoning-effort.md)

## Global Constraints

- All five gates from [CLAUDE.md](../CLAUDE.md) must pass before the work is
  done: release build with zero warnings; `swiftlint lint --strict --baseline
  .swiftlint-baseline.json`; markdown link check; `swift test --no-parallel`;
  the same suite under ThreadSanitizer.
- No model runs are required by any task. Do not download checkpoints,
  duplicate `.gturbo`s, or run `tools/golden-baseline.sh`.
- No new functions over 120 lines (swiftlint `function_body_length` warns at
  120). `OpenAIRequestValidator.validate` is already baselined; when editing
  it, extract new logic into helpers rather than growing it.
- Comments: default to none. Only a non-obvious WHY the code cannot show.
- **Pause after Task 2 (end of Phase A):** run the test gate, summarize, and
  wait for explicit go-ahead before starting Task 3.

---

### Task 1: Carry the unknown tool name into the structured-output failure log

**Files:**
- Modify: `sources/ShrikeServer/Core/ServerInference.swift` (`StructuredOutputFailure` at :254-263, the `structuredFailure` closure at :1126-1145, call sites :1147, :1154, :1159)
- Test: `tests/ShrikeServer/StructuredOutputDiagnosticsTests.swift` (existing test `parserCausesAreFixedAndUnknownToolNameIsDiscarded()` at :49)

**Interfaces:**
- Consumes: `ToolCallParserError.unknownTool(String)` (thrown by `HarmonyToolCallParser.parse`, sources/Shrike/Tokenization/HarmonyToolCallParser.swift:41).
- Produces: `StructuredOutputFailure` gains `let unknownToolName: String?`; its `debugDescription` includes `unknown_tool_name=<name>` when non-nil. `StructuredOutputFailureCause` gains `static func unknownToolName(_ error: Error) -> String?`.

- [ ] **Step 1: Read the existing test and rewrite it as a failing test**

Read `tests/ShrikeServer/StructuredOutputDiagnosticsTests.swift` in full (it is
~110 lines). Rewrite `parserCausesAreFixedAndUnknownToolNameIsDiscarded()` —
keeping its existing arrange code for building a `StructuredOutputFailure` from
`ToolCallParserError.unknownTool("missing_tool")`, whatever helper pattern the
file already uses — renamed and re-asserted as:

```swift
@Test func unknownToolNameSurvivesIntoTheFailureLog() throws {
    // arrange: same construction the old test used, with
    // ToolCallParserError.unknownTool("missing_tool") as the error,
    // now also passing
    //   unknownToolName: StructuredOutputFailureCause.unknownToolName(error)
    #expect(StructuredOutputFailureCause.classify(error) == .unknownTool)
    #expect(StructuredOutputFailureCause.unknownToolName(error) == "missing_tool")
    #expect(failure.debugDescription.contains("cause=unknown_tool"))
    #expect(failure.debugDescription.contains("unknown_tool_name=missing_tool"))
}
```

Also add the negative case (a `malformed` error must not emit the field):

```swift
@Test func nonUnknownToolFailuresOmitTheNameField() throws {
    // arrange with ToolCallParserError.malformed and unknownToolName: nil
    #expect(StructuredOutputFailureCause.unknownToolName(
        ToolCallParserError.malformed) == nil)
    #expect(!failure.debugDescription.contains("unknown_tool_name="))
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `swift test --no-parallel --filter StructuredOutputDiagnosticsTests`
Expected: compile failure — `unknownToolName` and the init parameter do not exist yet.

- [ ] **Step 3: Implement**

In `sources/ShrikeServer/Core/ServerInference.swift`:

Add to `StructuredOutputFailureCause` (after `classify`, ~line 59):

```swift
static func unknownToolName(_ error: Error) -> String? {
    if case ToolCallParserError.unknownTool(let name) = error { return name }
    return nil
}
```

Change `StructuredOutputFailure` (line 254) to:

```swift
struct StructuredOutputFailure: Error, CustomDebugStringConvertible, Sendable {
    let kind: StructuredOutputFailureKind
    let cause: StructuredOutputFailureCause
    let unknownToolName: String?
    let diagnostics: StructuredOutputFailureDiagnostics

    var debugDescription: String {
        let name = unknownToolName.map { " unknown_tool_name=\($0)" } ?? ""
        return "structured_output_failure kind=\(kind.rawValue) "
            + "cause=\(cause.rawValue)\(name) \(diagnostics.logDescription)"
    }
}
```

In the `structuredFailure` closure (line 1126), add a
`unknownToolName: String? = nil` parameter, pass it through to the init, and
update the two error-bearing call sites:

```swift
throw structuredFailure(kind: .decoderConsume,
                        cause: .classify(decodingError),
                        unknownToolName: StructuredOutputFailureCause
                            .unknownToolName(decodingError))
```

(same shape at the `.decoderFinish` site with `error`; the
`.orphanToolResponse` site at :1159 stays as-is, taking the nil default).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `swift test --no-parallel --filter StructuredOutputDiagnosticsTests`
Expected: PASS, including the untouched suite members.

- [ ] **Step 5: Commit**

```bash
git add sources/ShrikeServer/Core/ServerInference.swift tests/ShrikeServer/StructuredOutputDiagnosticsTests.swift
git commit -m "v7 1/6: carry the unknown tool name into the failure log"
```

---

### Task 2: `SHRIKE_GEN_DIAG` generated-token dump

**Files:**
- Modify: `sources/ShrikeServer/Core/ServerInference.swift` (new `ShrikeGenDiag` enum near the diagnostics section ~line 33; emission inside `generate` after the harmony boundary replay, i.e. after :1125)
- Test: `tests/ShrikeServer/StructuredOutputDiagnosticsTests.swift`

**Interfaces:**
- Consumes: `RawDecodeResult.prefillTokens: Int`, `.kvBackedTokenIDs: [Int32]`, `.uncommittedBoundaryTokenIDs: [Int32]` (already used by `StructuredOutputFailureDiagnostics.init` at :109-114).
- Produces: `ShrikeGenDiag.enabled: Bool` and `ShrikeGenDiag.line(prefillTokens:kvBackedTokenIDs:boundaryTokenIDs:) -> String`.

- [ ] **Step 1: Write the failing test**

Append to `StructuredOutputDiagnosticsTests.swift`:

```swift
@Test func genDiagLineListsGeneratedIDsAfterThePrefill() {
    let line = ShrikeGenDiag.line(
        prefillTokens: 2,
        kvBackedTokenIDs: [10, 11, 200005, 42],
        boundaryTokenIDs: [200008])
    #expect(line == "Shrike gen_diag prefill=2 generated=3 ids=[200005, 42, 200008]")
}

@Test func genDiagLineClampsAnInvalidPrefillCount() {
    let line = ShrikeGenDiag.line(
        prefillTokens: 9,
        kvBackedTokenIDs: [10, 11],
        boundaryTokenIDs: [])
    #expect(line == "Shrike gen_diag prefill=2 generated=0 ids=[]")
}
```

- [ ] **Step 2: Run to verify failure**

Run: `swift test --no-parallel --filter StructuredOutputDiagnosticsTests`
Expected: compile failure — `ShrikeGenDiag` does not exist.

- [ ] **Step 3: Implement**

In `ServerInference.swift`, after the `// MARK: - Structured Output
Diagnostics` section opener:

```swift
/// Opt-in generated-token dump: set SHRIKE_GEN_DIAG=1 to log every
/// completion's generated token IDs to stderr, so channel-marker questions
/// (which 2000xx token preceded a text region) are answerable post-hoc.
enum ShrikeGenDiag {
    static let enabled =
        ProcessInfo.processInfo.environment["SHRIKE_GEN_DIAG"] != nil

    static func line(prefillTokens: Int,
                     kvBackedTokenIDs: [Int32],
                     boundaryTokenIDs: [Int32]) -> String {
        let prefill = min(max(prefillTokens, 0), kvBackedTokenIDs.count)
        let generated = Array(kvBackedTokenIDs.dropFirst(prefill))
            + boundaryTokenIDs
        return "Shrike gen_diag prefill=\(prefill) "
            + "generated=\(generated.count) ids=\(generated)"
    }
}
```

In `generate`, immediately after the harmony boundary-replay block ends
(:1125) and **before** the `structuredFailure` closure / failure throws, so
both success and failure paths emit:

```swift
if ShrikeGenDiag.enabled {
    cacheDiag(ShrikeGenDiag.line(
        prefillTokens: result.prefillTokens,
        kvBackedTokenIDs: result.kvBackedTokenIDs,
        boundaryTokenIDs: result.uncommittedBoundaryTokenIDs))
}
```

(`cacheDiag(_:)` at :832 is a bare stderr write, not env-gated — the
`ShrikeGenDiag.enabled` guard is what gates this.)

- [ ] **Step 4: Run to verify pass**

Run: `swift test --no-parallel --filter StructuredOutputDiagnosticsTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sources/ShrikeServer/Core/ServerInference.swift tests/ShrikeServer/StructuredOutputDiagnosticsTests.swift
git commit -m "v7 2/6: SHRIKE_GEN_DIAG generated-token dump"
```

**PHASE A CHECKPOINT: run `swift test --no-parallel`, report, and wait for
explicit go-ahead before Task 3.**

---

### Task 3: `ReasoningEffort` enum and the parameterized Harmony render

**Files:**
- Modify: `sources/Shrike/Tokenization/Tokenizer.swift` (`ModelThinkingMode` neighborhood :37-56; `applyChatTemplate` :693; `harmonyChatTemplate` :735; `harmonySystemBlock` :770; `encodeToolChat` :1110)
- Test: `tests/Shrike/Core/Tokenization/HarmonyTemplateTests.swift` (`systemBlock` helper at :37, `render` helper at :31)

**Interfaces:**
- Produces (later tasks depend on these exact names):
  - `public enum ReasoningEffort: String, Codable, CaseIterable, Sendable { case low, medium, high }` with `public static func resolved(environment: [String: String] = ProcessInfo.processInfo.environment) -> ReasoningEffort?`
  - `public func applyChatTemplate(_ messages: [Message], reasoningEffort: ReasoningEffort = .medium) throws -> String`
  - `public func encodeToolChat(messages: [Message], tools: [FunctionDefinition], reasoningEffort: ReasoningEffort = .medium) throws -> [Int32]`
  - `func harmonyChatTemplate(_ messages: [Message], tools: [FunctionDefinition], reasoningEffort: ReasoningEffort = .medium, currentDate: String? = nil, addGenerationPrompt: Bool = true) throws -> String`

- [ ] **Step 1: Write the failing tests**

In `HarmonyTemplateTests.swift`, extend the `systemBlock` helper (:37) with an
effort parameter, replacing the hardcoded literal:

```swift
private func systemBlock(withTools: Bool = false,
                         effort: String = "medium") -> String {
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        + "Knowledge cutoff: 2024-06\n"
        + "Current date: 2026-08-26\n\n"
        + "Reasoning: \(effort)\n\n"
        + ...  // rest unchanged
}
```

Add a test alongside the existing render tests:

```swift
@Test("Reasoning effort parameterizes the system block")
func reasoningEffortParameterizesTheSystemBlock() throws {
    let messages = [Message(role: .user, content: "Hi")]
    for effort in ReasoningEffort.allCases {
        let rendered = try tok.harmonyChatTemplate(
            messages, tools: [],
            reasoningEffort: effort,
            currentDate: Self.goldenDate)
        #expect(rendered.contains("Reasoning: \(effort.rawValue)\n\n"))
        #expect(rendered.hasPrefix(systemBlock(effort: effort.rawValue)))
    }
}
```

(Adapt the `Message` construction to the file's existing style — its private
`Message` typealias at :28.) The existing tests keep passing unchanged because
the default stays `.medium`.

- [ ] **Step 2: Run to verify failure**

Run: `swift test --no-parallel --filter HarmonyTemplateTests`
Expected: compile failure — `ReasoningEffort` and the parameter do not exist.

- [ ] **Step 3: Implement**

In `Tokenizer.swift`, directly below `ModelThinkingMode` (:56):

```swift
/// Harmony's trained deliberation knob: the literal word after "Reasoning:"
/// in the system block. Meaningless for other dialects, which ignore it.
public enum ReasoningEffort: String, Codable, CaseIterable, Sendable {
    case low
    case medium
    case high

    public static func resolved(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ReasoningEffort? {
        environment["SHRIKE_REASONING_EFFORT"].flatMap {
            ReasoningEffort(rawValue: $0.lowercased())
        }
    }
}
```

Thread the parameter:
- `harmonySystemBlock(hasTools:currentDate:reasoningEffort:)` — replace
  `s += "Reasoning: medium\n\n"` (:776) with
  `s += "Reasoning: " + reasoningEffort.rawValue + "\n\n"`.
- `harmonyChatTemplate` (:735) gains `reasoningEffort: ReasoningEffort =
  .medium` after `tools:` and forwards it to `Self.harmonySystemBlock`.
- `applyChatTemplate` (:693) gains `reasoningEffort: ReasoningEffort =
  .medium`; only the `.harmony` case forwards it.
- `encodeToolChat` (:1110) gains `reasoningEffort: ReasoningEffort = .medium`;
  only the `dialect == .harmony` branch forwards it.

Defaults keep every existing call site (CLI, app, decode service, tests)
compiling unchanged.

- [ ] **Step 4: Run the full tokenization suites**

Run: `swift test --no-parallel --filter Tokenization`
Expected: PASS — including every golden Harmony test, byte-identical at the
default.

- [ ] **Step 5: Commit**

```bash
git add sources/Shrike/Tokenization/Tokenizer.swift tests/Shrike/Core/Tokenization/HarmonyTemplateTests.swift
git commit -m "v7 3/6: ReasoningEffort parameterizes the harmony system block"
```

---

### Task 4: Server default — flag, env, and the `--thinking off` mapping

**Files:**
- Modify: `sources/ShrikeServer/Core/ServerArguments.swift` (fields :4-44, parse loop :158-301, usage string :55-123)
- Modify: `sources/ShrikeServer/Core/ModelRegistry.swift` (:391 vicinity)
- Modify: `sources/ShrikeServer/Core/ModelSessionPlan.swift` (fields :35-50, `makeSession` :84-104)
- Modify: `sources/ShrikeServer/Core/ServerInference.swift` (`ServerModelSession.load` :535-561, `encodePrompt` :1711-1721, `generate`)
- Modify: `sources/ShrikeServer/Command/main.swift` (:83 launch line)
- Test: `tests/ShrikeServer/OpenAIValidationTests.swift` (`ServerArgumentTests` struct at :274), `tests/ShrikeServer/StructuredOutputDiagnosticsTests.swift` or a new `ReasoningEffortDefaultTests.swift`

**Interfaces:**
- Consumes: `ReasoningEffort` from Task 3.
- Produces:
  - `ServerArguments.reasoningEffort: ReasoningEffort?` (nil = unset), parsed from `--reasoning-effort low|medium|high`, env fallback `SHRIKE_REASONING_EFFORT`.
  - `ModelSessionPlan.reasoningEffort: ReasoningEffort?`.
  - `ServerModelSession.defaultReasoningEffort(explicit:dialect:thinkingMode:) -> (effort: ReasoningEffort, warning: String?)` — static, pure, and the session stores its result as `let defaultReasoningEffort: ReasoningEffort`.

- [ ] **Step 1: Write the failing tests**

In `ServerArgumentTests` (mirror the style of
`parsesOnlyBinaryThinkingModes()` at :360):

```swift
@Test func parsesReasoningEffortLevels() throws {
    let parsed = try ServerArguments.parse(
        ["--model", "m", "--reasoning-effort", "low"], environment: [:])
    #expect(parsed.reasoningEffort == .low)
    #expect(try ServerArguments.parse(["--model", "m"], environment: [:])
        .reasoningEffort == nil)
    #expect(try ServerArguments.parse(
        ["--model", "m"],
        environment: ["SHRIKE_REASONING_EFFORT": "HIGH"]).reasoningEffort == .high)
    #expect(throws: ServerArgumentError.self) {
        try ServerArguments.parse(
            ["--model", "m", "--reasoning-effort", "max"], environment: [:])
    }
}
```

For the mapping (new test struct in `StructuredOutputDiagnosticsTests.swift`'s
file or its own file):

```swift
struct ReasoningEffortDefaultTests {
    @Test func explicitEffortAlwaysWins() {
        let resolved = ServerModelSession.defaultReasoningEffort(
            explicit: .high, dialect: .harmony, thinkingMode: .off)
        #expect(resolved.effort == .high)
        #expect(resolved.warning == nil)
    }

    @Test func thinkingOffOnHarmonyWarnsAndLowers() {
        let resolved = ServerModelSession.defaultReasoningEffort(
            explicit: nil, dialect: .harmony, thinkingMode: .off)
        #expect(resolved.effort == .low)
        #expect(resolved.warning != nil)
    }

    @Test func otherDialectsAndModesStayMedium() {
        #expect(ServerModelSession.defaultReasoningEffort(
            explicit: nil, dialect: .chatml, thinkingMode: .off).effort == .medium)
        #expect(ServerModelSession.defaultReasoningEffort(
            explicit: nil, dialect: .harmony, thinkingMode: .adaptive).effort == .medium)
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `swift test --no-parallel --filter "ServerArgumentTests|ReasoningEffortDefaultTests"`
Expected: compile failure — the field, flag, and function do not exist.

- [ ] **Step 3: Implement**

`ServerArguments.swift`:
- Field `public let reasoningEffort: ReasoningEffort?` (plus memberwise-init
  threading matching the struct's existing pattern).
- In `parse`, seed `var reasoningEffort = ReasoningEffort.resolved(environment:
  environment)` next to the `thinkingMode` seeding (:148), and add the flag
  case following the `--thinking` pattern (:263):

```swift
case "--reasoning-effort":
    guard let parsed = ReasoningEffort(rawValue: value) else {
        throw ServerArgumentError.invalid(
            "--reasoning-effort must be low, medium or high")
    }
    reasoningEffort = parsed
```

- Usage string (:55-123), next to the `--thinking` bullet:
  `--reasoning-effort   Harmony deliberation level: low, medium or high (default medium; --thinking off on a Harmony model implies low)`.

`ModelRegistry.swift` :391 vicinity — pass
`reasoningEffort: arguments.reasoningEffort` into `ModelSessionPlan`.

`ModelSessionPlan.swift` — add `public let reasoningEffort: ReasoningEffort?`
and forward it in `makeSession` into `ServerModelSession.load`.

`ServerInference.swift`:
- `ServerModelSession.load` gains `reasoningEffort: ReasoningEffort? = nil`;
  after the tokenizer loads, resolve and warn:

```swift
let resolved = Self.defaultReasoningEffort(
    explicit: reasoningEffort,
    dialect: tokenizer.dialect,
    thinkingMode: thinkingMode)
if let warning = resolved.warning {
    FileHandle.standardError.write(Data((warning + "\n").utf8))
}
```

  storing `resolved.effort` on the session as `let defaultReasoningEffort`.
- The pure mapping:

```swift
static func defaultReasoningEffort(
    explicit: ReasoningEffort?,
    dialect: ChatDialect,
    thinkingMode: ModelThinkingMode
) -> (effort: ReasoningEffort, warning: String?) {
    if let explicit { return (explicit, nil) }
    guard dialect == .harmony, thinkingMode == .off else {
        return (.medium, nil)
    }
    return (.low, "Shrike reasoning_effort: harmony cannot disable thinking; "
        + "--thinking off maps to reasoning effort low")
}
```

- `encodePrompt` (:1711) gains `reasoningEffort: ReasoningEffort` and forwards
  it to both `encodeToolChat(messages:tools:reasoningEffort:)` and
  `applyChatTemplate(_:reasoningEffort:)`; `preparePrompt`/`generate` pass
  `defaultReasoningEffort` for now (per-request override lands in Task 5).

`main.swift` :83 — append to the ready line:
`reasoning_effort=\(effective.reasoningEffort?.rawValue ?? "auto")`.

- [ ] **Step 4: Run to verify pass**

Run: `swift test --no-parallel --filter "ServerArgumentTests|ReasoningEffortDefaultTests"`
Expected: PASS. Then `swift build` to confirm all `ModelSessionPlan`/`load`
call sites (including tests) still compile.

- [ ] **Step 5: Commit**

```bash
git add sources/ShrikeServer sources/Shrike tests/ShrikeServer
git commit -m "v7 4/6: --reasoning-effort server default and the --thinking off mapping"
```

---

### Task 5: Per-request `reasoning_effort` — parse, validate, thread, reject on wrong dialect

**Files:**
- Modify: `sources/ShrikeServer/Core/OpenAIModels.swift` (`OpenAIChatRequest` :153-189, `ValidatedChatRequest` :276-318, `validate` :324-453)
- Modify: `sources/ShrikeServer/Core/ServerInference.swift` (`generate` :962-971)
- Test: `tests/ShrikeServer/OpenAIValidationTests.swift`

**Interfaces:**
- Consumes: `ReasoningEffort` (Task 3), `ValidatedChatRequest` (existing), `ServerModelSession.defaultReasoningEffort` stored property (Task 4).
- Produces: `OpenAIChatRequest.reasoningEffort: String?` (wire key `reasoning_effort`); `ValidatedChatRequest.reasoningEffort: ReasoningEffort?` (init parameter defaulted to nil, preserved by `replacingMessages`).

- [ ] **Step 1: Write the failing tests**

In `OpenAIValidationTests` (matching the raw-JSON style at the top of that
suite):

```swift
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
```

- [ ] **Step 2: Run to verify failure**

Run: `swift test --no-parallel --filter OpenAIValidationTests`
Expected: compile failure (`reasoningEffort` member missing), then after
decode-only stubs an assertion failure.

- [ ] **Step 3: Implement parsing and validation**

`OpenAIChatRequest`: add `public let reasoningEffort: String?` and CodingKeys
entry `case reasoningEffort = "reasoning_effort"`.

`ValidatedChatRequest`: add `public let reasoningEffort: ReasoningEffort?`,
init parameter `reasoningEffort: ReasoningEffort? = nil`, and carry it through
`replacingMessages` (read its body at :300-318 and preserve the field the same
way the other fields are preserved).

In `validate`, as a small private helper to avoid growing the baselined
function body:

```swift
private static func validatedReasoningEffort(
    _ raw: String?
) throws -> ReasoningEffort? {
    guard let raw else { return nil }
    guard let effort = ReasoningEffort(rawValue: raw) else {
        throw invalid("reasoning_effort must be low, medium or high",
                      "reasoning_effort", "unsupported_value")
    }
    return effort
}
```

called from `validate` and passed into the `ValidatedChatRequest` it returns.

- [ ] **Step 4: Thread the override and the dialect guard**

In `ServerModelSession.generate` (:962), before `preparePrompt` runs (:971):

```swift
if request.reasoningEffort != nil, tokenizer.dialect != .harmony {
    throw ServerRequestError.invalid(
        message: "reasoning_effort is not supported by this model",
        param: "reasoning_effort",
        code: "unsupported_parameter")
}
```

and resolve the effective value used by `encodePrompt`:

```swift
let effectiveReasoningEffort = request.reasoningEffort ?? defaultReasoningEffort
```

Then verify the HTTP mapping: run
`rg -n "ServerRequestError" sources/ShrikeServer --type swift -l` and read the
route handler's catch to confirm a `ServerRequestError` thrown from `generate`
returns HTTP 400 like one thrown from `validate` (they share the error type;
if the handler only catches it around validation, widen that catch to the
generate call in the same style).

- [ ] **Step 5: Run to verify pass**

Run: `swift test --no-parallel --filter OpenAIValidationTests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add sources/ShrikeServer/Core/OpenAIModels.swift sources/ShrikeServer/Core/ServerInference.swift tests/ShrikeServer/OpenAIValidationTests.swift
git commit -m "v7 5/6: per-request reasoning_effort with strict validation"
```

---

### Task 6: Documentation and the five gates

**Files:**
- Modify: `README.md` (:90-91, the thinking-mode bullet)
- Modify: `docs/v7-reasoning-effort.md` (only if implementation diverged from the spec)

- [ ] **Step 1: Update README**

Replace the bullet at README.md:90-91 with:

```markdown
- **Thinking mode.** Off/on/adaptive for the Qwen-family templates. gpt-oss
  cannot disable thinking; its knob is `--reasoning-effort low|medium|high`
  (per-request via the OpenAI `reasoning_effort` field), and `--thinking off`
  on a Harmony model warns and maps to effort `low`.
```

- [ ] **Step 2: Run the five gates**

Delegate each run to a haiku subagent (return pass/fail plus ALL errors,
warnings and failing test names verbatim — nothing else):

1. `swift build -c release` — zero warnings.
2. `swiftlint lint --strict --baseline .swiftlint-baseline.json` — if it fails
   on a pre-existing baselined function whose lines shifted (e.g.
   `OpenAIRequestValidator.validate`), regenerate with
   `swiftlint lint --write-baseline .swiftlint-baseline.json` and re-run;
   never baseline a genuinely new violation.
3. The repo's markdown link check over all `*.md` (both v7 docs are new —
   their relative links must resolve).
4. `swift test --no-parallel`.
5. The same suite under ThreadSanitizer.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/v7-reasoning-effort.md docs/v7-implementation-plan.md
git commit -m "v7 6/6: document reasoning effort; retire the thinking-off no-op"
```

---

## Post-merge, outside this plan

- pi wiring on the client machine (`~/.pi/agent/models.json`): `"reasoning":
  true` + `"thinkingLevelMap": {"minimal": "low", "xhigh": "high"}` on
  `shrike/gpt-oss-20b`, then one live prompt per effort level to confirm the
  field arrives and the rendered block reflects it.
- The next mini deploy independently carries the rename checklist from
  [CLAUDE.md](../CLAUDE.md).
- Re-run the two gpt-oss repros with `SHRIKE_GEN_DIAG=1` to collect the Bug 1
  tool name and the Bug 3 channel-marker verdict.
