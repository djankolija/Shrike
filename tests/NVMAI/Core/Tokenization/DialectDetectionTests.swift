import Foundation
import Testing
@testable import NVMAI

/// Dialect detection order over the tokenizer's framing tokens: harmony
/// (`<|channel|>`) before kimi (`<|im_middle|>`) before chatml
/// (`<|im_start|>`), with no ChatML fallback for unrecognized vocabularies.
/// Each case patches the ChatML fixture's `tokenizer.json` in a temp copy.
@Suite("Dialect detection")
struct DialectDetectionTests {

    private func patchedFixture(rename: [String: String],
                                add: [String] = []) throws -> URL {
        let source = try ChatMLTemplateTests.fixtureFolder()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dialect-fixture-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: source, to: dir)

        let tokenizerJSON = dir.appendingPathComponent("tokenizer.json")
        var root = try JSONSerialization.jsonObject(
            with: Data(contentsOf: tokenizerJSON)) as! [String: Any]  // lint:allow-force fixture JSON
        var added = root["added_tokens"] as! [[String: Any]]  // lint:allow-force fixture JSON
        var maxID = added.compactMap { $0["id"] as? Int }.max() ?? 0
        for i in added.indices {
            if let content = added[i]["content"] as? String,
               let replacement = rename[content] {
                added[i]["content"] = replacement
            }
        }
        for token in add {
            maxID += 1
            added.append(["id": maxID, "content": token,
                          "special": true, "single_word": false, "lstrip": false,
                          "rstrip": false, "normalized": false])
        }
        root["added_tokens"] = added
        try JSONSerialization.data(withJSONObject: root)
            .write(to: tokenizerJSON)
        return dir
    }

    private func loadError(from dir: URL) async -> String? {
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try await GFTokenizer.load(from: dir)
            return nil
        } catch {
            return "\(error)"
        }
    }

    @Test func harmonyMarkRoutesToHarmonyDialect() async throws {
        // The patched ChatML fixture carries only the channel mark, so a load
        // that demands the remaining Harmony tokens proves the routing.
        let dir = try patchedFixture(rename: ["<|im_start|>": "<|channel|>"])
        let error = await loadError(from: dir)
        #expect(error?.contains("<|startoftext|>") == true)
    }

    @Test func harmonyFixtureLoadsAsHarmony() async throws {
        let tok = try await GFTokenizer.load(
            from: HarmonyTemplateTests.fixtureFolder())
        #expect(tok.dialect == .harmony)
    }

    @Test func kimiFixtureLoadsAsKimi() async throws {
        let dir = try #require(Bundle.module.url(
            forResource: "KimiTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
        let tok = try await GFTokenizer.load(from: dir)
        #expect(tok.dialect == .kimi)
    }

    @Test func kimiMarkWinsOverChatML() async throws {
        // The patched ChatML fixture carries only the middle mark, so a load
        // that demands the remaining Kimi tokens proves the routing.
        let dir = try patchedFixture(rename: [:], add: ["<|im_middle|>"])
        let error = await loadError(from: dir)
        #expect(error?.contains("[BOS]") == true)
    }

    @Test func unrecognizedFramingIsRejectedWithDialectError() async throws {
        let dir = try patchedFixture(rename: ["<|im_start|>": "<|zzz_unknown|>"])
        let error = await loadError(from: dir)
        #expect(error?.contains("no recognized chat framing tokens") == true)
    }
}
