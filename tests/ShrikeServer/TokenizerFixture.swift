import Foundation
import Testing

/// Shared ChatML tokenizer fixture for the ShrikeServer test target. The same
/// synthetic byte-level BPE fixture the Shrike/Core tokenizer tests use,
/// bundled here so server tests exercise the real ChatML pipeline offline.
enum TokenizerFixture {
    static func folder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "ChatMLTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    static func harmonyFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "HarmonyTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    static func kimiFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "KimiTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }
}
