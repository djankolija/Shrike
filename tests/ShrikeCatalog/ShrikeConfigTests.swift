import Foundation
import Testing

import ShrikeCatalog

@Suite struct ShrikeConfigTests {
    @Test func decodesTheDocumentedExample() throws {
        let json = """
        {
          "models_dir": "~/shrike-runtime/models",
          "defaults": { "max_context": 32768, "ram_budget": "6G", "idle_unload_seconds": 0 },
          "models": [
            { "dir": "kimi-linear-48b-a3b-4bit.gturbo", "id": "kimi-linear-48b-a3b" },
            { "dir": "gpt-oss-20b-mlx-4bit.gturbo", "id": "gpt-oss-20b", "default": true }
          ]
        }
        """
        let config = try ShrikeConfig.parse(Data(json.utf8))
        #expect(config.modelsDir == "~/shrike-runtime/models")
        #expect(config.defaults.maxContext == 32_768)
        #expect(config.defaults.ramBudget == "6G")
        #expect(config.defaults.idleUnloadSeconds == 0)
        #expect(config.models.count == 2)
        #expect(config.models[0].id == "kimi-linear-48b-a3b")
        #expect(!config.models[0].isDefault)
        #expect(config.models[1].isDefault)
    }

    @Test func absentSectionsDecodeToEmpty() throws {
        let config = try ShrikeConfig.parse(Data("{}".utf8))
        #expect(config.modelsDir == nil)
        #expect(config.defaults == ShrikeConfig.Defaults())
        #expect(config.models.isEmpty)
    }

    @Test func duplicateDirFails() throws {
        let json = """
        {"models": [{"dir": "a.gturbo"}, {"dir": "a.gturbo", "id": "b"}]}
        """
        #expect(throws: ShrikeConfigError.self) { try ShrikeConfig.parse(Data(json.utf8)) }
    }

    @Test func twoDefaultsFail() throws {
        let json = """
        {"models": [{"dir": "a.gturbo", "default": true}, {"dir": "b.gturbo", "default": true}]}
        """
        #expect(throws: ShrikeConfigError.self) { try ShrikeConfig.parse(Data(json.utf8)) }
    }

    @Test func unparseableRamBudgetFails() throws {
        let json = """
        {"defaults": {"ram_budget": "six gigs"}}
        """
        #expect(throws: ShrikeConfigError.self) { try ShrikeConfig.parse(Data(json.utf8)) }
    }

    @Test func negativeIdleFails() throws {
        let json = """
        {"defaults": {"idle_unload_seconds": -1}}
        """
        #expect(throws: ShrikeConfigError.self) { try ShrikeConfig.parse(Data(json.utf8)) }
    }

    @Test func emptyIDFails() throws {
        let json = """
        {"models": [{"dir": "a.gturbo", "id": ""}]}
        """
        #expect(throws: ShrikeConfigError.self) { try ShrikeConfig.parse(Data(json.utf8)) }
    }

    @Test func malformedJSONFails() throws {
        #expect(throws: ShrikeConfigError.self) {
            try ShrikeConfig.parse(Data("not json".utf8))
        }
    }

    @Test func loadReadsAFile() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-config-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: path) }
        try Data(#"{"models_dir": "/models"}"#.utf8).write(to: path)
        let config = try ShrikeConfig.load(path: path.path)
        #expect(config.modelsDir == "/models")
    }

    @Test func loadingAMissingFileFails() throws {
        do {
            _ = try ShrikeConfig.load(path: "/nonexistent/server-\(UUID().uuidString).json")
            Issue.record("expected an unreadable failure")
        } catch let error as ShrikeConfigError {
            guard case .unreadable = error else {
                Issue.record("expected .unreadable, got \(error)")
                return
            }
        }
    }
}
