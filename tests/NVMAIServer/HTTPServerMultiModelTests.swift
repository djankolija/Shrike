import Foundation
import Testing

@testable import NVMAI
@testable import NVMAIServerCore

private final class MultiModelRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _loads: [String] = []

    var loads: [String] { lock.withLock { _loads } }

    func record(_ id: String) { lock.withLock { _loads.append(id) } }
}

private struct EchoBackend: ServerInferenceBackend {
    let maximumContext = 262_144

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1,
                               completionTokens: 1,
                               totalTokens: 2,
                               cachedTokens: 0))
    }
}

@Suite("Multi-model HTTP server", .serialized)
struct HTTPServerMultiModelTests {
    private func plan(_ name: String) -> ModelSessionPlan {
        ModelSessionPlan(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/\(name).gturbo"),
            maxContext: 262_144,
            promptCacheMode: .multiPrefix,
            promptCacheMaximumEntries: 1,
            promptCacheMemoryLimitBytes: 1_048_576,
            promptCacheDiskDirectory: nil,
            promptCacheDiskLimitBytes: 1_048_576,
            prefillChunkTokens: nil,
            expertCacheSlots: nil,
            mtpModelDirectory: nil,
            mtpMemoryMiB: 0)
    }

    private func makeServer(
        _ names: [String],
        overrides: [ServerConfig.ModelOverride] = [],
        recorder: MultiModelRecorder
    ) throws -> (NVMAIHTTPServer, ModelRegistry) {
        let roster = try ModelRoster.resolve(
            candidates: names.map {
                RosterCandidate(bundleName: $0,
                                directory: URL(fileURLWithPath: "/nonexistent/\($0).gturbo"),
                                manifestModelID: "vendor/\($0)",
                                family: .qwen36)
            },
            overrides: overrides)
        let models = roster.entries.map { entry in
            ModelRegistry.Model(
                id: entry.id,
                plan: plan(entry.bundleName),
                facts: ModelSessionFacts(modelID: entry.id,
                                         prefillChunkTokens: 4_096,
                                         promptCacheMode: .multiPrefix))
        }
        let registry = ModelRegistry(
            models: models, roster: roster, idleTimeout: nil,
            loader: { plan, _ in
                recorder.record(plan.modelDirectory.deletingPathExtension().lastPathComponent)
                return EchoBackend()
            })
        return (NVMAIHTTPServer(registry: registry, queueLimit: 4), registry)
    }

    private func started(_ server: NVMAIHTTPServer) async throws -> Int {
        let channel = try await server.start(port: 0)
        return try #require(channel.localAddress?.port)
    }

    private func get(port: Int, path: String) async throws -> (Int, [String: Any]) {
        let url = try #require(URL(string: "http://127.0.0.1:\(port)\(path)"))
        let (data, response) = try await URLSession.shared.data(from: url)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        return (status, object)
    }

    private func post(port: Int, path: String,
                      body: [String: Any]) async throws -> (Int, [String: Any]) {
        let url = try #require(URL(string: "http://127.0.0.1:\(port)\(path)"))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        return (status, object)
    }

    private func chatBody(model: String?) -> [String: Any] {
        var body: [String: Any] = [
            "messages": [["role": "user", "content": "hi"]],
        ]
        if let model { body["model"] = model }
        return body
    }

    private func errorCode(_ object: [String: Any]) -> String? {
        (object["error"] as? [String: Any])?["code"] as? String
    }

    private func errorMessage(_ object: [String: Any]) -> String? {
        (object["error"] as? [String: Any])?["message"] as? String
    }

    @Test func modelsEndpointListsTheRosterWithoutFastAliases() async throws {
        let recorder = MultiModelRecorder()
        let (server, _) = try makeServer(["alpha", "beta"], recorder: recorder)
        let port = try await started(server)

        let (status, object) = try await get(port: port, path: "/v1/models")
        #expect(status == 200)
        let ids = (object["data"] as? [[String: Any]])?.compactMap { $0["id"] as? String }
        #expect(ids == ["alpha", "beta"])
        try await server.shutdown()
    }

    @Test func unknownModelIs404NamingTheValidIDs() async throws {
        let recorder = MultiModelRecorder()
        let (server, _) = try makeServer(["alpha", "beta"], recorder: recorder)
        let port = try await started(server)

        let (status, object) = try await post(
            port: port, path: "/v1/chat/completions", body: chatBody(model: "gamma"))
        #expect(status == 404)
        #expect(errorCode(object) == "model_not_found")
        #expect(errorMessage(object)?.contains("alpha, beta") == true)
        #expect(recorder.loads.isEmpty)
        try await server.shutdown()
    }

    @Test func theFastAliasIsGone() async throws {
        let recorder = MultiModelRecorder()
        let (server, _) = try makeServer(["alpha"], recorder: recorder)
        let port = try await started(server)

        let (status, object) = try await post(
            port: port, path: "/v1/chat/completions", body: chatBody(model: "alpha-fast"))
        #expect(status == 404)
        #expect(errorCode(object) == "model_not_found")
        try await server.shutdown()
    }

    @Test func omittedModelServesTheDefault() async throws {
        let recorder = MultiModelRecorder()
        let (server, _) = try makeServer(
            ["alpha", "beta"],
            overrides: [.init(dir: "alpha", isDefault: true)],
            recorder: recorder)
        let port = try await started(server)

        let (status, object) = try await post(
            port: port, path: "/v1/chat/completions", body: chatBody(model: nil))
        #expect(status == 200)
        #expect(object["model"] as? String == "alpha")
        #expect(recorder.loads == ["alpha"])
        try await server.shutdown()
    }

    @Test func omittedModelWithoutADefaultIsAnError() async throws {
        let recorder = MultiModelRecorder()
        let (server, _) = try makeServer(["alpha", "beta"], recorder: recorder)
        let port = try await started(server)

        let (status, object) = try await post(
            port: port, path: "/v1/chat/completions", body: chatBody(model: nil))
        #expect(status == 404)
        #expect(errorCode(object) == "model_not_found")
        #expect(errorMessage(object)?.contains("alpha, beta") == true)
        try await server.shutdown()
    }

    @Test func aBundleNameAliasResolvesToTheCanonicalID() async throws {
        let recorder = MultiModelRecorder()
        let (server, _) = try makeServer(
            ["kimi-linear-48b-a3b-4bit"],
            overrides: [.init(dir: "kimi-linear-48b-a3b-4bit.gturbo", id: "kimi-linear-48b-a3b")],
            recorder: recorder)
        let port = try await started(server)

        let (status, object) = try await post(
            port: port, path: "/v1/chat/completions",
            body: chatBody(model: "kimi-linear-48b-a3b-4bit"))
        #expect(status == 200)
        #expect(object["model"] as? String == "kimi-linear-48b-a3b")
        try await server.shutdown()
    }

    @Test func requestsForAnotherModelSwapOverHTTP() async throws {
        let recorder = MultiModelRecorder()
        let (server, registry) = try makeServer(["alpha", "beta"], recorder: recorder)
        let port = try await started(server)

        let (firstStatus, first) = try await post(
            port: port, path: "/v1/chat/completions", body: chatBody(model: "alpha"))
        #expect(firstStatus == 200)
        #expect(first["model"] as? String == "alpha")

        let (secondStatus, second) = try await post(
            port: port, path: "/v1/chat/completions", body: chatBody(model: "beta"))
        #expect(secondStatus == 200)
        #expect(second["model"] as? String == "beta")

        #expect(recorder.loads == ["alpha", "beta"])
        #expect(await registry.residentModelID == "beta")
        try await server.shutdown()
    }

    @Test func loadEndpointAcceptsAndLoadsWithoutGenerating() async throws {
        let recorder = MultiModelRecorder()
        let (server, registry) = try makeServer(["alpha", "beta"], recorder: recorder)
        let port = try await started(server)

        let (status, object) = try await post(
            port: port, path: "/v1/models/load", body: ["model": "beta"])
        #expect(status == 200)
        #expect(object["status"] as? String == "accepted")
        #expect(object["model"] as? String == "beta")

        let deadline = ContinuousClock.now + .seconds(5)
        while ContinuousClock.now < deadline {
            if await registry.residentModelID == "beta" { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(await registry.residentModelID == "beta")
        #expect(recorder.loads == ["beta"])
        try await server.shutdown()
    }

    @Test func loadEndpointRejectsAnUnknownModel() async throws {
        let recorder = MultiModelRecorder()
        let (server, _) = try makeServer(["alpha"], recorder: recorder)
        let port = try await started(server)

        let (status, object) = try await post(
            port: port, path: "/v1/models/load", body: ["model": "gamma"])
        #expect(status == 404)
        #expect(errorCode(object) == "model_not_found")
        #expect(recorder.loads.isEmpty)
        try await server.shutdown()
    }

    @Test func healthReportsResidencyLoadingAndCount() async throws {
        let recorder = MultiModelRecorder()
        let (server, _) = try makeServer(["alpha", "beta"], recorder: recorder)
        let port = try await started(server)

        let (idleStatus, idle) = try await get(port: port, path: "/health")
        #expect(idleStatus == 200)
        #expect(idle["status"] as? String == "ok")
        #expect(idle["resident"] is NSNull)
        #expect(idle["loading"] as? Bool == false)
        #expect(idle["models"] as? Int == 2)

        _ = try await post(port: port, path: "/v1/chat/completions",
                           body: chatBody(model: "alpha"))
        let (_, busy) = try await get(port: port, path: "/health")
        #expect(busy["resident"] as? String == "alpha")
        try await server.shutdown()
    }
}
