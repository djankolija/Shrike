import Foundation
import Metal
import Testing
import ShrikeValidationSupport

@testable import Shrike

extension RawCompletionLoopTests {
  final class BoundaryCountingProducer: BoundaryLogitProducer, @unchecked Sendable {
    struct NoPendingToken: Error {}

    let vocabSize: Int
    private let context: MetalContext
    private let step: @Sendable (Int32, Int) -> ScriptedLogitProducer.Step
    private var calls = 0
    private var pendingToken: Int32?
    private var lastAwaited: Int32 = 0
    private(set) var syncCalls = 0
    private(set) var boundaryCalls = 0
    private(set) var continuedCalls = 0
    private(set) var awaited: [Int32] = []

    init(vocabSize: Int, context: MetalContext,
         step: @escaping @Sendable (Int32, Int) -> ScriptedLogitProducer.Step) {
      self.vocabSize = vocabSize
      self.context = context
      self.step = step
    }

    func reset() {
      calls = 0
      pendingToken = nil
      syncCalls = 0
      boundaryCalls = 0
      continuedCalls = 0
      awaited = []
    }

    private func writeLogits(for token: Int32, into logits: MTLBuffer) {
      let spec = step(token, calls)
      calls += 1
      let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
      switch spec {
      case .argmax(let token):
        for i in 0..<vocabSize { ptr[i] = Float16(-30.0) }
        if Int(token) >= 0 && Int(token) < vocabSize { ptr[Int(token)] = Float16(30.0) }
      case .vector(let values):
        for i in 0..<vocabSize { ptr[i] = Float16(i < values.count ? values[i] : -30.0) }
      }
    }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
      syncCalls += 1
      writeLogits(for: token, into: logits)
    }

    func produce(token: Int32?, position: Int, into logits: MTLBuffer,
                 tokenWord: MTLBuffer,
                 sample: (MTLComputeCommandEncoder) throws -> Void) async throws {
      boundaryCalls += 1
      if token == nil { continuedCalls += 1 }
      writeLogits(for: token ?? lastAwaited, into: logits)
      guard let cb = context.queue.makeCommandBuffer(),
            let encoder = cb.makeComputeCommandEncoder() else {
        throw ModelError.residentBufferWrapFailed
      }
      try sample(encoder)
      encoder.endEncoding()
      runToCompletion(cb)
      pendingToken = Int32(bitPattern: tokenWord.contents().load(as: UInt32.self))
    }

    private func runToCompletion(_ cb: MTLCommandBuffer) {
      cb.commit()
      cb.waitUntilCompleted()
    }

    func awaitBoundaryToken() throws -> Int32 {
      guard let token = pendingToken else { throw NoPendingToken() }
      pendingToken = nil
      lastAwaited = token
      awaited.append(token)
      return token
    }
  }

  func runBoundaryLoop(seq: [Int32], end: Int32, prompt: String = "go",
                       config: GenerationConfig)
    async throws -> (Collected, RawDecodeResult, BoundaryCountingProducer, Int) {
    let ctx = try MetalContext()
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let producer = BoundaryCountingProducer(vocabSize: tok.vocabSize, context: ctx,
                                            step: automaton(seq, end: end))
    let promptIds = tok.encode(prompt, addBOS: true)
    let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
    var collected = Collected()
    let result = try await runRawCompletion(producer: producer, tokenizer: tok,
                                            promptIds: promptIds, config: config,
                                            context: ctx, scratch: scratch,
                                            prefillConfig: .off) { progress in
      switch progress {
      case .prefill(let done, let total): collected.prefills.append((done, total))
      case .token(let index, let id, let delta): collected.tokens.append((index, id, delta))
      case .tail(let text): collected.tails.append(text)
      }
    }
    return (collected, result, producer, promptIds.count)
  }

  @Test func boundaryPathStreamsTheSameTokensAsTheSynchronousPath() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let idB = tok.encode("b", addBOS: false).first!
    let idC = tok.encode("c", addBOS: false).first!
    let config = GenerationConfig(maxNewTokens: 50, temperature: 0)
    let (sync, syncResult) = try await runLoop(seq: [idA, idB, idC], end: tok.eosID, config: config)
    let (boundary, result, _, _) = try await runBoundaryLoop(
      seq: [idA, idB, idC], end: tok.eosID, config: config)
    #expect(boundary.tokens.map(\.1) == sync.tokens.map(\.1))
    #expect(boundary.tokens.map(\.2) == sync.tokens.map(\.2))
    #expect(boundary.tokens.map(\.0) == sync.tokens.map(\.0))
    #expect(result.reason == syncResult.reason)
    #expect(result.newTokens == syncResult.newTokens)
    #expect(result.kvPosition == syncResult.kvPosition)
    #expect(result.kvBackedTokenIDs == syncResult.kvBackedTokenIDs)
    #expect(result.uncommittedBoundaryTokenIDs == [tok.eosID])
  }

  @Test func boundaryPathStreamsTheSameTokensAsTheSynchronousPathWhenSampling() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let idB = tok.encode("b", addBOS: false).first!
    let idC = tok.encode("c", addBOS: false).first!
    let config = GenerationConfig(maxNewTokens: 40, temperature: 0.8, topK: 8, seed: 0x5EED_0001)
    let (sync, syncResult) = try await runLoop(seq: [idA, idB, idC], end: tok.eosID, config: config)
    let (boundary, result, _, _) = try await runBoundaryLoop(
      seq: [idA, idB, idC], end: tok.eosID, config: config)
    #expect(boundary.tokens.map(\.1) == sync.tokens.map(\.1))
    #expect(boundary.tokens.map(\.0) == sync.tokens.map(\.0))
    #expect(result.reason == syncResult.reason)
    #expect(result.newTokens == syncResult.newTokens)
  }

  @Test func boundaryPathContinuesEveryPassAfterTheFirst() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let idB = tok.encode("b", addBOS: false).first!
    let idC = tok.encode("c", addBOS: false).first!
    let (_, result, producer, promptCount) = try await runBoundaryLoop(
      seq: [idA, idB, idC], end: tok.eosID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0))
    #expect(result.newTokens == 4)
    #expect(producer.syncCalls == promptCount)
    #expect(producer.boundaryCalls == 3)
    #expect(producer.continuedCalls == 2)
    #expect(producer.awaited == [idB, idC, tok.eosID])
  }

  @Test func boundaryPathStopsOnTheStopTokenWithoutAnotherPass() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let (collected, result, producer, _) = try await runBoundaryLoop(
      seq: [idA], end: tok.endOfTurnID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0))
    #expect(result.reason == .endOfTurn)
    #expect(result.newTokens == 2)
    #expect(collected.tokens.map(\.1) == [idA])
    #expect(producer.boundaryCalls == 1)
    #expect(result.kvBackedTokenIDs.last == idA)
  }

  @Test func boundaryPathHonoursMaxTokens() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let (collected, result, producer, _) = try await runBoundaryLoop(
      seq: [idA, idA], end: idA,
      config: GenerationConfig(maxNewTokens: 5, temperature: 0))
    #expect(result.reason == .maxTokens)
    #expect(result.newTokens == 5)
    #expect(collected.tokens.map(\.0) == [0, 1, 2, 3, 4])
    #expect(producer.boundaryCalls == 4)
  }

  @Test func boundaryPathStopsOnAStopString() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let textA = tok.decode([idA], skipSpecialTokens: true)
    let (_, result, producer, _) = try await runBoundaryLoop(
      seq: [idA, idA], end: idA,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0, stopStrings: [textA]))
    #expect(result.reason == .stopString)
    #expect(producer.boundaryCalls == 0)
  }

  @Test func boundaryPathFallsBackWhenARepetitionPenaltyIsSet() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let idB = tok.encode("b", addBOS: false).first!
    let (collected, result, producer, promptCount) = try await runBoundaryLoop(
      seq: [idA, idB], end: tok.eosID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0, repetitionPenalty: 1.2))
    #expect(result.reason == .eos)
    #expect(collected.tokens.map(\.1) == [idA, idB])
    #expect(producer.boundaryCalls == 0)
    #expect(producer.syncCalls == promptCount + 2)
  }
}
