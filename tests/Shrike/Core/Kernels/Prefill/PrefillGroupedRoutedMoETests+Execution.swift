import Metal
import Testing
import ShrikeValidationSupport

@testable import Shrike

/// Mirrors `MPPPrefillInt4QMMTests.swift`'s `mppTensorOpsAvailable`: a
/// missing MPP path is a recorded skip here, never a vacuous pass.
private let mppTensorOpsAvailable: Bool = {
  guard let context = try? MetalContext() else { return false }
  return MPPPrefillInt4QMM(context: context).isAvailable
}()

extension PrefillGroupedRoutedMoETests {
  @Test(arguments: [4, 8])
  func streamedBatchedMatchesReferenceAcrossPartialMicrobatch(weightBits: Int) throws {
    let d = 64
    let f = 64
    let rows = 3
    let topK = 2
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      [
        Self.pair(token: 0, expert: 2, rank: 0),
        Self.pair(token: 0, expert: 0, rank: 1),
        Self.pair(token: 1, expert: 1, rank: 0),
        Self.pair(token: 1, expert: 2, rank: 1),
        Self.pair(token: 2, expert: 0, rank: 0),
        Self.pair(token: 2, expert: 1, rank: 1),
      ],
      queryCount: rows,
      topK: topK,
      numExperts: 16,
      tileExpertCount: 16)
    let pool = Self.makeSyntheticExpertPool(numExperts: 16,
                                            d: d,
                                            f: f,
                                            weightBits: weightBits)
    let hidden = (0..<(rows * d)).map { i in
      Float16(Float((i % 17) - 8))
    }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes,
      hidden: hidden,
      hiddenStride: d,
      pool: pool,
      topK: topK,
      d: d,
      f: f)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx,
                                              weightBits: weightBits)
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * 4 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let downScratch = ctx.device.makeBuffer(
        length: 4 * d * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }

    let expertIDs = Array(0..<16)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device,
        pool: pool,
        expertIDs: expertIDs))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(
      device: ctx.device,
      binding: binding)
    let microbatches = try grouped.encodeStreamedBatched(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: activationScratch,
      downScratch: downScratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      params: params,
      pairMicrobatchRows: 4)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let maxAbsoluteError = zip(actual, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(microbatches == 2)
    #expect(maxAbsoluteError <= 0.0001,
            "weightBits=\(weightBits) maxAbsoluteError=\(maxAbsoluteError)")
    #expect(binding.views.allSatisfy { $0.offset > 0 })
  }

  @Test func streamedBatchedGptOssBiasesMatchReference() throws {
    let d = 64
    let f = 64
    let rows = 2
    let topK = 2
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      [
        Self.pair(token: 0, expert: 1, rank: 0),
        Self.pair(token: 0, expert: 3, rank: 1),
        Self.pair(token: 1, expert: 0, rank: 0),
        Self.pair(token: 1, expert: 2, rank: 1),
      ],
      queryCount: rows,
      topK: topK,
      numExperts: 4,
      tileExpertCount: 4)
    let pool = Self.makeSyntheticExpertPool(numExperts: 4,
                                            d: d,
                                            f: f,
                                            weightBits: 4,
                                            additiveBiases: true)
    let hidden = (0..<(rows * d)).map { i in
      Float16(Float((i % 13) - 6) * 0.25)
    }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes,
      hidden: hidden,
      hiddenStride: d,
      pool: pool,
      topK: topK,
      d: d,
      f: f,
      additiveBiases: true,
      clampedSwiGLU: true)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx,
                                              weightBits: 4,
                                              expertAdditiveBiases: true,
                                              clampedSwiGLU: true)
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * 4 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let downScratch = ctx.device.makeBuffer(
        length: 4 * d * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }

    let expertIDs = Array(0..<4)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device,
        pool: pool,
        expertIDs: expertIDs))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(
      device: ctx.device,
      binding: binding)
    _ = try grouped.encodeStreamedBatched(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: activationScratch,
      downScratch: downScratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      params: params,
      pairMicrobatchRows: 4)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let maxAbsoluteError = zip(actual, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(maxAbsoluteError <= 0.001,
            "gpt-oss biases maxAbsoluteError=\(maxAbsoluteError)")
  }


  /// One synthetic tile of four experts holding 40 / 32 / 5 / 3 pairs. Every
  /// token below 32 carries a pair in both long experts, so the scatter has to
  /// keep the two `(token, rank)` rows apart.
  struct FourExpertTile {
    let d: Int
    let f: Int
    let rows = 40
    let topK = 2
    let ctx: MetalContext
    let mpp: MPPPrefillInt4QMM
    let grouped: PrefillGroupedRoutedMoE
    let routes: PrefillMoEGroupedRoutes
    let tile: PrefillMoETile
    let binding: PrefillStreamedTileBinding
    let argumentBuffer: PrefillStreamedTileArgumentBuffer
    let params: PrefillGroupedRoutedMoEStreamedParams
    let hiddenBuffer: MTLBuffer
    let pairBuffer: MTLBuffer
    let activationScratch: MTLBuffer
    let downScratch: MTLBuffer

    var partialElements: Int { rows * topK * d }

    init?(siluActivation: Bool,
          d: Int = 512,
          f: Int = 512,
          irregularHidden: Bool = false) throws {
      self.d = d
      self.f = f
      var pairs: [PrefillTokenExpertPair] = []
      for token in 0..<40 {
        pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: 0, rank: 0))
      }
      for token in 0..<32 {
        pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: 1, rank: 1))
      }
      for token in 32..<37 {
        pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: 2, rank: 1))
      }
      for token in 37..<40 {
        pairs.append(PrefillGroupedRoutedMoETests.pair(token: UInt32(token), expert: 3, rank: 1))
      }
      routes = try PrefillMoEGrouping.groupTokenExpertPairs(
        pairs,
        queryCount: rows,
        topK: topK,
        numExperts: 8,
        tileExpertCount: 16)
      let pool = PrefillGroupedRoutedMoETests.makeSyntheticExpertPool(numExperts: 8, d: d, f: f)
      var state: UInt64 = 0x9E37_79B9_7F4A_7C15
      let hidden = (0..<(rows * d)).map { i -> Float16 in
        guard irregularHidden else { return Float16(Float((i % 17) - 8)) }
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float16(Float(Int64(state >> 40) - (1 << 23)) / Float(1 << 21))
      }

      ctx = try MetalContext()
      mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 4)
      guard mpp.isAvailable else {
        Issue.record("""
          MPP prefill QMM pipeline unavailable; the GEMM path would silently \
          compare the scalar path with itself
          """)
        return nil
      }
      grouped = try PrefillGroupedRoutedMoE(context: ctx,
                                            siluActivation: siluActivation,
                                            weightBits: 4)
      let expertIDs = Array(0..<8)
      guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
        let pairBuffer = ctx.device.makeBuffer(
          bytes: routes.sortedPairs,
          length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
          options: .storageModeShared),
        let activationScratch = ctx.device.makeBuffer(
          length: 3 * 32 * f * MemoryLayout<Float16>.stride,
          options: .storageModePrivate),
        let downScratch = ctx.device.makeBuffer(
          length: 32 * d * MemoryLayout<Float16>.stride,
          options: .storageModePrivate)
      else {
        Issue.record("allocation failed")
        return nil
      }
      self.hiddenBuffer = hiddenBuffer
      self.pairBuffer = pairBuffer
      self.activationScratch = activationScratch
      self.downScratch = downScratch
      binding = try PrefillStreamedTileBinding(
        expertIDs: expertIDs,
        views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
          device: ctx.device,
          pool: pool,
          expertIDs: expertIDs))
      argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: ctx.device,
                                                              binding: binding)
      tile = routes.tiles[0]
      params = PrefillGroupedRoutedMoEStreamedParams(
        pairStart: tile.pairStart,
        pairCount: tile.pairCount,
        d: UInt32(d),
        routedIntermediate: UInt32(f),
        topK: UInt32(topK),
        hiddenStrideElements: UInt32(d),
        binding: binding,
        offsets: pool.offsets)
    }

    func sentinelPartials() -> MTLBuffer? {
      Fp16Buffer.make(ctx.device,
                      halves: [Float16](repeating: -77, count: partialElements))
    }

    func encodeScalar(on commandBuffer: MTLCommandBuffer,
                      into routePartials: MTLBuffer,
                      pairStart: UInt32,
                      pairCount: UInt32) throws {
      var scalarParams = params
      scalarParams.pairStart = pairStart
      scalarParams.pairCount = pairCount
      _ = try grouped.encodeStreamedBatched(
        commandBuffer: commandBuffer,
        hidden: hiddenBuffer,
        sortedPairs: pairBuffer,
        routePartials: routePartials,
        gateUpActScratch: activationScratch,
        downScratch: downScratch,
        argumentBuffer: argumentBuffer,
        binding: binding,
        params: scalarParams,
        pairMicrobatchRows: 32)
    }

    func scalarReference() throws -> [Float]? {
      guard let buffer = sentinelPartials() else { return nil }
      try run { commandBuffer in
        try encodeScalar(on: commandBuffer, into: buffer,
                         pairStart: tile.pairStart, pairCount: tile.pairCount)
      }
      return Fp16Buffer.read(buffer, count: partialElements)
    }

    func run(_ encode: (MTLCommandBuffer) throws -> Void) throws {
      guard let commandBuffer = ctx.queue.makeCommandBuffer() else {
        throw PrefillGroupedRoutedMoEError.allocationFailed("command buffer")
      }
      try encode(commandBuffer)
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      if let error = commandBuffer.error { throw error }
    }

    func rowElements(_ values: [Float], experts: Set<UInt32>) -> [Float] {
      routes.sortedPairs.filter { experts.contains($0.expert) }.flatMap { pair in
        let base = (Int(pair.token) * topK + Int(pair.rank)) * d
        return Array(values[base..<(base + d)])
      }
    }
  }

  @Test func groupedWavePlannerSplitsAndPadsOnRowTiles() throws {
    let ctx = try MetalContext()
    let binding = try PrefillStreamedTileBinding(
      expertIDs: [0, 1, 2, 3],
      views: Self.fakeTensorViews(device: ctx.device, count: 4))
    let ranges = [PrefillExpertPairRange(expert: 0, pairStart: 0, pairCount: 40),
                  PrefillExpertPairRange(expert: 1, pairStart: 40, pairCount: 32),
                  PrefillExpertPairRange(expert: 2, pairStart: 72, pairCount: 5),
                  PrefillExpertPairRange(expert: 3, pairStart: 77, pairCount: 3)]
    func block(_ slot: UInt32, _ pairStart: UInt32, _ rows: UInt32,
               _ stagingRow: UInt32) -> PrefillRoutedExpertBlock {
      PrefillRoutedExpertBlock(slot: slot, pairStart: pairStart, rows: rows,
                               stagingRow: stagingRow, rowTileStart: stagingRow / 64)
    }

    let oneWave = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges,
                                                              binding: binding,
                                                              stagingRows: 512)
    #expect(oneWave == [PrefillRoutedExpertWave(
      blocks: [block(0, 0, 40, 0), block(1, 40, 32, 64),
               block(2, 72, 5, 128), block(3, 77, 3, 192)],
      paddedRows: 256)])
    #expect(PrefillGroupedRoutedMoE.rowTileTable(for: oneWave[0]) == [0, 1, 2, 3])

    let twoWaves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges,
                                                               binding: binding,
                                                               stagingRows: 128)
    #expect(twoWaves == [
      PrefillRoutedExpertWave(blocks: [block(0, 0, 40, 0), block(1, 40, 32, 64)],
                              paddedRows: 128),
      PrefillRoutedExpertWave(blocks: [block(2, 72, 5, 0), block(3, 77, 3, 64)],
                              paddedRows: 128),
    ])

    let longExpert = try PrefillGroupedRoutedMoE.planExpertWaves(
      ranges: [PrefillExpertPairRange(expert: 2, pairStart: 0, pairCount: 600)],
      binding: binding,
      stagingRows: 512)
    #expect(longExpert == [
      PrefillRoutedExpertWave(blocks: [block(2, 0, 512, 0)], paddedRows: 512),
      PrefillRoutedExpertWave(blocks: [block(2, 512, 88, 0)], paddedRows: 128),
    ])
    #expect(PrefillGroupedRoutedMoE.rowTileTable(for: longExpert[0]).count == 8)
    #expect(PrefillGroupedRoutedMoE.rowTileTable(for: longExpert[1]) == [0, 0])

    #expect(throws: PrefillGroupedRoutedMoEError.self) {
      try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges,
                                                  binding: binding,
                                                  stagingRows: 96)
    }
  }

  /// 512 takes the kernel's own 256-wide K tile, 384 the K128 rung, 192 the K64 rung.
  @Test(arguments: [(d: 512, f: 512), (d: 384, f: 384), (d: 192, f: 192)])
  func groupedGEMMsMatchTheScalarPathAcrossWaves(shape: (d: Int, f: Int)) throws {
    guard let fixture = try FourExpertTile(siluActivation: true, d: shape.d, f: shape.f) else { return }
    guard let reference = try fixture.scalarReference(),
          let groupedBuffer = fixture.sentinelPartials() else {
      Issue.record("allocation failed")
      return
    }
    let ranges = try PrefillExpertPairRange.ranges(forTile: fixture.tile, routes: fixture.routes)
    let staging = try PrefillExpertStaging.allocate(device: fixture.ctx.device,
                                                    rowBlock: 64,
                                                    hiddenSize: fixture.d,
                                                    intermediate: fixture.f)
    let waves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges,
                                                            binding: fixture.binding,
                                                            stagingRows: 64)
    #expect(waves.count == 4)
    try fixture.run { commandBuffer in
      try fixture.grouped.encodeGroupedExpertGEMMs(
        commandBuffer: commandBuffer,
        mpp: fixture.mpp,
        hidden: fixture.hiddenBuffer,
        sortedPairs: fixture.pairBuffer,
        routePartials: groupedBuffer,
        binding: fixture.binding,
        argumentBuffer: fixture.argumentBuffer,
        waves: waves,
        staging: staging,
        params: fixture.params)
    }
    let actual = Fp16Buffer.read(groupedBuffer, count: fixture.partialElements)
    let allFinite = actual.allSatisfy { $0.isFinite }
    #expect(allFinite)
    let maxAbsDiff = RelError.maxAbsDiff(actual, reference)
    let relError = RelError.compute(actual: actual, reference: reference)
    #expect(maxAbsDiff <= 2e-2, "maxAbsDiff=\(maxAbsDiff)")
    #expect(relError <= 2e-2, "relError=\(relError)")
    let untouched = fixture.rowElements(actual, experts: [0, 1, 2, 3]).contains(-77)
    #expect(!untouched)
  }

  private static func groupedPartialsAcrossWaves(d: Int = 512,
                                                 f: Int = 512,
                                                 rowTile: MPPPrefillInt4QMM.GroupedRowTile = .m64,
                                                 tailTile: Int = 0,
                                                 irregular: Bool = false,
                                                 stagingRows: Int = 64) throws -> [Float16]? {
    guard let fixture = try FourExpertTile(siluActivation: true, d: d, f: f,
                                           irregularHidden: irregular) else { return nil }
    guard let groupedBuffer = fixture.sentinelPartials() else {
      Issue.record("allocation failed")
      return nil
    }
    let ranges = try PrefillExpertPairRange.ranges(forTile: fixture.tile, routes: fixture.routes)
    let staging = try PrefillExpertStaging.allocate(device: fixture.ctx.device,
                                                    rowBlock: stagingRows,
                                                    hiddenSize: fixture.d,
                                                    intermediate: fixture.f)
    let waves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges,
                                                            binding: fixture.binding,
                                                            stagingRows: stagingRows,
                                                            rowTile: rowTile.rawValue,
                                                            tailTile: tailTile)
    try fixture.run { commandBuffer in
      try fixture.grouped.encodeGroupedExpertGEMMs(
        commandBuffer: commandBuffer,
        mpp: fixture.mpp,
        hidden: fixture.hiddenBuffer,
        sortedPairs: fixture.pairBuffer,
        routePartials: groupedBuffer,
        binding: fixture.binding,
        argumentBuffer: fixture.argumentBuffer,
        waves: waves,
        staging: staging,
        params: fixture.params,
        rowTile: rowTile,
        tailTile: tailTile)
    }
    return Fp16Buffer.readHalf(groupedBuffer, count: fixture.partialElements)
  }

  /// 64 staging rows split the fixture into body-only and tail-only waves;
  /// 512 packs it into one wave with both regions on one encoder.
  @Test(.enabled(if: mppTensorOpsAvailable,
                 "Requires runtime MPP TensorOps support"),
        arguments: [64, 512])
  func tailTileIsBitIdenticalToTheSixtyFourRowPath(stagingRows: Int) throws {
    guard let plain = try Self.groupedPartialsAcrossWaves(irregular: true,
                                                          stagingRows: stagingRows),
          let tailed = try Self.groupedPartialsAcrossWaves(tailTile: 32,
                                                           irregular: true,
                                                           stagingRows: stagingRows) else { return }
    let finite = tailed.allSatisfy(\.isFinite)
    #expect(finite)
    let firstMismatch = zip(plain, tailed).enumerated().first { $0.element.0 != $0.element.1 }?.offset
    #expect(firstMismatch == nil, "first mismatch at \(firstMismatch ?? -1)")
    let untouched = tailed.contains(-77)
    #expect(!untouched)
  }

  @Test func tailTilePlannerPacksRemaindersIntoThirtyTwoRowTiles() throws {
    let ctx = try MetalContext()
    let binding = try PrefillStreamedTileBinding(
      expertIDs: [0, 1, 2, 3],
      views: Self.fakeTensorViews(device: ctx.device, count: 4))
    let ranges = [PrefillExpertPairRange(expert: 0, pairStart: 0, pairCount: 40),
                  PrefillExpertPairRange(expert: 1, pairStart: 40, pairCount: 32),
                  PrefillExpertPairRange(expert: 2, pairStart: 72, pairCount: 5),
                  PrefillExpertPairRange(expert: 3, pairStart: 77, pairCount: 3)]
    func block(_ slot: UInt32, _ pairStart: UInt32, _ rows: UInt32,
               _ stagingRow: UInt32, _ tileStart: UInt32) -> PrefillRoutedExpertBlock {
      PrefillRoutedExpertBlock(slot: slot, pairStart: pairStart, rows: rows,
                               stagingRow: stagingRow, rowTileStart: tileStart)
    }
    let waves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges, binding: binding,
                                                            stagingRows: 512, tailTile: 32)
    #expect(waves == [PrefillRoutedExpertWave(
      blocks: [block(0, 0, 40, 0, 0), block(1, 40, 32, 64, 0), block(2, 72, 5, 96, 1), block(3, 77, 3, 128, 2)],
      paddedRows: 160,
      tailRows: 96)])
    let tables = PrefillGroupedRoutedMoE.rowTileTables(for: waves[0])
    #expect(tables.gather == [0, 0, 1, 2, 3])
    #expect(tables.body == [0])
    #expect(tables.tail == [1, 2, 3])

    let split = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges, binding: binding,
                                                            stagingRows: 64, tailTile: 32)
    #expect(split == [
      PrefillRoutedExpertWave(blocks: [block(0, 0, 40, 0, 0)], paddedRows: 64, tailRows: 0),
      PrefillRoutedExpertWave(blocks: [block(1, 40, 32, 0, 0), block(2, 72, 5, 32, 1)],
                              paddedRows: 64, tailRows: 64),
      PrefillRoutedExpertWave(blocks: [block(3, 77, 3, 0, 0)], paddedRows: 32, tailRows: 32),
    ])

    let long = try PrefillGroupedRoutedMoE.planExpertWaves(
      ranges: [PrefillExpertPairRange(expert: 2, pairStart: 0, pairCount: 200)],
      binding: binding, stagingRows: 128, tailTile: 32)
    #expect(long == [
      PrefillRoutedExpertWave(blocks: [block(2, 0, 128, 0, 0)], paddedRows: 128, tailRows: 0),
      PrefillRoutedExpertWave(blocks: [block(2, 128, 64, 0, 0), block(2, 192, 8, 64, 0)],
                              paddedRows: 96, tailRows: 32),
    ])
  }

  @Test func tailTileRefusesARaggedK() throws {
    guard let fixture = try FourExpertTile(siluActivation: true, d: 192, f: 192) else { return }
    #expect(fixture.mpp.groupedRowTile32Available(forK: 2048))
    #expect(!fixture.mpp.groupedRowTile32Available(forK: 192))
    let ranges = try PrefillExpertPairRange.ranges(forTile: fixture.tile, routes: fixture.routes)
    let staging = try PrefillExpertStaging.allocate(device: fixture.ctx.device, rowBlock: 64,
                                                    hiddenSize: fixture.d, intermediate: fixture.f)
    let waves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges, binding: fixture.binding,
                                                            stagingRows: 64, tailTile: 32)
    guard let commandBuffer = fixture.ctx.queue.makeCommandBuffer(),
          let partials = fixture.sentinelPartials() else {
      Issue.record("allocation failed")
      return
    }
    let refusal = #expect(throws: MPPPrefillInt4QMMError.self) {
      try fixture.grouped.encodeGroupedExpertGEMMs(
        commandBuffer: commandBuffer, mpp: fixture.mpp, hidden: fixture.hiddenBuffer,
        sortedPairs: fixture.pairBuffer, routePartials: partials, binding: fixture.binding,
        argumentBuffer: fixture.argumentBuffer, waves: waves, staging: staging,
        params: fixture.params, tailTile: 32)
    }
    #expect(String(describing: refusal).contains("32-row instantiation"), "\(String(describing: refusal))")
  }

  @Test func tailTileIsSkippedWhenNoBlockHasARemainder() throws {
    let ctx = try MetalContext()
    let binding = try PrefillStreamedTileBinding(
      expertIDs: [0, 1, 2],
      views: Self.fakeTensorViews(device: ctx.device, count: 3))
    let ranges = [PrefillExpertPairRange(expert: 0, pairStart: 0, pairCount: 64),
                  PrefillExpertPairRange(expert: 1, pairStart: 64, pairCount: 128),
                  PrefillExpertPairRange(expert: 2, pairStart: 192, pairCount: 97)]
    let waves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges, binding: binding,
                                                            stagingRows: 512, tailTile: 32)
    #expect(waves.count == 1)
    #expect(waves[0].tailRows == 0)
    #expect(waves[0].paddedRows == 320)
    #expect(waves[0].blocks.map(\.rows) == [64, 128, 97])
    #expect(waves[0].blocks.map(\.stagingRow) == [0, 64, 192])
    let tables = PrefillGroupedRoutedMoE.rowTileTables(for: waves[0])
    #expect(tables.tail.isEmpty)
    #expect(tables.body == [0, 1, 1, 2, 2])
    #expect(tables.gather == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
  }

  @Test(.enabled(if: mppTensorOpsAvailable, "Requires runtime MPP TensorOps support"))
  func thirtyTwoRowGroupedTileIsBitIdenticalToTheSixtyFourRowTile() throws {
    guard let wide = try Self.groupedPartialsAcrossWaves(irregular: true),
          let narrow = try Self.groupedPartialsAcrossWaves(rowTile: .m32,
                                                           irregular: true) else { return }
    let finite = narrow.allSatisfy(\.isFinite)
    #expect(finite)
    let firstMismatch = zip(wide, narrow).enumerated().first { $0.element.0 != $0.element.1 }?.offset
    #expect(firstMismatch == nil, "first mismatch at \(firstMismatch ?? -1)")
    let untouched = narrow.contains(-77)
    #expect(!untouched)
  }

  @Test func thirtyTwoRowPlannerHalvesTheTailPadding() throws {
    let ranges = [
      PrefillExpertPairRange(expert: 0, pairStart: 0, pairCount: 40),
      PrefillExpertPairRange(expert: 1, pairStart: 40, pairCount: 32),
      PrefillExpertPairRange(expert: 2, pairStart: 72, pairCount: 5),
      PrefillExpertPairRange(expert: 3, pairStart: 77, pairCount: 3),
    ]
    let ctx = try MetalContext()
    let binding = try PrefillStreamedTileBinding(
      expertIDs: [0, 1, 2, 3],
      views: Self.fakeTensorViews(device: ctx.device, count: 4))
    let waves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges, binding: binding,
                                                            stagingRows: 512, rowTile: 32)
    #expect(waves.count == 1)
    #expect(waves[0].paddedRows == 160)
    #expect(waves[0].blocks.map(\.stagingRow) == [0, 64, 96, 128])
    #expect(waves[0].blocks.map(\.rowTileStart) == [0, 2, 3, 4])
    #expect(PrefillGroupedRoutedMoE.rowTileTable(for: waves[0], rowTile: 32) == [0, 0, 1, 2, 3])
  }

  @Test func groupedGEMMsHandleASingleOnePairExpert() throws {
    let d = 512
    let f = 512
    let sentinelRows = 64
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      [Self.pair(token: 0, expert: 5, rank: 0)],
      queryCount: 1,
      topK: 1,
      numExperts: 8,
      tileExpertCount: 16)
    let pool = Self.makeSyntheticExpertPool(numExperts: 8, d: d, f: f)
    let hidden = (0..<d).map { i in Float16(Float((i % 17) - 8)) }
    let ctx = try MetalContext()
    let mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 4)
    guard mpp.isAvailable else {
      Issue.record("MPP prefill QMM pipeline unavailable")
      return
    }
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, siluActivation: true, weightBits: 4)
    let partialElements = sentinelRows * d
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let referenceBuffer = Fp16Buffer.make(
        ctx.device, halves: [Float16](repeating: -77, count: partialElements)),
      let groupedBuffer = Fp16Buffer.make(
        ctx.device, halves: [Float16](repeating: -77, count: partialElements)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * 32 * f * MemoryLayout<Float16>.stride, options: .storageModePrivate),
      let downScratch = ctx.device.makeBuffer(
        length: 32 * d * MemoryLayout<Float16>.stride, options: .storageModePrivate)
    else {
      Issue.record("allocation failed")
      return
    }
    let binding = try PrefillStreamedTileBinding(
      expertIDs: [5],
      views: Self.streamedViewsWithNonzeroOffsets(device: ctx.device, pool: pool, expertIDs: [5]))
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: ctx.device, binding: binding)
    let tile = routes.tiles[0]
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: tile.pairStart,
      pairCount: tile.pairCount,
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: 1,
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    func run(_ encode: (MTLCommandBuffer) throws -> Void) throws {
      guard let commandBuffer = ctx.queue.makeCommandBuffer() else {
        throw PrefillGroupedRoutedMoEError.allocationFailed("command buffer")
      }
      try encode(commandBuffer)
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      if let error = commandBuffer.error { throw error }
    }
    try run { commandBuffer in
      _ = try grouped.encodeStreamedBatched(
        commandBuffer: commandBuffer,
        hidden: hiddenBuffer,
        sortedPairs: pairBuffer,
        routePartials: referenceBuffer,
        gateUpActScratch: activationScratch,
        downScratch: downScratch,
        argumentBuffer: argumentBuffer,
        binding: binding,
        params: params,
        pairMicrobatchRows: 32)
    }
    let ranges = try PrefillExpertPairRange.ranges(forTile: tile, routes: routes)
    let staging = try PrefillExpertStaging.allocate(device: ctx.device, rowBlock: 64,
                                                    hiddenSize: d, intermediate: f)
    let waves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges,
                                                            binding: binding,
                                                            stagingRows: 64)
    #expect(waves == [PrefillRoutedExpertWave(
      blocks: [PrefillRoutedExpertBlock(slot: 0, pairStart: 0, rows: 1,
                                        stagingRow: 0, rowTileStart: 0)],
      paddedRows: 64)])
    try run { commandBuffer in
      try grouped.encodeGroupedExpertGEMMs(
        commandBuffer: commandBuffer,
        mpp: mpp,
        hidden: hiddenBuffer,
        sortedPairs: pairBuffer,
        routePartials: groupedBuffer,
        binding: binding,
        argumentBuffer: argumentBuffer,
        waves: waves,
        staging: staging,
        params: params)
    }
    let reference = Fp16Buffer.read(referenceBuffer, count: partialElements)
    let actual = Fp16Buffer.read(groupedBuffer, count: partialElements)
    let row = Array(actual[0..<d])
    let rowFinite = row.allSatisfy { $0.isFinite }
    #expect(rowFinite)
    #expect(RelError.maxAbsDiff(row, Array(reference[0..<d])) <= 2e-2)
    #expect(RelError.compute(actual: row, reference: Array(reference[0..<d])) <= 2e-2)
    let paddedRowsUntouched = actual[d...].allSatisfy { $0 == -77 }
    #expect(paddedRowsUntouched, "the 63 padded rows must not leave the staging block")
  }
}
