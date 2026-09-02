import Metal
import Testing
import ShrikeValidationSupport

@testable import Shrike

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


  @Test func expertGEMMsMatchTheScalarPathInOneRowBlock() throws {
    try Self.runExpertGEMMsMatchTheScalarPath(rowBlock: 512, siluActivation: false)
  }

  @Test func expertGEMMsMatchTheScalarPathAcrossRowBlocks() throws {
    try Self.runExpertGEMMsMatchTheScalarPath(rowBlock: 24, siluActivation: true)
  }

  /// One synthetic tile of four experts holding 40 / 32 / 5 / 3 pairs: the two
  /// experts at or above `matrixPathMinimumRows` go through the GEMM path, the
  /// short two come back as leftovers for the scalar path. Every token below 32
  /// carries a pair in both matrix experts, so the scatter has to keep the two
  /// `(token, rank)` rows apart.
  static func runExpertGEMMsMatchTheScalarPath(rowBlock: Int,
                                               siluActivation: Bool) throws {
    let d = 64
    let f = 64
    let rows = 40
    let topK = 2
    var pairs: [PrefillTokenExpertPair] = []
    for token in 0..<40 {
      pairs.append(Self.pair(token: UInt32(token), expert: 0, rank: 0))
    }
    for token in 0..<32 {
      pairs.append(Self.pair(token: UInt32(token), expert: 1, rank: 1))
    }
    for token in 32..<37 {
      pairs.append(Self.pair(token: UInt32(token), expert: 2, rank: 1))
    }
    for token in 37..<40 {
      pairs.append(Self.pair(token: UInt32(token), expert: 3, rank: 1))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs,
      queryCount: rows,
      topK: topK,
      numExperts: 8,
      tileExpertCount: 16)
    let pool = Self.makeSyntheticExpertPool(numExperts: 8, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in Float16(Float((i % 17) - 8)) }

    let ctx = try MetalContext()
    let mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 4)
    guard mpp.isAvailable else {
      Issue.record("""
        MPP prefill QMM pipeline unavailable; the GEMM path would silently \
        compare the scalar path with itself
        """)
      return
    }
    let grouped = try PrefillGroupedRoutedMoE(context: ctx,
                                              siluActivation: siluActivation,
                                              weightBits: 4)
    let partialElements = rows * topK * d
    let expertIDs = Array(0..<8)
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let referenceBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: partialElements)),
      let matrixBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: partialElements)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * 32 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let downScratch = ctx.device.makeBuffer(
        length: 32 * d * MemoryLayout<Float16>.stride,
        options: .storageModePrivate)
    else {
      Issue.record("allocation failed")
      return
    }
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device,
        pool: pool,
        expertIDs: expertIDs))
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: ctx.device,
                                                                binding: binding)
    let tile = routes.tiles[0]
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: tile.pairStart,
      pairCount: tile.pairCount,
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)

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

    guard let referenceCB = ctx.queue.makeCommandBuffer() else {
      Issue.record("allocation failed")
      return
    }
    try encodeScalar(on: referenceCB,
                     into: referenceBuffer,
                     pairStart: tile.pairStart,
                     pairCount: tile.pairCount)
    referenceCB.commit()
    referenceCB.waitUntilCompleted()
    if let error = referenceCB.error { throw error }

    let staging = try PrefillExpertStaging.allocate(device: ctx.device,
                                                    rowBlock: rowBlock,
                                                    hiddenSize: d,
                                                    intermediate: f)
    guard let matrixCB = ctx.queue.makeCommandBuffer() else {
      Issue.record("allocation failed")
      return
    }
    let ranges = try PrefillExpertPairRange.ranges(forTile: tile, routes: routes)
    let leftovers = try grouped.encodeExpertGEMMs(
      commandBuffer: matrixCB,
      mpp: mpp,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: matrixBuffer,
      binding: binding,
      ranges: ranges,
      staging: staging,
      params: params)
    for leftover in leftovers {
      try encodeScalar(on: matrixCB,
                       into: matrixBuffer,
                       pairStart: UInt32(leftover.pairStart),
                       pairCount: UInt32(leftover.pairCount))
    }
    matrixCB.commit()
    matrixCB.waitUntilCompleted()
    if let error = matrixCB.error { throw error }

    #expect(ranges == [PrefillExpertPairRange(expert: 0, pairStart: 0, pairCount: 40),
                       PrefillExpertPairRange(expert: 1, pairStart: 40, pairCount: 32),
                       PrefillExpertPairRange(expert: 2, pairStart: 72, pairCount: 5),
                       PrefillExpertPairRange(expert: 3, pairStart: 77, pairCount: 3)])
    #expect(leftovers == [PrefillExpertPairRange(expert: 2, pairStart: 72, pairCount: 5),
                          PrefillExpertPairRange(expert: 3, pairStart: 77, pairCount: 3)])

    let reference = Fp16Buffer.read(referenceBuffer, count: partialElements)
    let actual = Fp16Buffer.read(matrixBuffer, count: partialElements)
    func rowElements(_ values: [Float], experts: Set<UInt32>) -> [Float] {
      routes.sortedPairs.filter { experts.contains($0.expert) }.flatMap { pair in
        let base = (Int(pair.token) * topK + Int(pair.rank)) * d
        return Array(values[base..<(base + d)])
      }
    }
    let matrixActual = rowElements(actual, experts: [0, 1])
    let matrixReference = rowElements(reference, experts: [0, 1])
    let maxAbsDiff = RelError.maxAbsDiff(matrixActual, matrixReference)
    let relError = RelError.compute(actual: matrixActual, reference: matrixReference)
    #expect(matrixActual.count == 72 * d)
    #expect(maxAbsDiff <= 2e-2,
            "rowBlock=\(rowBlock) silu=\(siluActivation) maxAbsDiff=\(maxAbsDiff)")
    #expect(relError <= 2e-2,
            "rowBlock=\(rowBlock) silu=\(siluActivation) relError=\(relError)")
    #expect(rowElements(actual, experts: [2, 3]) == rowElements(reference, experts: [2, 3]),
            "leftover pairs must come out of the untouched scalar path bit for bit")

    let rank0 = Array(actual[0..<d])
    let rank1 = Array(actual[d..<(2 * d)])
    #expect(RelError.maxAbsDiff(rank0, rank1) > 1e-3,
            "token 0 rides expert 0 at rank 0 and expert 1 at rank 1: two distinct pair rows")
  }

  /// `leftovers == ranges` with `routePartials` untouched is the seam the
  /// runner branches on, so this reproduces its whole-tile fallback exactly.
  @Test func tileBelowTheThresholdFallsBackToTheWholeTileScalarPath() throws {
    let d = 64
    let f = 64
    let rows = 24
    let topK = 2
    var pairs: [PrefillTokenExpertPair] = []
    for token in 0..<24 {
      pairs.append(Self.pair(token: UInt32(token), expert: UInt32(token % 4), rank: 0))
      pairs.append(Self.pair(token: UInt32(token), expert: UInt32(4 + token % 4), rank: 1))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs,
      queryCount: rows,
      topK: topK,
      numExperts: 8,
      tileExpertCount: 16)
    let pool = Self.makeSyntheticExpertPool(numExperts: 8, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in Float16(Float((i % 17) - 8)) }

    let ctx = try MetalContext()
    let mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 4)
    guard mpp.isAvailable else {
      Issue.record("""
        MPP prefill QMM pipeline unavailable; the GEMM path would silently \
        compare the scalar path with itself
        """)
      return
    }
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 4)
    let partialElements = rows * topK * d
    let sentinel = Float16(-77)
    let expertIDs = Array(0..<8)
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let referenceBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: sentinel, count: partialElements)),
      let fallbackBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: sentinel, count: partialElements)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * 32 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let downScratch = ctx.device.makeBuffer(
        length: 32 * d * MemoryLayout<Float16>.stride,
        options: .storageModePrivate)
    else {
      Issue.record("allocation failed")
      return
    }
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device,
        pool: pool,
        expertIDs: expertIDs))
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: ctx.device,
                                                                binding: binding)
    let tile = routes.tiles[0]
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: tile.pairStart,
      pairCount: tile.pairCount,
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)

    func encodeWholeTileScalar(on commandBuffer: MTLCommandBuffer,
                               into routePartials: MTLBuffer) throws {
      _ = try grouped.encodeStreamedBatched(
        commandBuffer: commandBuffer,
        hidden: hiddenBuffer,
        sortedPairs: pairBuffer,
        routePartials: routePartials,
        gateUpActScratch: activationScratch,
        downScratch: downScratch,
        argumentBuffer: argumentBuffer,
        binding: binding,
        params: params,
        pairMicrobatchRows: 32)
    }

    guard let referenceCB = ctx.queue.makeCommandBuffer() else {
      Issue.record("allocation failed")
      return
    }
    try encodeWholeTileScalar(on: referenceCB, into: referenceBuffer)
    referenceCB.commit()
    referenceCB.waitUntilCompleted()
    if let error = referenceCB.error { throw error }

    let staging = try PrefillExpertStaging.allocate(device: ctx.device,
                                                    rowBlock: 512,
                                                    hiddenSize: d,
                                                    intermediate: f)
    let ranges = try PrefillExpertPairRange.ranges(forTile: tile, routes: routes)
    guard let gemmCB = ctx.queue.makeCommandBuffer() else {
      Issue.record("allocation failed")
      return
    }
    let leftovers = try grouped.encodeExpertGEMMs(
      commandBuffer: gemmCB,
      mpp: mpp,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: fallbackBuffer,
      binding: binding,
      ranges: ranges,
      staging: staging,
      params: params)
    gemmCB.commit()
    gemmCB.waitUntilCompleted()
    if let error = gemmCB.error { throw error }

    #expect(ranges.allSatisfy { $0.pairCount < PrefillGroupedRoutedMoE.matrixPathMinimumRows })
    #expect(leftovers == ranges)
    #expect(Fp16Buffer.readHalf(fallbackBuffer, count: partialElements)
      == [Float16](repeating: sentinel, count: partialElements),
            "no expert cleared the threshold, so the GEMM path must write nothing")

    guard let fallbackCB = ctx.queue.makeCommandBuffer() else {
      Issue.record("allocation failed")
      return
    }
    try encodeWholeTileScalar(on: fallbackCB, into: fallbackBuffer)
    fallbackCB.commit()
    fallbackCB.waitUntilCompleted()
    if let error = fallbackCB.error { throw error }

    #expect(Fp16Buffer.readHalf(fallbackBuffer, count: partialElements)
      == Fp16Buffer.readHalf(referenceBuffer, count: partialElements))
  }
}
