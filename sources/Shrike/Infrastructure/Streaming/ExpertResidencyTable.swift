import Metal

@frozen
public struct ExpertResidencyEntry: Sendable, Equatable {
    public static let notResidentSlot = UInt32.max
    public static let empty: UInt32 = 0
    public static let loading: UInt32 = 1
    public static let resident: UInt32 = 2

    public var slot: UInt32
    public var state: UInt32

    public init(slot: UInt32 = ExpertResidencyEntry.notResidentSlot,
                state: UInt32 = ExpertResidencyEntry.empty) {
        self.slot = slot
        self.state = state
    }
}

/// unchecked-invariant: immutable resource handles are synchronized by cache plans.
public struct ExpertResidencyResources: @unchecked Sendable {
    public let table: MTLBuffer
    public let expertPool: MTLBuffer
    public let poolSlotStride: UInt64
    public let expertStride: UInt64
    public let expertCount: Int
}
