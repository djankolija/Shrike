import Metal

/// Keeps every routed-expert pool slab resident for the queue's lifetime, so
/// the first command buffer of a layer to name one does not pay for its
/// residency in the driver (v12 P12). The queue outlives the model, so the
/// holder detaches and empties the set on deinit or the pools would never be
/// freed on unload.
final class ExpertPoolResidency {
    private let queue: MTLCommandQueue
    private let residencySet: MTLResidencySet
    private var included: Set<ObjectIdentifier> = []

    init(device: MTLDevice, queue: MTLCommandQueue) throws {
        let descriptor = MTLResidencySetDescriptor()
        descriptor.label = "shrike.expert-pools"
        descriptor.initialCapacity = 64
        self.queue = queue
        self.residencySet = try device.makeResidencySet(descriptor: descriptor)
        queue.addResidencySet(residencySet)
    }

    deinit {
        queue.removeResidencySet(residencySet)
        residencySet.removeAllAllocations()
        residencySet.commit()
    }

    func include(_ buffer: MTLBuffer) {
        guard included.insert(ObjectIdentifier(buffer)).inserted else { return }
        residencySet.addAllocation(buffer)
        residencySet.commit()
    }

    var allocationCount: Int { residencySet.allocationCount }
    var allocatedBytes: UInt64 { residencySet.allocatedSize }
}
