#ifndef SHRIKE_EXPERT_IO_H
#define SHRIKE_EXPERT_IO_H

#include <stddef.h>
#include <stdint.h>

/// Parallel reader for routed-expert blocks.
///
/// v3.x fetched misses one at a time on the calling thread and reached ~0.62 GB/s.
/// Measured on the same machine and the same file, expert-sized (1.688 MiB) reads
/// sustain ~2.0 GB/s on one thread and ~3.2 GB/s on four, and random offsets are
/// as fast as sequential ones -- NVMe does not care about locality at this
/// granularity, so nothing needs reordering. The gap was concurrency, not the
/// device.
///
/// Reads go through `pread` on per-thread descriptors: no shared file offset, no
/// lseek, and no mmap, so a fetch cannot fault the GPU's address space or leave
/// page-cache pressure behind for the rest of the machine to pay. That matters
/// here because the point of streaming is to keep RAM free.
typedef struct shrike_expert_reader shrike_expert_reader;

/// Independent descriptors a reader can run; `threads` is clamped to
/// [1, SHRIKE_IO_MAX_THREADS].
#define SHRIKE_IO_MAX_THREADS 16

/// Batches the reader can hold published at once. A second caller's batch is
/// accepted while the first is still outstanding, so the drive can stay fed
/// between one caller's last claim and its batch actually landing instead of
/// draining while a single publisher's reads finish. `batch_depth` is clamped
/// to [1, SHRIKE_IO_MAX_BATCHES].
#define SHRIKE_IO_MAX_BATCHES 2

/// Opens `path` with `threads` independent descriptors and room for
/// `batch_depth` published batches.
///
/// Returns NULL if the file cannot be opened or the arguments are invalid;
/// `out_errno` receives the failure cause when non-NULL. `threads` is clamped
/// to [1, SHRIKE_IO_MAX_THREADS] -- four saturates this device at one batch in
/// flight; `batch_depth` is clamped to [1, SHRIKE_IO_MAX_BATCHES].
/// `bypass_cache` sets `F_NOCACHE`, keeping expert reads out of the unified
/// buffer cache.
///
/// This is a policy choice, not an optimisation. With the page cache in play,
/// repeated reads of a hot expert are served from RAM and are very fast -- but the
/// cache grows without bound and consumes exactly the memory this project exists to
/// leave free. With it bypassed, the slot cache is the *only* cache, so the
/// declared RAM budget is the true footprint and throughput is the device's.
///
/// Default to bypassing for a predictable footprint; allow the page cache when the
/// caller would rather have the free speed.
shrike_expert_reader *shrike_expert_reader_create(const char *path,
                                               size_t expert_stride,
                                               int threads,
                                               int batch_depth,
                                               int bypass_cache,
                                               int *out_errno);

/// The caller must ensure no other thread is inside `shrike_expert_reader_fetch`
/// or `shrike_expert_reader_fetch_offsets` when this runs, since a signalled
/// submitter can still be inside `pthread_cond_wait` on the reader's lock.
void shrike_expert_reader_destroy(shrike_expert_reader *reader);

/// Reads `count` experts, `expert_ids[i]` into `destinations[i]`, and blocks
/// until every one has completed.
///
/// Each destination must hold `expert_stride` bytes. Returns 0 on success, or the
/// first `errno` observed by any worker; on failure the contents of the
/// destinations are undefined and the caller must not use them.
int shrike_expert_reader_fetch(shrike_expert_reader *reader,
                              const uint32_t *expert_ids,
                              void *const *destinations,
                              size_t count);

/// As `shrike_expert_reader_fetch`, but the caller supplies absolute byte offsets.
///
/// Callers that lay experts out as `index * stride` can use the id form; the
/// streamer cannot, because its regions carry a per-layer base and a container
/// offset, so an id would silently address the wrong layer.
int shrike_expert_reader_fetch_offsets(shrike_expert_reader *reader,
                                     const uint64_t *offsets,
                                     void *const *destinations,
                                     size_t count);

/// Threads actually in use, after clamping.
int shrike_expert_reader_threads(const shrike_expert_reader *reader);

/// Published batches actually held, after clamping.
int shrike_expert_reader_batch_depth(const shrike_expert_reader *reader);

/// A slot's shape as the claim and free-slot rules below see it: enough to
/// decide which slot a worker claims a read from, or a submitter publishes
/// into, with no lock and no I/O. `sequence` is publication order -- lower is
/// older -- and is meaningless while `active` is 0.
typedef struct {
    int active;          // published, not yet reaped by its submitter
    uint64_t sequence;
    size_t next_index;   // next unclaimed read
    size_t count;        // total reads in the batch
} shrike_expert_batch_slot;

/// The FIFO claim: the active slot with the lowest `sequence` that still has
/// an unclaimed read (`next_index < count`). Returns the slot's index, or -1
/// if none is claimable -- every active slot is exhausted, or none is active.
/// A worker waits only when this returns -1; no read of a newer batch is ever
/// claimed while an older one still has an unclaimed read.
int shrike_expert_reader_claim_slot(const shrike_expert_batch_slot *slots,
                                   int slot_count);

/// The free-slot search a submitter uses to publish: the first inactive
/// slot's index, or -1 if every slot is active (the caller parks).
int shrike_expert_reader_free_slot(const shrike_expert_batch_slot *slots,
                                  int slot_count);

/// A slot's shape as `shrike_expert_reader_cancel_slot` sees it.
typedef struct {
    size_t count;
    size_t next_index;
    size_t outstanding;
    int first_errno;
} shrike_expert_batch_cancel_state;

/// Cancels one slot's unclaimed reads at shutdown. A slot with no unclaimed
/// read (`next_index == count`) is left untouched. Otherwise: `outstanding`
/// drops by exactly `count - next_index`, `next_index` becomes `count` so no
/// worker claims from it again, and `first_errno` becomes `ECANCELED` only if
/// it was unset -- a real read failure is preserved. Reads already claimed are
/// left for their worker to finish and count normally against the reduced
/// `outstanding`.
void shrike_expert_reader_cancel_slot(shrike_expert_batch_cancel_state *slot);

#endif /* SHRIKE_EXPERT_IO_H */
