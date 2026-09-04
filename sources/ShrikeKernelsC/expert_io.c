// Parallel routed-expert reader. See include/shrike_expert_io.h for why this
// exists: the device sustains ~3.2 GB/s on four concurrent expert-sized reads and
// v3.x was reaching ~0.62 GB/s by fetching one at a time on the calling thread.

#include "include/shrike_expert_io.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syslimits.h>
#include <unistd.h>

// One published (or reapable) batch.
typedef struct {
    const uint32_t *expert_ids;   // one of these two is non-NULL while active
    const uint64_t *offsets;
    void *const *destinations;
    size_t count;
    size_t next_index;            // claimed by workers
    size_t outstanding;           // published minus completed
    int first_errno;
    int active;                   // published, not yet reaped by its submitter
    uint64_t sequence;            // publication order; the FIFO claim key
    pthread_cond_t done;          // wakes only this slot's own submitter
} shrike_expert_batch_state;

struct shrike_expert_reader {
    int fds[SHRIKE_IO_MAX_THREADS];
    int threads;
    int depth;                    // clamped batch depth, 1..SHRIKE_IO_MAX_BATCHES
    size_t expert_stride;

    pthread_t workers[SHRIKE_IO_MAX_THREADS];
    pthread_mutex_t lock;
    pthread_cond_t work_ready;    // reader-wide: some slot has claimable work, or shutdown
    pthread_cond_t batch_idle;    // reader-wide: a slot was freed, or shutdown
    int shutting_down;
    uint64_t next_sequence;

    shrike_expert_batch_state slots[SHRIKE_IO_MAX_BATCHES];
};

int shrike_expert_reader_claim_slot(const shrike_expert_batch_slot *slots,
                                   int slot_count) {
    int best = -1;
    for (int i = 0; i < slot_count; ++i) {
        if (!slots[i].active || slots[i].next_index >= slots[i].count) {
            continue;
        }
        if (best < 0 || slots[i].sequence < slots[best].sequence) {
            best = i;
        }
    }
    return best;
}

int shrike_expert_reader_free_slot(const shrike_expert_batch_slot *slots,
                                  int slot_count) {
    for (int i = 0; i < slot_count; ++i) {
        if (!slots[i].active) {
            return i;
        }
    }
    return -1;
}

void shrike_expert_reader_cancel_slot(shrike_expert_batch_cancel_state *slot) {
    if (slot->next_index >= slot->count) {
        return;
    }
    size_t unclaimed = slot->count - slot->next_index;
    slot->outstanding -= unclaimed;
    slot->next_index = slot->count;
    if (slot->first_errno == 0) {
        slot->first_errno = ECANCELED;
    }
}

/// Best-effort, and never lowers an already-higher soft limit: a failure here
/// just surfaces later as a diagnosable EMFILE from open().
static void raise_nofile_limit(void) {
    struct rlimit lim;
    if (getrlimit(RLIMIT_NOFILE, &lim) != 0) {
        return;
    }
    rlim_t target = lim.rlim_max;
    if (target > (rlim_t)OPEN_MAX) {
        target = (rlim_t)OPEN_MAX;
    }
    if (lim.rlim_cur < target) {
        lim.rlim_cur = target;
        setrlimit(RLIMIT_NOFILE, &lim);
    }
}

static pthread_once_t raise_nofile_once_token = PTHREAD_ONCE_INIT;

/// Reads one expert with `pread`, looping because a short read is legal.
static int read_one(int fd, void *dst, size_t stride, uint64_t byte_offset) {
    unsigned char *out = (unsigned char *)dst;
    off_t base = (off_t)byte_offset;
    size_t done = 0;
    while (done < stride) {
        ssize_t got = pread(fd, out + done, stride - done, base + (off_t)done);
        if (got > 0) {
            done += (size_t)got;
            continue;
        }
        if (got == 0) {
            return EIO;               // short file: the caller's offsets are wrong
        }
        if (errno == EINTR) {
            continue;
        }
        return errno;
    }
    return 0;
}

static void *worker_main(void *arg) {
    shrike_expert_reader *r = (shrike_expert_reader *)arg;
    // Each worker owns one descriptor, so index it by position in the pool.
    int fd_slot = -1;
    pthread_mutex_lock(&r->lock);
    for (int i = 0; i < r->threads; ++i) {
        if (pthread_equal(r->workers[i], pthread_self())) {
            fd_slot = i;
            break;
        }
    }
    pthread_mutex_unlock(&r->lock);
    if (fd_slot < 0) {
        return NULL;
    }
    int fd = r->fds[fd_slot];

    pthread_mutex_lock(&r->lock);
    for (;;) {
        int claimed = -1;
        for (;;) {
            shrike_expert_batch_slot views[SHRIKE_IO_MAX_BATCHES];
            for (int i = 0; i < r->depth; ++i) {
                views[i].active = r->slots[i].active;
                views[i].sequence = r->slots[i].sequence;
                views[i].next_index = r->slots[i].next_index;
                views[i].count = r->slots[i].count;
            }
            claimed = shrike_expert_reader_claim_slot(views, r->depth);
            if (claimed >= 0 || r->shutting_down) {
                break;
            }
            pthread_cond_wait(&r->work_ready, &r->lock);
        }
        if (claimed < 0) {
            break;
        }
        shrike_expert_batch_state *slot = &r->slots[claimed];
        size_t index = slot->next_index++;
        uint64_t offset = slot->offsets != NULL
            ? slot->offsets[index]
            : (uint64_t)slot->expert_ids[index] * (uint64_t)r->expert_stride;
        void *dst = slot->destinations[index];
        pthread_mutex_unlock(&r->lock);

        int rc = read_one(fd, dst, r->expert_stride, offset);

        pthread_mutex_lock(&r->lock);
        // `claimed` still names our slot: this read is still counted in its
        // submitter's `outstanding`, so the slot cannot have been reaped.
        slot = &r->slots[claimed];
        if (rc != 0 && slot->first_errno == 0) {
            slot->first_errno = rc;
        }
        if (--slot->outstanding == 0) {
            pthread_cond_signal(&slot->done);
        }
    }
    pthread_mutex_unlock(&r->lock);
    return NULL;
}

shrike_expert_reader *shrike_expert_reader_create(const char *path,
                                               size_t expert_stride,
                                               int threads,
                                               int batch_depth,
                                               int bypass_cache,
                                               int *out_errno) {
    if (out_errno) {
        *out_errno = 0;
    }
    if (path == NULL || expert_stride == 0) {
        if (out_errno) { *out_errno = EINVAL; }
        return NULL;
    }
    if (threads < 1) { threads = 1; }
    if (threads > SHRIKE_IO_MAX_THREADS) { threads = SHRIKE_IO_MAX_THREADS; }
    if (batch_depth < 1) { batch_depth = 1; }
    if (batch_depth > SHRIKE_IO_MAX_BATCHES) { batch_depth = SHRIKE_IO_MAX_BATCHES; }

    pthread_once(&raise_nofile_once_token, raise_nofile_limit);

    shrike_expert_reader *r = (shrike_expert_reader *)calloc(1, sizeof(*r));
    if (r == NULL) {
        if (out_errno) { *out_errno = ENOMEM; }
        return NULL;
    }
    r->expert_stride = expert_stride;
    r->threads = threads;
    r->depth = batch_depth;
    for (int i = 0; i < SHRIKE_IO_MAX_THREADS; ++i) {
        r->fds[i] = -1;
    }

    // One descriptor per worker: pread does not use the shared offset, but
    // separate descriptors keep the kernel's per-fd state uncontended.
    for (int i = 0; i < threads; ++i) {
        r->fds[i] = open(path, O_RDONLY);
        if (r->fds[i] >= 0 && bypass_cache) {
            // Bounded memory is a correctness contract. Silently continuing
            // without F_NOCACHE would create an undeclared page-cache working
            // set, so an unsupported descriptor must fail the reader.
            if (fcntl(r->fds[i], F_NOCACHE, 1) != 0) {
                int err = errno;
                close(r->fds[i]);
                r->fds[i] = -1;
                for (int j = 0; j < i; ++j) { close(r->fds[j]); }
                free(r);
                if (out_errno) { *out_errno = err; }
                return NULL;
            }
        }
        if (r->fds[i] < 0) {
            int err = errno;
            for (int j = 0; j < i; ++j) { close(r->fds[j]); }
            free(r);
            if (out_errno) { *out_errno = err; }
            return NULL;
        }
    }

    if (pthread_mutex_init(&r->lock, NULL) != 0) {
        goto sync_init_failed;
    }
    if (pthread_cond_init(&r->work_ready, NULL) != 0) {
        pthread_mutex_destroy(&r->lock);
        goto sync_init_failed;
    }
    if (pthread_cond_init(&r->batch_idle, NULL) != 0) {
        pthread_cond_destroy(&r->work_ready);
        pthread_mutex_destroy(&r->lock);
        goto sync_init_failed;
    }
    {
        int slots_initialized = 0;
        for (int i = 0; i < SHRIKE_IO_MAX_BATCHES; ++i) {
            if (pthread_cond_init(&r->slots[i].done, NULL) != 0) {
                for (int j = 0; j < slots_initialized; ++j) {
                    pthread_cond_destroy(&r->slots[j].done);
                }
                pthread_cond_destroy(&r->batch_idle);
                pthread_cond_destroy(&r->work_ready);
                pthread_mutex_destroy(&r->lock);
                goto sync_init_failed;
            }
            slots_initialized++;
        }
    }

    // Publish thread identities under the lock before any worker looks for its
    // own slot, otherwise a fast worker can fail to find itself.
    pthread_mutex_lock(&r->lock);
    int started = 0;
    for (int i = 0; i < threads; ++i) {
        if (pthread_create(&r->workers[i], NULL, worker_main, r) != 0) {
            break;
        }
        started++;
    }
    if (started < threads) {
        // Bring up whatever started, then fail: a partial pool would silently
        // read at a fraction of the requested rate.
        r->shutting_down = 1;
        r->threads = started;
        pthread_cond_broadcast(&r->work_ready);
        pthread_mutex_unlock(&r->lock);
        for (int i = 0; i < started; ++i) { pthread_join(r->workers[i], NULL); }
        for (int j = 0; j < threads; ++j) { close(r->fds[j]); }
        pthread_cond_destroy(&r->work_ready);
        pthread_cond_destroy(&r->batch_idle);
        for (int i = 0; i < SHRIKE_IO_MAX_BATCHES; ++i) {
            pthread_cond_destroy(&r->slots[i].done);
        }
        pthread_mutex_destroy(&r->lock);
        free(r);
        if (out_errno) { *out_errno = EAGAIN; }
        return NULL;
    }
    pthread_mutex_unlock(&r->lock);
    return r;

sync_init_failed:
    for (int j = 0; j < threads; ++j) { close(r->fds[j]); }
    free(r);
    if (out_errno) { *out_errno = ENOMEM; }
    return NULL;
}

void shrike_expert_reader_destroy(shrike_expert_reader *r) {
    if (r == NULL) {
        return;
    }
    pthread_mutex_lock(&r->lock);
    r->shutting_down = 1;
    // Cancel each active slot's unclaimed reads so its submitter's wait
    // predicate can become true, and signal it directly since no worker will
    // if outstanding already reached zero right here.
    for (int i = 0; i < r->depth; ++i) {
        shrike_expert_batch_state *slot = &r->slots[i];
        if (!slot->active) {
            continue;
        }
        shrike_expert_batch_cancel_state cancel = {
            .count = slot->count,
            .next_index = slot->next_index,
            .outstanding = slot->outstanding,
            .first_errno = slot->first_errno,
        };
        shrike_expert_reader_cancel_slot(&cancel);
        slot->next_index = cancel.next_index;
        slot->outstanding = cancel.outstanding;
        slot->first_errno = cancel.first_errno;
        pthread_cond_signal(&slot->done);
    }
    pthread_cond_broadcast(&r->work_ready);
    pthread_cond_broadcast(&r->batch_idle);
    pthread_mutex_unlock(&r->lock);
    for (int i = 0; i < r->threads; ++i) {
        pthread_join(r->workers[i], NULL);
    }
    for (int i = 0; i < SHRIKE_IO_MAX_THREADS; ++i) {
        if (r->fds[i] >= 0) { close(r->fds[i]); }
    }
    pthread_cond_destroy(&r->work_ready);
    pthread_cond_destroy(&r->batch_idle);
    for (int i = 0; i < SHRIKE_IO_MAX_BATCHES; ++i) {
        pthread_cond_destroy(&r->slots[i].done);
    }
    pthread_mutex_destroy(&r->lock);
    free(r);
}

static int submit_batch(shrike_expert_reader *r,
                        const uint32_t *expert_ids,
                        const uint64_t *offsets,
                        void *const *destinations,
                        size_t count) {
    pthread_mutex_lock(&r->lock);
    // A caller owns its slot's published pointers until its own workers finish
    // and it clears the slot below. Without this predicate a later caller
    // could overwrite those pointers while an earlier one was still waiting,
    // corrupting destinations and leaving both callers blocked on the wrong
    // outstanding count.
    int slot_index;
    for (;;) {
        if (r->shutting_down) {
            pthread_mutex_unlock(&r->lock);
            return ECANCELED;
        }
        shrike_expert_batch_slot views[SHRIKE_IO_MAX_BATCHES];
        for (int i = 0; i < r->depth; ++i) {
            views[i].active = r->slots[i].active;
            views[i].sequence = r->slots[i].sequence;
            views[i].next_index = r->slots[i].next_index;
            views[i].count = r->slots[i].count;
        }
        slot_index = shrike_expert_reader_free_slot(views, r->depth);
        if (slot_index >= 0) {
            break;
        }
        pthread_cond_wait(&r->batch_idle, &r->lock);
    }

    shrike_expert_batch_state *slot = &r->slots[slot_index];
    slot->offsets = offsets;
    slot->expert_ids = expert_ids;
    slot->destinations = destinations;
    slot->count = count;
    slot->next_index = 0;
    slot->outstanding = count;
    slot->first_errno = 0;
    slot->active = 1;
    slot->sequence = r->next_sequence++;
    pthread_cond_broadcast(&r->work_ready);
    while (slot->outstanding > 0) {
        pthread_cond_wait(&slot->done, &r->lock);
    }
    int rc = slot->first_errno;
    // Leave the slot empty so idle workers block instead of spinning on a
    // consumed batch, and so the free-slot search finds it.
    slot->count = 0;
    slot->next_index = 0;
    slot->expert_ids = NULL;
    slot->offsets = NULL;
    slot->destinations = NULL;
    slot->active = 0;
    pthread_cond_broadcast(&r->batch_idle);
    pthread_mutex_unlock(&r->lock);
    return rc;
}

int shrike_expert_reader_fetch(shrike_expert_reader *r,
                              const uint32_t *expert_ids,
                              void *const *destinations,
                              size_t count) {
    if (r == NULL || (count > 0 && (expert_ids == NULL || destinations == NULL))) {
        return EINVAL;
    }
    if (count == 0) { return 0; }
    return submit_batch(r, expert_ids, NULL, destinations, count);
}

int shrike_expert_reader_fetch_offsets(shrike_expert_reader *r,
                                     const uint64_t *offsets,
                                     void *const *destinations,
                                     size_t count) {
    if (r == NULL || (count > 0 && (offsets == NULL || destinations == NULL))) {
        return EINVAL;
    }
    if (count == 0) { return 0; }
    return submit_batch(r, NULL, offsets, destinations, count);
}

int shrike_expert_reader_threads(const shrike_expert_reader *r) {
    return r == NULL ? 0 : r->threads;
}

int shrike_expert_reader_batch_depth(const shrike_expert_reader *r) {
    return r == NULL ? 0 : r->depth;
}
