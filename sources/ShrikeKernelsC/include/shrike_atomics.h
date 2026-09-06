#ifndef SHRIKE_ATOMICS_H
#define SHRIKE_ATOMICS_H

#include <stdint.h>

/// An acquire load of a shared word a GPU kernel writes while the host polls it.
static inline uint32_t shrike_load_acquire_u32(const uint32_t *word) {
    return __atomic_load_n(word, __ATOMIC_ACQUIRE);
}

#endif
