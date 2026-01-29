#include <x86intrin.h>

static inline uint64_t get_cycles(void)
{
    unsigned aux;
    _mm_lfence();
    const uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}
