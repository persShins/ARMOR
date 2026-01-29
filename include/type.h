#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define FP_FRACTIONAL_BITS 6

typedef uint8_t s;
typedef uint32_t share_fixed;

typedef int32_t fixed_point;

//Model Spec
#define n 784
#define n1 512
#define n2 10