#include "type.h"

//Types of secret shares
typedef uint8_t s;
typedef uint32_t share_fixed;
typedef struct share { share_fixed d[2]; } share;

//Parameters
#define size 2
#define k 32

//Parameters for LuTs
#define num_cLuT (k/size -2)
#define ran_output (k/size / 8)
#define ran_input (k/size / 4 )

#define mod (size + 1)
#define entry 1 << size

void SecSub(share* input_a, share* input_b, share* output, share_fixed add_r);
void SecADD(share* input_a, share* input_b, share *output, share_fixed add_r);
void SecMUL(share* input_a, share* input_b, share* output, share_fixed mul_r);
void SecReLU_prime(share* x, share_fixed input_r, s output_r, share_fixed m);
void SecLinear(share* input, share* weight, share* bias, share* output, uint32_t cnt1, uint32_t cnt2, share_fixed mul_r);
void BtoA(share* x, share* output, share_fixed dim, share_fixed btoa_r0, share_fixed btoa_r1);
void SecReLU_prime_cmp(share* x, share* output, s output_r);
void SecArgmax(share* input, share* max_index, share_fixed input_r, s output_r, share_fixed mul_r, share_fixed add_r, share_fixed btoa_r0, share_fixed btoa_r1, share_fixed index_r);
void genLuT(s* input_r, s* output_r);