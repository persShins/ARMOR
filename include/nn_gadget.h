#include "nn_subgadget.h"
void mask_twoshare(uint8_t* input, uint32_t num, share* minput, share_fixed fresh_r);

void inputTohidden(share* x, share* w, share* bias, share* output, share* node_value, share_fixed input_r, s output_r, share_fixed mul_r, share_fixed btoa_r0, share_fixed btoa_r1);
void hiddenTooutput(share* x, share* w, share* bias, share* output, share* node_value, share_fixed input_r, s output_r, share_fixed mul_r, share_fixed add_r, share_fixed btoa_r0, share_fixed btoa_r1, share_fixed index_r);

void BNN_gadget_accuracy(share* x, share* w1, share* w2, share* bias1, share* bias2, share* output, share_fixed* cnt, share_fixed* answer_hidden_node, share_fixed* answer_hidden_output, share_fixed* answer_output_node, share_fixed* answer_output);
void BNN_gadget_clock(share* x, share* w1, share* w2, share* bias1, share* bias2, share* output);

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, NAME, ...) NAME
#define BNN_gadget(...) GET_MACRO(__VA_ARGS__, BNN_gadget_accuracy, BAD, BAD, BAD, BAD, BNN_gadget_clock)(__VA_ARGS__)