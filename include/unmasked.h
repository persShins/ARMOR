#include <stdio.h>

void unmask_input(fixed_point* input, fixed_point* weight, fixed_point* bias, fixed_point* output, fixed_point cnt1, fixed_point cnt2);
void unmask_input_accuracy(fixed_point* input, fixed_point* weight, fixed_point* bias, fixed_point* output, fixed_point cnt1, fixed_point cnt2, share_fixed* output1, share_fixed* output2);

void unmask_hidden(fixed_point* input, fixed_point* weight, fixed_point* bias, fixed_point* output, fixed_point* index, fixed_point cnt1, fixed_point cnt2);
void unmask_hidden_accuracy(fixed_point* input, fixed_point* weight, fixed_point* bias, fixed_point* output, fixed_point* index, fixed_point cnt1, fixed_point cnt2, share_fixed* output1, share_fixed* output2);

#define GET_MACRO_hidden(_1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME
#define GET_MACRO_output(_1, _2, _3, _4, _5, _6, _7, _8, _9, NAME, ...) NAME

#define hidden(...) GET_MACRO_hidden(__VA_ARGS__, unmask_input_accuracy, BAD, unmask_input)(__VA_ARGS__)
#define output(...) GET_MACRO_output(__VA_ARGS__, unmask_hidden_accuracy, BAD, unmask_hidden)(__VA_ARGS__)