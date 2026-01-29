#include "nn_gadget.h"

/*********************************************************************************
 * Name:        mask_twoshare
 *
 * Description: First-Order masking for the inputs with a fresh random input
 *
 * Arguments:   - uint8_t* input[]: Inputs to mask
 *              - uint32_t num: Number of inputs
 *              - share* minput: Shares of each input (First-Order masking)
 *              - share_fixed fresh_r: Fresh random for the input
 **********************************************************************************/
void mask_twoshare(uint8_t* input, uint32_t num, share* minput, share_fixed fresh_r)
{
    int i;
    share_fixed temp;
    int8_t tmp;
    for (i = 0; i < num; i++)
    {
        tmp = input[i];
        temp = (share_fixed)tmp;
        minput[i].d[0] = temp + fresh_r;
        minput[i].d[1] = -fresh_r;
    }
}

/*********************************************************************************
 * Name:        inputTohidden
 *
 * Description: Masked Input Layer with SecLinear, SecReLU_prime, which calculates
 *              the hidden layer nodes.
 *
 * Arguments:   - share* input[]: Input shares for the input layer
 *              - share* w[]: Weight shares for the input layer
 *              - share* bias[]: Bias shares for the input layer
 *              - share* output[]: Returns the input shares for the hidden layer
 *              - share* node_value[]: Returns the input shares for the hidden layer
 *              - share_fixed input_r: Random used for the input shares of masked LuTs
 *              - s output_r: Random used for the output of SecReLU' gadget
 *              - share_fixed mul_r: Random for SecMUL gadget
 *              - share_fixed btoa_r0: Random for BtoA gadget
 *              - share_fixed btoa_r1: Random for BtoA gadget
 **********************************************************************************/
void inputTohidden(
    share* x, share* w, share* bias, share* output, \
    share* node_value, share_fixed input_r, s output_r, \
    share_fixed mul_r, share_fixed btoa_r0, share_fixed btoa_r1
)
{
    share z[n1];
	int cnt;

    // z = <z-r, r> = x * w + b
	SecLinear(x, w, bias, z, n1, n, mul_r);

    // Saves the output node values for comparison
    for(int i=0;i<n1;i++)
    {
        node_value[i].d[0] = z[i].d[0];
        node_value[i].d[1] = z[i].d[1];
    }

    // output= <output-r, r> = SecReLU'(<z>)
	SecReLU_prime(z, input_r, output_r, n1);
	BtoA(z, output, n1, btoa_r0, btoa_r1);
	for (cnt = 0; cnt < n1; cnt++)
	{
		output[cnt].d[0] = 2 * output[cnt].d[0] - 1;
		output[cnt].d[1] = 2 * output[cnt].d[1];
	}
}

/*********************************************************************************
 * Name:        hiddenTooutput
 *
 * Description: Masked Hidden Layer with SecLinear, SecArgmax, which calculates
 *              the hidden layer nodes.
 *
 * Arguments:   - share* x[]: Masked inputs for the hidden layer
 *              - share* w[]: Masked weights for the hidden layer
 *              - share* bias[]: Masked biases for the hidden layer
 *              - share* output[]: Returns the output of the masked NN
 *              - share* node_value[]: Returns the output share of the output nodes
 *              - share_fixed input_r: Random used for the input shares of masked LuTs
 *              - s output_r: Random used for the output of SecReLU' gadget
 *              - share_fixed mul_r: Random for SecMUL gadget
 *              - share_fixed add_r: Random for SecADD gadget
 *              - share_fixed btoa_r0: Random for BtoA gadget
 *              - share_fixed btoa_r1: Random for BtoA gadget
 *              - share_fixed index_r: Fresh random for masking index
 **********************************************************************************/
void hiddenTooutput(
    share* x, share* w, share* bias, \
    share* output, share* node_value, share_fixed input_r, \
    s output_r, share_fixed mul_r, share_fixed add_r, share_fixed btoa_r0, \
    share_fixed btoa_r1, share_fixed index_r
)
{
	share z[n2];

    // z = <z-r, r> = x * w + b
	SecLinear(x, w, bias, z, n2, n1, mul_r);

    // Saves the output node values for comparison
    for(int i = 0; i < n2; i++)
    {
        node_value[i].d[0] = z[i].d[0];
        node_value[i].d[1] = z[i].d[1];
    }

    // output = <output-r, r> = SecArgmax(z)
	SecArgmax(z, output, input_r, output_r, mul_r, add_r, btoa_r0, btoa_r1, index_r);

}

/*****************************************************************************************
 * Name:        BNN_gadget_accuracy
 *
 * Description: Masked BNN inference for information loss calculation (See Section 6.1.4)
 *
 * Arguments:   - share* x[]: Input shares for Masked BNN
 *              - share* w1[]: Weight shares for masked input layer
 *              - share* w2[]: Weight shares for masked hidden layer
 *              - share* bias1[]: Bias shares for masked input layer
 *              - share* bias2[]: Biase shares for masked hidden layer
 *              - share* output[]: Returns the output of the masked NN
 *              - share_fixed* cnt[]: Returns the information loss of the masked NN
 *              - shared_fixed* answer_input_node: Output of input layer linear gadget
 *              - shared_fixed* answer_input_output: Output of input layer
 *              - shared_fixed* answer_hidden_node: Output of hidden layer linear gadget
 *              - shared_fixed* answer_hidden_output: Output of hidden layer
 ******************************************************************************************/
void BNN_gadget_accuracy(
    share* x, share* w1, share* w2, share* bias1, \
    share* bias2, share* output, share_fixed* cnt, share_fixed* answer_input_node, \
    share_fixed* answer_input_output, share_fixed* answer_hidden_node, \
    share_fixed* answer_hidden_output
)
{
    share_fixed tmp, mul_r, add_r, btoa_r0, btoa_r1, index_r;
    share_fixed input_32r = 0;
    share z[n1], node_value[n2], hidden_output[n1];
    s temp, output_r[(k-size)], input_r[(k-size)];
    int i;

    // Random for output of the masked LuTs
    for(int i = 0; i < ran_output; i++)
    {
        temp = rand();
        output_r[8 * i + 0] = temp & 0x1;
        output_r[8 * i + 1] = (temp >> 1) & 0x1;
        output_r[8 * i + 2] = (temp >> 2) & 0x1;
        output_r[8 * i + 3] = (temp >> 3) & 0x1;
        output_r[8 * i + 4] = (temp >> 4) & 0x1;
        output_r[8 * i + 5] = (temp >> 5) & 0x1;
        output_r[8 * i + 6] = (temp >> 6) & 0x1;
        output_r[8 * i + 7] = (temp >> 7) & 0x1;
    }

    // Random for input of the masked LuTs
    for (i = 0; i < ran_input; i++)
    {
        temp = rand();
        input_r[i * 4] = temp & mod;
        input_r[i * 4 + 1] = (temp >> mod) & mod;
        input_r[i * 4 + 2] = (temp >> (mod * 2)) & mod;
        input_r[i * 4 + 3] = (temp >> (mod * 3)) & mod;
    }

    // The input random of the masked LuTs are calculated at once
    for(int i = 0; i < (k/size); i++)
    {
        input_32r ^= (input_r[i] << (i * 2));
    }

    // Generate three kinds of LuTs (See Section 4.1)
    genLuT(input_r, output_r);

    // Fresh random for masked gadgets
    mul_r = rand();
    add_r = rand();
    btoa_r0 = rand();
    btoa_r1 = rand();
    index_r = rand();

    inputTohidden(x, w1, bias1,  z, hidden_output, input_32r, output_r[8 * ran_output -1], mul_r, btoa_r0, btoa_r1);

    // Compare unmasked function and the masked gadget
    for(int i = 0; i < n1; i++)
    {
        tmp = (share_fixed)hidden_output[i].d[0] + hidden_output[i].d[1];
        if(tmp != answer_input_node[i])
        {
            cnt[0]++;
        }
    }

    // Compare unmasked function and the masked gadget
    for(int i = 0; i < n1; i++)
    {
        tmp = (share_fixed)z[i].d[0] + z[i].d[1];
        if(tmp != answer_input_output[i])
        {
            cnt[1]++;
        }
    }
    
    hiddenTooutput(z, w2, bias2, output, node_value, input_32r, output_r[8 * ran_output -1], mul_r, add_r, btoa_r0, btoa_r1, index_r);

    // Compare unmasked function and the masked gadget
    for(int i = 0; i < 10; i++)
    {
        tmp = (share_fixed)node_value[i].d[0] + node_value[i].d[1];
        if(tmp != answer_hidden_node[i])
        {
            cnt[2]++;
        }
    }

    // Compare unmasked function and the masked gadget
    for(int i = 0; i < 1; i++)
    {
        tmp = (share_fixed)output[i].d[0] + output[i].d[1];
        if(tmp != answer_hidden_output[i])
           {
            cnt[3]++;
        }
    }
}

/*****************************************************************************************
 * Name:        BNN_gadget_clock
 *
 * Description: Masked BNN inference (See Section 6.1.3)
 *
 * Arguments:   - share* x[]: Input shares for Masked BNN
 *              - share* w1[]: Weight shares for masked input layer
 *              - share* w2[]: Weight shares for masked hidden layer
 *              - share* bias1[]: Bias shares for masked input layer
 *              - share* bias2[]: Biase shares for masked hidden layer
 *              - share* output[]: Returns the output of the masked NN
 *              - share_fixed* cnt[]: Returns the information loss of the masked NN
 *              - shared_fixed* answer_input_node: Output of input layer linear gadget
 *              - shared_fixed* answer_input_output: Output of input layer
 *              - shared_fixed* answer_hidden_node: Output of hidden layer linear gadget
 *              - shared_fixed* answer_hidden_output: Output of hidden layer
 ******************************************************************************************/
void BNN_gadget_clock(share* x, share* w1, share* w2, share* bias1, share* bias2, share* output)
{
    share_fixed mul_r, add_r, btoa_r0, btoa_r1, index_r;
    share_fixed input_32r = 0;
    share z[n1], node_value[n2], hidden_output[n1];
    s temp, output_r[(k-size)], input_r[(k-size)];
    int i;

    // Random for output of the masked LuTs
    for(int i = 0; i < ran_output; i++)
    {
        temp = rand();
        output_r[8 * i + 0] = temp & 0x1;
        output_r[8 * i + 1] = (temp >> 1) & 0x1;
        output_r[8 * i + 2] = (temp >> 2) & 0x1;
        output_r[8 * i + 3] = (temp >> 3) & 0x1;
        output_r[8 * i + 4] = (temp >> 4) & 0x1;
        output_r[8 * i + 5] = (temp >> 5) & 0x1;
        output_r[8 * i + 6] = (temp >> 6) & 0x1;
        output_r[8 * i + 7] = (temp >> 7) & 0x1;
    }

    // Random for input of the masked LuTs
    for (i = 0; i < ran_input; i++)
    {
        temp = rand();
        input_r[i * 4] = temp & mod;
        input_r[i * 4 + 1] = (temp >> mod) & mod;
        input_r[i * 4 + 2] = (temp >> (mod * 2)) & mod;
        input_r[i * 4 + 3] = (temp >> (mod * 3)) & mod;
    }

    // The input random of the masked LuTs are calculated at once
    for(int i = 0; i < (k/size); i++)
    {
        input_32r ^= (input_r[i] << (i * 2));
    }

    // Generate three kinds of LuTs (See Section 4.1)
    genLuT(input_r, output_r);

    // Fresh random for masked gadgets
    mul_r = rand();
    add_r = rand();
    btoa_r0 = rand();
    btoa_r1 = rand();
    index_r = rand();

    inputTohidden(x, w1, bias1,  z, hidden_output, input_32r, output_r[8 * ran_output -1], mul_r, btoa_r0, btoa_r1);
    
    hiddenTooutput(z, w2, bias2, output, node_value, input_32r, output_r[8 * ran_output -1], mul_r, add_r, btoa_r0, btoa_r1, index_r);
}