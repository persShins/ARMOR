#include "nn_subgadget.h"

// Define ILuT, cLuT and sLuT
s iLuT[entry][entry];
s cLuT[num_cLuT][entry][2][entry];
s sLuT[entry][2][entry];

/*****************************************************************************************
 * Name:        SecSub
 *
 * Description: Masked subtraction with input_a and input_b utilizing add_r
 *              output = input_a - input_b
 *
 * Arguments:   - share* input_a: First masked input share
 *              - share* input_b: Second masked input share
 *              - share* output: Masked output share
 *              - share* add_r: Random for the SecSub gadget
 ******************************************************************************************/
void SecSub(share* input_a, share* input_b, share* output, share_fixed add_r)
{
    output[0].d[0] = add_r;
    output[0].d[1] = - output[0].d[0];
    
    output[0].d[0] += input_a[0].d[0];
    output[0].d[0] -= input_b[0].d[0];
    
    output[0].d[0] += input_a[0].d[1];
    output[0].d[0] -= input_b[0].d[1];
}

/*****************************************************************************************
 * Name:        SecADD
 *
 * Description: Masked addition with input_a and input_b utilizing add_r
 *              output = input_a + input_b
 *
 * Arguments:   - share* input_a: First masked input share
 *              - share* input_b: Second masked input share
 *              - share* output: Masked output share
 *              - share* add_r: Random for the SecADD gadget
 ******************************************************************************************/
void SecADD(share* input_a, share* input_b, share *output, share_fixed add_r)
{
    output[0].d[0] = add_r;
    output[0].d[1] = - output[0].d[0];
    
    output[0].d[0] += input_a[0].d[0];
    output[0].d[0] += input_b[0].d[0];
    
    output[0].d[0] += input_a[0].d[1];
    output[0].d[0] += input_b[0].d[1];
}

/*****************************************************************************************
 * Name:        SecMUL
 *
 * Description: Masked multiplication with input_a and input_b utilizing add_r
 *              output = input_a * input_b
 *
 * Arguments:   - share* input_a: First masked input share
 *              - share* input_b: Second masked input share
 *              - share* output: Masked output share
 *              - share* add_r: Random for the SecMUL gadget
 ******************************************************************************************/
void SecMUL(share* input_a, share* input_b, share* output, share_fixed mul_r)
{
    share_fixed mul1, mul2;
    output[0].d[0] = mul_r;
    output[0].d[1] = -output[0].d[0];

    mul1 = input_a[0].d[1] * input_b[0].d[1];
    mul2 = input_a[0].d[0] * input_b[0].d[0];

    output[0].d[0] += mul1;
    output[0].d[0] += mul2;

    mul1 = input_a[0].d[1] * input_b[0].d[0];
    mul2 = input_a[0].d[0] * input_b[0].d[1];

    output[0].d[1] += mul1;
    output[0].d[1] += mul2;
    
}

/*****************************************************************************************
 * Name:        SecReLU_prime
 *
 * Description: Masked Activation function for BNN (i.e. SecReLU')
 *
 * Arguments:   - share* x: Masked input share for BNN activation function
 *              - share_fixed input_r: Input mask for the SecReLU' gadget
 *              - s output_r: Output mask for the SecReLU' gadget
 *              - share_fixed m: The number of activation functions to execute
 ******************************************************************************************/
void SecReLU_prime(share* x, share_fixed input_r, s output_r, share_fixed m)
{
    share_fixed input0, input1, carry, temp;
    int cnt, cnt_size;
    for (cnt = 0; cnt < m; cnt++)
    {
        temp = x[cnt].d[0] ^ input_r;
        
        input0 = temp & mod;
        input1 = x[cnt].d[1] & mod;
        
        carry = iLuT[input0][input1];
        for (cnt_size = 1; cnt_size < num_cLuT + 1; cnt_size++)
        {
            input0 = (temp >> (2 * cnt_size)) & mod;
            input1 = (x[cnt].d[1] >> (2 * cnt_size)) & mod;
            carry = cLuT[cnt_size - 1][input0][carry][input1];
        }

        input0 = (temp >> (k - 2)) & mod;
        input1 = (x[cnt].d[1] >> (k - 2)) & mod;
        x[cnt].d[0] = sLuT[input0][carry][input1];
        x[cnt].d[1] = output_r;
    }
}

/*****************************************************************************************
 * Name:        SecLinear
 *
 * Description: Masked fully-connected layer for BNN
 *
 * Arguments:   - share* input: Masked input shares for SecLinear gadget
 *              - share* weight: Masked weight shares for SecLinear gadget
 *              - share* bias: Masked biase shares for SecLinear gadget
 *              - share* output: Masked output share
 *              - uint32_t cnt1: The number of input layer 
 *              - uint32_t cnt2: The number of output layer
 *              - share_fixed mul_r: Random for SecMUL gadget
 ******************************************************************************************/
void SecLinear(
    share* input, share* weight, share* bias, share* output, \
    uint32_t cnt1, uint32_t cnt2, share_fixed mul_r
)
{
    share_fixed mul1, mul2;
    uint32_t index;
    share z;

    for (int cnt_w = 0; cnt_w < cnt1; cnt_w++)
    {
        z.d[0] = mul_r;
        z.d[1] = -z.d[0];
        for (int cnt_i = 0; cnt_i < cnt2; cnt_i++)
        {
            //index for weight
            index = cnt_w * cnt2 + cnt_i;

            mul1 = input[cnt_i].d[1] * weight[index].d[1];
            mul2 = input[cnt_i].d[0] * weight[index].d[0];

            z.d[0] += mul1;
            z.d[0] += mul2;

            mul1 = input[cnt_i].d[1] * weight[index].d[0];
            mul2 = input[cnt_i].d[0] * weight[index].d[1];

            z.d[1] += mul1;
            z.d[1] += mul2;
        }
        output[cnt_w].d[0] = z.d[0] + bias[cnt_w].d[0];
        output[cnt_w].d[1] = z.d[1] + bias[cnt_w].d[1];
    }
}

/*****************************************************************************************
 * Name:        BtoA
 *
 * Description: Boolean to Arithmetic conversion for boolean input share
 *
 * Arguments:   - share* x: Masked inputs for BtoA conversion gadget
 *              - share* output: Masked arithmetic output share
 *              - share_fixed* dim: The number of masked inputs
 *              - share_fixed* btoa_r0: Random for BtoA conversion gadget
 *              - share_fixed btoa_r1: Random for BtoA conversion gadget
 ******************************************************************************************/
void BtoA(
    share* x, share* output, share_fixed dim, share_fixed btoa_r0, \
    share_fixed btoa_r1
    )
{
    share_fixed t, r_temp, u, u_temp;
    for (int cnt = 0; cnt < dim; cnt++)
    {
        x[cnt].d[0] ^= btoa_r0;
        x[cnt].d[1] ^= btoa_r0;
        
        t = x[cnt].d[0] ^ btoa_r1;
        t = t - btoa_r1;
        t = t ^ x[cnt].d[0];
        r_temp = btoa_r1 ^ x[cnt].d[1];
        u = -r_temp;
        u_temp = x[cnt].d[0] ^ r_temp;
        u = u + u_temp;
        u = u ^ t;
        output[cnt].d[0] = u;
        output[cnt].d[1] = x[cnt].d[1];
    }
}

/*****************************************************************************************
 * Name:        SecReLU_prime_cmp
 *
 * Description: A single masked input for the SecReLU' gadget (output layer only).
 *
 * Arguments:   - share* x: The input for the gadget
 *              - share* output: Masked output share of SecReLU' gadget
 *              - s* output_r: Random for the output share
 ******************************************************************************************/
void SecReLU_prime_cmp(share* x, share* output, s output_r)
{
    s input0, input1, carry;
    int cnt_size;
    
    input0 = x[0].d[0] & mod;
    input1 = x[0].d[1] & mod;

    carry = iLuT[input0][input1];
    for (cnt_size = 1; cnt_size < num_cLuT + 1; cnt_size++)
    {
        input0 = (x[0].d[0] >> (2 * cnt_size)) & mod;
        input1 = (x[0].d[1] >> (2 * cnt_size)) & mod;
        carry = cLuT[cnt_size - 1][input0][carry][input1];
    }

    input0 = (x[0].d[0] >> (k - 2)) & mod;
    input1 = (x[0].d[1] >> (k - 2)) & mod;
    output[0].d[0] = sLuT[input0][carry][input1];
    output[0].d[1] = output_r;
}

/*****************************************************************************************
 * Name:        SecArgmax
 *
 * Description: A single masked input for the SecReLU' gadget (output layer only).
 *
 * Arguments:   - share* input: The input share the SecArgmax gadget
 *              - share* max_index: Masked output share of SecArgmax gadget
 *              - share_fixed input_r: Input share mask for SecReLU' gadget
 *              - s output_r: Random for the output share of SecReLU' gadget
 *              - share_fixed mul_r: Random for SecMUL gadget
 *              - share_fixed add_r: Random for SecADD gadget
 *              - share_fixed btoa_r0: Random for BtoA conversion gadget
 *              - share_fixed btoa_r1: Random for BtoA conversion gadget
 *              - share_fixed index_r: Random for the index share
 ******************************************************************************************/
void SecArgmax(share* input, share* max_index, share_fixed input_r, s output_r, \
        share_fixed mul_r, share_fixed add_r, share_fixed btoa_r0, share_fixed btoa_r1, \
        share_fixed index_r
    )
{
    share_fixed i;
    share temp[1], max_temp[1], ReLu_temp[1], sign_arith[1], sign_boolean[1], output[1];
    share index[n2], temp_index[1], max_temp_index[1];
    for (i = 0; i < n2; i++)
    {
        index[i].d[0] = index_r;
        index[i].d[1] = i - index[i].d[0];
    }
    output[0].d[0] = input[0].d[0];
    output[0].d[1] = input[0].d[1];

    max_index[0].d[0] = index[0].d[0];
    max_index[0].d[1] = index[0].d[1];

    for (i = 1; i < n2; i++)
    {
        SecSub(output, (input + i), temp, add_r);
        SecSub(max_index, (index + i), temp_index, add_r);
        ReLu_temp[0].d[0] = temp[0].d[0] ^ input_r;
        ReLu_temp[0].d[1] = temp[0].d[1];

        SecReLU_prime_cmp(ReLu_temp, sign_boolean, output_r);
        BtoA(sign_boolean, sign_arith, 1, btoa_r0, btoa_r1);

        SecMUL(temp, sign_arith, max_temp, mul_r);
        SecMUL(temp_index, sign_arith, max_temp_index, mul_r);

        SecADD((input + i), max_temp, output, add_r);
        SecADD((index + i), max_temp_index, max_index, add_r);
    }
    
}

/*****************************************************************************************
 * Name:        genLuT
 *
 * Description: Building the masked LuT (e.g. ILuT, cLuT, sLuT) with input and output random
 *
 * Arguments:   - s* input_r: The input masks for the LuTs
 *              - s* output_r: The output masks for the LuTs
 ******************************************************************************************/
void genLuT(s* input_r, s* output_r)
{
    uint8_t x_temp, temp, carry_temp;

    //generating initial LuT
    //output: < (carry ^ output_r), output_r>
    for (int a = 0; a < entry; a++)
    {
        x_temp = a ^ input_r[0];

        for (int b = 0; b < entry; b++)
        {
            temp = ((x_temp + b) >> 2) & 1;
            iLuT[a][b] = temp ^ output_r[0];
        }
    }
    //generating carry LuT with carry bit
    //output: < (carry ^ output_r), output_r>
    for (int i = 0; i < num_cLuT; i++)
    {
        for (int a = 0; a < entry; a++)
        {
            x_temp = a ^ input_r[i + 1];
            for (int carry = 0; carry < 2; carry++)
            {
                carry_temp = carry ^ output_r[i];
                for (int b = 0; b < entry; b++)
                {
                    temp = ((x_temp + b + carry_temp) >> 2) & 1;
                    cLuT[i][a][carry][b] = temp ^ output_r[i + 1];
                }
            }
        }
    }
    //generating sign LuT with carry bit
    //output: < (sign ^ output_r), output_r>
    for (int a = 0; a < entry; a++)
    {
        x_temp = a ^ input_r[num_cLuT + 1];
        for (int carry = 0; carry < 2; carry++)
        {
            carry_temp = carry ^ output_r[num_cLuT];
            for (int b = 0; b < entry; b++)
            {
                temp = ((x_temp + b + carry_temp) >> 1) & 1;
                temp = temp ^ 1;
                sLuT[a][carry][b] = temp ^ output_r[num_cLuT + 1];
            }
        }
    }
}