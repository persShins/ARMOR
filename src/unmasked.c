#include "type.h"

/*****************************************************************************************
 * Name:        unmask_input
 *
 * Description: Unmasked input layer of BNN, the hidden layer nodes are returned.
 *
 * Arguments:   - fixed_point* input[]: Inputs for input layer
 *              - fixed_point* weight[]: Weights for input layer
 *              - fixed_point* bias[]: Biases for input layer
 *              - fixed_point* output[]: Output of unmasked input layer
 *              - fixed_point cnt1: Number of nodes in input layer
 *              - fixed_point cnt2: Number of nodes in hidden layer
 ******************************************************************************************/
void unmask_input(
    fixed_point* input, fixed_point* weight, fixed_point* bias, \
    fixed_point* output, fixed_point cnt1, fixed_point cnt2
    )
{
    fixed_point result;
    for (int cnt_w = 0; cnt_w < cnt2; cnt_w++)
    {
        // x * w
        result = input[0] * weight[cnt_w * cnt1];
        for(int cnt_i = 1; cnt_i < cnt1; cnt_i++)
        {
            result += input[cnt_i] * weight[cnt_w * cnt1 + cnt_i];
        }

        // x * w + b
        result += bias[cnt_w];

        // ReLU'(x * w + b)
        output[cnt_w] = result >=0 ? 1 : -1;
    }
}

/*****************************************************************************************
 * Name:        unmask_input_accuracy
 *
 * Description: Unmasked input layer of BNN, the hidden layer nodes are returned.
 *              The results of each function are returned as well.
 *
 * Arguments:   - fixed_point* input[]: Inputs for input layer
 *              - fixed_point* weight[]: Weights for input layer
 *              - fixed_point* bias[]: Biases for input layer
 *              - fixed_point* output[]: Output of unmasked input layer
 *              - fixed_point cnt1: Number of nodes in input layer
 *              - fixed_point cnt2: Number of nodes in hidden layer
 *              - share_fixed* output1: Linear function output
 *              - share_fixed* output2: ReLU' function output
 ******************************************************************************************/
void unmask_input_accuracy(
    fixed_point* input, fixed_point* weight, fixed_point* bias, \
    fixed_point* output, fixed_point cnt1, fixed_point cnt2, share_fixed* output1, \
    share_fixed* output2
    )
{
    fixed_point result;
    for (int cnt_w = 0; cnt_w < cnt2; cnt_w++)
    {
        // x * w
        result = input[0] * weight[cnt_w * cnt1];
        for(int cnt_i = 1; cnt_i < cnt1; cnt_i++)
        {
            result += input[cnt_i] * weight[cnt_w * cnt1 + cnt_i];
        }

        // x * w + b
        result += bias[cnt_w];
        
        // Record the linear functions' output for comparison with Masked NN
        output1[cnt_w] = (share_fixed)result;
        
        // ReLU'(x * w + b)
        output[cnt_w] = result >=0 ? 1 : -1;

        // Record the output for comparison with Masked NN's hidden layer
        output2[cnt_w] = (share_fixed)output[cnt_w];
    }
}

/*****************************************************************************************
 * Name:        unmask_hidden
 *
 * Description: Unmasked hidden layer of BNN, the prediction of the label is returned
 *
 * Arguments:   - fixed_point* input[]: Inputs for hidden layer
 *              - fixed_point* weight[]: Weights for hidden layer
 *              - fixed_point* bias[]: Biases for hidden layer
 *              - fixed_point* output[]: Output of unmasked hidden layer
 *              - fixed_point* index[]: Prediction of unmasked BNN
 *              - fixed_point cnt1: Number of nodes in hidden layer
 *              - fixed_point cnt2: Number of nodes in output layer
 ******************************************************************************************/
void unmask_hidden(
    fixed_point* input, fixed_point* weight, fixed_point* bias, \
    fixed_point* output, fixed_point* index, fixed_point cnt1, \
    fixed_point cnt2
    )
{
    fixed_point result, max;
    max = -n;
    index[0] = 0;
    for (int cnt_w = 0; cnt_w < cnt2; cnt_w++)
    {
        // x * w
        result = input[0] * weight[cnt_w * cnt1];
        for (int cnt_i = 1; cnt_i < cnt1; cnt_i++)
        {
            result += input[cnt_i] * weight[cnt_w * cnt1 + cnt_i];
        }

        // x * w + b
        result += bias[cnt_w];

        // Result of the Output Layer
        output[cnt_w] = result;

        // Argmax(output_nodes)
        if (max < output[cnt_w])
        {
            max = output[cnt_w];
            index[0] = cnt_w;
        }
    }
}

/*****************************************************************************************
 * Name:        unmask_hidden
 *
 * Description: Unmasked hidden layer of BNN, the prediction of the label is returned
 *              The results of each function are returned as well.
 *
 * Arguments:   - fixed_point* input[]: Inputs for hidden layer
 *              - fixed_point* weight[]: Weights for hidden layer
 *              - fixed_point* bias[]: Biases for hidden layer
 *              - fixed_point* output[]: Output of unmasked hidden layer
 *              - fixed_point* index[]: Prediction of unmasked BNN
 *              - fixed_point cnt1: Number of nodes in hidden layer
 *              - fixed_point cnt2: Number of nodes in output layer
 *              - share_fixed* output1: Linear function output
 *              - share_fixed* output2: Argmax function output
 ******************************************************************************************/
void unmask_hidden_accuracy(
    fixed_point* input, fixed_point* weight, fixed_point* bias, fixed_point* output, \
    fixed_point* index, fixed_point cnt1, fixed_point cnt2, share_fixed* output1, \
    share_fixed* output2
    )
{
    fixed_point result, max;
    max = -n;
    index[0] = 0;
    for (int cnt_w = 0; cnt_w < cnt2; cnt_w++)
    {
        // x * w
        result = input[0] * weight[cnt_w * cnt1];
        for (int cnt_i = 1; cnt_i < cnt1; cnt_i++)
        {
            result += input[cnt_i] * weight[cnt_w * cnt1 + cnt_i];
        }

        // x * w + b
        result += bias[cnt_w];

        // Result of the Output Layer
        output[cnt_w] = result;

        // Record the linear functions' output for comparison with Masked NN
        output1[cnt_w] = output[cnt_w];

        // Argmax(output_nodes)
        if (max < output[cnt_w])
        {
            max = output[cnt_w];
            index[0] = cnt_w;
        }

        // Record the NN for comparison with Masked NN
        output2[0] = index[0];
    }
}