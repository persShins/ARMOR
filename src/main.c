#include "main.h"
#include <time.h>
#include "read_mnist.h"
#include <math.h>
/* Shares of ARMOR */
share minput[n], mweight1[n * n1], mweight2[n1 * n2], mbias1[n1], mbias2[n2];
uint8_t r_input[n];

/* Variables for Unmasked BNN */
int32_t rtrain[n], rweight1[n * n1], rweight2[n1 * n2], rbias1[n1], rbias2[n2];

/*********************************************************************************
 * Name:        float_to_fixed8
 *
 * Description: Casting input from double to unsigned int8
 *
 * Arguments:   - double a: Input to cast
 **********************************************************************************/
uint8_t float_to_fixed8(double a)
{
	float z = a * (1 << FP_FRACTIONAL_BITS);
	uint8_t res = (uint8_t)round(z);
	return res;
}

/*********************************************************************************
 * Name:        fixed8_to_fixed_point
 *
 * Description: Casting variable for inference from unsigned int8 to fixed_point
 *
 **********************************************************************************/
void fixed8_to_fixed_point()
{
    int8_t temp;
    for (int c = 0; c < n; c++)
    {
        temp = r_input[c];
        rtrain[c] = (fixed_point)temp;
    }
    for (int c = 0; c < n * n1; c++)
    {
        temp = bfirst_layer_weight[c];
        rweight1[c] = (fixed_point)temp;
    }
    for (int c = 0; c < n1 * n2; c++)
    {
        temp = bsecond_layer_weight[c];
        rweight2[c] = (fixed_point)temp;
    }

    for (int c = 0; c < n1; c++)
    {
        temp = fc1_bias_fp[c];
        rbias1[c] = (fixed_point)temp;
    }

    for (int c = 0; c < n2; c++)
    {
        temp = fc2_bias_fp[c];
        rbias2[c] = (fixed_point)temp;
    }
}

/*********************************************************************************
 * Name:        run_unmasked_accuracy
 *
 * Description: Evaluates each unmasked NN function 
 *              with the input and stores the results.
 *
 * Arguments:   - int32_t* rtrain: 10k or less MNIST Dataset
 *              - shared_fixed output1[]: Result of Hidden Layer's linear function
 *              - shared_fixed output2[]: Result of Hidden Layer
 *              - shared_fixed output3[]: Result of Output Layer's linear function
 *              - shared_fixed output4[]: Result of Output Layer (i.e. label)
 **********************************************************************************/
void run_unmasked_accuracy(int32_t* rtrain, share_fixed* output1, share_fixed* output2, share_fixed* output3, share_fixed* output4)
{
    int32_t z[n1], outputs[n2], index;

    hidden(rtrain, rweight1, rbias1, z, n, n1, output1, output2);

    output(z, rweight2, rbias2, outputs, &index, n1, n2, output3, output4);
}

/*********************************************************************************
 * Name:        run_unmasked_accuracy
 *
 * Description: Computes the unmasked NN functions for the given input rtrain.
 *              The Argamx index is returned.
 *
 * Arguments:   - int32_t* rtrain: 10k MNIST Dataset
 **********************************************************************************/
int32_t run_unmasked(int32_t* rtrain)
{
    int32_t z[n1], outputs[n2], index;

    hidden(rtrain, rweight1, rbias1, z, n, n1);

    output(z, rweight2, rbias2, outputs, &index, n1, n2);

    return index;
}

/*********************************************************************************
 * Name:        run_accuracy
 *
 * Description: Evaluates information loss between unmasked BNN and our BNN
 *
 * Arguments:   - int32_t counter: The number of MNIST dataset (less then 10k)
 **********************************************************************************/
void run_accuracy(int32_t counter)
{
    srand(time(NULL));

    // The results of the unmasked functions are stored in each array
    share_fixed output1[n1];
    share_fixed output2[n1];
    share_fixed output3[n2];
    share_fixed output4[1];

    int cnt_input;

    // The information loss is stored here
    share_fixed cnt[5] ={0,0,0,0,0};

    for(int i = 0; i < counter; i++)
    {
        for(cnt_input = 0 ; cnt_input < n; cnt_input++)
            r_input[cnt_input] = float_to_fixed8(test_image[i][cnt_input]);

        /* 1. Unmasked NN */
        // Values of inputs, weights, and biases are converted from one type to another
        fixed8_to_fixed_point();
        
        // Process the unmasked NN and record the outputs
        run_unmasked_accuracy(rtrain, output1, output2, output3, output4);

        /* 2. Masked NN */
        // Values of inputs, weights, and biases are masked with fresh_r 
        share_fixed fresh_r = rand();
        mask_twoshare(r_input, n, minput, fresh_r);

        fresh_r = rand();
        mask_twoshare(bfirst_layer_weight, (n * n1), mweight1, fresh_r);

        fresh_r = rand();
        mask_twoshare(fc1_bias_fp, n1, mbias1, fresh_r);

        fresh_r = rand();
        mask_twoshare(bsecond_layer_weight, (n1 * n2), mweight2, fresh_r);

        fresh_r = rand();
        mask_twoshare(fc2_bias_fp, n2, mbias2, fresh_r);
        
        share outputs[1];

        // Process the masked NN and compare each gadgets
        BNN_gadget(minput, mweight1, mweight2, mbias1, mbias2, outputs, cnt, output1, output2, output3, output4);

        // Compare the result with the label
        if(test_label[i] != output4[0])
        {
            cnt[4] += 1;
        }

    }

    // Shows the information loss
    printf("hidden layer node loss: %d\n", cnt[0]);
    printf("hidden layer loss: %d\n", cnt[1]);
    printf("output layer loss: %d\n", cnt[2]);
    printf("output loss: %d\n", cnt[3]);
    printf("label loss: %d\n", cnt[4]);
}

/*********************************************************************************
 * Name:        run_clock
 *
 * Description: Evaluates the mean clock cycle for masked BNN
 *
 * Arguments:   - int32_t counter: The number of MNIST dataset (less then 10k)
 **********************************************************************************/
void run_clock(int counter)
{
    srand(time(NULL));
    uint64_t start, end, avg = 0;
    int cnt_input, cnt = 0;
    
    for(int i = 0; i < counter; i++)
    {
        for(cnt_input = 0; cnt_input < 784; cnt_input++)
        {
            r_input[cnt_input] = float_to_fixed8(test_image[i][cnt_input]);
        }

        start = get_cycles();
        
        share outputs[1];

        share_fixed fresh_r = rand();
        mask_twoshare(r_input, n, minput, fresh_r);

        fresh_r = rand();
        mask_twoshare(bfirst_layer_weight, n * n1, mweight1, fresh_r);

        fresh_r = rand();
        mask_twoshare(fc1_bias_fp, n1, mbias1, fresh_r);

        fresh_r = rand();
        mask_twoshare(bsecond_layer_weight, n1 * n2, mweight2, fresh_r);
    
        fresh_r = rand();
        mask_twoshare(fc2_bias_fp, n2, mbias2, fresh_r);

        BNN_gadget(minput, mweight1, mweight2, mbias1, mbias2, outputs);

        end = get_cycles();

        avg += (end - start);

        if((outputs[0].d[0] + outputs[0].d[1]) == test_label[i])
            cnt++;
    }
    
    double accuracy = (double)cnt/counter * 100;
    printf("Accuracy of the masked BNN: %.2f%%\n", accuracy);
    
    printf("Clock cycles: %lu\n", avg/counter);
}

int main(int argc, char *argv[]) {
    load_mnist_test();

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <mode>\n", argv[0]);
        fprintf(stderr, "  mode=1: run_accuracy (information loss)\n");
        fprintf(stderr, "  mode=0: run_clock (mean cycles)\n");
        return 1;
    }

    int mode = atoi(argv[1]);
    int counter = atoi(argv[2]);
    
    // Counter should be at most 10k
    if(counter > 10000)
    {
        fprintf(stderr, "Invalid Counter: %s (use less than 10k)\n", argv[2]);
        return 1;
    }
    //Information loss analysis for Masked BNN (784-512-10)
    if(mode == 1) 
    {
        run_accuracy(counter);
    } 

    //Clock cycle measurement for BNN (784-512-10)
    else if(mode == 0) 
    {
        run_clock(counter);
    } 
    else 
    {
        fprintf(stderr, "Invalid mode: %s (use 1 or 0)\n", argv[1]);
        return 1;
    }
    return 0;
}