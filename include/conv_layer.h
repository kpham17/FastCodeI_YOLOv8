#include "silu.h"
#include "conv2D.h"


float *conv_layer(
    float* input, 
    int H, // input height
    int W, // input width
    int input_C, // input channel size
    float *kernel, 
    int kernel_size, // 3 or 1 
    int output_C, // output channel size (multiple of 8)
    int stride // 1 or 2
)
{    
    float *w_input = input;
    if (input_C % 8 != 0){
        w_input = padinput_8x(input, H, W, input_C);
    } 

    float *conv2d_output;
    if (kernel_size == 3){ // Assumes that padding = 1 for kernel_size = 3
        conv2d_output = conv_3x3(w_input, H, W, input_C, kernel, output_C, stride);
    }
    else if (kernel_size == 1){ // Assumes that padding = 0 and stride = 1 for kernel_size = 1
        conv2d_output = conv_1x1(w_input, H, W, output_C,kernel, output_C);
    }

    if (input_C % 8 != 0){
        _mm_free(w_input);
    }

    float *batchnorm_output;
    batchnorm(conv2d_output, H, W, output_C); //placeholder

    float *output;
    output = fast_SILU(conv2d_output, H, W, output_C);
    _mm_free(conv2d_output);
    mm_free(batchnorm_output);
    
    return output;
}


