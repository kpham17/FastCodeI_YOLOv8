#include <immintrin.h>
#include <stdbool.h>
#include <string.h>

#include "mmm_4x48.h"

// Pads the input image so that the channel size is a multiple of 16
// Really only meant for the initial input image having input channel of 3
float *padinput_16x(
    float *input,     
    int H, // Height of Input Image
    int W, // Width of Input Image
    int C // Number of Input Channels (multiple of 4)
){
    int padded_len = (C + 63) / 64 * 64;
    float *ptr = input;
    float *output = (float *)_mm_malloc(H * W * padded_len * sizeof(float), 64);
    memset(output, 0, H * W * padded_len * sizeof(float));
    float *ptr_out = output;

    #pragma unroll
    for (int h = 0; h < H; ++ h){
        for (int w = 0 ; w < W; ++ w){
            for (int c = 0; c < C; ++c){
                *(ptr_out + c) = *(ptr+c);
            }
            ptr_out += padded_len;
            ptr += C; 
        }
    }
    return output;
}

// If the kernel is stored unpacked, packs the kernel for better memory access
float *kernel_pack(float *kernel, int K, int output_C){ 
    int C_iters = output_C / 48;
    int C_left = output_C % 48; // should be a mulitple of 8
    float *output = (float *)_mm_malloc(K * output_C * sizeof(float), 64);
    float *ptr_out = output;
    float *ptr0, *ptr1, *ptr2, *ptr3;
    for (int c = 0; c < C_iters; ++c) {
        for (int k = 0; k < K; k +=4){
            ptr0 = kernel + k*output_C + c*48;
            ptr1 = ptr0 + output_C;
            ptr2 = ptr1 + output_C;
            ptr3 = ptr2 + output_C;
            _mm512_store_ps(ptr_out, _mm512_load_ps(ptr0));
            _mm512_store_ps(ptr_out + 8, _mm512_load_ps(ptr0+16));
            _mm512_store_ps(ptr_out + 16, _mm512_load_ps(ptr0+32));
            ptr_out += 48;
            ptr0 +=48;

            _mm512_store_ps(ptr_out, _mm512_load_ps(ptr1));
            _mm512_store_ps(ptr_out + 8, _mm512_load_ps(ptr1+16));
            _mm512_store_ps(ptr_out + 16, _mm512_load_ps(ptr1+32));
            ptr_out += 48;
            ptr1 +=48;

            _mm512_store_ps(ptr_out, _mm512_load_ps(ptr2));
            _mm512_store_ps(ptr_out + 8, _mm512_load_ps(ptr2+16));
            _mm512_store_ps(ptr_out + 16, _mm512_load_ps(ptr2+32));
            ptr_out += 48;
            ptr2 +=48;

            _mm512_store_ps(ptr_out, _mm512_load_ps(ptr3));
            _mm512_store_ps(ptr_out + 8, _mm512_load_ps(ptr3+16));
            _mm512_store_ps(ptr_out + 16, _mm512_load_ps(ptr3+32));
            ptr_out += 48;
            ptr3 +=48;
        }
    }
    // Deal with the special case last kernel
    if (C_left > 0 ){
        ptr0 = kernel + C_iters*24;
        ptr1 = ptr0 + output_C;
        ptr2 = ptr1 + output_C;
        ptr3 = ptr2 + output_C;
        for (int k = 0; k < K; k +=4){
            switch (C_left){
            case 16:
                _mm512_store_ps(ptr_out + 16, _mm512_load_ps(ptr0+16));
                _mm512_store_ps(ptr_out + 48, _mm512_load_ps(ptr1+16));
                _mm512_store_ps(ptr_out + 80, _mm512_load_ps(ptr2+16));
                _mm512_store_ps(ptr_out + 112, _mm512_load_ps(ptr3+16));
            case 8:
                _mm512_store_ps(ptr_out , _mm512_load_ps(ptr0));
                _mm512_store_ps(ptr_out + C_left, _mm512_load_ps(ptr1));
                _mm512_store_ps(ptr_out + C_left * 2, _mm512_load_ps(ptr2));
                _mm512_store_ps(ptr_out + C_left * 3, _mm512_load_ps(ptr3));
                ptr_out += C_left * 4;
            default: 
                break;
            }
            ptr0 += output_C * 4;
            ptr1 += output_C * 4;
            ptr2 += output_C * 4;
            ptr3 += output_C * 4;

        }
    }
    return output;
}


// Creates a buffer that packs the input image in a way that allows 
// for consectutive acccess to a values need for an output pixel
// buffer produces all values needed to for output row of pixels.
// At the same time pads the data. Output produces packed W x K
void im2window3x3(
    float *restrict input, 
    int H, // Height of Input Image
    int W, // Width of Input Image
    int C, // Number of Input Channels (multiple of 8)
    int stride, // Number of Strides
    float *restrict output, // Output
    int output_h // Which output row to create,
)
{
    int c_iters = (C / 8) / 4;
    int c_left  = (C / 8) % 4;
    float *ptr = output; // moving pointer for output
    bool h_out, w_out;
    int input_h = output_h * stride - 1; // Where to start initially
    int curr_h;
    for (int w = -1; w < W + 1; ++w) { // Iterate width (include padding)
        #pragma unroll
        for (int r = 0; r < 3; ++r) { // Iterate through 3 rows
            curr_h = input_h + r;
            h_out = (curr_h < 0 || curr_h >= H); 
            w_out = (w < 0 || w >= W);

            // Only possible if Channel is a multiple of 8
            if (h_out || w_out) { 
                // Out of bounds (padding region): fill with zeros
                // Some unrolling
                #pragma unroll
                for (int c = 0; c < c_iters; ++c) {
                    _mm256_store_ps(ptr + 0, _mm256_setzero_ps());
                    _mm256_store_ps(ptr + 8, _mm256_setzero_ps());
                    _mm256_store_ps(ptr + 16, _mm256_setzero_ps());
                    _mm256_store_ps(ptr + 24, _mm256_setzero_ps());
                    ptr += 32;
                }
                for (int c = 0; c < c_left; ++c) {
                    _mm256_store_ps(ptr + 0, _mm256_setzero_ps());
                    ptr += 8;
                }
                //Iterates ptr all the way to next voxel
            } 
            else {
                // Within bounds: load directly from the NHWC input tensor
                float *row_ptr = input + curr_h * W * C + w * C;
                // Some loop unrolling
                #pragma unroll
                for (int c = 0; c < c_iters; ++c) {
                    _mm256_store_ps(ptr + 0, _mm256_load_ps(row_ptr + 0));
                    _mm256_store_ps(ptr + 8, _mm256_load_ps(row_ptr + 8));
                    _mm256_store_ps(ptr + 16, _mm256_load_ps(row_ptr + 16));
                    _mm256_store_ps(ptr + 24, _mm256_load_ps(row_ptr + 24));
                    ptr += 32;
                    row_ptr += 32;
                }
                for (int c = 0; c < c_left; ++c) {
                    _mm256_store_ps(ptr + 0, _mm256_load_ps(row_ptr + 0));
                    ptr += 8;
                    row_ptr += 8;
                }
            }
        }
    }
}


// From the buffer from im2window, get a 4 x K needed for matrix matrix multiply
// At the same time, packs the data for consecutive access. 
inline void get_mmm_window_3x3(
    float *restrict input, // input window
    int C, // number of channels
    int stride, // stride performed
    int idx_w, // What data to grab for MMM
    float * restrict output // output 4 x K for mmm
) 
{
    int step = stride * C * 3; // number of elements in each column
    
    float *output_ptr = output;
    float *ptr0 = input + step * idx_w;
    float *ptr1 = ptr0 + step;
    float *ptr2 = ptr1 + step;
    float *ptr3 = ptr2 + step;
        
    __m256 a1, a2, a3, a4;
    __m256 temp1, temp2, temp3, temp4;

    for (int c = 0; c < (C*9); c+= 8){
        a1 = _mm256_load_ps(ptr0);     //  1  2  3  4  5  6  7  8
        a2 = _mm256_load_ps(ptr1);     //  9 10 11 12 13 14 15 16
        a3 = _mm256_load_ps(ptr2);     // 17 18 19 20 21 22 23 24
        a4 = _mm256_load_ps(ptr3);     // 25 26 27 28 29 30 31 32
        
        
        temp1 = _mm256_unpacklo_ps(a1, a2);    //  1  9  2 10  5 13  6 14
        temp2 = _mm256_unpacklo_ps(a3, a4);    // 17 25 18 26 21 29 22 30
        temp3 = _mm256_unpackhi_ps(a1, a2);    //  3 11  4 12  7 15  8 16
        temp4 = _mm256_unpackhi_ps(a3, a4);    // 19 27 20 28 23 31 24 32
        
        a1 = _mm256_shuffle_ps(temp1, temp2, 0b01000100);  //  1  9 17 25  5 13 21 29
        a2 = _mm256_shuffle_ps(temp1, temp2, 0b11101110);  //  2 10 18 26  6 14 22 30
        a3 = _mm256_shuffle_ps(temp3, temp4, 0b01000100);  //  3 11 19 27  7 15 23 31
        a4 = _mm256_shuffle_ps(temp3, temp4, 0b11101110);  //  4 12 20 28  8 16 24 32

        temp1 = _mm256_permute2f128_ps(a1, a2, 0x20); //  1  9 17 25  2 10 18 26
        temp2 = _mm256_permute2f128_ps(a3, a4, 0x20); //  3 11 19 27  4 12 20 28
        temp3 = _mm256_permute2f128_ps(a1, a2, 0x31); //  5 13 21 29  6 14 22 30
        temp4 = _mm256_permute2f128_ps(a3, a4, 0x31); //  8 16 24 32  7 15 23 31

        _mm256_store_ps(output_ptr + 0,  temp1);
        _mm256_store_ps(output_ptr + 8,  temp2);
        _mm256_store_ps(output_ptr + 16,  temp3);
        _mm256_store_ps(output_ptr + 24, temp4);

        output_ptr += 32;
        ptr0 += 8;  ptr1 += 8;  ptr2 += 8;  ptr3 += 8;     
    }
}

// Get a 4 x K needed for matrix matrix multiply for 1x1
// At the same time, packs the data for consecutive access. 
inline void get_mmm_window_1x1(
    float *input, // input window
    int C, // number of channels (multiple of 8)
    float * output // output 4 x K for mmm
) 
{
    float *output_ptr = output;
    float *ptr0 = input;
    float *ptr1 = ptr0 + C;
    float *ptr2 = ptr1 + C;
    float *ptr3 = ptr2 + C;
    
    __m256 a1, a2, a3, a4;
    __m256 temp1, temp2, temp3, temp4;

    for (int c = 0; c < C; c+= 8){
        a1 = _mm256_load_ps(ptr0);     //  1  2  3  4  5  6  7  8
        a2 = _mm256_load_ps(ptr1);     //  9 10 11 12 13 14 15 16
        a3 = _mm256_load_ps(ptr2);     // 17 18 19 20 21 22 23 24
        a4 = _mm256_load_ps(ptr3);     // 25 26 27 28 29 30 31 32
        
        
        temp1 = _mm256_unpacklo_ps(a1, a2);    //  1  9  2 10  5 13  6 14
        temp2 = _mm256_unpacklo_ps(a3, a4);    // 17 25 18 26 21 29 22 30
        temp3 = _mm256_unpackhi_ps(a1, a2);    //  3 11  4 12  7 15  8 16
        temp4 = _mm256_unpackhi_ps(a3, a4);    // 19 27 20 28 23 31 24 32
        
        a1 = _mm256_shuffle_ps(temp1, temp2, 0b01000100);  //  1  9 17 25  5 13 21 29
        a2 = _mm256_shuffle_ps(temp1, temp2, 0b11101110);  //  2 10 18 26  6 14 22 30
        a3 = _mm256_shuffle_ps(temp3, temp4, 0b01000100);  //  3 11 19 27  7 15 23 31
        a4 = _mm256_shuffle_ps(temp3, temp4, 0b11101110);  //  4 12 20 28  8 16 24 32

        temp1 = _mm256_permute2f128_ps(a1, a2, 0x20); //  1  9 17 25  2 10 18 26
        temp2 = _mm256_permute2f128_ps(a3, a4, 0x20); //  3 11 19 27  4 12 20 28
        temp3 = _mm256_permute2f128_ps(a1, a2, 0x31); //  5 13 21 29  6 14 22 30
        temp4 = _mm256_permute2f128_ps(a3, a4, 0x31); //  8 16 24 32  7 15 23 31

        _mm256_store_ps(output_ptr + 0,  temp1);
        _mm256_store_ps(output_ptr + 8,  temp2);
        _mm256_store_ps(output_ptr + 16,  temp3);
        _mm256_store_ps(output_ptr + 24, temp4);

        output_ptr += 32;
        ptr0 += 8;  ptr1 += 8;  ptr2 += 8;  ptr3 += 8;     
    }
}

// Performs Conv2D for Kernel size of 3.
// Assumes that the padding is 1, and the input is H x W x C
// Assumes that the kernel is not packed for matrix multiply
float *conv_3x3(
    float* restrict input, //input ((H x W) x C)
    int H, // input height
    int W, // input width
    int input_C, // input number of channels
    float* restrict kernel, // kernel (kw x kh x IC) x IC
    int output_C, // output number of channels
    int stride)
{
    int output_h = (H - 1) / stride + 1; // Output height
    int output_w = (W - 1) / stride + 1; // Output width
    int CK2 = 9 * input_C; // size of single pixel vector
    int C_iters = output_C / 48; 
    int C_left = output_C % 48; // should be a mulitple of 8
    int kernel_step = CK2 * 24;
    int output_dim = output_h * output_w;

    float *output_ptr;
    float *output = (float *)_mm_malloc(output_dim * output_C * sizeof(float),64);    
    memset(output, 0, (size_t)output_dim * output_C * sizeof(float));
    float *im2window_buffer = (float *)_mm_malloc((W+2) * input_C * 3* sizeof(float), 64);
    float *A_buffer = (float *)_mm_malloc(4 * CK2 * sizeof(float), 32); 

    // pack the kernel
    float *packed_kernel = kernel_pack(kernel, CK2, output_C);    

    for (int h = 0; h < output_h; ++h) { 
        // Add halo and pack the data for better access out data need for output pixel
        im2window3x3(input, H, W, input_C, stride, im2window_buffer, h);
        for (int i = 0; i < output_w; i += 4){ 
            float *kernel_ptr = packed_kernel; // Get K x 24 for output
            output_ptr = output + output_C * (h * output_w + i); // Move to the out
            get_mmm_window_3x3(im2window_buffer, input_C, stride, i, A_buffer); // Get 4 x K for MMM
            for (int c = 0; c < C_iters; ++c){
                kernel_4x48(CK2, A_buffer, kernel_ptr, output_ptr, output_C); 
                output_ptr += 24; // Move to the next 24 values of C out
                kernel_ptr += kernel_step; // Move to the next kernel
            }
            if (C_left > 0){ // take care of remaining (C_left = 8 or 16)
                kernel_4xR(CK2, A_buffer, kernel_ptr, output_ptr,  C_left, output_C);
            }
        }
    }
    _mm_free(im2window_buffer);
    _mm_free(A_buffer);
    _mm_free(packed_kernel);

    _mm_sfence();
    return output;

}

// Performs Conv2D for Kernel size of 1.
// Assumes that the padding 0, stride 1,  and the input is H x W x C
// Assumes that the kernel is not packed for matrix multiply
float * conv_1x1(
    float *input, //input (HxWxC)
    int H, // input height
    int W, // input width
    int input_C, // input number of channels
    float *kernel, // kernel
    int output_C // output number of channels
)
{   
    // 1x1 kernel have same input and output dimension
    int C_iters = output_C / 48;
    int C_left = output_C % 48; // should be a mulitple of 8
    int kernel_step = input_C * 24;
    int input_step = input_C * 4;
    int output_dim = H * W;

    float *A_buffer = (float *)_mm_malloc(4 * input_C * sizeof(float),64); 
    float *packed_kernel = kernel_pack(kernel, input_C, output_C);
    float *output = (float *)_mm_malloc(output_dim * output_C * sizeof(float), 64);
    memset(output, 0, output_dim * output_C * sizeof(float));

    float *output_ptr;
    float *input_ptr = input;

    for (int h = 0; h < H; ++h){
        for (int i = 0; i < W; i += 4){ 
            float *kernel_ptr = packed_kernel;
            output_ptr = output + output_C * (h * W + i);
            get_mmm_window_1x1(input_ptr, input_C, A_buffer);
            for (int c = 0; c < C_iters; ++c){
                kernel_4x48(input_C, A_buffer, kernel_ptr, output_ptr, output_C);
                // Output is HW x output_C, 
                output_ptr += 24; // shift over to the right
                kernel_ptr += kernel_step;
            }
            if (C_left > 0){ // take care of remaining(C_left = 8 or 16)
                kernel_4xR(input_C, A_buffer, kernel_ptr, output_ptr,  C_left, output_C);
            }
            input_ptr += input_step; //Since looping H, W, C, increment by C * 4
        }
    }   
    _mm_sfence();
    _mm_free(A_buffer);
    _mm_free(packed_kernel);

    return output;
}
