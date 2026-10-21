#include <immintrin.h>

// Schraudolph exponential approximation
// Magic multiplier: 2^23 / ln(2)
#define MULTIPLIER 12102203.0f
// Exponent bias adjustment offset (127 * 2^23 - C)
// (127 << 23)
#define BIAS_OFFSET 1065353216
#define C_OFFSET 60801

// Addition  latency 4, throughput 0.5 
// Division  slow! latency 11 throughput 5
// Use reciprocal instead(Latency 4) throughput 1
// Multiplication latency 4 throughput 0.5 
// Convert latency 4 throughput 0.5
// bottleneck is the reciprocal which requires 4 independent instructions to maximize
// Need to fix to do 8 things at once

// Sigmoid using Schraudolph exponential approximation
// Assuem C is multiple of 8 and data is aligned

// Assume that C is a multiple of 8
float *fast_SILU(float* input, int H, int W, int C){
    int dim = H * W * C;
    int d_iters = dim / 64;
    int d_left = (dim / 8) % 8;

    float *input_ptr = input;
    float* output = (float *)_mm_malloc(dim * sizeof(float), 32);
    float *output_ptr = output;

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    // Offset is combination of Bias(127*2^23) and C which minimize error in the mantissa/fraction of float
    __m256i bias = _mm256_set1_epi32(BIAS_OFFSET - C_OFFSET); 
    __m256 neg_mask = _mm256_set1_ps(-0.0f); // mask for negate
    __m256 mult = _mm256_set1_ps(MULTIPLIER);
    __m256 one = _mm256_set1_ps(1.0f);

    for (int i  = 0; i < d_iters; ++i){
        // load
        ymm0 = _mm256_load_ps(input_ptr);
        ymm1 = _mm256_load_ps(input_ptr + 8);
        ymm2 = _mm256_load_ps(input_ptr + 16);
        ymm3 = _mm256_load_ps(input_ptr + 24);
        ymm4 = _mm256_load_ps(input_ptr + 32);
        ymm5 = _mm256_load_ps(input_ptr + 40);
        ymm6 = _mm256_load_ps(input_ptr + 48);
        ymm7 = _mm256_load_ps(input_ptr + 56);

        // Negate using xor
        ymm8  =  _mm256_xor_ps(ymm0, neg_mask);
        ymm9  =  _mm256_xor_ps(ymm1, neg_mask);
        ymm10 =  _mm256_xor_ps(ymm2, neg_mask);
        ymm11 =  _mm256_xor_ps(ymm3, neg_mask);
        ymm12 =  _mm256_xor_ps(ymm4, neg_mask);
        ymm13 =  _mm256_xor_ps(ymm5, neg_mask);
        ymm14 =  _mm256_xor_ps(ymm6, neg_mask);
        ymm15 =  _mm256_xor_ps(ymm7, neg_mask);

        // Do e^x using Schraudolph exponential approximation
        // Step 1: Multiply x by (2^23 / ln(2)) can be done inplace
        // e^x = 2^k, x = kln2, k = x/ln(2)
        // Multiply by 2^23 to shift k over to expontial position of float
        ymm8 = _mm256_mul_ps(mult, ymm8);
        ymm9 = _mm256_mul_ps(mult, ymm9);
        ymm10 = _mm256_mul_ps(mult, ymm10);
        ymm11 = _mm256_mul_ps(mult, ymm11);
        ymm12 = _mm256_mul_ps(mult, ymm12);
        ymm13 = _mm256_mul_ps(mult, ymm13);
        ymm14 = _mm256_mul_ps(mult, ymm14);
        ymm15 = _mm256_mul_ps(mult, ymm15);
        
        // Cast to ps so can still use same register, sake of compiler
        // Step 2: Converts from float to int to add bias
        ymm8 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm8));
        ymm9 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm9));
        ymm10 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm10));
        ymm11 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm11));
        ymm12 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm12));
        ymm13 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm13));
        ymm14 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm14));
        ymm15 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm15));
        
        // Step 3: add the offset
        ymm8 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm8), bias));
        ymm9 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm9), bias));
        ymm10 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm10), bias));
        ymm11 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm11), bias));
        ymm12 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm12), bias));
        ymm13 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm13), bias));
        ymm14 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm14), bias));
        ymm15 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm15), bias));
        
        // Add 1
        ymm8  = _mm256_add_ps(ymm8, one);
        ymm9  = _mm256_add_ps(ymm9, one);
        ymm10 = _mm256_add_ps(ymm10, one);
        ymm11 = _mm256_add_ps(ymm11, one);
        ymm12 = _mm256_add_ps(ymm12, one);
        ymm13 = _mm256_add_ps(ymm13, one);
        ymm14 = _mm256_add_ps(ymm14, one);
        ymm15 = _mm256_add_ps(ymm15, one);

        // Find Reciprocal
        ymm8 = _mm256_rcp_ps(ymm8);
        ymm9 = _mm256_rcp_ps(ymm9);
        ymm10 = _mm256_rcp_ps(ymm10);
        ymm11 = _mm256_rcp_ps(ymm11);
        ymm12 = _mm256_rcp_ps(ymm12);
        ymm13 = _mm256_rcp_ps(ymm13);
        ymm14 = _mm256_rcp_ps(ymm14);
        ymm15 = _mm256_rcp_ps(ymm15);

        // Final calculation
        ymm0 = _mm256_mul_ps(ymm0, ymm8);
        ymm1 = _mm256_mul_ps(ymm1, ymm9);
        ymm2 = _mm256_mul_ps(ymm2, ymm10);
        ymm3 = _mm256_mul_ps(ymm3, ymm11);
        ymm4 = _mm256_mul_ps(ymm4, ymm12);
        ymm5 = _mm256_mul_ps(ymm5, ymm13);
        ymm6 = _mm256_mul_ps(ymm6, ymm14);
        ymm7 = _mm256_mul_ps(ymm7, ymm15);

        // Store outputs
        _mm256_store_ps(output_ptr, ymm0);
        _mm256_store_ps(output_ptr + 8, ymm1);
        _mm256_store_ps(output_ptr + 16, ymm2);
        _mm256_store_ps(output_ptr + 24, ymm3);
        _mm256_store_ps(output_ptr + 32, ymm4);
        _mm256_store_ps(output_ptr + 40, ymm5);
        _mm256_store_ps(output_ptr + 48, ymm6);
        _mm256_store_ps(output_ptr + 56, ymm7);
        
        input_ptr += 64;
        output_ptr += 64;
    }

    // Deal with the remaining use switch to interleave and not have dependency chains

    switch (d_left) {// loads
    case 7: ymm6 = _mm256_load_ps(input_ptr + 48);
    case 6: ymm5 = _mm256_load_ps(input_ptr + 40);
    case 5: ymm4 = _mm256_load_ps(input_ptr + 32);
    case 4: ymm3 = _mm256_load_ps(input_ptr + 24);
    case 3: ymm2 = _mm256_load_ps(input_ptr + 16);
    case 2: ymm1 = _mm256_load_ps(input_ptr + 8);
    case 1: ymm0 = _mm256_load_ps(input_ptr);
    default: break;
    }

    switch (d_left) { // Negate
    case 7: ymm14 =  _mm256_xor_ps(ymm6, neg_mask);
    case 6: ymm13 =  _mm256_xor_ps(ymm5, neg_mask);
    case 5: ymm12 =  _mm256_xor_ps(ymm4, neg_mask);
    case 4: ymm11 =  _mm256_xor_ps(ymm3, neg_mask);
    case 3: ymm10 =  _mm256_xor_ps(ymm2, neg_mask);
    case 2: ymm9 =  _mm256_xor_ps(ymm1, neg_mask);
    case 1: ymm8 =  _mm256_xor_ps(ymm0, neg_mask);
    default: break;
    }

    switch (d_left) { // Step 1 exponent approx
    case 7: ymm14 =  _mm256_mul_ps(mult, ymm14);
    case 6: ymm13 =  _mm256_mul_ps(mult, ymm13);
    case 5: ymm12 =  _mm256_mul_ps(mult, ymm12);
    case 4: ymm11 =  _mm256_mul_ps(mult, ymm11);
    case 3: ymm10 =  _mm256_mul_ps(mult, ymm10);
    case 2: ymm9 =  _mm256_mul_ps(mult, ymm9);
    case 1: ymm8 =  _mm256_mul_ps(mult, ymm8);
    default: break;
    }
    switch (d_left) { // Step 2 convert float to int
    case 7: ymm14 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm14));
    case 6: ymm13 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm13));
    case 5: ymm12 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm12));
    case 4: ymm11 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm11));
    case 3: ymm10 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm10));
    case 2: ymm9 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm9));
    case 1: ymm8 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm8));
    default: break;
    }
    switch (d_left) { // Step 3
    case 7: ymm14 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm14), bias));
    case 6: ymm13 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm13), bias));
    case 5: ymm12 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm12), bias));
    case 4: ymm11 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm11), bias));
    case 3: ymm10 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm10), bias));
    case 2: ymm9 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm9), bias));
    case 1: ymm8 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm8), bias));
    default: break;
    }

    switch (d_left) {// add
    case 7: ymm14 = _mm256_add_ps(one, ymm14);
    case 6: ymm13 = _mm256_add_ps(one, ymm13);
    case 5: ymm12 = _mm256_add_ps(one, ymm12);
    case 4: ymm11 = _mm256_add_ps(one, ymm11);
    case 3: ymm10 = _mm256_add_ps(one, ymm10);
    case 2: ymm9 = _mm256_add_ps(one, ymm9);
    case 1: ymm8 = _mm256_add_ps(one, ymm8);
    default: break;
    }

    switch (d_left) { // reciprocal
    case 7: ymm14 = _mm256_rcp_ps(ymm14);
    case 6: ymm13 = _mm256_rcp_ps(ymm13);
    case 5: ymm12 = _mm256_rcp_ps(ymm12);
    case 4: ymm11 = _mm256_rcp_ps(ymm11);
    case 3: ymm10 = _mm256_rcp_ps(ymm10);
    case 2: ymm9 = _mm256_rcp_ps(ymm9);
    case 1: ymm8 = _mm256_rcp_ps(ymm8);
    default: break;
    }

    switch (d_left) { // multiply
    case 7: ymm6 = _mm256_mul_ps(ymm6, ymm14);
    case 6: ymm5 = _mm256_mul_ps(ymm5, ymm13);
    case 5: ymm4 = _mm256_mul_ps(ymm4, ymm12);
    case 4: ymm3 = _mm256_mul_ps(ymm3, ymm11);
    case 3: ymm2 = _mm256_mul_ps(ymm2, ymm10);
    case 2: ymm1 = _mm256_mul_ps(ymm1, ymm9);
    case 1: ymm0 = _mm256_mul_ps(ymm0, ymm8);
    default: break;
    }

    switch (d_left) { // store
    case 7: _mm256_store_ps(output_ptr + 48, ymm6);
    case 6: _mm256_store_ps(output_ptr + 40, ymm5);
    case 5: _mm256_store_ps(output_ptr + 32, ymm4);
    case 4: _mm256_store_ps(output_ptr + 24, ymm3);
    case 3: _mm256_store_ps(output_ptr + 16, ymm2);
    case 2: _mm256_store_ps(output_ptr + 8, ymm1);
    case 1: _mm256_store_ps(output_ptr , ymm0);
    default: break;
    }
    
    return output; 
}

// Sigmoid using Taylor expansion
// Assume C is multiple of 8 and data is aligned
float *taylor_SILU(float* input, int H, int W, int C){
    int dim = H * W * C;
    int d_iters = (dim / 8) / 8;
    int d_left = (dim / 8) % 8;

    float *input_ptr = input;
    float* output = (float *)_mm_malloc(dim * sizeof(float), 32);   
    float *output_ptr = output;

    // 32 Vector registers // as tempCan switch out
    __m256 neg_mask  = _mm256_set1_ps(-0.0f);
    __m256 log2e  = _mm256_set1_ps(1.4426950408889634f);
    __m256 ln2    = _mm256_set1_ps(0.6931471805599453f);
    __m256i exp_bias = _mm256_set1_epi32(BIAS_OFFSET);
    // Taylor Series Coefficients
    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c23; // reuse
    __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
    __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
    __m256 ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23;

    for (int i  = 0; i < d_iters; ++i){
        // load
        ymm0 = _mm256_load_ps(input_ptr);
        ymm1 = _mm256_load_ps(input_ptr + 8);
        ymm2 = _mm256_load_ps(input_ptr + 16);
        ymm3 = _mm256_load_ps(input_ptr + 24);
        ymm4 = _mm256_load_ps(input_ptr + 32);
        ymm5 = _mm256_load_ps(input_ptr + 40);
        ymm6 = _mm256_load_ps(input_ptr + 48);
        ymm7 = _mm256_load_ps(input_ptr + 56);

        // Negate using xor
        ymm8  =  _mm256_xor_ps(ymm0, neg_mask); 
        ymm9  =  _mm256_xor_ps(ymm1, neg_mask);
        ymm10 =  _mm256_xor_ps(ymm2, neg_mask);
        ymm11 =  _mm256_xor_ps(ymm3, neg_mask);
        ymm12 =  _mm256_xor_ps(ymm4, neg_mask);
        ymm13 =  _mm256_xor_ps(ymm5, neg_mask);
        ymm14 =  _mm256_xor_ps(ymm6, neg_mask);
        ymm15 =  _mm256_xor_ps(ymm7, neg_mask);

        // Calculating e^x, Taylor expansion centers around x = 0, so adjust to get better accuracy(especially neg values)
        // Step 1: e^x = 2^k * e^r,  k = round(x*log2e)
        ymm16 = _mm256_mul_ps(ymm8, log2e);
        ymm17 = _mm256_mul_ps(ymm9, log2e);
        ymm18 = _mm256_mul_ps(ymm10, log2e);
        ymm19 = _mm256_mul_ps(ymm11, log2e);
        ymm20 = _mm256_mul_ps(ymm12, log2e);
        ymm21 = _mm256_mul_ps(ymm13, log2e);
        ymm22 = _mm256_mul_ps(ymm14, log2e);
        ymm23 = _mm256_mul_ps(ymm15, log2e);

        ymm16 =_mm256_round_ps(ymm16, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        ymm17 =_mm256_round_ps(ymm17, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        ymm18 =_mm256_round_ps(ymm18, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        ymm19 =_mm256_round_ps(ymm19, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        ymm20 =_mm256_round_ps(ymm20, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        ymm21 =_mm256_round_ps(ymm21, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        ymm22 =_mm256_round_ps(ymm22, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        ymm23 =_mm256_round_ps(ymm23, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

        // Step 2: 
        // e^x = e^(kln2 + r), r = x - kln2        
        ymm8 = _mm256_fnmadd_ps(ymm16, ln2, ymm8);
        ymm9 = _mm256_fnmadd_ps(ymm17, ln2, ymm9);
        ymm10 = _mm256_fnmadd_ps(ymm18, ln2, ymm10);
        ymm11 = _mm256_fnmadd_ps(ymm19, ln2, ymm11);
        ymm12 = _mm256_fnmadd_ps(ymm20, ln2, ymm12);
        ymm13 = _mm256_fnmadd_ps(ymm21, ln2, ymm13);
        ymm14 = _mm256_fnmadd_ps(ymm22, ln2, ymm14);
        ymm15 = _mm256_fnmadd_ps(ymm23, ln2, ymm15);

        // Convert to int to do 2^k quickly
        ymm16 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm16)); 
        ymm17 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm17)); 
        ymm18 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm18)); 
        ymm19 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm19)); 
        ymm20 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm20)); 
        ymm21 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm21)); 
        ymm22 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm22)); 
        ymm23 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm23)); 

        // shift k to the exponent bits
        ymm16 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm16), 23)); 
        ymm17 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm17), 23)); 
        ymm18 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm18), 23)); 
        ymm19 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm19), 23)); 
        ymm20 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm20), 23)); 
        ymm21 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm21), 23)); 
        ymm22 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm22), 23)); 
        ymm23 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm23), 23)); 
        // Add the bias
        ymm16 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm16), exp_bias));
        ymm17 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm17), exp_bias));
        ymm18 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm18), exp_bias));
        ymm19 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm19), exp_bias));
        ymm20 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm20), exp_bias));
        ymm21 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm21), exp_bias));
        ymm22 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm22), exp_bias));
        ymm23 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm23), exp_bias));

        //Taylor expansion: 1 + r +r^2/2 + r^3/6 + r^4/24+ r^5/120
        // 1 + r(1 + r(1/2 + r(1/6+ r(1/23 + r*(1/120))))) 
        // i = 4
        ymm0 = _mm256_fmadd_ps(ymm8, c5, c4);
        ymm1 = _mm256_fmadd_ps(ymm9, c5, c4);
        ymm2 = _mm256_fmadd_ps(ymm10, c5, c4);
        ymm3 = _mm256_fmadd_ps(ymm11, c5, c4);
        ymm4 = _mm256_fmadd_ps(ymm12, c5, c4);
        ymm5 = _mm256_fmadd_ps(ymm13, c5, c4);
        ymm6 = _mm256_fmadd_ps(ymm14, c5, c4);
        ymm7 = _mm256_fmadd_ps(ymm15, c5, c4);
        // i = 3
       
        c23 = _mm256_set1_ps(1.0f / 6.0f);
        ymm0 = _mm256_fmadd_ps(ymm8,  ymm0, c23);
        ymm1 = _mm256_fmadd_ps(ymm9,  ymm1, c23);
        ymm2 = _mm256_fmadd_ps(ymm10, ymm2, c23);
        ymm3 = _mm256_fmadd_ps(ymm11, ymm3, c23);
        ymm4 = _mm256_fmadd_ps(ymm12, ymm4, c23);
        ymm5 = _mm256_fmadd_ps(ymm13, ymm5, c23);
        ymm6 = _mm256_fmadd_ps(ymm14, ymm6, c23);
        ymm7 = _mm256_fmadd_ps(ymm15, ymm7, c23);
        // i = 2
        c23 = _mm256_set1_ps(0.5f);
        ymm0 = _mm256_fmadd_ps(ymm8,  ymm0, c23);
        ymm1 = _mm256_fmadd_ps(ymm9,  ymm1, c23);
        ymm2 = _mm256_fmadd_ps(ymm10, ymm2, c23);
        ymm3 = _mm256_fmadd_ps(ymm11, ymm3, c23);
        ymm4 = _mm256_fmadd_ps(ymm12, ymm4, c23);
        ymm5 = _mm256_fmadd_ps(ymm13, ymm5, c23);
        ymm6 = _mm256_fmadd_ps(ymm14, ymm6, c23);
        ymm7 = _mm256_fmadd_ps(ymm15, ymm7, c23);
        // i = 1
        ymm0 = _mm256_fmadd_ps(ymm8,  ymm0, c1);
        ymm1 = _mm256_fmadd_ps(ymm9,  ymm1, c1);
        ymm2 = _mm256_fmadd_ps(ymm10, ymm2, c1);
        ymm3 = _mm256_fmadd_ps(ymm11, ymm3, c1);
        ymm4 = _mm256_fmadd_ps(ymm12, ymm4, c1);
        ymm5 = _mm256_fmadd_ps(ymm13, ymm5, c1);
        ymm6 = _mm256_fmadd_ps(ymm14, ymm6, c1);
        ymm7 = _mm256_fmadd_ps(ymm15, ymm7, c1);
        // i = 0
        ymm0 = _mm256_fmadd_ps(ymm8,  ymm0, c1);
        ymm1 = _mm256_fmadd_ps(ymm9,  ymm1, c1);
        ymm2 = _mm256_fmadd_ps(ymm10, ymm2, c1);
        ymm3 = _mm256_fmadd_ps(ymm11, ymm3, c1);
        ymm4 = _mm256_fmadd_ps(ymm12, ymm4, c1);
        ymm5 = _mm256_fmadd_ps(ymm13, ymm5, c1);
        ymm6 = _mm256_fmadd_ps(ymm14, ymm6, c1);
        ymm7 = _mm256_fmadd_ps(ymm15, ymm7, c1);

        // Reload x from storage
        ymm8 = _mm256_load_ps(input_ptr);
        ymm9 = _mm256_load_ps(input_ptr + 8);
        ymm10 = _mm256_load_ps(input_ptr + 16);
        ymm11 = _mm256_load_ps(input_ptr + 24);
        ymm12 = _mm256_load_ps(input_ptr + 32);
        ymm13 = _mm256_load_ps(input_ptr + 40);
        ymm14 = _mm256_load_ps(input_ptr + 48);
        ymm15 = _mm256_load_ps(input_ptr + 56);

        // Final calculattion of e^x (2^k * e^r)
        ymm16 = _mm256_mul_ps(ymm0, ymm16);
        ymm17 = _mm256_mul_ps(ymm1, ymm17);
        ymm18 = _mm256_mul_ps(ymm2, ymm18);
        ymm19 = _mm256_mul_ps(ymm3, ymm19);
        ymm20 = _mm256_mul_ps(ymm4, ymm20);
        ymm21 = _mm256_mul_ps(ymm5, ymm21);
        ymm22 = _mm256_mul_ps(ymm6, ymm22);
        ymm23 = _mm256_mul_ps(ymm7, ymm23);

        // Add 1 for denom
        ymm16 = _mm256_add_ps(ymm16, c1);
        ymm17 = _mm256_add_ps(ymm17, c1);
        ymm18 = _mm256_add_ps(ymm18, c1);
        ymm19 = _mm256_add_ps(ymm19, c1);
        ymm20 = _mm256_add_ps(ymm20, c1);
        ymm21 = _mm256_add_ps(ymm21, c1);
        ymm22 = _mm256_add_ps(ymm22, c1);
        ymm23 = _mm256_add_ps(ymm23, c1);

    
        // Find Reciprocal
        ymm16 = _mm256_rcp_ps(ymm16);
        ymm17 = _mm256_rcp_ps(ymm17);
        ymm18 = _mm256_rcp_ps(ymm18);
        ymm19 = _mm256_rcp_ps(ymm19);
        ymm20 = _mm256_rcp_ps(ymm20);
        ymm21 = _mm256_rcp_ps(ymm21);
        ymm22 = _mm256_rcp_ps(ymm22);
        ymm23 = _mm256_rcp_ps(ymm23);

        // Final calculation
        ymm0 = _mm256_mul_ps(ymm8,  ymm16);
        ymm1 = _mm256_mul_ps(ymm9,  ymm17);
        ymm2 = _mm256_mul_ps(ymm10, ymm18);
        ymm3 = _mm256_mul_ps(ymm11, ymm19);
        ymm4 = _mm256_mul_ps(ymm12, ymm20);
        ymm5 = _mm256_mul_ps(ymm13, ymm21);
        ymm6 = _mm256_mul_ps(ymm14, ymm22);
        ymm7 = _mm256_mul_ps(ymm15, ymm23);

        // Store outputs
        _mm256_store_ps(output_ptr, ymm0);
        _mm256_store_ps(output_ptr + 8, ymm1);
        _mm256_store_ps(output_ptr + 16, ymm2);
        _mm256_store_ps(output_ptr + 24, ymm3);
        _mm256_store_ps(output_ptr + 32, ymm4);
        _mm256_store_ps(output_ptr + 40, ymm5);
        _mm256_store_ps(output_ptr + 48, ymm6);
        _mm256_store_ps(output_ptr + 56, ymm7);
        
        input_ptr += 64;
        output_ptr += 64;
    }

    // Deal with the remaining use switch to interleave and not have dependency chains

    switch (d_left) {// loads
        case 7: ymm6 = _mm256_load_ps(input_ptr + 48);
        case 6: ymm5 = _mm256_load_ps(input_ptr + 40);
        case 5: ymm4 = _mm256_load_ps(input_ptr + 32);
        case 4: ymm3 = _mm256_load_ps(input_ptr + 24);
        case 3: ymm2 = _mm256_load_ps(input_ptr + 16);
        case 2: ymm1 = _mm256_load_ps(input_ptr + 8);
        case 1: ymm0 = _mm256_load_ps(input_ptr);
        default: break;
    }

    switch (d_left) {    // Negate using xor
        case 7: ymm14 =  _mm256_xor_ps(ymm6, neg_mask);
        case 6: ymm13 =  _mm256_xor_ps(ymm5, neg_mask);
        case 5: ymm12 =  _mm256_xor_ps(ymm4, neg_mask);
        case 4: ymm11 =  _mm256_xor_ps(ymm3, neg_mask);
        case 3: ymm10 =  _mm256_xor_ps(ymm2, neg_mask);
        case 2: ymm9  =  _mm256_xor_ps(ymm1, neg_mask);
        case 1: ymm8  =  _mm256_xor_ps(ymm0, neg_mask);
        default: break;
    }
    // Calculating e^x, Taylor expansion centers around x = 0, so adjust to get better accuracy(especially neg values)
    switch (d_left) {    // Step 1: e^x = 2^k * e^r,  k = round(x*log2e)
        case 7: ymm22 = _mm256_mul_ps(ymm14, log2e);
        case 6: ymm21 = _mm256_mul_ps(ymm13, log2e);
        case 5: ymm20 = _mm256_mul_ps(ymm12, log2e);
        case 4: ymm19 = _mm256_mul_ps(ymm11, log2e);
        case 3: ymm18 = _mm256_mul_ps(ymm10, log2e);
        case 2: ymm17 = _mm256_mul_ps(ymm9, log2e);
        case 1: ymm16 = _mm256_mul_ps(ymm8, log2e);
        default: break;
    }

    switch (d_left) {
        case 7: ymm22 =_mm256_round_ps(ymm22, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        case 6: ymm21 =_mm256_round_ps(ymm21, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        case 5: ymm20 =_mm256_round_ps(ymm20, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        case 4: ymm19 =_mm256_round_ps(ymm19, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        case 3: ymm18 =_mm256_round_ps(ymm18, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        case 2: ymm17 =_mm256_round_ps(ymm17, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        case 1: ymm16 =_mm256_round_ps(ymm16, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        default: break;
    }
    
    switch (d_left) {   // Step 2: e^x = e^(kln2 + r), r = x - kln2        
        case 7: ymm14 = _mm256_fnmadd_ps(ymm22, ln2, ymm14);
        case 6: ymm13 = _mm256_fnmadd_ps(ymm21, ln2, ymm13);
        case 5: ymm12 = _mm256_fnmadd_ps(ymm20, ln2, ymm12);
        case 4: ymm11 = _mm256_fnmadd_ps(ymm19, ln2, ymm11);
        case 3: ymm10 = _mm256_fnmadd_ps(ymm18, ln2, ymm10);
        case 2: ymm9 = _mm256_fnmadd_ps(ymm17, ln2, ymm9);
        case 1: ymm8 = _mm256_fnmadd_ps(ymm16, ln2, ymm8);
        default: break;
    }

    switch (d_left) {    // Convert to int to do 2^k quickly
        case 7: ymm22 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm22)); 
        case 6: ymm21 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm21)); 
        case 5: ymm20 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm20)); 
        case 4: ymm19 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm19)); 
        case 3: ymm18 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm18)); 
        case 2: ymm17 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm17)); 
        case 1: ymm16 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm16)); 
        default: break;
    }
    
    switch (d_left) {    // shift k to the exponent bits
        case 7: ymm22 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm22), 23)); 
        case 6: ymm21 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm21), 23)); 
        case 5: ymm20 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm20), 23)); 
        case 4: ymm19 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm19), 23)); 
        case 3: ymm18 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm18), 23)); 
        case 2: ymm17 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm17), 23)); 
        case 1: ymm16 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(ymm16), 23)); 
        default: break;
    }
    
    switch (d_left) {    // Add the bias
        case 7: ymm22 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm22), exp_bias));
        case 6: ymm21 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm21), exp_bias));
        case 5: ymm20 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm20), exp_bias));
        case 4: ymm19 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm19), exp_bias));
        case 3: ymm18 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm18), exp_bias));
        case 2: ymm17 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm17), exp_bias));
        case 1: ymm16 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm16), exp_bias));
        default: break;
    }

    switch (d_left) {  //Taylor expansion: 1 + r +r^2/2 + r^3/6 + r^4/24+ r^5/120
        // i = 4, 1/24 + r/120
        case 7: ymm6 = _mm256_fmadd_ps(ymm14, c5, c4);
        case 6: ymm5 = _mm256_fmadd_ps(ymm13, c5, c4);
        case 5: ymm4 = _mm256_fmadd_ps(ymm12, c5, c4);
        case 4: ymm3 = _mm256_fmadd_ps(ymm11, c5, c4);
        case 3: ymm2 = _mm256_fmadd_ps(ymm10, c5, c4);
        case 2: ymm1 = _mm256_fmadd_ps(ymm9, c5, c4);
        case 1: ymm0 = _mm256_fmadd_ps(ymm8, c5, c4);
        default: break;
    }
       
    c23 = _mm256_set1_ps(1.0f / 6.0f);
    switch (d_left) {   // i = 3, 1/6 + r/24 + r^2/120
        case 7: ymm6 = _mm256_fmadd_ps(ymm14, ymm6, c23);
        case 6: ymm5 = _mm256_fmadd_ps(ymm13, ymm5, c23);
        case 5: ymm4 = _mm256_fmadd_ps(ymm12, ymm4, c23);
        case 4: ymm3 = _mm256_fmadd_ps(ymm11, ymm3, c23);
        case 3: ymm2 = _mm256_fmadd_ps(ymm10, ymm2, c23);
        case 2: ymm1 = _mm256_fmadd_ps(ymm9,  ymm1, c23);
        case 1: ymm0 = _mm256_fmadd_ps(ymm8,  ymm0, c23);
        default: break;
    }
    
    c23 = _mm256_set1_ps(0.5f);
    switch (d_left) {    // i = 2, 1/2 + r/6 + r^2/24 + r^3/120
        case 7: ymm6 = _mm256_fmadd_ps(ymm14, ymm6, c23);
        case 6: ymm5 = _mm256_fmadd_ps(ymm13, ymm5, c23);
        case 5: ymm4 = _mm256_fmadd_ps(ymm12, ymm4, c23);
        case 4: ymm3 = _mm256_fmadd_ps(ymm11, ymm3, c23);
        case 3: ymm2 = _mm256_fmadd_ps(ymm10, ymm2, c23);
        case 2: ymm1 = _mm256_fmadd_ps(ymm9,  ymm1, c23);
        case 1: ymm0 = _mm256_fmadd_ps(ymm8,  ymm0, c23);
        default: break;
    }

    switch (d_left) {    // i = 1, 1 + r/2 + r^2/6 + r^3/24 + r^4/120
        case 7: ymm6 = _mm256_fmadd_ps(ymm14, ymm6, c1);
        case 6: ymm5 = _mm256_fmadd_ps(ymm13, ymm5, c1);
        case 5: ymm4 = _mm256_fmadd_ps(ymm12, ymm4, c1);
        case 4: ymm3 = _mm256_fmadd_ps(ymm11, ymm3, c1);
        case 3: ymm2 = _mm256_fmadd_ps(ymm10, ymm2, c1);
        case 2: ymm1 = _mm256_fmadd_ps(ymm9,  ymm1, c1);
        case 1: ymm0 = _mm256_fmadd_ps(ymm8,  ymm0, c1);
        default: break;
    }

    switch (d_left) {    // i = 0, 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
        case 7: ymm6 = _mm256_fmadd_ps(ymm14, ymm6, c1);
        case 6: ymm5 = _mm256_fmadd_ps(ymm13, ymm5, c1);
        case 5: ymm4 = _mm256_fmadd_ps(ymm12, ymm4, c1);
        case 4: ymm3 = _mm256_fmadd_ps(ymm11, ymm3, c1);
        case 3: ymm2 = _mm256_fmadd_ps(ymm10, ymm2, c1);
        case 2: ymm1 = _mm256_fmadd_ps(ymm9,  ymm1, c1);
        case 1: ymm0 = _mm256_fmadd_ps(ymm8,  ymm0, c1);
        default: break;
    }

    switch (d_left) {    // Reload x from storage
        case 7: ymm14 = _mm256_load_ps(input_ptr + 48);
        case 6: ymm13 = _mm256_load_ps(input_ptr + 40);
        case 5: ymm12 = _mm256_load_ps(input_ptr + 32);
        case 4: ymm11 = _mm256_load_ps(input_ptr + 24);
        case 3: ymm10 = _mm256_load_ps(input_ptr + 16);
        case 2: ymm9 = _mm256_load_ps(input_ptr + 8);
        case 1: ymm8 = _mm256_load_ps(input_ptr);
        default: break;
    }

    switch (d_left) {   // Final calculattion of e^x (2^k * e^r)
        case 7: ymm22 = _mm256_mul_ps(ymm6, ymm22);
        case 6: ymm21 = _mm256_mul_ps(ymm5, ymm21);
        case 5: ymm20 = _mm256_mul_ps(ymm4, ymm20);
        case 4: ymm19 = _mm256_mul_ps(ymm3, ymm19);
        case 3: ymm18 = _mm256_mul_ps(ymm2, ymm18);
        case 2: ymm17 = _mm256_mul_ps(ymm1, ymm17);
        case 1: ymm16 = _mm256_mul_ps(ymm0, ymm16);
        default: break;
    }

    switch (d_left) {    // Add 1 for denom
        case 7: ymm22 = _mm256_add_ps(ymm22, c1);
        case 6: ymm21 = _mm256_add_ps(ymm21, c1);
        case 5: ymm20 = _mm256_add_ps(ymm20, c1);
        case 4: ymm19 = _mm256_add_ps(ymm19, c1);
        case 3: ymm18 = _mm256_add_ps(ymm18, c1);
        case 2: ymm17 = _mm256_add_ps(ymm17, c1);
        case 1: ymm16 = _mm256_add_ps(ymm16, c1);
        default: break;
    }
    
    switch (d_left) {    // Find Reciprocal
        case 7: ymm22 = _mm256_rcp_ps(ymm22);
        case 6: ymm21 = _mm256_rcp_ps(ymm21);
        case 5: ymm20 = _mm256_rcp_ps(ymm20);
        case 4: ymm19 = _mm256_rcp_ps(ymm19);
        case 3: ymm18 = _mm256_rcp_ps(ymm18);
        case 2: ymm17 = _mm256_rcp_ps(ymm17);
        case 1: ymm16 = _mm256_rcp_ps(ymm16);
        default: break;
    }
    
    switch (d_left) {    // Final calculation
        case 7: ymm6 = _mm256_mul_ps(ymm14, ymm22);
        case 6: ymm5 = _mm256_mul_ps(ymm13, ymm21);
        case 5: ymm4 = _mm256_mul_ps(ymm12, ymm20);
        case 4: ymm3 = _mm256_mul_ps(ymm11, ymm19);
        case 3: ymm2 = _mm256_mul_ps(ymm10, ymm18);
        case 2: ymm1 = _mm256_mul_ps(ymm9,  ymm17);
        case 1: ymm0 = _mm256_mul_ps(ymm8,  ymm16);
        default: break;
    }

    switch (d_left) {    // Store outputs
        case 7: _mm256_store_ps(output_ptr + 48, ymm6);
        case 6: _mm256_store_ps(output_ptr + 40, ymm5);
        case 5: _mm256_store_ps(output_ptr + 32, ymm4);
        case 4: _mm256_store_ps(output_ptr + 24, ymm3);
        case 3: _mm256_store_ps(output_ptr + 16, ymm2);
        case 2: _mm256_store_ps(output_ptr + 8, ymm1);
        case 1: _mm256_store_ps(output_ptr, ymm0);
        default: break;
    }
    return output; 
}

