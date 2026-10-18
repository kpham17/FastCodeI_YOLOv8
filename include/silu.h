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

// Sigmoid using Schraudolph exponential approximation
// Assuem C is multiple of 8 and data is aligned
float *fast_SILU(float* input, int H, int W, int C){
    int dim = H * W * C;
    int d_iters = (dim / 8) / 6;
    int d_left = (dim / 8) % 6;

    float *input_ptr = input;
    float* output = (float *)_mm_malloc(dim * sizeof(float), 32);
    float *output_ptr = output;

    __m256i ymm1 = _mm256_set1_epi32(BIAS_OFFSET - C_OFFSET); 
    __m256 ymm0, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    for (int i  = 0; i < d_iters; ++i){
        // load
        ymm15 = _mm256_load_ps(input_ptr);
        ymm14 = _mm256_load_ps(input_ptr + 8);
        ymm13 = _mm256_load_ps(input_ptr + 16);
        ymm12 = _mm256_load_ps(input_ptr + 24);
        ymm11 = _mm256_load_ps(input_ptr + 32);
        ymm10 = _mm256_load_ps(input_ptr + 40);

        // Negate using xor
        ymm0 = _mm256_set1_ps(-0.0f); // mask for negate
        ymm2 =  _mm256_xor_ps(ymm15, ymm0);
        ymm3 =  _mm256_xor_ps(ymm14, ymm0);
        ymm4 =  _mm256_xor_ps(ymm13, ymm0);
        ymm5 =  _mm256_xor_ps(ymm12, ymm0);
        ymm6 =  _mm256_xor_ps(ymm11, ymm0);
        ymm7 =  _mm256_xor_ps(ymm10, ymm0);

        // Do e^x using Schraudolph exponential approximation
        // Step 1: Multiply x by (2^23 / ln(2)) can be done inplace
        ymm0 = _mm256_set1_ps(MULTIPLIER);
        ymm2 = _mm256_mul_ps(ymm0, ymm2);
        ymm3 = _mm256_mul_ps(ymm0, ymm3);
        ymm4 = _mm256_mul_ps(ymm0, ymm4);
        ymm5 = _mm256_mul_ps(ymm0, ymm5);
        ymm6 = _mm256_mul_ps(ymm0, ymm6);
        ymm7 = _mm256_mul_ps(ymm0, ymm7);
        // Cast to ps so can still use same register, sake of compiler
        // Step 2: Converts from float to int
        ymm2 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm2));
        ymm3 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm3));
        ymm4 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm4));
        ymm5 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm5));
        ymm6 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm6));
        ymm7 = _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm7));
        
        // Step 3: add the offset
        ymm2 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm2), ymm1));
        ymm3 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm3), ymm1));
        ymm4 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm4), ymm1));
        ymm5 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm5), ymm1));
        ymm6 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm6), ymm1));
        ymm7 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm7), ymm1));
        
        // Add 1
        ymm0 = _mm256_set1_ps(1.0f);
        ymm2 = _mm256_add_ps(ymm0, ymm2);
        ymm3 = _mm256_add_ps(ymm0, ymm3);
        ymm4 = _mm256_add_ps(ymm0, ymm4);
        ymm5 = _mm256_add_ps(ymm0, ymm5);
        ymm6 = _mm256_add_ps(ymm0, ymm6);
        ymm7 = _mm256_add_ps(ymm0, ymm7);

        // Find Reciprocal
        ymm2 = _mm256_rcp_ps(ymm2);
        ymm3 = _mm256_rcp_ps(ymm3);
        ymm4 = _mm256_rcp_ps(ymm4);
        ymm5 = _mm256_rcp_ps(ymm5);
        ymm6 = _mm256_rcp_ps(ymm6);
        ymm7 = _mm256_rcp_ps(ymm7);

        // Final calculation
        ymm15 = _mm256_mul_ps(ymm15, ymm2);
        ymm14 = _mm256_mul_ps(ymm14, ymm3);
        ymm13 = _mm256_mul_ps(ymm13, ymm4);
        ymm12 = _mm256_mul_ps(ymm12, ymm5);
        ymm11 = _mm256_mul_ps(ymm11, ymm6);
        ymm10 = _mm256_mul_ps(ymm10, ymm7);

        // Store outputs
        _mm256_store_ps(output_ptr, ymm15);
        _mm256_store_ps(output_ptr + 8, ymm14);
        _mm256_store_ps(output_ptr + 16, ymm13);
        _mm256_store_ps(output_ptr + 24, ymm12);
        _mm256_store_ps(output_ptr + 32, ymm11);
        _mm256_store_ps(output_ptr + 40, ymm10);
        
        input_ptr += 48;
        output_ptr += 48;
    }

    // Deal with the remaining use switch to interleave and not have dependency chains

    switch (d_left) {// loads
    case 5: ymm11 = _mm256_load_ps(input_ptr + 32);
    case 4: ymm12 = _mm256_load_ps(input_ptr + 24);
    case 3: ymm13 = _mm256_load_ps(input_ptr + 16);
    case 2: ymm14 = _mm256_load_ps(input_ptr + 8);
    case 1: ymm15 = _mm256_load_ps(input_ptr);
    default: break;
    }

    ymm0 = _mm256_set1_ps(-0.0f); // mask for negate
    switch (d_left) { // Negate
    case 5: ymm6 =  _mm256_xor_ps(ymm11, ymm0);
    case 4: ymm5 =  _mm256_xor_ps(ymm12, ymm0);
    case 3: ymm4 =  _mm256_xor_ps(ymm13, ymm0);
    case 2: ymm3 =  _mm256_xor_ps(ymm14, ymm0);
    case 1: ymm2 =  _mm256_xor_ps(ymm15, ymm0);
    default: break;
    }

    ymm0 = _mm256_set1_ps(MULTIPLIER); // multiplier
    switch (d_left) { // Step 1 exponent approx
    case 5: ymm6 =  _mm256_mul_ps(ymm0, ymm6);
    case 4: ymm5 =  _mm256_mul_ps(ymm0, ymm5);
    case 3: ymm4 =  _mm256_mul_ps(ymm0, ymm4);
    case 2: ymm3 =  _mm256_mul_ps(ymm0, ymm3);
    case 1: ymm2 =  _mm256_mul_ps(ymm0, ymm2);
    default: break;
    }
    switch (d_left) { // Step 2 convert float to int
    case 5: ymm6 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm6));
    case 4: ymm5 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm5));
    case 3: ymm4 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm4));
    case 2: ymm3 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm3));
    case 1: ymm2 =  _mm256_castsi256_ps(_mm256_cvtps_epi32(ymm2));
    default: break;
    }
    switch (d_left) { // Step 3
    case 5: ymm6 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm6), ymm1));
    case 4: ymm5 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm5), ymm1));
    case 3: ymm4 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm4), ymm1));
    case 2: ymm3 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm3), ymm1));
    case 1: ymm2 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm2), ymm1));
    default: break;
    }

    ymm0 = _mm256_set1_ps(1.0f); 
    switch (d_left) {// add
    case 5: ymm6 = _mm256_add_ps(ymm0, ymm6);
    case 4: ymm5 = _mm256_add_ps(ymm0, ymm5);
    case 3: ymm4 = _mm256_add_ps(ymm0, ymm4);
    case 2: ymm3 = _mm256_add_ps(ymm0, ymm3);
    case 1: ymm2 = _mm256_add_ps(ymm0, ymm2);
    default: break;
    }

    switch (d_left) { // reciprocal
    case 5: ymm6 = _mm256_rcp_ps(ymm6);
    case 4: ymm5 = _mm256_rcp_ps(ymm5);
    case 3: ymm4 = _mm256_rcp_ps(ymm4);
    case 2: ymm3 = _mm256_rcp_ps(ymm3);
    case 1: ymm2 = _mm256_rcp_ps(ymm2);
    default: break;
    }

    switch (d_left) { // multiply
    case 5: ymm11 = _mm256_mul_ps(ymm11, ymm6);
    case 4: ymm12 = _mm256_mul_ps(ymm12, ymm5);
    case 3: ymm13 = _mm256_mul_ps(ymm13, ymm4);
    case 2: ymm14 = _mm256_mul_ps(ymm14, ymm3);
    case 1: ymm15 = _mm256_mul_ps(ymm15, ymm2);
    default: break;
    }

    switch (d_left) { // store
    case 5: _mm256_store_ps(output_ptr + 32, ymm11);
    case 4: _mm256_store_ps(output_ptr + 24, ymm12);
    case 3: _mm256_store_ps(output_ptr + 16, ymm13);
    case 2: _mm256_store_ps(output_ptr + 8, ymm14);
    case 1: _mm256_store_ps(output_ptr , ymm15);
    default: break;
    }
    
    return output; 
}


