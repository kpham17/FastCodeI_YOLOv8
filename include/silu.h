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

// Sigmoid using Taylor expansion
// Assume C is multiple of 8 and data is aligned
void *taylor_SILU(float* input, int H, int W, int C, float* output){
    int dim = H * W * C;
    int d_iters = (dim / 8) / 8;
    int d_left = (dim / 8) % 8;

    float *input_ptr = input;
    //float* output = (float *)_mm_malloc(dim * sizeof(float), 64);   
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

    __m256 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m256 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m256 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;

    for (int i  = 0; i < d_iters; ++i){
        // load
        zmm0 = _mm256_load_ps(input_ptr);
        zmm1 = _mm256_load_ps(input_ptr + 8);
        zmm2 = _mm256_load_ps(input_ptr + 16);
        zmm3 = _mm256_load_ps(input_ptr + 24);
        zmm4 = _mm256_load_ps(input_ptr + 32);
        zmm5 = _mm256_load_ps(input_ptr + 40);
        zmm6 = _mm256_load_ps(input_ptr + 48);
        zmm7 = _mm256_load_ps(input_ptr + 56);
    
        // 2. Negate via bitwise XOR
        zmm8  = _mm256_xor_ps(zmm0, neg_mask);
        zmm9  = _mm256_xor_ps(zmm1, neg_mask);
        zmm10 = _mm256_xor_ps(zmm2, neg_mask);
        zmm11 = _mm256_xor_ps(zmm3, neg_mask);
        zmm12 = _mm256_xor_ps(zmm4, neg_mask);
        zmm13 = _mm256_xor_ps(zmm5, neg_mask);
        zmm14 = _mm256_xor_ps(zmm6, neg_mask);
        zmm15 = _mm256_xor_ps(zmm7, neg_mask);

        // 3. k = round(-x * log2e)
        zmm16 = _mm256_mul_ps(zmm8,  log2e);
        zmm17 = _mm256_mul_ps(zmm9,  log2e);
        zmm18 = _mm256_mul_ps(zmm10, log2e);
        zmm19 = _mm256_mul_ps(zmm11, log2e);
        zmm20 = _mm256_mul_ps(zmm12, log2e);
        zmm21 = _mm256_mul_ps(zmm13, log2e);
        zmm22 = _mm256_mul_ps(zmm14, log2e);
        zmm23 = _mm256_mul_ps(zmm15, log2e);

        // Convert to int (rounds automatically)
        zmm0 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm16));
        zmm1 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm17));
        zmm2 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm18));
        zmm3 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm19));
        zmm4 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm20));
        zmm5 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm21));
        zmm6 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm22));
        zmm7 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm23));

        // Convert back to float for argument reduction
        zmm16 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm0));
        zmm17 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm1));
        zmm18 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm2));
        zmm19 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm3));
        zmm20 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm4));
        zmm21 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm5));
        zmm22 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm6));
        zmm23 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm7));

        // Compute 2^k via bit manipulation in IEEE floating-point format
        zmm0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm0), 23));
        zmm1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm1), 23));
        zmm2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm2), 23));
        zmm3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm3), 23));
        zmm4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm4), 23));
        zmm5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm5), 23));
        zmm6 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm6), 23));
        zmm7 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm7), 23));

        zmm0 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm0), exp_bias));
        zmm1 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm1), exp_bias));
        zmm2 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm2), exp_bias));
        zmm3 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm3), exp_bias));
        zmm4 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm4), exp_bias));
        zmm5 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm5), exp_bias));
        zmm6 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm6), exp_bias));
        zmm7 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm7), exp_bias));

        // 4. Argument reduction: r = -x - k * ln2
        zmm8  = _mm256_fnmadd_ps(zmm16, ln2, zmm8);
        zmm9  = _mm256_fnmadd_ps(zmm17, ln2, zmm9);
        zmm10 = _mm256_fnmadd_ps(zmm18, ln2, zmm10);
        zmm11 = _mm256_fnmadd_ps(zmm19, ln2, zmm11);
        zmm12 = _mm256_fnmadd_ps(zmm20, ln2, zmm12);
        zmm13 = _mm256_fnmadd_ps(zmm21, ln2, zmm13);
        zmm14 = _mm256_fnmadd_ps(zmm22, ln2, zmm14);
        zmm15 = _mm256_fnmadd_ps(zmm23, ln2, zmm15);

        // 5. Horner's Scheme for Taylor Series: e^r = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120)))))
        // Stage 4 (i = 4)
        zmm16 = _mm256_fmadd_ps(zmm8,  c5, c4);
        zmm17 = _mm256_fmadd_ps(zmm9,  c5, c4);
        zmm18 = _mm256_fmadd_ps(zmm10, c5, c4);
        zmm19 = _mm256_fmadd_ps(zmm11, c5, c4);
        zmm20 = _mm256_fmadd_ps(zmm12, c5, c4);
        zmm21 = _mm256_fmadd_ps(zmm13, c5, c4);
        zmm22 = _mm256_fmadd_ps(zmm14, c5, c4);
        zmm23 = _mm256_fmadd_ps(zmm15, c5, c4);

        // Stage 3 (i = 3)
        c23 = _mm256_set1_ps(1.0f / 6.0f);
        zmm16 = _mm256_fmadd_ps(zmm8,  zmm16, c23);
        zmm17 = _mm256_fmadd_ps(zmm9,  zmm17, c23);
        zmm18 = _mm256_fmadd_ps(zmm10, zmm18, c23);
        zmm19 = _mm256_fmadd_ps(zmm11, zmm19, c23);
        zmm20 = _mm256_fmadd_ps(zmm12, zmm20, c23);
        zmm21 = _mm256_fmadd_ps(zmm13, zmm21, c23);
        zmm22 = _mm256_fmadd_ps(zmm14, zmm22, c23);
        zmm23 = _mm256_fmadd_ps(zmm15, zmm23, c23);

        // Stage 2 (i = 2)
        c23 = _mm256_set1_ps(0.5f);
        zmm16 = _mm256_fmadd_ps(zmm8,  zmm16, c23);
        zmm17 = _mm256_fmadd_ps(zmm9,  zmm17, c23);
        zmm18 = _mm256_fmadd_ps(zmm10, zmm18, c23);
        zmm19 = _mm256_fmadd_ps(zmm11, zmm19, c23);
        zmm20 = _mm256_fmadd_ps(zmm12, zmm20, c23);
        zmm21 = _mm256_fmadd_ps(zmm13, zmm21, c23);
        zmm22 = _mm256_fmadd_ps(zmm14, zmm22, c23);
        zmm23 = _mm256_fmadd_ps(zmm15, zmm23, c23);

        // Stage 1 (i = 1)
        zmm16 = _mm256_fmadd_ps(zmm8,  zmm16, c1);
        zmm17 = _mm256_fmadd_ps(zmm9,  zmm17, c1);
        zmm18 = _mm256_fmadd_ps(zmm10, zmm18, c1);
        zmm19 = _mm256_fmadd_ps(zmm11, zmm19, c1);
        zmm20 = _mm256_fmadd_ps(zmm12, zmm20, c1);
        zmm21 = _mm256_fmadd_ps(zmm13, zmm21, c1);
        zmm22 = _mm256_fmadd_ps(zmm14, zmm22, c1);
        zmm23 = _mm256_fmadd_ps(zmm15, zmm23, c1);

        // Stage 0 (i = 0) -> Yields e^r
        zmm16 = _mm256_fmadd_ps(zmm8,  zmm16, c1);
        zmm17 = _mm256_fmadd_ps(zmm9,  zmm17, c1);
        zmm18 = _mm256_fmadd_ps(zmm10, zmm18, c1);
        zmm19 = _mm256_fmadd_ps(zmm11, zmm19, c1);
        zmm20 = _mm256_fmadd_ps(zmm12, zmm20, c1);
        zmm21 = _mm256_fmadd_ps(zmm13, zmm21, c1);
        zmm22 = _mm256_fmadd_ps(zmm14, zmm22, c1);
        zmm23 = _mm256_fmadd_ps(zmm15, zmm23, c1);

        // Reload x from memory
        zmm8  = _mm256_loadu_ps(input_ptr);
        zmm9  = _mm256_loadu_ps(input_ptr + 8);
        zmm10 = _mm256_loadu_ps(input_ptr + 16);
        zmm11 = _mm256_loadu_ps(input_ptr + 24);
        zmm12 = _mm256_loadu_ps(input_ptr + 32);
        zmm13 = _mm256_loadu_ps(input_ptr + 40);
        zmm14 = _mm256_loadu_ps(input_ptr + 48);
        zmm15 = _mm256_loadu_ps(input_ptr + 56);

        // Denominator = 1 + (2^k * e^r)
        zmm16 = _mm256_fmadd_ps(zmm16, zmm0, c1);
        zmm17 = _mm256_fmadd_ps(zmm17, zmm1, c1);
        zmm18 = _mm256_fmadd_ps(zmm18, zmm2, c1);
        zmm19 = _mm256_fmadd_ps(zmm19, zmm3, c1);
        zmm20 = _mm256_fmadd_ps(zmm20, zmm4, c1);
        zmm21 = _mm256_fmadd_ps(zmm21, zmm5, c1);
        zmm22 = _mm256_fmadd_ps(zmm22, zmm6, c1);
        zmm23 = _mm256_fmadd_ps(zmm23, zmm7, c1);

        // 6. AVX-256 Reciprocal 
        zmm16 = _mm256_rcp14_ps(zmm16);
        zmm17 = _mm256_rcp14_ps(zmm17);
        zmm18 = _mm256_rcp14_ps(zmm18);
        zmm19 = _mm256_rcp14_ps(zmm19);
        zmm20 = _mm256_rcp14_ps(zmm20);
        zmm21 = _mm256_rcp14_ps(zmm21);
        zmm22 = _mm256_rcp14_ps(zmm22);
        zmm23 = _mm256_rcp14_ps(zmm23);

        // 7. Final Output: x * sigmoid(x)
        zmm0 = _mm256_mul_ps(zmm8,  zmm16);
        zmm1 = _mm256_mul_ps(zmm9,  zmm17);
        zmm2 = _mm256_mul_ps(zmm10, zmm18);
        zmm3 = _mm256_mul_ps(zmm11, zmm19);
        zmm4 = _mm256_mul_ps(zmm12, zmm20);
        zmm5 = _mm256_mul_ps(zmm13, zmm21);
        zmm6 = _mm256_mul_ps(zmm14, zmm22);
        zmm7 = _mm256_mul_ps(zmm15, zmm23);

        // Store 128 elements back to memory
        _mm256_storeu_ps(output_ptr,       zmm0);
        _mm256_storeu_ps(output_ptr + 8,  zmm1);
        _mm256_storeu_ps(output_ptr + 16,  zmm2);
        _mm256_storeu_ps(output_ptr + 24,  zmm3);
        _mm256_storeu_ps(output_ptr + 32,  zmm4);
        _mm256_storeu_ps(output_ptr + 40,  zmm5);
        _mm256_storeu_ps(output_ptr + 48,  zmm6);
        _mm256_storeu_ps(output_ptr + 56, zmm7);

        input_ptr  += 64;
        output_ptr += 64;
    }

    // Deal with the remaining use switch to interleave and not have dependency chains

    switch (d_left) {// loads
        case 7: zmm6 = _mm256_loadu_ps(input_ptr + 48);  // fallthrough
        case 6: zmm5 = _mm256_loadu_ps(input_ptr + 40);  // fallthrough
        case 5: zmm4 = _mm256_loadu_ps(input_ptr + 32);  // fallthrough
        case 4: zmm3 = _mm256_loadu_ps(input_ptr + 24);  // fallthrough
        case 3: zmm2 = _mm256_loadu_ps(input_ptr + 16);  // fallthrough
        case 2: zmm1 = _mm256_loadu_ps(input_ptr + 8);  // fallthrough
        case 1: zmm0 = _mm256_loadu_ps(input_ptr);
    }

    switch (d_left) {    // Negate using xor
        case 7: zmm14 = _mm256_xor_ps(zmm6, neg_mask); // fallthrough
        case 6: zmm13 = _mm256_xor_ps(zmm5, neg_mask); // fallthrough
        case 5: zmm12 = _mm256_xor_ps(zmm4, neg_mask); // fallthrough
        case 4: zmm11 = _mm256_xor_ps(zmm3, neg_mask); // fallthrough
        case 3: zmm10 = _mm256_xor_ps(zmm2, neg_mask); // fallthrough
        case 2: zmm9  = _mm256_xor_ps(zmm1, neg_mask); // fallthrough
        case 1: zmm8  = _mm256_xor_ps(zmm0, neg_mask);
    }
    // Calculating e^x, Taylor expansion centers around x = 0, so adjust to get better accuracy(especially neg values)
    switch (d_left) {    // Step 1: e^x = 2^k * e^r,  k = round(x*log2e), Find k
        case 7: zmm22 = _mm256_mul_ps(zmm14, log2e); // fallthrough
        case 6: zmm21 = _mm256_mul_ps(zmm13, log2e); // fallthrough
        case 5: zmm20 = _mm256_mul_ps(zmm12, log2e); // fallthrough
        case 4: zmm19 = _mm256_mul_ps(zmm11, log2e); // fallthrough
        case 3: zmm18 = _mm256_mul_ps(zmm10, log2e); // fallthrough
        case 2: zmm17 = _mm256_mul_ps(zmm9,  log2e); // fallthrough
        case 1: zmm16 = _mm256_mul_ps(zmm8,  log2e);
    }

    switch (d_left) { // Convert to int to do 2^k while also rounding
        case 7: zmm6 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm22)); // fallthrough
        case 6: zmm5 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm21)); // fallthrough
        case 5: zmm4 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm20)); // fallthrough
        case 4: zmm3 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm19)); // fallthrough
        case 3: zmm2 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm18)); // fallthrough
        case 2: zmm1 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm17)); // fallthrough
        case 1: zmm0 = _mm256_castsi256_ps(_mm256_cvtps_epi32(zmm16));
    }
    
    switch (d_left) { // Convert back to float to finish rounding
        case 7: zmm22 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm6)); // fallthrough
        case 6: zmm21 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm5)); // fallthrough
        case 5: zmm20 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm4)); // fallthrough
        case 4: zmm19 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm3)); // fallthrough
        case 3: zmm18 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm2)); // fallthrough
        case 2: zmm17 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm1)); // fallthrough
        case 1: zmm16 = _mm256_cvtepi32_ps(_mm256_castps_si256(zmm0));
    }

    switch (d_left) {    // shift k to the exponent bits
        case 7: zmm6 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm6), 23)); // fallthrough
        case 6: zmm5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm5), 23)); // fallthrough
        case 5: zmm4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm4), 23)); // fallthrough
        case 4: zmm3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm3), 23)); // fallthrough
        case 3: zmm2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm2), 23)); // fallthrough
        case 2: zmm1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm1), 23)); // fallthrough
        case 1: zmm0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(zmm0), 23));
    }
    
    switch (d_left) {    // Add the bias
        case 7: zmm6 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm6), exp_bias)); // fallthrough
        case 6: zmm5 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm5), exp_bias)); // fallthrough
        case 5: zmm4 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm4), exp_bias)); // fallthrough
        case 4: zmm3 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm3), exp_bias)); // fallthrough
        case 3: zmm2 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm2), exp_bias)); // fallthrough
        case 2: zmm1 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm1), exp_bias)); // fallthrough
        case 1: zmm0 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(zmm0), exp_bias));
    }

    switch (d_left) {   // Step 2: e^x = e^(kln2 + r), r = x - kln2  
        case 7: zmm14 = _mm256_fnmadd_ps(zmm22, ln2, zmm14); // fallthrough
        case 6: zmm13 = _mm256_fnmadd_ps(zmm21, ln2, zmm13); // fallthrough
        case 5: zmm12 = _mm256_fnmadd_ps(zmm20, ln2, zmm12); // fallthrough
        case 4: zmm11 = _mm256_fnmadd_ps(zmm19, ln2, zmm11); // fallthrough
        case 3: zmm10 = _mm256_fnmadd_ps(zmm18, ln2, zmm10); // fallthrough
        case 2: zmm9  = _mm256_fnmadd_ps(zmm17, ln2, zmm9);  // fallthrough
        case 1: zmm8  = _mm256_fnmadd_ps(zmm16, ln2, zmm8);
    }

    switch (d_left) {  //Taylor expansion: 1 + r +r^2/2 + r^3/6 + r^4/24+ r^5/120
        // i = 4, 1/24 + r/120      
        case 7: zmm22 = _mm256_fmadd_ps(zmm14, c5, c4); // fallthrough
        case 6: zmm21 = _mm256_fmadd_ps(zmm13, c5, c4); // fallthrough
        case 5: zmm20 = _mm256_fmadd_ps(zmm12, c5, c4); // fallthrough
        case 4: zmm19 = _mm256_fmadd_ps(zmm11, c5, c4); // fallthrough
        case 3: zmm18 = _mm256_fmadd_ps(zmm10, c5, c4); // fallthrough
        case 2: zmm17 = _mm256_fmadd_ps(zmm9,  c5, c4); // fallthrough
        case 1: zmm16 = _mm256_fmadd_ps(zmm8,  c5, c4);
    }
    
    c23 = _mm256_set1_ps(1.0f / 6.0f);
    switch (d_left) {   // i = 3, 1/6 + r/24 + r^2/120 
        case 7: zmm22 = _mm256_fmadd_ps(zmm14, zmm22, c23); // fallthrough
        case 6: zmm21 = _mm256_fmadd_ps(zmm13, zmm21, c23); // fallthrough
        case 5: zmm20 = _mm256_fmadd_ps(zmm12, zmm20, c23); // fallthrough
        case 4: zmm19 = _mm256_fmadd_ps(zmm11, zmm19, c23); // fallthrough
        case 3: zmm18 = _mm256_fmadd_ps(zmm10, zmm18, c23); // fallthrough
        case 2: zmm17 = _mm256_fmadd_ps(zmm9,  zmm17, c23); // fallthrough
        case 1: zmm16 = _mm256_fmadd_ps(zmm8,  zmm16, c23);
    }   
    
    c23 = _mm256_set1_ps(0.5f);
    switch (d_left) {    // i = 2, 1/2 + r/6 + r^2/24 + r^3/120
        case 7: zmm22 = _mm256_fmadd_ps(zmm14, zmm22, c23); // fallthrough
        case 6: zmm21 = _mm256_fmadd_ps(zmm13, zmm21, c23); // fallthrough
        case 5: zmm20 = _mm256_fmadd_ps(zmm12, zmm20, c23); // fallthrough
        case 4: zmm19 = _mm256_fmadd_ps(zmm11, zmm19, c23); // fallthrough
        case 3: zmm18 = _mm256_fmadd_ps(zmm10, zmm18, c23); // fallthrough
        case 2: zmm17 = _mm256_fmadd_ps(zmm9,  zmm17, c23); // fallthrough
        case 1: zmm16 = _mm256_fmadd_ps(zmm8,  zmm16, c23);
    }

    switch (d_left) {    // i = 1, 1 + r/2 + r^2/6 + r^3/24 + r^4/120
        case 7: zmm22 = _mm256_fmadd_ps(zmm14, zmm22, c1); // fallthrough
        case 6: zmm21 = _mm256_fmadd_ps(zmm13, zmm21, c1); // fallthrough
        case 5: zmm20 = _mm256_fmadd_ps(zmm12, zmm20, c1); // fallthrough
        case 4: zmm19 = _mm256_fmadd_ps(zmm11, zmm19, c1); // fallthrough
        case 3: zmm18 = _mm256_fmadd_ps(zmm10, zmm18, c1); // fallthrough
        case 2: zmm17 = _mm256_fmadd_ps(zmm9,  zmm17, c1); // fallthrough
        case 1: zmm16 = _mm256_fmadd_ps(zmm8,  zmm16, c1);

    }

    switch (d_left) {    // i = 0, 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
        case 7: zmm22 = _mm256_fmadd_ps(zmm14, zmm22, c1); // fallthrough
        case 6: zmm21 = _mm256_fmadd_ps(zmm13, zmm21, c1); // fallthrough
        case 5: zmm20 = _mm256_fmadd_ps(zmm12, zmm20, c1); // fallthrough
        case 4: zmm19 = _mm256_fmadd_ps(zmm11, zmm19, c1); // fallthrough
        case 3: zmm18 = _mm256_fmadd_ps(zmm10, zmm18, c1); // fallthrough
        case 2: zmm17 = _mm256_fmadd_ps(zmm9,  zmm17, c1); // fallthrough
        case 1: zmm16 = _mm256_fmadd_ps(zmm8,  zmm16, c1);
    }

    switch (d_left) {    // Reload x from storage
        case 7: zmm14 = _mm256_loadu_ps(input_ptr + 48);  // fallthrough
        case 6: zmm13 = _mm256_loadu_ps(input_ptr + 40);  // fallthrough
        case 5: zmm12 = _mm256_loadu_ps(input_ptr + 32);  // fallthrough
        case 4: zmm11 = _mm256_loadu_ps(input_ptr + 24);  // fallthrough
        case 3: zmm10 = _mm256_loadu_ps(input_ptr + 16);  // fallthrough
        case 2: zmm9 = _mm256_loadu_ps(input_ptr + 8);  // fallthrough
        case 1: zmm8 = _mm256_loadu_ps(input_ptr);
    }

    switch (d_left) {    // denom calculation
        case 7: zmm22 = _mm256_fmadd_ps(zmm22, zmm6, c1);  // fallthrough
        case 6: zmm21 = _mm256_fmadd_ps(zmm21, zmm5, c1);  // fallthrough
        case 5: zmm20 = _mm256_fmadd_ps(zmm20, zmm4, c1);  // fallthrough
        case 4: zmm19 = _mm256_fmadd_ps(zmm19, zmm3, c1);  // fallthrough
        case 3: zmm18 = _mm256_fmadd_ps(zmm18, zmm2, c1);  // fallthrough
        case 2: zmm17 = _mm256_fmadd_ps(zmm17, zmm1, c1);  // fallthrough
        case 1: zmm16 = _mm256_fmadd_ps(zmm16, zmm0, c1);  // fallthrough
    }
    
    switch (d_left) {    // Find Reciprocal
        case 7: zmm22 = _mm256_rcp14_ps(zmm22); // fallthrough
        case 6: zmm21 = _mm256_rcp14_ps(zmm21); // fallthrough
        case 5: zmm20 = _mm256_rcp14_ps(zmm20); // fallthrough
        case 4: zmm19 = _mm256_rcp14_ps(zmm19); // fallthrough
        case 3: zmm18 = _mm256_rcp14_ps(zmm18); // fallthrough
        case 2: zmm17 = _mm256_rcp14_ps(zmm17); // fallthrough
        case 1: zmm16 = _mm256_rcp14_ps(zmm16);
    }
    
    switch (d_left) {    // Final calculation
        case 7: zmm6 = _mm256_mul_ps(zmm14, zmm22); // fallthrough
        case 6: zmm5 = _mm256_mul_ps(zmm13, zmm21); // fallthrough
        case 5: zmm4 = _mm256_mul_ps(zmm12, zmm20); // fallthrough
        case 4: zmm3 = _mm256_mul_ps(zmm11, zmm19); // fallthrough
        case 3: zmm2 = _mm256_mul_ps(zmm10, zmm18); // fallthrough
        case 2: zmm1 = _mm256_mul_ps(zmm9,  zmm17); // fallthrough
        case 1: zmm0 = _mm256_mul_ps(zmm8,  zmm16);
    }

    switch (d_left) {    // Store outputs
        case 7: _mm256_storeu_ps(output_ptr + 48, zmm6); // fallthrough
        case 6: _mm256_storeu_ps(output_ptr + 40, zmm5); // fallthrough
        case 5: _mm256_storeu_ps(output_ptr + 32, zmm4); // fallthrough
        case 4: _mm256_storeu_ps(output_ptr + 24, zmm3); // fallthrough
        case 3: _mm256_storeu_ps(output_ptr + 16, zmm2); // fallthrough
        case 2: _mm256_storeu_ps(output_ptr + 8, zmm1); // fallthrough
        case 1: _mm256_storeu_ps(output_ptr,      zmm0);
    }
    //return output; 
}

