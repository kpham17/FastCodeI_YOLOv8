#include <immintrin.h>

// 4x24 kernel for mxm multiplication
inline void kernel_4x24(
  int k, 
  float* restrict a, 
  float* restrict b, 
  float* restrict c, 
  int out_channel
)
 {

  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
  __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
    
  float* bptr = b;
  float* aptr = a;

  int k_iter = k / 4;
  int k_left = k % 4;

  ymm0 = _mm256_setzero_ps();  ymm1 = _mm256_setzero_ps();
  ymm2 = _mm256_setzero_ps();  ymm3 = _mm256_setzero_ps();
  ymm4 = _mm256_setzero_ps();  ymm5 = _mm256_setzero_ps();
  ymm6 = _mm256_setzero_ps();  ymm7 = _mm256_setzero_ps();
  ymm8 = _mm256_setzero_ps();  ymm9 = _mm256_setzero_ps();
  ymm10 = _mm256_setzero_ps();  ymm11 = _mm256_setzero_ps();

  for (int p = 0; p != k_iter; ++p) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      ymm15 = _mm256_load_ps(bptr);
      ymm14 = _mm256_load_ps(bptr+8);
      ymm13 = _mm256_load_ps(bptr+16);
      
      ymm12 = _mm256_broadcast_ss(aptr+0);
      ymm0 = _mm256_fmadd_ps(ymm12, ymm15, ymm0);
      ymm1 = _mm256_fmadd_ps(ymm12, ymm14, ymm1);
      ymm2 = _mm256_fmadd_ps(ymm12, ymm13, ymm2);
      
      ymm12 = _mm256_broadcast_ss(aptr+1);
      ymm3 = _mm256_fmadd_ps(ymm12, ymm15, ymm3);
      ymm4 = _mm256_fmadd_ps(ymm12, ymm14, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm12, ymm13, ymm5);
 
      ymm12 = _mm256_broadcast_ss(aptr+2);
      ymm6 = _mm256_fmadd_ps(ymm12, ymm15, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm12, ymm14, ymm7);
      ymm8 = _mm256_fmadd_ps(ymm12, ymm13, ymm8);
 
      ymm12 = _mm256_broadcast_ss(aptr+3);
      ymm9 = _mm256_fmadd_ps(ymm12, ymm15, ymm9);
      ymm10 = _mm256_fmadd_ps(ymm12, ymm14, ymm10);
      ymm11 = _mm256_fmadd_ps(ymm12, ymm13, ymm11);

      aptr += 4;
      bptr += 24;
    }
  }

  for (int p = 0; p != k_left; ++p)
    {
      ymm15 = _mm256_load_ps(bptr);
      ymm14 = _mm256_load_ps(bptr+8);
      ymm13 = _mm256_load_ps(bptr+16);
      
      ymm12 = _mm256_broadcast_ss(aptr+0);
      ymm0 = _mm256_fmadd_ps(ymm12, ymm15, ymm0);
      ymm1 = _mm256_fmadd_ps(ymm12, ymm14, ymm1);
      ymm2 = _mm256_fmadd_ps(ymm12, ymm13, ymm2);
      
      ymm12 = _mm256_broadcast_ss(aptr+1);
      ymm3 = _mm256_fmadd_ps(ymm12, ymm15, ymm3);
      ymm4 = _mm256_fmadd_ps(ymm12, ymm14, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm12, ymm13, ymm5);
 
      ymm12 = _mm256_broadcast_ss(aptr+2);
      ymm6 = _mm256_fmadd_ps(ymm12, ymm15, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm12, ymm14, ymm7);
      ymm8 = _mm256_fmadd_ps(ymm12, ymm13, ymm8);
 
      ymm12 = _mm256_broadcast_ss(aptr+3);
      ymm9 = _mm256_fmadd_ps(ymm12, ymm15, ymm9);
      ymm10 = _mm256_fmadd_ps(ymm12, ymm14, ymm10);
      ymm11 = _mm256_fmadd_ps(ymm12, ymm13, ymm11);

      aptr += 4;
      bptr += 24;

    }

  float* c0, *c1, *c2, *c3;
  // store output not in blocks but for each vec
  c0 = c;                  c1 = c0 + out_channel; 
  c2 = c1 + out_channel  ; c3 = c2 + out_channel;
    
  // store outputs
  _mm256_stream_ps(c0+0, ymm0);        _mm256_stream_ps(c0+8, ymm1);        _mm256_stream_ps(c0+16, ymm2);
  _mm256_stream_ps(c1+0, ymm3);        _mm256_stream_ps(c1+8, ymm4);        _mm256_stream_ps(c1+16, ymm5);
  _mm256_stream_ps(c2+0, ymm6);        _mm256_stream_ps(c2+8, ymm7);        _mm256_stream_ps(c2+16, ymm8);
  _mm256_stream_ps(c3+0, ymm9);        _mm256_stream_ps(c3+8, ymm10);        _mm256_stream_ps(c3+16, ymm11);

 }


// output channel is definitely a multiple of 8 
// 4xR kernel for edge cases
inline void kernel_4xR
(
 int k,
 float* restrict a, // input A matrix
 float* restrict b, // input B matrix (pointer), assume that is already in the correct format
 float* restrict c, // pointer to location on output matrix
 int R, // value is only 8 or 16
 int out_channel
 ){
  if ((R != 8) && (R!=16)) return;

  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
  __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
    
  float* bptr = b;
  float* aptr = a;

  int k_iter = k / 4;
  int k_left = k % 4;

  ymm0 = _mm256_setzero_ps();  ymm1 = _mm256_setzero_ps();
  ymm2 = _mm256_setzero_ps();  ymm3 = _mm256_setzero_ps();
  ymm4 = _mm256_setzero_ps();  ymm5 = _mm256_setzero_ps();
  ymm6 = _mm256_setzero_ps();  ymm7 = _mm256_setzero_ps();
  ymm8 = _mm256_setzero_ps();  ymm9 = _mm256_setzero_ps();
  ymm10 = _mm256_setzero_ps();  ymm11 = _mm256_setzero_ps();

  for (int p = 0; p != k_iter; ++p) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      ymm15 = _mm256_load_ps(bptr);
      
      ymm13 = _mm256_broadcast_ss(aptr+0);
      ymm12 = _mm256_broadcast_ss(aptr+1);
      ymm11 = _mm256_broadcast_ss(aptr+2);
      ymm10 = _mm256_broadcast_ss(aptr+3);
      switch (R) {
        case 16: // Do Upper Channel
          ymm14 = _mm256_load_ps(bptr+8);
          ymm1 = _mm256_fmadd_ps(ymm13, ymm14, ymm1);
          ymm3 = _mm256_fmadd_ps(ymm12, ymm14, ymm3);
          ymm5 = _mm256_fmadd_ps(ymm11, ymm14, ymm5);
          ymm7 = _mm256_fmadd_ps(ymm10, ymm14, ymm7);
        default: // Fall through and do lower channel
          ymm0 = _mm256_fmadd_ps(ymm13, ymm15, ymm0);
          ymm2 = _mm256_fmadd_ps(ymm12, ymm15, ymm2);
          ymm4 = _mm256_fmadd_ps(ymm11, ymm15, ymm4);
          ymm6 = _mm256_fmadd_ps(ymm10, ymm15, ymm6);
          break;
      }
      aptr += 4;
      bptr += R;
    }
  }

  for (int p = 0; p != k_left; ++p)
    {
      ymm15 = _mm256_load_ps(bptr);
      
      ymm13 = _mm256_broadcast_ss(aptr+0);
      ymm12 = _mm256_broadcast_ss(aptr+1);
      ymm11 = _mm256_broadcast_ss(aptr+2);
      ymm10 = _mm256_broadcast_ss(aptr+3);
      switch (R) {
        case 16: 
          ymm14 = _mm256_load_ps(bptr+8);
          ymm1 = _mm256_fmadd_ps(ymm13, ymm14, ymm1);
          ymm3 = _mm256_fmadd_ps(ymm12, ymm14, ymm3);
          ymm5 = _mm256_fmadd_ps(ymm11, ymm14, ymm5);
          ymm7 = _mm256_fmadd_ps(ymm10, ymm14, ymm7);
        default:
          ymm0 = _mm256_fmadd_ps(ymm13, ymm15, ymm0);
          ymm2 = _mm256_fmadd_ps(ymm12, ymm15, ymm2);
          ymm4 = _mm256_fmadd_ps(ymm11, ymm15, ymm4);
          ymm6 = _mm256_fmadd_ps(ymm10, ymm15, ymm6);
          break;
      }

      aptr += 4;
      bptr += R;

    }

  float* c0, *c1, *c2, *c3;
  // store output not in blocks but for each vec
  c0 = c;                  c1 = c0 + out_channel; 
  c2 = c1 + out_channel  ; c3 = c2 + out_channel;
    
  // store outputs
  switch (R) {
    case 16: 
      _mm256_stream_ps(c0+8, ymm1);        
      _mm256_stream_ps(c1+8, ymm3);      
      _mm256_stream_ps(c2+8, ymm5);      
      _mm256_stream_ps(c3+8, ymm7);        
    case 8: // store lower channel
      _mm256_stream_ps(c0, ymm0);        
      _mm256_stream_ps(c1, ymm2);      
      _mm256_stream_ps(c2, ymm4);      
      _mm256_stream_ps(c3, ymm6);        
    default: // store lower channel
      break;
  }  
}
