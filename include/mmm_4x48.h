#include <immintrin.h>

// 4x48 kernel for mxm multiplication
// Make use of AVX 512
inline void kernel_4x48(
  int k, 
  float* restrict a, 
  float* restrict b, 
  float* restrict c, 
  int out_channel
)
 {

  __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
  __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
  __m512 zmm16;
    
  float* bptr = b;
  float* aptr = a;

  int k_iter = k / 4;
  int k_left = k % 4;

  zmm0 = _mm512_setzero_ps();  zmm1 = _mm512_setzero_ps();
  zmm2 = _mm512_setzero_ps();  zmm3 = _mm512_setzero_ps();
  zmm4 = _mm512_setzero_ps();  zmm5 = _mm512_setzero_ps();
  zmm6 = _mm512_setzero_ps();  zmm7 = _mm512_setzero_ps();
  zmm8 = _mm512_setzero_ps();  zmm9 = _mm512_setzero_ps();
  zmm10 = _mm512_setzero_ps();  zmm11 = _mm512_setzero_ps();
  zmm12 = _mm512_setzero_ps();  zmm13 = _mm512_setzero_ps();
  zmm14 = _mm512_setzero_ps();  zmm15 = _mm512_setzero_ps();
  zmm16 = _mm512_setzero_ps(); 
  for (int p = 0; p != k_iter; ++p) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      zmm15 = _mm512_load_ps(bptr);
      zmm14 = _mm512_load_ps(bptr+16);
      zmm13 = _mm512_load_ps(bptr+32);
      
      zmm12 = _mm512_set1_ps(*(aptr+0));
      zmm0  = _mm512_fmadd_ps(zmm12, zmm15, zmm0);
      zmm1  = _mm512_fmadd_ps(zmm12, zmm14, zmm1);
      zmm2  = _mm512_fmadd_ps(zmm12, zmm13, zmm2);
      
      zmm16 = _mm512_set1_ps(*(aptr+1));
      zmm3  = _mm512_fmadd_ps(zmm16, zmm15, zmm3);
      zmm4  = _mm512_fmadd_ps(zmm16, zmm14, zmm4);
      zmm5  = _mm512_fmadd_ps(zmm16, zmm13, zmm5);
 
      zmm12 = _mm512_set1_ps(*(aptr+2));
      zmm6  = _mm512_fmadd_ps(zmm12, zmm15, zmm6);
      zmm7  = _mm512_fmadd_ps(zmm12, zmm14, zmm7);
      zmm8  = _mm512_fmadd_ps(zmm12, zmm13, zmm8);
 
      zmm16 = _mm512_set1_ps(*(aptr+3));
      zmm9  = _mm512_fmadd_ps(zmm16, zmm15, zmm9);
      zmm10 = _mm512_fmadd_ps(zmm16, zmm14, zmm10);
      zmm11 = _mm512_fmadd_ps(zmm16, zmm13, zmm11);

      aptr += 4;
      bptr += 48;
    }
  }

  for (int p = 0; p != k_left; ++p)
    {
      zmm15 = _mm512_load_ps(bptr);
      zmm14 = _mm512_load_ps(bptr+16);
      zmm13 = _mm512_load_ps(bptr+32);
      
      zmm12 = _mm512_set1_ps(*(aptr+0));
      zmm0  = _mm512_fmadd_ps(zmm12, zmm15, zmm0);
      zmm1  = _mm512_fmadd_ps(zmm12, zmm14, zmm1);
      zmm2  = _mm512_fmadd_ps(zmm12, zmm13, zmm2);
      
      zmm16 = _mm512_set1_ps(*(aptr+1));  
      zmm3  = _mm512_fmadd_ps(zmm16, zmm15, zmm3);
      zmm4  = _mm512_fmadd_ps(zmm16, zmm14, zmm4);
      zmm5  = _mm512_fmadd_ps(zmm16, zmm13, zmm5);
 
      zmm12 = _mm512_set1_ps(*(aptr+2));
      zmm6  = _mm512_fmadd_ps(zmm12, zmm15, zmm6);
      zmm7  = _mm512_fmadd_ps(zmm12, zmm14, zmm7);
      zmm8  = _mm512_fmadd_ps(zmm12, zmm13, zmm8);
 
      zmm16 = _mm512_set1_ps(*(aptr+3));
      zmm9  = _mm512_fmadd_ps(zmm16, zmm15, zmm9);
      zmm10 = _mm512_fmadd_ps(zmm16, zmm14, zmm10);
      zmm11 = _mm512_fmadd_ps(zmm16, zmm13, zmm11);

      aptr += 4;
      bptr += 48;

    }

  float* c0, *c1, *c2, *c3;
  // store output not in blocks but for each vec
  c0 = c;                  c1 = c0 + out_channel; 
  c2 = c1 + out_channel  ; c3 = c2 + out_channel;
    
  // store outputs
  _mm512_store_ps(c0+0, zmm0);        _mm512_store_ps(c0+16, zmm1);        _mm512_store_ps(c0+32, zmm2);
  _mm512_store_ps(c1+0, zmm3);        _mm512_store_ps(c1+16, zmm4);        _mm512_store_ps(c1+32, zmm5);
  _mm512_store_ps(c2+0, zmm6);        _mm512_store_ps(c2+16, zmm7);        _mm512_store_ps(c2+32, zmm8);
  _mm512_store_ps(c3+0, zmm9);        _mm512_store_ps(c3+16, zmm10);        _mm512_store_ps(c3+32, zmm11);

 }


// output channel is definitely a multiple of 16 
// 4xR kernel for edge cases
inline void kernel_4xR (
 int k,
 float* restrict a, // input A matrix
 float* restrict b, // input B matrix (pointer), assume that is already in the correct format
 float* restrict c, // pointer to location on output matrix
 int R, // value is only 8 or 16
 int out_channel
 ){
  if ((R != 16) && (R!=32)) return;

  __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
  __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    
  float* bptr = b;
  float* aptr = a;

  int k_iter = k / 4;
  int k_left = k % 4;
  
  zmm0 = _mm512_setzero_ps();  zmm1 = _mm512_setzero_ps();
  zmm2 = _mm512_setzero_ps();  zmm3 = _mm512_setzero_ps();
  zmm4 = _mm512_setzero_ps();  zmm5 = _mm512_setzero_ps();
  zmm6 = _mm512_setzero_ps();  zmm7 = _mm512_setzero_ps();
  zmm8 = _mm512_setzero_ps();  zmm9 = _mm512_setzero_ps();
  zmm10 = _mm512_setzero_ps();  zmm11 = _mm512_setzero_ps();
  zmm12 = _mm512_setzero_ps();  zmm13 = _mm512_setzero_ps();
  zmm14 = _mm512_setzero_ps();  zmm15 = _mm512_setzero_ps();


  for (int p = 0; p != k_iter; ++p) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      zmm15 = _mm512_load_ps(bptr);
      
      zmm13 = _mm512_set1_ps(*(aptr+0));
      zmm12 = _mm512_set1_ps(*(aptr+1));
      zmm11 = _mm512_set1_ps(*(aptr+2));
      zmm10 = _mm512_set1_ps(*(aptr+3));
      switch (R) {
        case 32: // Do Upper Channel
          zmm14 = _mm512_load_ps(bptr+16);
          zmm1 = _mm512_fmadd_ps(zmm13, zmm14, zmm1);
          zmm3 = _mm512_fmadd_ps(zmm12, zmm14, zmm3);
          zmm5 = _mm512_fmadd_ps(zmm11, zmm14, zmm5);
          zmm7 = _mm512_fmadd_ps(zmm10, zmm14, zmm7);
        default: // Fall through and do lower channel
          zmm0 = _mm512_fmadd_ps(zmm13, zmm15, zmm0);
          zmm2 = _mm512_fmadd_ps(zmm12, zmm15, zmm2);
          zmm4 = _mm512_fmadd_ps(zmm11, zmm15, zmm4);
          zmm6 = _mm512_fmadd_ps(zmm10, zmm15, zmm6);
          break;
      }
      aptr += 4;
      bptr += R;
    }
  }

  for (int p = 0; p != k_left; ++p)
    {
      zmm15 = _mm512_load_ps(bptr);
      
      zmm13 = _mm512_set1_ps(*(aptr+0));
      zmm12 = _mm512_set1_ps(*(aptr+1));
      zmm11 = _mm512_set1_ps(*(aptr+2));
      zmm10 = _mm512_set1_ps(*(aptr+3));
      switch (R) {
        case 32: 
          zmm14 = _mm512_load_ps(bptr+16);
          zmm1 = _mm512_fmadd_ps(zmm13, zmm14, zmm1);
          zmm3 = _mm512_fmadd_ps(zmm12, zmm14, zmm3);
          zmm5 = _mm512_fmadd_ps(zmm11, zmm14, zmm5);
          zmm7 = _mm512_fmadd_ps(zmm10, zmm14, zmm7);
        case 16:
          zmm0 = _mm512_fmadd_ps(zmm13, zmm15, zmm0);
          zmm2 = _mm512_fmadd_ps(zmm12, zmm15, zmm2);
          zmm4 = _mm512_fmadd_ps(zmm11, zmm15, zmm4);
          zmm6 = _mm512_fmadd_ps(zmm10, zmm15, zmm6);
        default:
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
    case 32: 
      _mm512_store_ps(c0+16, zmm1);        
      _mm512_store_ps(c1+16, zmm3);      
      _mm512_store_ps(c2+16, zmm5);      
      _mm512_store_ps(c3+16, zmm7);        
    default: // store lower channel
      _mm512_store_ps(c0, zmm0);        
      _mm512_store_ps(c1, zmm2);      
      _mm512_store_ps(c2, zmm4);      
      _mm512_store_ps(c3, zmm6);        
      break;
  }  
}
