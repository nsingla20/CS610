// Compile: g++ -msse4 -mavx2 -mavx512f -march=native -O3 -fopt-info-vec-optimized -fopt-info-vec-missed -o problem3 problem3.cpp

#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <stdint.h>
#include <string>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 30)
#define SSE_WIDTH_BITS (128)
#define AVX2_WIDTH_BITS (256)
#define AVX512_WIDTH_BITS (512)
#define ALIGN (64)

void print_array(int* array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}

__attribute__((optimize("no-tree-vectorize"))) int ref_version(int* __restrict__ source,
                                                               int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
  for (int i = 0; i < N; i++) {
    tmp += source[i];
    dest[i] = tmp;
  }
  return tmp;
}

int omp_version(int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
#pragma omp simd reduction(inscan, + : tmp)
  for (int i = 0; i < N; i++) {
    tmp += source[i];
#pragma omp scan inclusive(tmp)
    dest[i] = tmp;
  }
  return tmp;
}

void print128i_u32(__m128i var, int start) {
  alignas(ALIGN) uint32_t val[4];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [" << start << ":" << start + 3 << "]: " << val[0] << " " << val[1] << " "
       << val[2] << " " << val[3] << "\n";
}

void print128i_u64(__m128i var) {
  alignas(ALIGN) uint64_t val[2];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [0:2]: " << val[0] << " " << val[1] << "\n";
}

// Tree reduction idea on every 128 bits vector data, involves 2 shifts, 3 adds, one broadcast
int sse4_version(int* __restrict__ source, int* __restrict__ dest) {
  source = (int*)__builtin_assume_aligned(source, ALIGN);
  dest = (int*)__builtin_assume_aligned(dest, ALIGN);
  // _MM_SHUFFLE(z, y, x, w) macro forms an integer mask according to the formula (z << 6) | (y <<
  // 4) | (x << 2) | w.
  const int mask = _MM_SHUFFLE(3, 3, 3, 3);

  // Return vector of type __m128i with all elements set to zero, to be added as previous sum for
  // the first four elements.
  __m128i offset = _mm_setzero_si128();

  const int stride = SSE_WIDTH_BITS / (sizeof(int) * 8);
  for (int i = 0; i < N; i += stride) {
    // Load 128-bits of integer data from memory into x. source_addr must be aligned on a 16-byte
    // boundary to be safe.
    __m128i x = _mm_load_si128((__m128i*)&source[i]);
    // Let the numbers in x be [d,c,b,a], where a is at source[i].
    __m128i tmp1 = _mm_slli_si128(x, 4); // Shift x left by 4 bytes while shifting in zeros.
    // tmp1 becomes [c,b,a,0].
    x = _mm_add_epi32(x, tmp1); // Add packed 32-bit integers in x and tmp1
    // x becomes [c+d,b+c,a+b,a].
    tmp1 = _mm_slli_si128(x, 8); // Shift x left by 8 bytes while shifting in zeros.
    // So, tmp1 becomes [a+b,a,0,0].
    __m128i out = _mm_add_epi32(x, tmp1); // Add packed 32-bit integers in x and tmp1.
    // out contains [a+b+c+d,a+b+c,a+b,a].
    out = _mm_add_epi32(out, offset);
    // out now includes the sum from the previous set of numbers, given by offset.
    // Store 128-bits of integer data from out into memory. dest_addr must be aligned on a 16-byte
    // boundary to be safe.
    _mm_store_si128((__m128i*)&dest[i], out);
    // Bits [7:0] of mask are 11111111 to pick the third integer (11) from out (i.e., a+b+c+d).
    // Shuffle 32-bit integers in out using the control in mask.
    offset = _mm_shuffle_epi32(out, mask);
    // offset now contains 4 copies of a+b+c+d.
  }
  return dest[N - 1];
}

int avx2_version(int* source, int* dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  const int mask = _MM_SHUFFLE(3, 3, 3, 3);

  const int stride = AVX2_WIDTH_BITS / (sizeof(int) * 8);
  __m128i offset = _mm_setzero_si128();

  for (int i = 0; i < N; i += stride) {

    __m256i x;
    x = _mm256_load_si256((__m256i*)&source[i]);

    // parallel 128 bits
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));

    __m128i l,h;
    l = _mm256_extractf128_si256(x,0);
    l = _mm_add_epi32(l, offset);
    offset = _mm_shuffle_epi32(l, mask);

    h = _mm256_extractf128_si256(x,1);
    h = _mm_add_epi32(h, offset);
    offset = _mm_shuffle_epi32(h, mask);

    x = _mm256_set_m128i(h, l);

    _mm256_store_si256((__m256i*)&dest[i], x);


  }
  return dest[N - 1];
}
int avx2_version3(int* source, int* dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  const int mask = _MM_SHUFFLE(3, 3, 3, 3);

  const int stride = AVX2_WIDTH_BITS / (sizeof(int) * 8);
  // __m128i offset = _mm_setzero_si128();
  __m256i offset = _mm256_setzero_si256();
  for (int i = 0; i < N; i += stride) {

    __m256i x;
    x = _mm256_load_si256((__m256i*)&source[i]);

    // parallel 128 bits
    x = _mm256_add_epi32(x, _mm256_bslli_epi128(x, 4));
    x = _mm256_add_epi32(x, _mm256_bslli_epi128(x, 8));

    __m256i tmp = _mm256_permutevar8x32_epi32(x, _mm256_set_epi32(3, 3, 3, 3, 0, 0, 0, 0));
    tmp = tmp = _mm256_and_si256(tmp, _mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0));
    x = _mm256_add_epi32(x, tmp);

    // x = _mm256_set_m128i(h, l);
    x = _mm256_add_epi32(x, offset);
    offset = _mm256_set1_epi32(_mm256_extract_epi32(x, 7));

    _mm256_store_si256((__m256i*)&dest[i], x);
  }
  return dest[N - 1];
}
int avx2_version2(int* source, int* dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  const int mask = _MM_SHUFFLE(3, 3, 3, 3);

 __m256i offset = _mm256_setzero_si256();

  const int stride = AVX2_WIDTH_BITS / (sizeof(int) * 8);
  for (int i = 0; i < N; i += stride) {

    __m256i x,tmp,mask;
    x = _mm256_load_si256((__m256i*)&source[i]);
    // Shift 4 bytes
    tmp = _mm256_permute2x128_si256(x, x, _MM_SHUFFLE(0,0,3,0) );
    tmp = _mm256_alignr_epi8(x,tmp,12);
    x = _mm256_add_epi32(x, tmp);

    //Shift 8 bytes
    tmp = _mm256_permute2x128_si256(x, x, _MM_SHUFFLE(0,0,3,0) );
    tmp = _mm256_alignr_epi8(x,tmp,8);
    x = _mm256_add_epi32(x, tmp);

    //Shift 16 bytes
    tmp = _mm256_permute2x128_si256(x, x, 41);
    x = _mm256_add_epi32(x, tmp);

    x = _mm256_add_epi32(x, offset);
    offset = _mm256_set1_epi32(_mm256_extract_epi32(x, 7));
    _mm256_store_si256((__m256i*)&dest[i], x);

  }

  return dest[N - 1];
}
int avx2_version1(int* source, int* dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  const int stride = AVX2_WIDTH_BITS / (sizeof(int) * 8);

  __m256i offset = _mm256_setzero_si256();

  for (int i = 0; i < N; i += stride) {

    __m256i x, tmp;
    x = _mm256_load_si256((__m256i*)&source[i]);

    //4 bytes
    tmp = _mm256_permutevar8x32_epi32(x, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0));
    tmp = _mm256_and_si256(tmp, _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, 0));
    x = _mm256_add_epi32(x, tmp);

    //8 bytes
    tmp = _mm256_permutevar8x32_epi32(x, _mm256_set_epi32(5, 4, 3, 2, 1, 0, 0, 0));
    tmp = _mm256_and_si256(tmp, _mm256_set_epi32(-1, -1, -1, -1, -1, -1, 0, 0));
    x = _mm256_add_epi32(x, tmp);

    //16 bytes
    tmp = _mm256_permutevar8x32_epi32(x, _mm256_set_epi32(4, 3, 2, 1, 0, 0, 0, 0));
    tmp = _mm256_and_si256(tmp, _mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0));
    x = _mm256_add_epi32(x, tmp);

    x = _mm256_add_epi32(x, offset);

    _mm256_store_si256((__m256i*)&dest[i], x);

    offset = _mm256_set1_epi32(_mm256_extract_epi32(x, 7));
  }

  return dest[N - 1];
}

int avx512_version(int* source, int* dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  __m512i offset = _mm512_setzero_si512();
  // __mmask16 _cvtu32_mask16 (unsigned int a)
  const __mmask16 s4 = _cvtu32_mask16(0b1111111111111110);
  const __m512i p4 = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0);

  const __mmask16 s8 = _cvtu32_mask16(0b1111111111111100);
  const __m512i p8 = _mm512_set_epi32(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0);

  const __mmask16 s16 = _cvtu32_mask16(0b1111111111110000);
  const __m512i p16 = _mm512_set_epi32(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0);

  const __mmask16 s32 = _cvtu32_mask16(0b1111111100000000);
  const __m512i p32 = _mm512_set_epi32(8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0);

  const __m512i po = _mm512_set_epi32(15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15);

  const int stride = AVX512_WIDTH_BITS / (sizeof(int) * 8);
  for (int i = 0; i < N; i += stride) {
    __m512i x = _mm512_load_si512((__m512i*)&source[i]);

    // _mm512_bslli_epi128

    // _mm512_maskz_permutexvar_epi32 (__mmask16 k, __m512i idx, __m512i a)
    // shift 4 bytes
    __m512i tmp = _mm512_maskz_permutexvar_epi32(s4, p4, x);
    x = _mm512_add_epi32(x, tmp);

    // shift 8 bytes
    tmp = _mm512_maskz_permutexvar_epi32(s8, p8, x);
    x = _mm512_add_epi32(x, tmp);

    // shift 16 bytes
    tmp = _mm512_maskz_permutexvar_epi32(s16, p16, x);
    x = _mm512_add_epi32(x, tmp);

    // shift 32 bytes
    tmp = _mm512_maskz_permutexvar_epi32(s32, p32, x);
    x = _mm512_add_epi32(x, tmp);

    x = _mm512_add_epi32(x, offset);

    _mm512_store_si512((__m512i*)&dest[i], x);

    offset = _mm512_permutexvar_epi32(po, x);
  }

  return dest[N - 1];
}


__attribute__((optimize("no-tree-vectorize"))) int main() {
  int* array = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(array, array + N, 1);

  int* ref_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(ref_res, ref_res + N, 0);
  HRTimer start = HR::now();
  int val_ser = ref_version(array, ref_res);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial version: " << val_ser << " time: " << duration << endl;

  int* omp_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(omp_res, omp_res + N, 0);
  start = HR::now();
  int val_omp = omp_version(array, omp_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_omp || printf("OMP result is wrong!\n"));
  cout << "OMP version: " << val_omp << " time: " << duration << endl;
  delete[] omp_res;

  int* sse_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(sse_res, sse_res + N, 0);
  start = HR::now();
  int val_sse = sse4_version(array, sse_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_sse || printf("SSE result is wrong!\n"));
  cout << "SSE version: " << val_sse << " time: " << duration << endl;
  delete[] sse_res;

  int* avx2_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  int val_avx2;
  std::fill(avx2_res, avx2_res + N, 0);
  start = HR::now();
  val_avx2 = avx2_version(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version: " << val_avx2 << " time: " << duration << endl;

  std::fill(avx2_res, avx2_res + N, 0);
  start = HR::now();
  val_avx2 = avx2_version1(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version1: " << val_avx2 << " time: " << duration << endl;

  std::fill(avx2_res, avx2_res + N, 0);
  start = HR::now();
  val_avx2 = avx2_version2(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version2: " << val_avx2 << " time: " << duration << endl;

  std::fill(avx2_res, avx2_res + N, 0);
  start = HR::now();
  val_avx2 = avx2_version3(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version3: " << val_avx2 << " time: " << duration << endl;

  int val_avx512;
  std::fill(avx2_res, avx2_res + N, 0);
  start = HR::now();
  val_avx512 = avx512_version(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx512 || printf("AVX512 result is wrong!\n"));
  cout << "AVX512 version: " << val_avx512 << " time: " << duration << endl;

  return EXIT_SUCCESS;
}
