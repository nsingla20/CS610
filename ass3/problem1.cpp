// Compile: g++ -O2 -mavx -mavx2 -march=native -o problem1 problem1.cpp

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <emmintrin.h>
#include <immintrin.h>

using std::cout;
using std::endl;
using std::ios;

const int N = (1 << 13);
const int Niter = 10;
const double THRESHOLD = 0.000001;

__attribute__((aligned(16))) double A_align[N][N],z_opt_align[N],x_align[N],y_opt_align[N];

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double** A, const double* x, double* y_ref, double* z_ref) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(const double* w_ref, const double* w_opt) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

// TODO: INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THE CODE
// You can create multiple versions of the optimized() function to test your changes
void optimized(double** __restrict__ A, const double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  __builtin_assume_aligned(A, 16);
  __builtin_assume_aligned(x, 16);
  __builtin_assume_aligned(y_opt, 16);
  __builtin_assume_aligned(z_opt, 16);

  int i, j;
  for (i = 0; i < N; i++) {
    #pragma GCC ivdep
    for (j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];

    }
  }
  for(j = 0; j < N; j++){
    #pragma GCC ivdep
    for(i=0;  i < N; i++) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }
}

void avx_version(double** A, double* x, double* y_opt, double* z_opt) {

  int i, j;
  double* base;



  for (i = 0; i < N; i++) {
    __m256d x_avx, y_avx, A_avx;

    base = A[i];
    x_avx = _mm256_set1_pd(x[i]);

    for (j = 0; j < N; j += 4) {
      A_avx = _mm256_load_pd(base+j);
      y_avx = _mm256_load_pd(y_opt+j);

      y_avx = _mm256_fmadd_pd(A_avx, x_avx, y_avx);
      _mm256_storeu_pd(y_opt+j, y_avx);
    }
  }

  for (j = 0; j < N; j++) {
    __m256d x_avx, z_avx, A_avx;

    base = A[j];
    x_avx = _mm256_set1_pd(x[j]);

    for (i = 0; i < N; i += 4) {
      A_avx = _mm256_load_pd(base+i);
      z_avx = _mm256_load_pd(z_opt+i);

      z_avx = _mm256_fmadd_pd(A_avx, x_avx, z_avx);
      _mm256_storeu_pd(z_opt+i, z_avx);
    }
  }

  return;
}

int main() {
  double clkbegin, clkend;
  double t;

  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double** A;
  A = new double*[N];
  for (int i = 0; i < N; i++) {
    A[i] = &A_align[i][0];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = &x_align[0];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = &y_opt_align[0];
  z_opt = &z_opt_align[0];

  for (int i = 0; i < N; i++) {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (int j = 0; j < N; j++) {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    reference(A, x, y_ref, z_ref);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n";

  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    optimized(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly

  // Version with intinsics

  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    avx_version(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Intrinsics Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  return EXIT_SUCCESS;
}
