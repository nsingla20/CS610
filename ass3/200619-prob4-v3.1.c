// Compile : gcc -O3 -std=c17 -D_POSIX_C_SOURCE=199309L -fopenmp 200619-prob4-v3.c
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define NSEC_SEC_MUL (1.0e9)

void gridloopsearch(double b[30], double a[120], double kk);

struct timespec begin_grid, end_main;

// to store values of disp.txt
__attribute__((aligned(16))) double a[120];

// to store values of grid.txt
__attribute__((aligned(16))) double b[30];

int main() {
  int i, j;

  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // read grid file
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  // grid value initialize
  // initialize value of kk;
  double kk = 0.3;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  gridloopsearch(b, a, kk);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}

// grid search function with loop variables

inline void gridloopsearch(double b[30], double a[120], double kk) {
  __builtin_assume_aligned(a, 16);
  __builtin_assume_aligned(b, 16);

  // constraint values
  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  // results points
  long pnts = 0;

  // re-calculated limits
  double e[11];

  __attribute__((aligned(16))) double c[12][10];

  for(int i=0; i<120; i++){
    c[i%12][i/12] = a[i];
  }

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v3.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  // initialization of re calculated limits, xi's.
  for(int i=1; i<11; ++i)
    e[i] = kk * c[11][i-1];

  __attribute__((aligned(16))) double qb[10];

  int s[10];
  for(int i=0;i<10;++i)
    s[i] = floor((b[3*i+1] - b[3*i]) / b[3*i+2]);
  // double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;
  __attribute__((aligned(16))) double x[10];
  __attribute__((aligned(16))) double r[10];

  // grid search starts
  #pragma omp parallel for collapse(10) default(none) shared(c,b,fptr,s,e) private(x,qb) reduction(+:pnts)
  for (long r0 = 0; r0 < s[0]; ++r0) {

    for (long r1 = 0; r1 < s[1]; ++r1) {

      for (long r2 = 0; r2 < s[2]; ++r2) {

        for (long r3 = 0; r3 < s[3]; ++r3) {

          for (long r4 = 0; r4 < s[4]; ++r4) {

            for (long r5 = 0; r5 < s[5]; ++r5) {

              for (long r6 = 0; r6 < s[6]; ++r6) {

                for (long r7 = 0; r7 < s[7]; ++r7) {

                  for (long r8 = 0; r8 < s[8]; ++r8) {

                    for (long r9 = 0; r9 < s[9]; ++r9) {

                      x[0] = b[3*0] + r0*b[3*0+2];
                      x[1] = b[3*1] + r1*b[3*1+2];
                      x[2] = b[3*2] + r2*b[3*2+2];
                      x[3] = b[3*3] + r3*b[3*3+2];
                      x[4] = b[3*4] + r4*b[3*4+2];
                      x[5] = b[3*5] + r5*b[3*5+2];
                      x[6] = b[3*6] + r6*b[3*6+2];
                      x[7] = b[3*7] + r7*b[3*7+2];
                      x[8] = b[3*8] + r8*b[3*8+2];
                      x[9] = b[3*9] + r9*b[3*9+2];

                      // for(int i=0; i<10; i++)
                      //   x[i] = b[3*i] + s[i]*b[3*i+2];

                      int i;
                      for(i=0; i<10; i++){
                        qb[i] = -c[10][i];
                        for(int j=0;j<10;++j)
                          qb[i] += c[i][j] * (x[j]);
                        if(fabs(qb[i])>e[i+1])break;
                      }

                      if(i!=10)continue;

                      ++pnts;

                      // ri's which satisfy the constraints to be written in file
                      #pragma omp critical
                      {
                        fprintf(fptr, "%lf\t", x[0]);
                        fprintf(fptr, "%lf\t", x[1]);
                        fprintf(fptr, "%lf\t", x[2]);
                        fprintf(fptr, "%lf\t", x[3]);
                        fprintf(fptr, "%lf\t", x[4]);
                        fprintf(fptr, "%lf\t", x[5]);
                        fprintf(fptr, "%lf\t", x[6]);
                        fprintf(fptr, "%lf\t", x[7]);
                        fprintf(fptr, "%lf\t", x[8]);
                        fprintf(fptr, "%lf\n", x[9]);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}
