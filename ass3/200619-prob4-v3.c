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

  __attribute__((aligned(16))) double qb[11][12];

  int s = floor((b[1] - b[0]) / b[2]);
  // double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  omp_set_nested(1);
  // grid search starts
  #pragma omp parallel for default(none) shared(c,b,fptr,s,e) private(qb) reduction(+:pnts)
  for (int s1 = 0 ; s1 < s; s1++) {
    // double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;
    double r1 = b[0] + s1 * b[2];

    for(int i=0; i<10; i++)
      qb[1][i] = c[0][i]*r1 - c[10][i];

    for (double r2 = b[3]; r2 < b[4]; r2+=b[5]) {

      // #pragma simd
      for(int i=0; i<10; i++)
        qb[2][i] = qb[1][i] + c[1][i] * r2;

      for (double r3 = b[6]; r3 < b[7]; r3+=b[8]) {

        // #pragma simd
        for(int i=0; i<10; i++)
          qb[3][i] = qb[2][i] + c[2][i] * r3;

        for (double r4 = b[9]; r4 < b[10]; r4+=b[11]) {

          // #pragma simd
          for(int i=0; i<10; i++)
            qb[4][i] = qb[3][i] + c[3][i] * r4;

          for (double r5 = b[12]; r5 < b[13]; r5+=b[14]) {

            // #pragma simd
            for(int i=0; i<10; i++)
              qb[5][i] = qb[4][i] + c[4][i] * r5;

            for (double r6 = b[15]; r6 < b[16]; r6+=b[17]) {

              // #pragma simd
              for(int i=0; i<10; i++)
                qb[6][i] = qb[5][i] + c[5][i] * r6;

              for (double r7 = b[18]; r7 < b[19]; r7+=b[20]) {

                // #pragma simd
                for(int i=0; i<10; i++)
                  qb[7][i] = qb[6][i] + c[6][i] * r7;

                for (double r8 = b[21]; r8 < b[22]; r8+=b[23]) {

                  // #pragma simd
                  for(int i=0; i<10; i++)
                    qb[8][i] = qb[7][i] + c[7][i] * r8;

                  for (double r9 = b[24]; r9 < b[25]; r9+=b[26]) {

                    // #pragma simd
                    for(int i=0; i<10; i++)
                      qb[9][i] = qb[8][i] + c[8][i] * r9;

                    for (double r10 = b[27]; r10 < b[28]; r10+=b[29]) {

                      for(int i=0; i<10; i++)
                        qb[10][i] = qb[9][i] + c[9][i] * r10;

                      if(fabs(qb[10][0]) > e[1] ||
                      fabs(qb[10][1]) > e[2] ||
                      fabs(qb[10][2]) > e[3] ||
                      fabs(qb[10][3]) > e[4] ||
                      fabs(qb[10][4]) > e[5] ||
                      fabs(qb[10][5]) > e[6] ||
                      fabs(qb[10][6]) > e[7] ||
                      fabs(qb[10][7]) > e[8] ||
                      fabs(qb[10][8]) > e[9] ||
                      fabs(qb[10][9]) > e[10])continue;

                      ++pnts;

                      // ri's which satisfy the constraints to be written in file
                      #pragma omp critical
                      {
                        fprintf(fptr, "%lf\t", r1);
                        fprintf(fptr, "%lf\t", r2);
                        fprintf(fptr, "%lf\t", r3);
                        fprintf(fptr, "%lf\t", r4);
                        fprintf(fptr, "%lf\t", r5);
                        fprintf(fptr, "%lf\t", r6);
                        fprintf(fptr, "%lf\t", r7);
                        fprintf(fptr, "%lf\t", r8);
                        fprintf(fptr, "%lf\t", r9);
                        fprintf(fptr, "%lf\n", r10);
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
