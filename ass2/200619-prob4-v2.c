#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)

void gridloopsearch(double b[30], double a[120], double kk);

struct timespec begin_grid, end_main;

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

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

  // constraint values
  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  // results points
  long pnts = 0;

  // re-calculated limits
  double e1, e2, e3, e4, e5, e6, e7, e8, e9, e10;

  __attribute__((aligned(16))) double c[12][10];

  for(int i=0; i<120; i++){
    c[i%12][i/12] = a[i];
  }

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v2.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  // initialization of re calculated limits, xi's.
  e1 = kk * c[11][0];
  e2 = kk * c[11][1];
  e3 = kk * c[11][2];
  e4 = kk * c[11][3];
  e5 = kk * c[11][4];
  e6 = kk * c[11][5];
  e7 = kk * c[11][6];
  e8 = kk * c[11][7];
  e9 = kk * c[11][8];
  e10 = kk * c[11][9];

  double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  __attribute__((aligned(16))) double qb[11][10];

  // grid search starts
  for (r1 = b[0]; r1 < b[1]; r1+=b[2]) {

    #pragma ivdep
    for(int i=0; i<10; ++i)
      qb[1][i] = c[0][i]*r1 - c[10][i];

    for (r2 = b[3]; r2 < b[4]; r2+=b[5]) {

      #pragma ivdep
      for(int i=0; i<10; ++i)
        qb[2][i] = qb[1][i] + c[1][i] * r2;

      for (r3 = b[6]; r3 < b[7]; r3+=b[8]) {

        #pragma ivdep
        for(int i=0; i<10; ++i)
          qb[3][i] = qb[2][i] + c[2][i] * r3;

        for (r4 = b[9]; r4 < b[10]; r4+=b[11]) {

          #pragma ivdep
          for(int i=0; i<10; ++i)
            qb[4][i] = qb[3][i] + c[3][i] * r4;

          for (r5 = b[12]; r5 < b[13]; r5+=b[14]) {

            #pragma ivdep
            for(int i=0; i<10; ++i)
              qb[5][i] = qb[4][i] + c[4][i] * r5;

            for (r6 = b[15]; r6 < b[16]; r6+=b[17]) {

              #pragma ivdep
              for(int i=0; i<10; ++i)
                qb[6][i] = qb[5][i] + c[5][i] * r6;

              for (r7 = b[18]; r7 < b[19]; r7+=b[20]) {

                #pragma ivdep
                for(int i=0; i<10; ++i)
                  qb[7][i] = qb[6][i] + c[6][i] * r7;

                for (r8 = b[21]; r8 < b[22]; r8+=b[23]) {

                  #pragma ivdep
                  for(int i=0; i<10; ++i)
                    qb[8][i] = qb[7][i] + c[7][i] * r8;

                  for (r9 = b[24]; r9 < b[25]; r9+=b[26]) {

                    #pragma ivdep
                    for(int i=0; i<10; ++i)
                      qb[9][i] = qb[8][i] + c[8][i] * r9;

                    for (r10 = b[27]; r10 < b[28]; r10+=b[29]) {

                      #pragma ivdep
                      for(int i=0; i<10; ++i)
                        qb[10][i] = qb[9][i] + c[9][i] * r10;

                      if(fabs(qb[10][0]) > e1 ||
                      fabs(qb[10][1]) > e2 ||
                      fabs(qb[10][2]) > e3 ||
                      fabs(qb[10][3]) > e4 ||
                      fabs(qb[10][4]) > e5 ||
                      fabs(qb[10][5]) > e6 ||
                      fabs(qb[10][6]) > e7 ||
                      fabs(qb[10][7]) > e8 ||
                      fabs(qb[10][8]) > e9 ||
                      fabs(qb[10][9]) > e10)continue;

                      ++pnts;

                      // ri's which satisfy the constraints to be written in file
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

  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}
