#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)

void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk);

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
  gridloopsearch(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
                 b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23], b[24],
                 b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                 a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19],
                 a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31],
                 a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
                 a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55],
                 a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63], a[64], a[65], a[66], a[67],
                 a[68], a[69], a[70], a[71], a[72], a[73], a[74], a[75], a[76], a[77], a[78], a[79],
                 a[80], a[81], a[82], a[83], a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91],
                 a[92], a[93], a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102],
                 a[103], a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
                 a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}

// grid search function with loop variables

void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk) {

  // constraint values
  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  // results points
  long pnts = 0;

  // re-calculated limits
  double e1, e2, e3, e4, e5, e6, e7, e8, e9, e10;

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v1.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  // initialization of re calculated limits, xi's.
  e1 = kk * ey1;
  e2 = kk * ey2;
  e3 = kk * ey3;
  e4 = kk * ey4;
  e5 = kk * ey5;
  e6 = kk * ey6;
  e7 = kk * ey7;
  e8 = kk * ey8;
  e9 = kk * ey9;
  e10 = kk * ey10;

  double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  double qb[11][11];

  // grid search starts
  for (r1 = dd1; r1 < dd2; r1+=dd3) {

    qb[1][1] = c11 * r1 - d1;
    qb[1][2] = c21 * r1 - d2;
    qb[1][3] = c31 * r1 - d3;
    qb[1][4] = c41 * r1 - d4;
    qb[1][5] = c51 * r1 - d5;
    qb[1][6] = c61 * r1 - d6;
    qb[1][7] = c71 * r1 - d7;
    qb[1][8] = c81 * r1 - d8;
    qb[1][9] = c91 * r1 - d9;
    qb[1][10] = c101 * r1 - d10;

    for (r2 = dd4; r2 < dd5; r2+=dd6) {

      qb[2][1] = qb[1][1] + c12 * r2;
      qb[2][2] = qb[1][2] + c22 * r2;
      qb[2][3] = qb[1][3] + c32 * r2;
      qb[2][4] = qb[1][4] + c42 * r2;
      qb[2][5] = qb[1][5] + c52 * r2;
      qb[2][6] = qb[1][6] + c62 * r2;
      qb[2][7] = qb[1][7] + c72 * r2;
      qb[2][8] = qb[1][8] + c82 * r2;
      qb[2][9] = qb[1][9] + c92 * r2;
      qb[2][10] = qb[1][10] + c102 * r2;

      for (r3 = dd7; r3 < dd8; r3+=dd9) {

        qb[3][1] = qb[2][1] + c13 * r3;
        qb[3][2] = qb[2][2] + c23 * r3;
        qb[3][3] = qb[2][3] + c33 * r3;
        qb[3][4] = qb[2][4] + c43 * r3;
        qb[3][5] = qb[2][5] + c53 * r3;
        qb[3][6] = qb[2][6] + c63 * r3;
        qb[3][7] = qb[2][7] + c73 * r3;
        qb[3][8] = qb[2][8] + c83 * r3;
        qb[3][9] = qb[2][9] + c93 * r3;
        qb[3][10] = qb[2][10] + c103 * r3;

        for (r4 = dd10; r4 < dd11; r4+=dd12) {

          qb[4][1] = qb[3][1] + c14 * r4;
          qb[4][2] = qb[3][2] + c24 * r4;
          qb[4][3] = qb[3][3] + c34 * r4;
          qb[4][4] = qb[3][4] + c44 * r4;
          qb[4][5] = qb[3][5] + c54 * r4;
          qb[4][6] = qb[3][6] + c64 * r4;
          qb[4][7] = qb[3][7] + c74 * r4;
          qb[4][8] = qb[3][8] + c84 * r4;
          qb[4][9] = qb[3][9] + c94 * r4;
          qb[4][10] = qb[3][10] + c104 * r4;

          for (r5 = dd13; r5 < dd14; r5+=dd15) {
            qb[5][1] = qb[4][1] + c15 * r5;
            qb[5][2] = qb[4][2] + c25 * r5;
            qb[5][3] = qb[4][3] + c35 * r5;
            qb[5][4] = qb[4][4] + c45 * r5;
            qb[5][5] = qb[4][5] + c55 * r5;
            qb[5][6] = qb[4][6] + c65 * r5;
            qb[5][7] = qb[4][7] + c75 * r5;
            qb[5][8] = qb[4][8] + c85 * r5;
            qb[5][9] = qb[4][9] + c95 * r5;
            qb[5][10] = qb[4][10] + c105 * r5;

            for (r6 = dd16; r6 < dd17; r6+=dd18) {

              qb[6][1] = qb[5][1] + c16 * r6;
              qb[6][2] = qb[5][2] + c26 * r6;
              qb[6][3] = qb[5][3] + c36 * r6;
              qb[6][4] = qb[5][4] + c46 * r6;
              qb[6][5] = qb[5][5] + c56 * r6;
              qb[6][6] = qb[5][6] + c66 * r6;
              qb[6][7] = qb[5][7] + c76 * r6;
              qb[6][8] = qb[5][8] + c86 * r6;
              qb[6][9] = qb[5][9] + c96 * r6;
              qb[6][10] = qb[5][10] + c106 * r6;

              for (r7 = dd19; r7 < dd20; r7+=dd21) {

                qb[7][1] = qb[6][1] + c17 * r7;
                qb[7][2] = qb[6][2] + c27 * r7;
                qb[7][3] = qb[6][3] + c37 * r7;
                qb[7][4] = qb[6][4] + c47 * r7;
                qb[7][5] = qb[6][5] + c57 * r7;
                qb[7][6] = qb[6][6] + c67 * r7;
                qb[7][7] = qb[6][7] + c77 * r7;
                qb[7][8] = qb[6][8] + c87 * r7;
                qb[7][9] = qb[6][9] + c97 * r7;
                qb[7][10] = qb[6][10] + c107 * r7;

                for (r8 = dd22; r8 < dd23; r8+=dd24) {

                  qb[8][1] = qb[7][1] + c18 * r8;
                  qb[8][2] = qb[7][2] + c28 * r8;
                  qb[8][3] = qb[7][3] + c38 * r8;
                  qb[8][4] = qb[7][4] + c48 * r8;
                  qb[8][5] = qb[7][5] + c58 * r8;
                  qb[8][6] = qb[7][6] + c68 * r8;
                  qb[8][7] = qb[7][7] + c78 * r8;
                  qb[8][8] = qb[7][8] + c88 * r8;
                  qb[8][9] = qb[7][9] + c98 * r8;
                  qb[8][10] = qb[7][10] + c108 * r8;

                  for (r9 = dd25; r9 < dd26; r9+=dd27) {

                    qb[9][1] = qb[8][1] + c19 * r9;
                    qb[9][2] = qb[8][2] + c29 * r9;
                    qb[9][3] = qb[8][3] + c39 * r9;
                    qb[9][4] = qb[8][4] + c49 * r9;
                    qb[9][5] = qb[8][5] + c59 * r9;
                    qb[9][6] = qb[8][6] + c69 * r9;
                    qb[9][7] = qb[8][7] + c79 * r9;
                    qb[9][8] = qb[8][8] + c89 * r9;
                    qb[9][9] = qb[8][9] + c99 * r9;
                    qb[9][10] = qb[8][10] + c109 * r9;

                    for (r10 = dd28; r10 < dd29; r10+=dd30) {

                      // constraints

                      qb[10][1] = qb[9][1] + c110 * r10;
                      qb[10][2] = qb[9][2] + c210 * r10;
                      qb[10][3] = qb[9][3] + c310 * r10;
                      qb[10][4] = qb[9][4] + c410 * r10;
                      qb[10][5] = qb[9][5] + c510 * r10;
                      qb[10][6] = qb[9][6] + c610 * r10;
                      qb[10][7] = qb[9][7] + c710 * r10;
                      qb[10][8] = qb[9][8] + c810 * r10;
                      qb[10][9] = qb[9][9] + c910 * r10;
                      qb[10][10] = qb[9][10] + c1010 * r10;



                      q1 = fabs(qb[10][1]);
                      q2 = fabs(qb[10][2]);
                      q3 = fabs(qb[10][3]);
                      q4 = fabs(qb[10][4]);
                      q5 = fabs(qb[10][5]);
                      q6 = fabs(qb[10][6]);
                      q7 = fabs(qb[10][7]);
                      q8 = fabs(qb[10][8]);
                      q9 = fabs(qb[10][9]);
                      q10 = fabs(qb[10][10]);

                      if ((q1 <= e1) && (q2 <= e2) && (q3 <= e3) && (q4 <= e4) && (q5 <= e5) &&
                          (q6 <= e6) && (q7 <= e7) && (q8 <= e8) && (q9 <= e9) && (q10 <= e10)) {
                        pnts = pnts + 1;

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
  }

  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}
