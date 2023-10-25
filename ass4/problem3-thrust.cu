// Compile : gcc -O3 -std=c17 -D_POSIX_C_SOURCE=199309L -fopenmp 200619-prob4-v3.c
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <time.h>
#include <vector>
#include <iostream>
using namespace std;
#define NSEC_SEC_MUL (1.0e9)

#define TPB 512
#define BPG 32768

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

struct myFunctor {
  const double *c;
  const double *e;
  const int *s;
  // bool *result;
  // double c[120];
  // double e[10];

  myFunctor(thrust::device_vector<double> const& _c, thrust::device_vector<double> const& _e){
    // for(int i=0;i<120;++i){
    //   c[i]=_c[i];
    // }
    // for(int i=0;i<10;++i){
    //   e[i]=_e[i];
    // }
    c = thrust::raw_pointer_cast(_c.data());
    e = thrust::raw_pointer_cast(_e.data());
    // result = thrust::raw_pointer_cast(_result.data());
  }

  __device__
  bool operator()(double*x) const {

      for (int i = 0; i < 10; i++) {
          double qb = -c[10*10 +i];
          for (int j = 0; j < 10; ++j)
              qb += c[j*10+i] * x[j];
          if (fabs(qb) > e[i]){
            return false;
          }
      }

      return true;
  }
};

inline void gridloopsearch(double b[30], double a[120], double kk) {
  // __builtin_assume_aligned(a, 16);
  // __builtin_assume_aligned(b, 16);

  // constraint values
  // double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  // results points
  long pnts = 0;

  // re-calculated limits
  double e[10];
  double c[12][10];

  thrust::host_vector<double> c_host(120);
  thrust::host_vector<double> e_host(10);


  for(int i=0; i<120; i++){
    c[i%12][i/12] = a[i];
  }

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v31.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  // initialization of re calculated limits, xi's.
  for(int i=0; i<10; ++i)
    e[i] = kk * c[11][i];
  int s[10];

  for(int i=0;i<10;++i)
    s[i] = floor((b[3*i+1] - b[3*i]) / b[3*i+2]);
  // double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  long nitr=1;
  for(int i=0;i<10;++i) nitr*=s[i];

  thrust::device_vector<bool> result_dev(nitr);

  for(int i=0;i<120;++i){
    c_host[i]=(&c[0][0])[i];
  }
  for(int i=0;i<10;++i){
    e_host[i]=e[i];
  }
  thrust::device_vector<double> c_dev = c_host;
  thrust::device_vector<double> e_dev = e_host;

  thrust::host_vector<thrust::host_vector<double>> data_host;

  // // grid search starts
  // for (double r1 = b[0]; r1 < b[1]; r1+=b[2]) {

  //   for (double r2 = b[3]; r2 < b[4]; r2+=b[5]) {

  //     for (double r3 = b[6]; r3 < b[7]; r3+=b[8]) {

  //       for (double r4 = b[9]; r4 < b[10]; r4+=b[11]) {

  //         for (double r5 = b[12]; r5 < b[13]; r5+=b[14]) {

  //           for (double r6 = b[15]; r6 < b[16]; r6+=b[17]) {

  //             for (double r7 = b[18]; r7 < b[19]; r7+=b[20]) {

  //               for (double r8 = b[21]; r8 < b[22]; r8+=b[23]) {

  //                 for (double r9 = b[24]; r9 < b[25]; r9+=b[26]) {

  //                   for (double r10 = b[27]; r10 < b[28]; r10+=b[29]) {
  //                     thrust::host_vector<double> v;
  //                     v.push_back(r1);
  //                     v.push_back(r2);
  //                     v.push_back(r3);
  //                     v.push_back(r4);
  //                     v.push_back(r5);
  //                     v.push_back(r6);
  //                     v.push_back(r7);
  //                     v.push_back(r8);
  //                     v.push_back(r9);
  //                     v.push_back(r10);
  //                     data_host.push_back(v);
  //                   }
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }

  // }
  thrust::device_vector<thrust::device_vector<double>> data_dev = data_host;


  // thrust::host_vector<bool> results(points.size());

  // // Use Thrust to apply the constraints and write to the file
  // thrust::transform(data_dev.begin(), data_dev.end(), result_dev.begin(), myFunctor());

  // // Count how many points met the constraints
  // int numValidPoints = thrust::count(results.begin(), results.end(), true);



  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}
