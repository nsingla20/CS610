// Compile : gcc -O3 -std=c17 -D_POSIX_C_SOURCE=199309L -fopenmp 200619-prob4-v3.c
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
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

__global__ void navie_CUDA(double* out, double*b, double* c, double* e, int* s, long offset) {
  long ind = blockIdx.x * blockDim.x + threadIdx.x;

  long t = ind+offset;
  int r[10];
  for(int i=9;i>=0;--i){
    r[i]=t%s[i];
    t/=s[i];
  }

  double com[10];
  __shared__ double comc[120];


  ind*=10;

  for(int i=0;i<10;++i){
    com[i] = b[3*i]+r[i]*b[3*i+2];
  }

  if(threadIdx.x<120){
    comc[threadIdx.x] = c[threadIdx.x];
  }
  __syncthreads();

  double q[10];

  for(int i=0;i<10;++i){
    q[i] = -comc[10*10+i];
    for(int j=0;j<10;++j){
      q[i] += comc[j*10+i]*com[j];
    }
    if(fabs(q[i])>e[i]){
      return;
    }
  }

  for(int i=0;i<10;++i)
    out[ind+i] = com[i];

}

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
  // __builtin_assume_aligned(a, 16);
  // __builtin_assume_aligned(b, 16);

  // constraint values
  // double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  // results points
  long pnts = 0;

  double* bd;
  cudaMalloc((void**)&bd,30*sizeof(double));
  cudaMemcpy(bd, b, 30*sizeof(double), cudaMemcpyHostToDevice);

  // re-calculated limits
  double e[10];
  double* ed;
  cudaMalloc((void**)&ed,10*sizeof(double));

  double c[12][10];

  double* cd;
  cudaMalloc((void**)&cd,120*sizeof(double));
  for(int i=0; i<120; i++){
    c[i%12][i/12] = a[i];
  }
  cudaMemcpy(cd, c, 120*sizeof(double), cudaMemcpyHostToDevice);

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v31.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  // initialization of re calculated limits, xi's.
  for(int i=0; i<10; ++i)
    e[i] = kk * c[11][i];
  cudaMemcpy(ed, e, 10*sizeof(double), cudaMemcpyHostToDevice);

  int s[10];
  int* sd;
  cudaMalloc((void**)&sd, 10*sizeof(int));
  for(int i=0;i<10;++i)
    s[i] = floor((b[3*i+1] - b[3*i]) / b[3*i+2]);
  // double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;
  cudaMemcpy(sd, s, 10*sizeof(int), cudaMemcpyHostToDevice);
  // grid search starts
  long nitr=1;
  for(int i=0;i<10;++i) nitr*=s[i];

  long offset = 0;
  int tpb = TPB;
  int bpg = BPG;

  // vector<double*> data;
  // vector<int> datal;
  // vector<long> dataof;
  double* hdata,*ddata;
  cudaMallocManaged(&ddata, tpb*bpg*10*sizeof(double));
  // hdata = static_cast<double*>(malloc(tpb*bpg*10*sizeof(double)));
  // cudaHostAlloc((void**)&ddata,tpb*bpg*10*sizeof(double),cudaHostAllocDefault);
  // cudaMalloc((void**)&ddata,tpb*bpg*10*sizeof(double));
  while(nitr){
    int step = tpb*bpg;
    if(step>nitr){
      if(bpg==1){
        tpb >>=1;
      }else{
        bpg >>=1;
      }
    }else{
      cout<<nitr<<endl;
      // cudaStream_t stream;
      // cudaStreamCreate(&stream);

      // dataof.push_back(offset);
      // datal.push_back(tpb*bpg);
      // hdata = static_cast<double*>(malloc(tpb*bpg*10*sizeof(double)));
      // data.push_back(hdata);

      cudaMemset(ddata, 0, tpb*bpg*10*sizeof(double));
      navie_CUDA<<<bpg,tpb>>>(ddata,bd,cd,ed,sd,offset);
      // cudaMemcpy(hdata, ddata, tpb*bpg*10*sizeof(double), cudaMemcpyDeviceToHost);

      for(long j=0;j<bpg*tpb;++j){
        if(ddata[j*10]==0)continue;
        ++pnts;
        // long t = dataof[i]+j;
        for(int k=0;k<10;++k){
          fprintf(fptr, "%lf\t", ddata[j*10+k]);
        }
        fprintf(fptr, "\n");

      }

      offset+=step;
      nitr-=step;
    }
  }

  cudaDeviceSynchronize();
  // int n = data.size();
  // int r[10];
  // double x[10];
  // long ind;



  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}
