// nvcc -ccbin /usr/bin/g++-10 -arch=sm_80 -std=c++11 200619_prob1.cu -o 200619_prob1

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

#define N (64)
#define BLOCK_SIDE (8)     // Should be divisor of N
#define GRID_SIDE (N/BLOCK_SIDE)
#define THRESHOLD (0.000001)

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

__global__ void navie_CUDA(float* in, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i==0||i==N-1||j==0||j==N-1||k==0||k==N-1){
    out[i * N * N + j * N + k] = 0;
  }else{
    out[i * N * N + j * N + k] = 0.8 * (in[(i-1) * N * N + j * N + k] +
                                            in[(i+1) * N * N + j * N + k] +
                                            in[i * N * N + (j-1) * N + k] +
                                            in[i * N * N + (j+1) * N + k] +
                                            in[i * N * N + j * N + (k-1)] +
                                            in[i * N * N + j * N + (k+1)]);
  }
}

__global__ void memTile_CUDA(float* in, float* out) {
  const int i = blockIdx.x * BLOCK_SIDE + threadIdx.x;
  const int j = blockIdx.y * BLOCK_SIDE + threadIdx.y;
  const int k = blockIdx.z * BLOCK_SIDE + threadIdx.z;

  const int is = threadIdx.x+1;
  const int js = threadIdx.y+1;
  const int ks = threadIdx.z+1;

  __shared__ float data[(BLOCK_SIDE+2)*(BLOCK_SIDE+2)*(BLOCK_SIDE+2)];

  const int n = BLOCK_SIDE+2;

  float ans = 0;

  data[is * n * n + js * n + ks] = in[i * N * N + j * N + k];

  if(i!=0&&i!=N-1&&j!=0&&j!=N-1&&k!=0&&k!=N-1){
    if(is==1) data[(is-1) * n * n + js * n + ks] = in[(i-1) * N * N + j * N + k];
    if(js==1) data[is * n * n + (js-1) * n + ks] = in[i * N * N + (j-1) * N + k];
    if(ks==1) data[is * n * n + js * n + (ks-1)] = in[i * N * N + j * N + (k-1)];

    if(is==BLOCK_SIDE) data[(is+1) * n * n + js * n + ks] = in[(i+1) * N * N + j * N + k];
    if(js==BLOCK_SIDE) data[is * n * n + (js+1) * n + ks] = in[i * N * N + (j+1) * N + k];
    if(ks==BLOCK_SIDE) data[is * n * n + js * n + (ks+1)] = in[i * N * N + j * N + (k+1)];
  }


  __syncthreads();

  if(i!=0&&i!=N-1&&j!=0&&j!=N-1&&k!=0&&k!=N-1){
    ans += data[(is-1) * n * n + js * n + ks];
    ans += data[(is+1) * n * n + js * n + ks];
    ans += data[is * n * n + (js-1) * n + ks];
    ans += data[is * n * n + (js+1) * n + ks];
    ans += data[is * n * n + js * n + (ks-1)];
    ans += data[is * n * n + js * n + (ks+1)];
  }

  out[i * N * N + j * N + k] = 0.8 * ans;
}

__global__ void loopOP_CUDA(float* in, float* out) {
  const int i = blockIdx.z * BLOCK_SIDE + threadIdx.z;
  const int j = blockIdx.y * BLOCK_SIDE + threadIdx.y;
  const int k = blockIdx.x * BLOCK_SIDE + threadIdx.x;

  const int is = threadIdx.z+1;
  const int js = threadIdx.y+1;
  const int ks = threadIdx.x+1;

  __shared__ float data[(BLOCK_SIDE+2)*(BLOCK_SIDE+2)*(BLOCK_SIDE+2)];

  const int n = BLOCK_SIDE+2;

  float ans = 0;

  __syncthreads();
  data[is * n * n + js * n + ks] = in[i * N * N + j * N + k];

  if(i!=0&&i!=N-1&&j!=0&&j!=N-1&&k!=0&&k!=N-1){
    if(is==1) data[(is-1) * n * n + js * n + ks] = in[(i-1) * N * N + j * N + k];
    if(js==1) data[is * n * n + (js-1) * n + ks] = in[i * N * N + (j-1) * N + k];
    if(ks==1) data[is * n * n + js * n + (ks-1)] = in[i * N * N + j * N + (k-1)];

    if(is==BLOCK_SIDE) data[(is+1) * n * n + js * n + ks] = in[(i+1) * N * N + j * N + k];
    if(js==BLOCK_SIDE) data[is * n * n + (js+1) * n + ks] = in[i * N * N + (j+1) * N + k];
    if(ks==BLOCK_SIDE) data[is * n * n + js * n + (ks+1)] = in[i * N * N + j * N + (k+1)];
  }


  __syncthreads();

  if(i!=0&&i!=N-1&&j!=0&&j!=N-1&&k!=0&&k!=N-1){
    ans += data[(is-1) * n * n + js * n + ks];
    ans += data[(is+1) * n * n + js * n + ks];
    ans += data[is * n * n + (js-1) * n + ks];
    ans += data[is * n * n + (js+1) * n + ks];
    ans += data[is * n * n + js * n + (ks-1)];
    ans += data[is * n * n + js * n + (ks+1)];
  }

  out[i * N * N + j * N + k] = 0.8 * ans;

}

__host__ void stencil(float* in, float* out) {
  for (int i = 1; i < N-1; ++i) {
    for (int j = 1; j < N-1; ++j) {
      for (int k = 1; k < N-1; ++k) {
        out[i * N * N + j * N + k] = 0.8 * (in[(i-1) * N * N + j * N + k] +
                                            in[(i+1) * N * N + j * N + k] +
                                            in[i * N * N + (j-1) * N + k] +
                                            in[i * N * N + (j+1) * N + k] +
                                            in[i * N * N + j * N + (k-1)] +
                                            in[i * N * N + j * N + (k+1)]);
      }
    }
  }
}

__host__ void check_result(const float* w_ref, const float* w_opt, const uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        double ref = w_ref[i + N * j + N * N * k];
        double test = w_opt[i + N * j + N * N * k];
        this_diff = ref - test;
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  }
}

void print_mat(double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  uint64_t SIZE = N * N * N;
  size_t size = SIZE*sizeof(float);

  float* in = static_cast<float*>(malloc(size));
  memset(in,0,size);
  for (int i = 1; i < N-1; ++i) {
    for (int j = 1; j < N-1; ++j) {
      for (int k = 1; k < N-1; ++k) {
        in[i * N * N + j * N + k] = fabs(rand());
      }
    }
  }

  float* out_cpu = static_cast<float*>(malloc(size));
  memset(out_cpu,0,size);
  double clkbegin = rtclock();
  stencil(in, out_cpu);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  float *d_in, *d_out;
  float kernel_time;
  cudaMalloc((void**)&d_in, size);
  cudaMalloc((void**)&d_out, size);
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

  // Navie CUDA
  float* out_navie_CUDA = static_cast<float*>(malloc(size));
  memset(out_navie_CUDA,0,size);
  cudaMemset(d_out, 0, size);
  dim3 navieGrid(GRID_SIDE,GRID_SIDE,GRID_SIDE);
  dim3 navieBlock(BLOCK_SIDE,BLOCK_SIDE,BLOCK_SIDE);
  cudaEventRecord(start, 0);
  navie_CUDA<<<navieGrid, navieBlock>>>(d_in, d_out);
  cudaMemcpy(out_navie_CUDA, d_out, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  check_result(out_cpu, out_navie_CUDA, N);
  std::cout << "(i) Navie CUDA time (ms): " << kernel_time << "\n";

  // MemTile CUDA
  float* out_memTile_CUDA = static_cast<float*>(malloc(size));
  memset(out_memTile_CUDA,0,size);
  cudaMemset(d_out, 0, size);
  dim3 memTileGrid(GRID_SIDE,GRID_SIDE,GRID_SIDE);
  dim3 memTileBlock(BLOCK_SIDE,BLOCK_SIDE,BLOCK_SIDE);
  cudaEventRecord(start, 0);
  memTile_CUDA<<<memTileGrid, memTileBlock>>>(d_in, d_out);
  cudaMemcpy(out_memTile_CUDA, d_out, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  check_result(out_cpu, out_memTile_CUDA, N);
  std::cout << "(ii) MemTile CUDA time (ms): " << kernel_time << "\n";

  // LoopOP CUDA
  float* out_loopOP_CUDA = static_cast<float*>(malloc(size));
  memset(out_loopOP_CUDA,0,size);
  cudaMemset(d_out, 0, size);
  dim3 loopOPGrid(GRID_SIDE,GRID_SIDE,GRID_SIDE);
  dim3 loopOPBlock(BLOCK_SIDE,BLOCK_SIDE,BLOCK_SIDE);
  cudaEventRecord(start, 0);
  loopOP_CUDA<<<loopOPGrid, loopOPBlock>>>(d_in, d_out);
  cudaMemcpy(out_loopOP_CUDA, d_out, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  check_result(out_cpu, out_loopOP_CUDA, N);
  std::cout << "(iii) Loop Optimised CUDA time (ms): " << kernel_time << "\n";

  // LoopOP CUDA
  float* out_pin_CUDA;
  cudaHostAlloc(&out_pin_CUDA, size, cudaHostAllocDefault);
  memset(out_pin_CUDA,0,size);
  cudaMemset(d_out, 0, size);
  dim3 pinGrid(GRID_SIDE,GRID_SIDE,GRID_SIDE);
  dim3 pinBlock(BLOCK_SIDE,BLOCK_SIDE,BLOCK_SIDE);
  cudaEventRecord(start, 0);
  loopOP_CUDA<<<pinGrid, pinBlock>>>(d_in, d_out);
  cudaMemcpy(out_pin_CUDA, d_out, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  check_result(out_cpu, out_pin_CUDA, N);
  std::cout << "(iv) Pinned CUDA time (ms): " << kernel_time << "\n";
  cudaFreeHost(out_pin_CUDA);

  // LoopOP CUDA
  float* out_manage_CUDA;
  cudaMallocManaged(&out_manage_CUDA, size);
  memset(out_manage_CUDA,0,size);
  cudaMemset(out_manage_CUDA, 0, size);
  dim3 manageGrid(GRID_SIDE,GRID_SIDE,GRID_SIDE);
  dim3 manageBlock(BLOCK_SIDE,BLOCK_SIDE,BLOCK_SIDE);
  cudaEventRecord(start, 0);
  loopOP_CUDA<<<manageGrid, manageBlock>>>(d_in, out_manage_CUDA);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  check_result(out_cpu, out_manage_CUDA, N);
  std::cout << "(v) Managed CUDA time (ms): " << kernel_time << "\n";
  cudaFree(out_manage_CUDA);


  // TODO: Free memory
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(d_in);
  cudaFree(d_out);
  free(in);
  free(out_cpu);
  free(out_navie_CUDA);
  free(out_memTile_CUDA);
  free(out_loopOP_CUDA);

  return EXIT_SUCCESS;
}
