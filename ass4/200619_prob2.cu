// Compile: nvcc -arch=sm_80 -ccbin /usr/bin/g++-10 200619_prob2.cu -o 200619_prob2

#include <cuda.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define THRESHOLD 0

#define NUM_TH_PER_BLCK 512
#define ELE_PER_BLOCK (2*NUM_TH_PER_BLCK)
#define N 20000000		// Length of array

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

__host__ void thrust_sum(uint32_t *out, uint32_t *in, int n) {

	thrust::device_vector<uint32_t> deviceInput(in, in+n);
	thrust::device_vector<uint32_t> deviceOutput(n);
	thrust::exclusive_scan(deviceInput.begin(), deviceInput.end(), deviceOutput.begin());

	thrust::copy(deviceOutput.begin(), deviceOutput.end(), out);

}

__global__ void cuda_sum(uint32_t *output, uint32_t *input, uint32_t *sums, int n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int offset = blockID * ELE_PER_BLOCK;

	__shared__ uint32_t data[ELE_PER_BLOCK];

	__syncthreads();

	if(2*threadID+offset<n){
		data[2*threadID] = input[offset + 2*threadID];
	}else{
		data[2*threadID] = 0;
	}
	__syncthreads();
	if(2*threadID+offset+1<n){
		data[2*threadID + 1] = input[offset + 2*threadID + 1];
	}else{
		data[2*threadID + 1] = 0;
	}
	__syncthreads();



	int distance = 1;
	for (int i = NUM_TH_PER_BLCK; i > 0; i >>= 1)
	{
		__syncthreads();
		if (threadID < i)
		{
			int ai = distance * (2 * threadID + 1) - 1;
			int bi = distance * (2 * threadID + 2) - 1;
			data[bi] += data[ai];
		}
		distance *= 2;
	}


	if (threadID == 0) {
		sums[blockID] = data[2*NUM_TH_PER_BLCK - 1];
		data[2*NUM_TH_PER_BLCK - 1] = 0;
	}

	for (int i = 1; i < 2*NUM_TH_PER_BLCK; i *= 2) // traverse down tree & build scan
	{
		distance >>= 1;
		__syncthreads();
		if (threadID < i)
		{
			int ai = distance * (2 * threadID + 1) - 1;
			int bi = distance * (2 * threadID + 2) - 1;
			uint32_t t = data[ai];
			data[ai] = data[bi];
			data[bi] += t;
		}
	}
	__syncthreads();

	if(2*threadID+offset<n){
		output[offset + (2 * threadID)] = data[2 * threadID];
	}
	__syncthreads();
	if(2*threadID+offset+1<n){
		output[offset + (2 * threadID) + 1] = data[2 * threadID + 1];
	}
	__syncthreads();



}

__global__ void post_cuda_sum(uint32_t *output, uint32_t *input, uint32_t *sum_offset) {

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int offset = blockID * ELE_PER_BLOCK;

	output[offset + threadID] = sum_offset[blockID] + input[offset + threadID];

}


__host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt, const uint64_t size) {
	uint32_t maxdiff = 0.0, this_diff = 0.0;
	int numdiffs = 0;

	for (uint64_t i = 0; i < size; i++) {
		this_diff = w_ref[i] - w_opt[i];
		if (std::fabs(this_diff) > THRESHOLD) {
			numdiffs++;
			if (this_diff > maxdiff) {
				maxdiff = this_diff;
			}
			cout<<w_ref[i]<<" "<<w_opt[i]<<endl;
		}
	}

	if (numdiffs > 0) {
		cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
		<< endl;
	} else {
		cout << "No differences found between base and test versions\n";
	}
}

int les_go(uint32_t* d_in, uint32_t* d_out, int n){
	uint32_t *sums, *sumso;
	int nBlocks = (n+ELE_PER_BLOCK-1)/ELE_PER_BLOCK;

	cudaMalloc((void**)&sums, nBlocks*sizeof(uint32_t));
	cudaMalloc((void**)&sumso, nBlocks*sizeof(uint32_t));

	cuda_sum<<<nBlocks, NUM_TH_PER_BLCK>>>(d_out, d_in, sums, n);
	if(nBlocks==1){
		cudaFree(sums);
		cudaFree(sumso);
		return 0;
	}
	cudaMemcpy(d_in, d_out, n*sizeof(uint32_t),cudaMemcpyDeviceToDevice);
	les_go(sums, sumso, nBlocks);
	post_cuda_sum<<<nBlocks, 2*NUM_TH_PER_BLCK>>>(d_out, d_in, sumso);

	cudaFree(sums);
	cudaFree(sumso);

	return 0;

}

int main() {

	// Setting array
	// const int N = N%(2*NUM_TH_PER_BLCK)==0?N:N+2*NUM_TH_PER_BLCK-N%(2*NUM_TH_PER_BLCK);
	size_t size = N * sizeof(uint32_t);

	// Input array (change this, keep its element uint32_t)
	uint32_t* in = (uint32_t*)malloc(size);
	std::fill_n(in, N, 1);			// Filling with 1

	// Using Thrust ref
	uint32_t* thrust_res = (uint32_t*)malloc(size);
	std::fill_n(thrust_res, N, 0);
	thrust_sum(thrust_res, in, N);

	// Using CUDA
  cudaEvent_t start, end;
  float kernel_time;

  uint32_t *d_in, *d_out, *sums;
  cudaMalloc((void**)&d_in, size);
  cudaMalloc((void**)&d_out, size);
	cudaMalloc((void**)&sums, size);

	uint32_t* cuda_res;
	cudaHostAlloc(&cuda_res, size, cudaHostAllocDefault);
	memset(cuda_res,0,size);

  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&end);


  cudaEventRecord(start, 0);

	les_go(d_in,d_out,N);

  cudaMemcpy(cuda_res, d_out, size, cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  check_result(thrust_res, cuda_res, N);
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Cuda time (ms): " << kernel_time << "\n";

	// Free device memory
	cudaFree(d_in);
	cudaFree(d_out);

	cudaFreeHost(cuda_res);
	free(thrust_res);
	free(in);

	return 0;
}
