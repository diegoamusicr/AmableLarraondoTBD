#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "reduction.h"

using namespace std;

__global__
void vecDiffKernel(double *A, double *B, double *C, int n)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) C[i] = abs(A[i] - B[i]);
}

template <unsigned int blockSize>
__device__ 
void warpReduce(volatile double *sdata, unsigned int tid)
{
	if (blockSize >= 64) sdata[tid] += sdata[tid+32];
	if (blockSize >= 32) sdata[tid] += sdata[tid+16];
	if (blockSize >= 16) sdata[tid] += sdata[tid+ 8];
	if (blockSize >=  8) sdata[tid] += sdata[tid+ 4];
	if (blockSize >=  4) sdata[tid] += sdata[tid+ 2];
	if (blockSize >=  2) sdata[tid] += sdata[tid+ 1];
}

template <unsigned int blockSize>
__global__
void reduceKernel(double * g_idata, double * g_odata, int n)
{
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	while (i < n)
	{
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) {if (tid < 256) {sdata[tid] += sdata[tid + 256];} __syncthreads();}
	if (blockSize >= 256) {if (tid < 128) {sdata[tid] += sdata[tid + 128];} __syncthreads();}
	if (blockSize >= 128) {if (tid <  64) {sdata[tid] += sdata[tid +  64];} __syncthreads();}

	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void vecDiffWrapper(double * A, double * B, double * C, int n)
{
	int size = n * sizeof(double);

	double *d_A, *d_B, *d_C;

	cudaMalloc((void**) &d_A, size);
	cudaMalloc((void**) &d_B, size);
	cudaMalloc((void**) &d_C, size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(n/256.0), 1, 1);
	dim3 dimBlock(256, 1, 1);

	vecDiffKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

double reduceSumWrapper(double * A, int n)
{
	int size = n * sizeof(double);

	double *d_A, *d_out;

	cudaMalloc((void**) &d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	int num_threads = 256;
	int num_blocks = ceil(n/(float)num_threads);

	int size_out = num_blocks*sizeof(double);

	cudaMalloc((void**) &d_out, size_out);
	double * h_out = new double[num_blocks];

	dim3 dimGrid(num_blocks, 1, 1);
	dim3 dimBlock(num_threads, 1, 1);

	reduce<double>(n, num_threads, num_blocks, 6, d_A, d_out);

	cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

	double sum = 0.0;

	for (int i=0; i < num_blocks; i++)
	{
		sum += h_out[i];
	}

	cudaFree(d_A); cudaFree(d_out);

	delete h_out;

	return sum;
}