#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "reduction.h"

using namespace std;

__global__
void vecDiffKernel(double *A, double *B, double *C, int n)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) C[i] = fabs(A[i] - B[i]);
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