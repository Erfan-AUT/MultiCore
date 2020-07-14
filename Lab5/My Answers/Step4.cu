
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

cudaError_t addWithCuda(int* output, int size);

__global__ void addKernel(int* output)
{
	const int blockId = blockIdx.x //1D
		+ blockIdx.y * gridDim.x //2D
		+ gridDim.x * gridDim.y * blockIdx.z; //3D
	const int threadId = threadIdx.x //1D
		+ threadIdx.y * blockDim.x //2D
		+ blockDim.x * blockDim.y * threadIdx.z; //3D
	const int warpID = threadId / warpSize;
	const int globalThreadID = blockId * blockDim.x * blockDim.y * blockDim.z + threadId;
	const int arrIdx = globalThreadID * 4;
	output[arrIdx] = globalThreadID;
	output[arrIdx + 1] = blockId;
	output[arrIdx + 2] = warpID;
	output[arrIdx + 3] = threadId;
}

int main()
{
	const int size = 128;
	const int size2 = 4;
	int* output = new int[size * size2];
	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(output, size * size2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	for (int i = 0; i < size * size2; i += 4)
	{
		printf("Calculated Thread: %d - Block: %d - Warp: %d - Thread: %d \n", output[i], output[i+1], output[i+2], output[i+3]);
	}
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* output, int size)
{
	int* dev_out = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_out, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_out, output, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <2, 64 >> > (dev_out);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output, dev_out, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_out);

	return cudaStatus;
}

