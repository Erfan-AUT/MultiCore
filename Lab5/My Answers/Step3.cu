
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

cudaError_t addWithCuda_1(int* c, const int* a, const int* b, unsigned int matSizeX, unsigned int matSizeY, unsigned int N, int baseSize);
cudaError_t addWithCuda_2(int* c, const int* a, const int* b, unsigned int matSizeX, unsigned int matSizeY, unsigned int N, int baseSize);
void fillMat(int* v, int matSizeX, int matSizeY);
void printMat(int* v, int matSizeX, int matSizeY);


__global__ void addKernel_nSums(int* c, const int* a, const int* b, unsigned int N)
{

	int threadID = threadIdx.x + (threadIdx.y * blockDim.x);
	threadID *= N;

	for (int x = threadID; x < N + threadID; x++)
	{
		c[x] = a[x] + b[x];
	}
}


__global__ void addKernel_nBlocks(int* c, const int* a, const int* b)
{
	const int blockId = blockIdx.x //1D
		+ blockIdx.y * gridDim.x //2D
		+ gridDim.x * gridDim.y * blockIdx.z; //3D
	const int threadId = threadIdx.x //1D
		+ threadIdx.y * blockDim.x //2D
		+ blockDim.x * blockDim.y * threadIdx.z; //3D
	const int globalThreadID = blockId * blockDim.x * blockDim.y * blockDim.z + threadId;

	c[globalThreadID] = a[globalThreadID] + b[globalThreadID];
}


int main()
{
	const int squareOfN = 10;
	const int baseSize = 32;
	const int matSizeX = baseSize * squareOfN;
	const int matSizeY = baseSize * squareOfN;
	int* a;
	int* b;
	int* c;
	int* d;
	a = (int*)malloc(sizeof(int) * matSizeX * matSizeY);
	b = (int*)malloc(sizeof(int) * matSizeX * matSizeY);
	c = (int*)malloc(sizeof(int) * matSizeX * matSizeY);
	d = (int*)malloc(sizeof(int) * matSizeX * matSizeY);

	fillMat(a, matSizeX, matSizeY);
	fillMat(b, matSizeX, matSizeY);


	// Add vectors in parallel.
	printf("N Blocks: \n");
	cudaError_t cudaStatus = addWithCuda_1(c, a, b, matSizeX, matSizeY, squareOfN * squareOfN, baseSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	printf("N Sums: \n");
	cudaStatus = addWithCuda_2(d, a, b, matSizeX, matSizeY, squareOfN * squareOfN, baseSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	 /*printMat(a, matSizeX, matSizeY);
	 printMat(b, matSizeX, matSizeY);
	printMat(d, matSizeX, matSizeY);*/

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
void fillMat(int* v, int matSizeX, int matSizeY) {
	static int L = 0;
	for (int i = 0; i < matSizeX; i++) {
		for (int j = 0; j < matSizeY; j++)
			v[i * matSizeY + j] = L++;
	}
}
void printMat(int* v, int matSizeX, int matSizeY) {
	int i;
	printf("[-] Vector elements: \n");
	for (int i = 0; i < matSizeX; i++) {
		for (int j = 0; j < matSizeY; j++)
			printf("%d	", v[i * matSizeY + j]);
		printf("\n");

	}
	printf("\b\b  \n");
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda_1(int* c, const int* a, const int* b, unsigned int matSizeX, unsigned int matSizeY, unsigned int N, int baseSize)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaEventRecord(start, NULL);
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, matSizeX * matSizeY * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, matSizeX * matSizeY * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, matSizeX * matSizeY * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, matSizeX * matSizeY * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, matSizeX * matSizeY * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// KEEP THIS HERE
	//dim3 block_size = dim3(matSizeX, matSizeY / N, 1);
	//// Launch a kernel on the GPU with one thread for each element.
	//addKernel << <1, block_size >> > (dev_c, dev_a, dev_b, N);

	/*dim3 block_size = dim3(matSizeX, matSizeY / baseSize, 1);
	addKernel_nSums << <1, block_size >> > (dev_c, dev_a, dev_b, N);*/

	addKernel_nBlocks << <N, 1024 >> > (dev_c, dev_b, dev_a);

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
	cudaStatus = cudaMemcpy(c, dev_c, matSizeX * matSizeY * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaEventRecord(stop, NULL);
	cudaStatus = cudaEventSynchronize(stop);
	float mSecTotal;
	cudaStatus = cudaEventElapsedTime(&mSecTotal, start, stop);
	printf("Time: %f \n\n", mSecTotal);

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
cudaError_t addWithCuda_2(int* c, const int* a, const int* b, unsigned int matSizeX, unsigned int matSizeY, unsigned int N, int baseSize)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaEventRecord(start, NULL);
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, matSizeX * matSizeY * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, matSizeX * matSizeY * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, matSizeX * matSizeY * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, matSizeX * matSizeY * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, matSizeX * matSizeY * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// KEEP THIS HERE
	//dim3 block_size = dim3(matSizeX, matSizeY / N, 1);
	//// Launch a kernel on the GPU with one thread for each element.
	//addKernel << <1, block_size >> > (dev_c, dev_a, dev_b, N);

	dim3 block_size = dim3(matSizeX / baseSize, matSizeY / baseSize, 1);
	addKernel_nSums << <1, block_size >> > (dev_c, dev_a, dev_b, N);

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
	cudaStatus = cudaMemcpy(c, dev_c, matSizeX * matSizeY * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaEventRecord(stop, NULL);
	cudaStatus = cudaEventSynchronize(stop);
	float mSecTotal;
	cudaStatus = cudaEventElapsedTime(&mSecTotal, start, stop);
	printf("Time: %f \n\n", mSecTotal);

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}