// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/
#define TILE_WIDTH 4
__global__ void
matrixMulCUDA(float* C, float* A, float* B, int n)
{
	int start_row = blockDim.y * blockIdx.y * TILE_WIDTH + threadIdx.y * TILE_WIDTH;  	
	int end_row = start_row + TILE_WIDTH;
	int start_col = blockDim.x * blockIdx.x * TILE_WIDTH + threadIdx.x * TILE_WIDTH;  	
	int end_col = start_col + TILE_WIDTH;  	
	for (int row = start_row; row < end_row; row++) {
		for (int col = start_col; col < end_col; col++) {
			float C_val = 0;
			for (int k = 0; k < n; ++k) {
				float A_elem = A[row * n + k];  	 	 	 	
				float B_elem = B[k * n + col];
				C_val += A_elem * B_elem;
			}
			C[row * n + col] = C_val;
		}
	}
}


void constantInit(float* data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

void printMat(float* v, int matSizeX, int matSizeY) {
	printf("[-] Vector elements: \n");
	for (int i = 0; i < matSizeX; i++) {
		for (int j = 0; j < matSizeY; j++)
			cout << v[i * matSizeY + j] << "   ";
		cout << endl;
	}
}

void checkCorrectness(float* v, int matSizeX, int matSizeY) {
	float value = v[0];
	for (int i = 0; i < matSizeX; i++) {
		for (int j = 0; j < matSizeY; j++) {
			if (v[i * matSizeY + j] != value)
			{
				cout << "Something has gone wrong!" << endl;
				return;
			}
		}
	}
	cout << "All is right with the world." << endl;
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
float matrixMultiply(int argc, char** argv, int n)
{
	// Allocate host memory for matrices A and B
	unsigned int size_A = n * n;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);
	unsigned int size_B = n * n;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);

	// Initialize host memory
	const float valA = 1.0f;
	const float valB = 0.01f;
	constantInit(h_A, size_A, valA);
	constantInit(h_B, size_B, valB);

	// Allocate device memory
	float* d_A, * d_B, * d_C;

	// Allocate host matrix C
	unsigned int mem_size_C = n * n * sizeof(float);
	float* h_C = (float*)malloc(mem_size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void**)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**)&d_C, mem_size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters
	//dim3 threads(n, n, 1);
	/*dim3 threads(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 grid(n/TILE_WIDTH, n/TILE_WIDTH, 1);*/
	/*int a_b = 32, b_c = 128;
	dim3 threads(a_b, a_b, 1);
	dim3 grid(b_c, b_c, 1);*/
	dim3 threads(16, 16, 1);
	dim3 grid(n/(TILE_WIDTH*16), n/(TILE_WIDTH*16), 1);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Execute the kernel
	matrixMulCUDA << < grid, threads >> > (d_C, d_A, d_B, n);

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	printf("CUDA elapsed time in msec = %f\n", msecTotal);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	checkCorrectness(h_C, n, n);

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return msecTotal;

}

float serialMultiply(int n) {
	// Allocate host memory for matrices A and B
	unsigned int size_A = n * n;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);
	unsigned int size_B = n * n;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);

	// Initialize host memory
	const float valA = 1.0f;
	const float valB = 0.01f;
	constantInit(h_A, size_A, valA);
	constantInit(h_B, size_B, valB);

	// Allocate host matrix C
	unsigned int mem_size_C = n * n * sizeof(float);
	float* h_C = (float*)malloc(mem_size_C);
	chrono::steady_clock::time_point begin = chrono::steady_clock::now();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < n; k++)
				sum += h_A[i * n + k] * h_B[k * n + j];
			h_C[i * n + j] = sum;
		}
	}
	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
	cout << "Serial time difference = " << elapsed_time << " ms" << endl;
	checkCorrectness(h_C, n, n);
	return elapsed_time;
}

/**
* Program main
*/
int main(int argc, char** argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	// By default, we use device 0
	int devID = 0;
	cudaSetDevice(devID);

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	// Size of square matrices
	size_t n = 0;
	printf("[-] N = ");
	scanf("%u", &n);
	cout << "Tile Width: " << TILE_WIDTH << endl;
	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", n, n, n, n);

	float cuda_time = matrixMultiply(argc, argv, n);
	int serial_time = serialMultiply(n);
	cout << "Speed up: " << serial_time / cuda_time << endl;

	exit(0);
}
