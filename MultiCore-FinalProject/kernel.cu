
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <Windows.h>
#include <stdlib.h>
#include <unordered_map>
#include <assert.h>
#include <fstream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <omp.h>
#include <string>

using namespace std;

struct Matrix {
	double* A;
	int m;
};

vector<string> get_all_files_names_within_folder(string folder_name)
{
	vector<string> names;
	string search_path = folder_name + "/*.txt*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				//auto a = fd.cFileName;
				names.push_back(folder_name + "\\" + fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

void printMatrix(int m, int n, const double* A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row + col * lda];
			cout << Areg << " ";
			//printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
		cout << endl;
	}
}

double call_to_cusolver_with_stream(Matrix matrix, cudaStream_t& stream) {

	cusolverDnHandle_t cusolverH = NULL;
	//cudaStream_t stream = NULL;

	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	int m = matrix.m;
	int lda = m;

	double* A = matrix.A;
	double* LU = new double[lda * m];
	int* Ipiv = new int[m];      /* host copy of pivoting sequence */
	int info = 0;     /* host copy of error info */

	double* d_A = NULL; /* device copy of A */
	int* d_info = NULL; /* error info */
	int  lwork = 0;     /* size of workspace */
	double* d_work = NULL; /* device workspace for getrf */

	printf("pivot is off: compute A = L*U (not numerically stable)\n");

	printf("A = \n");
	printMatrix(m, m, A, lda, "A");
	printf("=====\n");

	/* step 1: create cusolver handle, bind a stream */
	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	//cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	//assert(cudaSuccess == cudaStat1);

	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: copy A to device */
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
	cudaStat4 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);

	/* step 3: query working space of getrf */
	status = cusolverDnDgetrf_bufferSize(
		cusolverH,
		m,
		m,
		d_A,
		lda,
		&lwork);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double) * lwork);
	assert(cudaSuccess == cudaStat1);

	/* step 4: LU factorization */

	status = cusolverDnDgetrf(
		cusolverH,
		m,
		m,
		d_A,
		lda,
		d_work,
		NULL,
		d_info);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	cudaStat2 = cudaMemcpy(LU, d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	if (0 > info) {
		printf("%d-th parameter is wrong \n", -info);
		exit(1);
	}
	printf("L and U = (matlab base-1)\n");
	printMatrix(m, m, LU, lda, "LU");
	printf("=====\n");

	double determinant = 1.0;
#pragma omp parallel for reduction(*: determinant)
	for (int i = 0; i < lda * lda; i += lda) {
		determinant *= LU[i];
	}

	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_info) cudaFree(d_info);
	if (d_work) cudaFree(d_work);

	if (cusolverH) cusolverDnDestroy(cusolverH);
	return determinant;
}

Matrix string_to_matrix(string matrix_str) {
	int len = matrix_str.length();
	// Assuming no \0 at its end.
	int n = (len - 1) / 2;
	double* A = new double[n];
	for (int i = 0; i < len; i += 2) {
		A[i / 2] = matrix_str[i] - '0';
		cout << A[i / 2] << " ";
	}
	Matrix matrix = {
		A,
		sqrt(n)
	};
	return matrix;
}

unordered_map<string, vector<double>> call_cuda(unordered_map<string, vector<Matrix>> matrices_of_files) {
	unordered_map<string, vector<double>> file_determinants;
	for (auto const& x : matrices_of_files) {
		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		vector<double> dets_of_file;
#pragma omp parallel for
		for (int i = 0; i < x.second.size(); i++) {
			Matrix matrix = x.second.at(i);
			double det = call_to_cusolver_with_stream(matrix, stream);
#pragma omp critical
			dets_of_file.push_back(det);
		}
		if (stream) cudaStreamDestroy(stream);
		file_determinants[x.first] = dets_of_file;
	}
	return file_determinants;
}

int main(int argc, char* argv[])
{
	vector<string> files = get_all_files_names_within_folder("data_in");
	unordered_map<string, vector<Matrix>> matrices_of_files;
#pragma omp parallel for
	for (int i = 0; i < files.size(); i++) {
		string file_name = files.at(i);
		vector<Matrix> file_matrices;
		ifstream input(file_name);
		for (string line; getline(input, line);) {
			if (line.length() > 2) {
				file_matrices.push_back(string_to_matrix(line));
			}
		}
//#pragma omp critical
		matrices_of_files[file_name] = file_matrices;
	}


	//call_to_cusolver_with_stream(4);
	//cudaDeviceReset();
	return 0;
}