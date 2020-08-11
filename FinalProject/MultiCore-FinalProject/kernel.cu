
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <Windows.h>
#include <unordered_map>
#include <assert.h>
#include <fstream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <omp.h>
#include <string>
#include <chrono>
#include <sstream>

using namespace std;

// A square m*m matrix implemented as a dynamic "double" array and its m.
struct Matrix {
	double* A;
	int m;
};

// To find all the input files' names.
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

// Prints matrix (Used solely for debugging purposes)
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

// Gets a matrix and its corresponding stream and computes its determinant.
double call_to_cusolver_with_stream(Matrix matrix, cudaStream_t& stream) {

	cusolverDnHandle_t cusolverH = NULL;

	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	int m = matrix.m;
	int lda = m;
	int lm = m * lda;

	double* A = matrix.A;
	double* LU = new double[lm];
	int* Ipiv = new int[m];      /* host copy of pivoting sequence */
	int info = 0;     /* host copy of error info */

	double* d_A = NULL; /* device copy of A */
	int* d_Ipiv = NULL; /* pivoting sequence */
	int* d_info = NULL; /* error info */
	int  lwork = 0;     /* size of workspace */
	double* d_work = NULL; /* device workspace for getrf */

	const bool pivot_on = 1;
	/*if (pivot_on) {
		printf("pivot is on : compute P*A = L*U \n");
	}
	else {
		printf("pivot is off: compute A = L*U (not numerically stable)\n");
	}*/

	/*printf("A = \n");
	printMatrix(m, m, A, lda, "A");
	printf("=====\n");*/

	/* step 1: create cusolver handle, bind a stream */
	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	//cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	//assert(cudaSuccess == cudaStat1);

	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: copy A to device */
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
	cudaStat2 = cudaMalloc((void**)&d_Ipiv, sizeof(int) * m);
	cudaStat4 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat4);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

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
		d_Ipiv,
		d_info);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	cudaStat2 = cudaMemcpy(LU, d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	cudaStat4 = cudaMemcpy(Ipiv, d_Ipiv, sizeof(int) * m, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	if (0 > info) {
		//printf("%d-th parameter is wrong \n", -info);
		exit(1);
	}
	/*printf("L and U = (matlab base-1)\n");
	printMatrix(m, m, LU, lda, "LU");
	printf("=====\n");*/

	// Calculating the determinant using OpenMP for acceleration.
	int swaps = 0;
	double determinant = 1.0;
#pragma omp parallel for reduction(*: determinant)
	for (int i = 0; i < m; i++) {
		if (Ipiv[i] != i + 1) {
#pragma omp critical
			++swaps;
		}
		determinant *= LU[i * m + i];
	}
	if (swaps % 2 == 1) {
		determinant *= -1;
	}

	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_info) cudaFree(d_info);
	if (d_work) cudaFree(d_work);

	if (cusolverH) cusolverDnDestroy(cusolverH);
	return determinant;
}

// Converts a string read from a file into a Matrix struct.
Matrix string_to_matrix(string matrix_str) {
	stringstream ss(matrix_str);
	vector<double> doubles;
	double buf;
	while (ss >> buf) {
		doubles.push_back(buf);
	}
	int n = doubles.size();
	double* A = new double[n];
	for (int i = 0; i < n; i++) {
		A[i] = doubles[i];
	}
	Matrix matrix = {
		A,
		sqrt(n)
	};
	//cout << "m is: " << sqrt(n) << endl;
	return matrix;
}

// Asynchronous write to all the found files.
void write_to_file(unordered_map<string, unordered_map<int, double>>& results, string folder_name) {
	vector<string> keys;
	for (auto item = results.begin(); item != results.end(); ++item) {
		keys.push_back(item->first);
	}
	CreateDirectory(folder_name.c_str(), NULL);
#pragma omp parallel for 
	for (int i = 0; i < keys.size(); i++) {
		string file_name = keys.at(i);
		unordered_map<int, double> file_results = results.at(file_name);
		auto full_file_name = folder_name + '\\' + file_name;
		ofstream file(full_file_name);
		for (int j = 0; j < file_results.size(); j++) {
			double result = file_results[j];
			// Because the matrix is all integers, the answer should also be an integer,
			// but since the library gives the answer as an array of doubles, we round the small values down to zero.
			if ((result < 0.5) && (result > 0))
				result = 0;
			file << result << endl;
		}
		file.close();
	}
}

// Creates a stream for each matrix and passes it to call_to_cusolver_with_stream for computation.
unordered_map<string, unordered_map<int, double>> call_cuda(unordered_map<string, vector<Matrix>>& matrices_of_files) {
	unordered_map<string, unordered_map<int, double>> file_determinants;
	// Files are processed serially.
	for (auto const& x : matrices_of_files) {
		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		// Save it in a HashMap to avoid inconsistencies.
		unordered_map<int, double> dets_of_file;
		// Matrices in one file are processed in parallel.
#pragma omp parallel for
		for (int i = 0; i < x.second.size(); i++) {
			Matrix matrix = x.second.at(i);
			double det = call_to_cusolver_with_stream(matrix, stream);
			//
#pragma omp critical
			dets_of_file[i] = det;
		}
		if (stream) cudaStreamDestroy(stream);
		string file_name = x.first;
		string key = file_name.substr(file_name.find("\\") + 1, file_name.length());
		file_determinants[key] = dets_of_file;
	}
	return file_determinants;
}

// Asynchronous read from all the found files.
unordered_map<string, vector<Matrix>> read_from_file(string folder_name) {
	unordered_map<string, vector<Matrix>> matrices_of_files;
	vector<string> files = get_all_files_names_within_folder(folder_name);
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
#pragma omp critical
		matrices_of_files[file_name] = file_matrices;
	}
	return matrices_of_files;
}

int main(int argc, char* argv[])
{
	auto t1 = chrono::high_resolution_clock::now();
	auto matrices_of_files = read_from_file("data_in");
	auto results = call_cuda(matrices_of_files);
	write_to_file(results, "data_out");
	auto t2 = chrono::high_resolution_clock::now();
	cudaDeviceReset();
	auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
	cout << "The program's execution time was: " << duration << " ms" << endl;
	return 0;
}