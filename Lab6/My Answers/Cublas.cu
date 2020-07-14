#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128 

#include <thrust/device_vector.h>
#include "cublas_v2.h"
#include <iostream>

using namespace std;

// C-style indexing
int ci(int row, int column, int nColumns) {
    return row * nColumns + column;
}

int main(void)
{
    size_t n = 1;
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "!!!! CUBLAS initialization error\n";
    }
    while (n > 0)
    {
        printf("[-] N = ");
        scanf("%u", &n);
        printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", n, n, n, n);

        // initialize data
        thrust::device_vector<float> D(n * n);
        thrust::device_vector<float> E(n * n);
        thrust::device_vector<float> F(n * n);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                D[ci(i, j, n)] = 1;
                //cout << D[ci(i, j, n)] << " ";
            }
            //cout << "\n";
        }

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                E[ci(i, j, n)] = 0.1;
                //cout << E[ci(i, j, n)] << " ";
            }
            //cout << "\n";
        }

        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                F[ci(i, j, n)] = 0;

        /* Initialize CUBLAS */
        cudaEvent_t start;
        cudaError_t error = cudaEventCreate(&start);
        cudaEvent_t stop;
        error = cudaEventCreate(&stop);
        

        float alpha = 1.0f, beta = 0.0f;
        error = cudaEventRecord(start, NULL);
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &alpha, thrust::raw_pointer_cast(&E[0]), n,
            thrust::raw_pointer_cast(&D[0]), n,
            &beta, thrust::raw_pointer_cast(&F[0]), n);// colE  x rowD
        if (status != CUBLAS_STATUS_SUCCESS) {
            cerr << "!!!! kernel execution error.\n";
        }
        error = cudaEventRecord(stop, NULL);
        error = cudaEventSynchronize(stop);
        float msecTotal = 0.0f;
        error = cudaEventElapsedTime(&msecTotal, start, stop);
        cout << "Elapsed time in msec = " << msecTotal << endl;

        /*for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                cout << F[ci(i, j, n)] << " ";
            }
            cout << "\n";
        }*/

    }
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "!!!! shutdown error (A)\n";
    }

    return 0;
}