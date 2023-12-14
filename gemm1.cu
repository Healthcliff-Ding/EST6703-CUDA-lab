#include <iostream>
#include <cuda_runtime.h>

const int M = 20480;
const int N = 4096;
const int K = 4096;

void fill_mat(float* mat, const int m, const int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      mat[i * n + j] = 1;
    }
  }
}

void check_mat(const float* mat, const int m, const int n, const int k) {
  size_t sum = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (mat[i * n + j] != k) {
        std::cout << "Error! index (" << i << ", " << j << ") is " << mat[i * n + j] << std::endl;
        sum++;
      }
    }
  }
  if (sum == 0)
    std::cout << "OK." << std::endl;
}

/*!
 *  \brief gemm version 0. 1 thread for k MAC
 *  \note gridDim control Matrix size; blockDim Matters performance
 *  \todo change blockDim to tune performance
 */
__global__ void gemm0_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

__global__ void hello_kernel() {
  if (threadIdx.x == 0)
    printf("Hello\n");
}

const int TILE_WIDTH = 32;

__global__ void gemm1_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float pSum = 0.;
  for (int p = 0; p < K / TILE_WIDTH; ++p) {
    /*! \note load data into shared memory */
    Mds[ty][tx] = A[row * K + p * TILE_WIDTH + tx];
    Nds[ty][tx] = B[(ty + p * TILE_WIDTH) * N + col];
    __syncthreads();

    /*! \todo ILP, loop unroll */
    for (int k = 0; k < TILE_WIDTH; ++k) {
      pSum += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  /*! \note write partial sum back to global memory */
  C[row * N + col] = pSum;
}

__global__ void gemm1_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
  
}

float* gemm0() {
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  fill_mat(A, M, K);
  fill_mat(B, K, N);

  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * K * sizeof(float));
  cudaMalloc((void **)&cuB, K * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(32, 32);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  gemm0_kernel<<<gridDim, blockDim>>>(cuA, cuB, cuC, M, N, K);
  // cudaError_t tmp = cudaDeviceSynchronize();

  cudaMemcpy(C, cuC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(cuA);
  cudaFree(cuB);
  cudaFree(cuC);
  free(A);
  free(B);
  return C;
}

float* gemm1() {
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  fill_mat(A, M, K);
  fill_mat(B, K, N);

  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * K * sizeof(float));
  cudaMalloc((void **)&cuB, K * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(32, 32);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  gemm1_kernel<<<gridDim, blockDim>>>(cuA, cuB, cuC, M, N, K);

  cudaMemcpy(C, cuC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(cuA);
  cudaFree(cuB);
  cudaFree(cuC);
  free(A);
  free(B);
  return C;
}

int main()
{
  float *mat0 = gemm0();
  check_mat(mat0, M, N, K);
  free(mat0);
  float *mat1 = gemm1();
  check_mat(mat1, M, N, K);
  free(mat1);
  return 0;
}