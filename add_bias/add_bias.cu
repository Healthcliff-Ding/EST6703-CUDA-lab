#include <iostream>
#include <cuda_runtime.h>
#include <random>

constexpr int M = 4096;
constexpr int N = 4096;
float t0;
cudaError_t cu_err;
cudaStream_t stream;

// #define TEST

void fill_mat_rng(float* mat, const int m, const int n) {
  // size_t tmp = 0;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<> dis(-1., 1.);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      mat[i * n + j] = dis(rng);
    }
  }
}

#ifdef TEST
void fill_mat_1(float* mat, const int m, const int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      mat[i * n + j] = 1.;
    }
  }
}

void check_mat1(const float* mat, const int m, const int n, const int k) {
  size_t sum = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if ((int)mat[i * n + j] != k) {
        std::cout << "Error! index (" << i << ", " << j << ") is " << mat[i * n + j] << std::endl;
        sum++;
      }
      if (sum > 128) return;
    }
  }
  if (sum == 0)
    std::cout << "OK." << std::endl;
}
#endif // TEST

//TODO tune blockDim
constexpr uint32_t THREAD_SIZE = 256;
__global__ void add_bias0_kernel(const float* A, const float* B, float* C) {
  uint32_t bid = blockIdx.x;
  uint32_t tid = threadIdx.x;
  float4 A_reg;
  const float* A_ldg_ptr = A + bid * N + tid * 4;
  float* A_stg_ptr = C + bid * N + tid * 4;
  const float bias = B[bid];

  #pragma unroll
  for (int i = 0; i < N / (blockDim.x * 4); ++i) {
    A_reg = ((float4*)A_ldg_ptr)[0];
    A_reg.x += bias;
    A_reg.y += bias;
    A_reg.z += bias;
    A_reg.w += bias;
    ((float4*)A_stg_ptr)[0] = A_reg;
    A_ldg_ptr += 4 * blockDim.x;
    A_stg_ptr += 4 * blockDim.x;
  }
}

float* add_bias0() {
  float* A = (float*)malloc(M * N * sizeof(float));
  float* B = (float*)malloc(N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  #ifndef TEST
  fill_mat_rng(A, M, N);
  fill_mat_rng(B, M, 1);
  #else
  fill_mat_1(A, M, N);
  fill_mat_1(B, M, 1);
  #endif // TEST

  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * N * sizeof(float));
  cudaMalloc((void **)&cuB, N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(THREAD_SIZE);
  dim3 gridDim(M);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  add_bias0_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuB, cuC);
  cudaEventRecord(end, stream);

  cu_err = cudaEventSynchronize(end);
  cudaEventElapsedTime(&t0, begin, end);

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
  cudaStreamCreate(&stream);
  constexpr size_t TFLOP = (size_t)M * (size_t)N;
  std::cout << "start..." << std::endl;
  float *mat0 = add_bias0();
  #ifdef TEST
  check_mat1(mat0, M, N, 2);
  #endif // TEST
  std::cout << "kernel0: " << t0 << "ms  TFLOPs: "
          << TFLOP / t0 / 1e9
          << std::endl;  
}