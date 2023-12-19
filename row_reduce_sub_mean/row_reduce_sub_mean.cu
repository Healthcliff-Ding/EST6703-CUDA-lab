#include <iostream>
#include <cuda_runtime.h>
#include <random>

constexpr int N = 4096;
float t0;
cudaError_t cu_err;
cudaStream_t stream;

// #define TEST

__device__ __forceinline__ float4 operator+=(float4& a, float4 b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; 
  return a;
}

__device__ __forceinline__ float4 operator-=(float4& a, float b) {
  a.x -= b; a.y -= b; a.z -= b; a.w -= b; 
  return a;
}

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
      mat[i * n + j] = 0;
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
constexpr uint32_t THREAD_SIZE = 512;
__global__ void row_reduce_sub_mean0_kernel(const float* A, float* C) {
  __shared__ float partial_sum[THREAD_SIZE];
  uint32_t bid = blockIdx.x;
  uint32_t tid = threadIdx.x;
  float4 A_reg;
  float res = 0.;
  const float* A_ldg_ptr = A + bid * N + tid * 4;
  float* A_stg_ptr = C + bid * N + tid * 4;

  // #pragma unroll
  for (int i = 0; i < N / (blockDim.x * 4); ++i) {
    //TODO initialization
    A_reg = ((float4*)A_ldg_ptr)[0];
    A_reg.x = (A_reg.x + A_reg.y + A_reg.z + A_reg.w) / 4;
    partial_sum[tid] = A_reg.x;  
    
    // #pragma unroll
    // stride = 32, 4, 0
    for (int stride = blockDim.x / 8; stride >= 1; stride /= 8) {
      __syncthreads();
      if (tid < stride) {
        float4 tmp4;
        A_reg = ((float4*)partial_sum)[tid];
        tmp4 = ((float4*)(partial_sum))[stride + tid];
        A_reg += tmp4;
        // partial_sum[tid] += (A_reg.x + A_reg.y + A_reg.z + A_reg.w) / 8;
        partial_sum[tid] = (A_reg.x + A_reg.y + A_reg.z + A_reg.w) / 8;
      }
    }

    res += partial_sum[0];
    A_ldg_ptr += 4 * blockDim.x;
    #ifdef DEBUG
    // if (bid == 0 && tid == 0) printf("i=%d: res=%f\n", i, res);
    #endif
  }

  res = res / (N / (blockDim.x * 4));
  A_ldg_ptr = A + bid * N + tid * 4;

  // #pragma unroll
  for (int i = 0; i < N / (blockDim.x * 4); ++i) {
    A_reg = ((float4*)A_ldg_ptr)[0];
    A_reg -= res;
    ((float4*)A_stg_ptr)[0] = A_reg;
    A_ldg_ptr += 4 * blockDim.x;
    A_stg_ptr += 4 * blockDim.x;
  }
}

float* row_reduce_sub_mean0_kernel() {
  float* A = (float*)malloc(N * N * sizeof(float));
  float* C = (float*)malloc(N * N * sizeof(float));

  #ifndef TEST
  fill_mat_rng(A, N, N);
  #else
  fill_mat_1(A, N, N);
  #endif // TEST

  float *cuA, *cuC;
  cudaMalloc((void **)&cuA, N * N * sizeof(float));
  cudaMalloc((void **)&cuC, N * N * sizeof(float));

  cudaMemcpy(cuA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(THREAD_SIZE);
  dim3 gridDim(N);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  row_reduce_sub_mean0_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuC);
  cudaEventRecord(end, stream);

  cu_err = cudaEventSynchronize(end);
  cudaEventElapsedTime(&t0, begin, end);

  if (cu_err != cudaSuccess) {
    std::cerr << "CUDA error in kernel2"
              << ": " << cudaGetErrorString(cu_err) << std::endl;
  }

  cudaMemcpy(C, cuC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(cuA);
  cudaFree(cuC);
  free(A);
  return C;
}

int main()
{
  cudaStreamCreate(&stream);
  constexpr size_t TFLOP = (size_t)N * (size_t)N;
  std::cout << "start..." << std::endl;
  float *mat0 = row_reduce_sub_mean0_kernel();
  #ifdef TEST
  check_mat1(mat0, N, N, 0);
  #endif // TEST
  std::cout << "kernel0: " << t0 << "ms  TFLOPs: "
          << TFLOP / t0 / 1e9
          << std::endl;  
}