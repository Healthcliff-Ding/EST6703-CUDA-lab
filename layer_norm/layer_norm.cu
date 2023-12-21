#include <iostream>
#include <cuda_runtime.h>
#include <random>

constexpr int B = 10;
constexpr int M = 128;
constexpr int N = 128;
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
void fill_mat_1(float* mat, const int b, const int m, const int n) {
  for (int i = 0; i < b; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < n; ++k) {
        mat[i * (m * n) + j * n + k] = 1.;
      }
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

constexpr uint32_t THREAD_SIZE = 512; //! MUST be exponential of 8

__device__ __forceinline__ float4 operator+=(float4& a, float4 b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; 
  return a;
}

//* nvcc always inline a __device__ funciton when deemed appropriate
__device__ float row_reduce_mean_kernel(const float val) {
  __shared__ float partial_sum[THREAD_SIZE];
  uint32_t tid = threadIdx.x;
  float4 A_reg;

  partial_sum[tid] = val;

  // #pragma unroll
  // 512 threads -> 64 -> 8 -> 1
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

  return partial_sum[0];
}

__device__ float row_reduce_sum_kernel(const float val) {
  __shared__ float partial_sum[THREAD_SIZE];
  uint32_t tid = threadIdx.x;
  float4 A_reg;

  partial_sum[tid] = val;

  // #pragma unroll
  // 512 threads -> 64 -> 8 -> 1
  for (int stride = blockDim.x / 8; stride >= 1; stride /= 8) {
    __syncthreads();
    if (tid < stride) {
      float4 tmp4;
      A_reg = ((float4*)partial_sum)[tid];
      tmp4 = ((float4*)(partial_sum))[stride + tid];
      A_reg += tmp4;
      // partial_sum[tid] += (A_reg.x + A_reg.y + A_reg.z + A_reg.w) / 8;
      partial_sum[tid] = (A_reg.x + A_reg.y + A_reg.z + A_reg.w);
    }
  }

  return partial_sum[0];
}

__global__ void layer_norm0_kernel(const float* A, float* C, const int b, const int m, const int n) {
  //TODO 1.先用 rowReduceSum 算一次 mean, 把结果存到 shared 里面
  //TODO 2.再用 rowReduceSum 算一次 variance
  float mean     = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  float local_var = 0.0f;
  float4 local_reg;
  
  const int line_size = m * n;
  const float* A_ldg_ptr = A + blockIdx.x * line_size;

  /*! \note use row reduce sum to compute mean */
  /*! \note now we have blockDim.x threads each with local sum */
  for (int i = threadIdx.x; i < line_size / 128; i += blockDim.x) {
    local_reg = ((const float4*)A_ldg_ptr)[i];
    local_sum = local_reg.x + local_reg.y + local_reg.z + local_reg.w;
  }
  mean = row_reduce_sum_kernel(local_sum);
  mean = mean / line_size;
  #ifdef DEBUG
  if (blockIdx.x == 0 && threadIdx.x == 0) printf("mean is: %f\n", mean);
  #endif
  __syncthreads();

  //TODO 从shared memory里弄数据进来 -> 懒得弄了
  for (int i = threadIdx.x; i < line_size / 128; i += blockDim.x) {
    local_reg = ((const float4*)A_ldg_ptr)[i];
    local_var += (local_reg.x - mean) * (local_reg.x - mean)
               + (local_reg.y - mean) * (local_reg.y - mean)
               + (local_reg.z - mean) * (local_reg.z - mean)
               + (local_reg.w - mean) * (local_reg.w - mean);
  }
  variance = row_reduce_sum_kernel(local_var);
  variance = rsqrtf(variance + 1e6);
  #ifdef DEBUG
  if (blockIdx.x == 0 && threadIdx.x == 0) printf("var is: %f\n", variance);
  #endif
  __syncthreads();

  /*! \note write back */
  float* C_ldg_ptr = C + blockIdx.x * line_size; 
  for (int i = threadIdx.x; i < line_size / 128; i += blockDim.x) {
    local_reg.x = (local_reg.x - mean) * variance;
    local_reg.y = (local_reg.y - mean) * variance;
    local_reg.z = (local_reg.z - mean) * variance;
    local_reg.w = (local_reg.w - mean) * variance;
    ((float4*)C_ldg_ptr)[i] = local_reg;
  }
}

float* layer_norm0() {
  float* A = (float*)malloc(B * M * N * sizeof(float));
  float* C = (float*)malloc(B * M * N * sizeof(float));

  #ifndef TEST
  // fill_mat_rng(A, B, M, N);
  #else
  fill_mat_1(A, B, M, N);
  #endif // TEST

  float *cuA, *cuC;
  cudaMalloc((void **)&cuA, B* M * N * sizeof(float));
  cudaMalloc((void **)&cuC, B* M * N * sizeof(float));

  cudaMemcpy(cuA, A, B* M * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(THREAD_SIZE);
  dim3 gridDim(B);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  layer_norm0_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuC, B, M, N);
  cudaEventRecord(end, stream);

  cu_err = cudaEventSynchronize(end);
  cudaEventElapsedTime(&t0, begin, end);

  if (cu_err != cudaSuccess) {
  std::cerr << "CUDA error in kernel2"
            << ": " << cudaGetErrorString(cu_err) << std::endl;
  }

  cudaMemcpy(C, cuC, B * M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(cuA);
  cudaFree(cuC);
  free(A);
  return C;
}

int main()
{
  cudaStreamCreate(&stream);
  std::cout << "start..." << std::endl;
  float *mat0 = layer_norm0();
  #ifdef TEST
  // check_mat1(mat0, M, N, 2);
  for (int i = 0; i < 10; ++i) std::cout << mat0[i] << ", " << std::endl;
  #endif // TEST
}