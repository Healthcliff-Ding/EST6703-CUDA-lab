#include <iostream>
#include <cuda_runtime.h>
#include <random>

constexpr int M = 4096;
constexpr int N = 4096;
float t0;
cudaError_t cu_err;
cudaStream_t stream;

#define TEST

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
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < n; ++k) {
        mat[ + j * n + k] = 1.;
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

constexpr uint32_t THREAD_SIZE = 64; //! MUST be exponential of 8

__device__ __forceinline__ float4 operator+=(float4& a, float4 b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; 
  return a;
}

//* nvcc always inline a __device__ funciton when deemed appropriate
__device__ float row_reduce_max_kernel(const float val) {
  __shared__ float partial_max[THREAD_SIZE];
  uint32_t tid = threadIdx.x;
  float4 A_reg;

  partial_max[tid] = val;

  // #pragma unroll
  // 64 threads -> 8 -> 1
  for (int stride = blockDim.x / 8; stride >= 1; stride /= 8) {
    __syncthreads();
    if (tid < stride) {
      float4 tmp4;
      A_reg = ((float4*)partial_max)[tid];
      tmp4 = ((float4*)(partial_max))[stride + tid];
      A_reg.x = fmaxf(A_reg.x, tmp4.x);
      A_reg.y = fmaxf(A_reg.y, tmp4.y);
      A_reg.z = fmaxf(A_reg.z, tmp4.z);
      A_reg.w = fmaxf(A_reg.w, tmp4.w);
      partial_max[tid] = fmaxf(fmaxf(A_reg.x, A_reg.y), fmaxf(A_reg.z, A_reg.w));
    }
  }

  return partial_max[0];
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

/*! \note 4096 * 4096, 256 threads each row */
__global__ void softmax0_kernel(const float* A, float* C, const int m, const int n) {
    //TODO 1.行加载算max
    //TODO 2.逐个算一下exp
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float* A_ldg_ptr = A + bid * n + tid * 4;
    float* C_stg_ptr = C + bid * n + tid * 4;
    float4 tmp4;
    float local_max = -1e20f;
    float max_val;
    float local_sum = 0.0f;
    float sum_val;

    for (int i = 0; i < n / (blockDim.x * 4); ++i) {
        tmp4 = ((const float4*)A_ldg_ptr)[0];
        local_max = fmaxf(local_max, tmp4.x);
        local_max = fmaxf(local_max, tmp4.y);
        local_max = fmaxf(local_max, tmp4.z);
        local_max = fmaxf(local_max, tmp4.w);
        A_ldg_ptr += blockDim.x * 4;
    }
    max_val = row_reduce_max_kernel(local_max);
    __syncthreads();

    A_ldg_ptr = A + bid * n + tid * 4;
    for (int i = 0; i < n / (blockDim.x * 4); ++i) {
        tmp4 = ((const float4*)A_ldg_ptr)[0];
        tmp4.x = __expf(tmp4.x - max_val);
        tmp4.y = __expf(tmp4.y - max_val);
        tmp4.z = __expf(tmp4.z - max_val);
        tmp4.w = __expf(tmp4.w - max_val);

        ((float4*)C_stg_ptr)[0] = tmp4;
        local_sum += tmp4.x + tmp4.y + tmp4.z + tmp4.w;
        A_ldg_ptr += blockDim.x * 4;
        C_stg_ptr += blockDim.x * 4;
    }
    sum_val = row_reduce_sum_kernel(local_sum);
    __syncthreads();

    C_stg_ptr = C + bid * n + tid * 4;
    for (int i = 0; i < n / (blockDim.x * 4); ++i) {
        tmp4 =  ((float4*)C_stg_ptr)[0];
        tmp4.x = __fdividef(tmp4.x, sum_val);
        tmp4.y = __fdividef(tmp4.y, sum_val);
        tmp4.z = __fdividef(tmp4.z, sum_val);
        tmp4.w = __fdividef(tmp4.w, sum_val);

        ((float4*)C_stg_ptr)[0] = tmp4;
        C_stg_ptr += blockDim.x * 4;
    }
}

float* softmax0() {
  float* A = (float*)malloc(M * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  #ifndef TEST
  // fill_mat_rng(A, M, N);
  #else
  fill_mat_1(A, M, N);
  #endif // TEST

  float *cuA, *cuC;
  cudaMalloc((void **)&cuA, M * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(THREAD_SIZE);
  dim3 gridDim(M);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  softmax0_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuC, M, N);
  cudaEventRecord(end, stream);

  cu_err = cudaEventSynchronize(end);
  cudaEventElapsedTime(&t0, begin, end);

  if (cu_err != cudaSuccess) {
  std::cerr << "CUDA error in kernel2"
            << ": " << cudaGetErrorString(cu_err) << std::endl;
  }

  cudaMemcpy(C, cuC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(cuA);
  cudaFree(cuC);
  free(A);
  return C;
}

int main()
{
  cudaStreamCreate(&stream);
  std::cout << "start..." << std::endl;
  float *mat0 = softmax0();
  #ifdef TEST
  // check_mat1(mat0, M, N, 2);
  for (int i = 0; i < 10; ++i) std::cout << mat0[i] << ", " << std::endl;
  #endif // TEST
}