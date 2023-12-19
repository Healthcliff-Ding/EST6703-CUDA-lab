#include <iostream>
#include <cuda_runtime.h>
#include <random>

constexpr int M = 20480;
constexpr int N = 4096;
constexpr int K = 4096;

static cudaStream_t stream;
static cudaError_t cu_err;
static float t0, t1, t2;

#define BUG

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

void fill_mat_1(float* mat, const int m, const int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      mat[i * n + j] = 1.;
    }
  }
}

void check_mat(const float* mat, const int m, const int n, const int k) {
  size_t sum = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if ((int)mat[i * n + j] != k) {
        std::cout << "Error! index (" << i << ", " << j << ") is " << mat[i * n + j] << std::endl;
        sum++;
      }
    }
  }
  if (sum == 0)
    std::cout << "OK." << std::endl;
}

#ifndef TO_PTX
/*!
 *  \brief gemm version 0: 1 thread for k MAC
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

const int TILE_WIDTH = 16;

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
#endif // TO_PTX

/*!  \note 99KB per CTA, 100KB per SM */
constexpr int M_TILE_SIZE = 128;
constexpr int N_TILE_SIZE = 128;
constexpr int K_TILE_SIZE = 8;

constexpr int M_WARP_SIZE = 4;
constexpr int N_WARP_SIZE = 2;

constexpr int M_THREAD = 4;
constexpr int N_THREAD = 8;

#ifndef BUG
__device__ __forceinline__ void lds128(
  float& reg0, float& reg1, float& reg2, float& reg3,
  const float* addr
) {
  asm volatile (
    "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
    : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
    : "l"(addr)
  );
}
#endif

__device__ __forceinline__ void stg128(
  float& reg0, float& reg1, float& reg2, float& reg3,
  float* addr
) {
  asm volatile (
    "st.global.v4.f32 [%4], {%0, %1, %2, %3};\n"
    :
    : "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3), "l"(addr)
  );
}
__device__ __forceinline__ void null_asm(int& reg) {
  asm volatile(
    "and.b32 %0 %0 %0;\n"
    : "=r"(reg)
    : "r"(reg)
  );
}

/*! \brief a block compute 128 * 128 tile */
__global__ void gemm2_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {

  __shared__ __align__(M_TILE_SIZE * K_TILE_SIZE * 4) float Mtile[M_TILE_SIZE * K_TILE_SIZE];
  __shared__ __align__(N_TILE_SIZE * K_TILE_SIZE * 4) float Ntile[K_TILE_SIZE * N_TILE_SIZE];

  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t warp_id = threadIdx.x / 32;

  float A_frag[8];
  float B_frag[8];
  float C_frag[8][8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
          C_frag[i][j] = 0.;
      }
  }

  /*! \note A_ldg_ptr: start of M tile */
  const float* A_ldg_ptr = (const float*)(
    //* version = 2
    // A + (blockIdx.y * M_TILE_SIZE + threadIdx.x / K_TILE_SIZE * 4) * K + threadIdx.x % K_TILE_SIZE
    A + (blockIdx.y * M_TILE_SIZE + threadIdx.x / 2) * K + threadIdx.x % 2 * 4
  );
  /*! \note B_ldg_ptr: start of N tile */
  const float* B_ldg_ptr = (const float*)(
    // B + (threadIdx.x / 32) * N + blockIdx.x * N_TILE_SIZE + lane_id
    //* ldgsts 失败的版本 Version = 0
    B + (threadIdx.x / 32) * N + blockIdx.x * N_TILE_SIZE + lane_id * 4
  );  

  float* C_stg_ptr = (C 
    + (blockIdx.y * M_TILE_SIZE // y-dim start of block
      + warp_id / N_WARP_SIZE * M_THREAD * 8 // y-dim start of warp
      + lane_id / N_THREAD * 4) * N // y-dim start of thread
    + (blockIdx.x * N_TILE_SIZE // x-dim start of block 
      + warp_id % N_WARP_SIZE * N_THREAD * 8 // x-dim start of warp
      + lane_id % N_THREAD * 4) // x-dim start of thread
  );

  /*! \note compute Mtile by Ntile by Ktile GEMM */
  for (int p = 0; p < K / K_TILE_SIZE; ++p) {

    //* Version = 2
    // float* A_sts_ptr = Mtile + (threadIdx.x % 8) * M_TILE_SIZE + (threadIdx.x / 8) * 4;
    float* A_sts_ptr = Mtile + (threadIdx.x % 2) * 4 * M_TILE_SIZE + threadIdx.x / 2;
    //* ldgsts 失败的版本 Version = 0
    // float* B_sts_ptr = Ntile + (threadIdx.x / 32) * N_TILE_SIZE + lane_id;
    float* B_sts_ptr = Ntile + (threadIdx.x / 32) * N_TILE_SIZE + lane_id * 4;
    
    /*! \todo Zig-zag to avoid bank conflict */
    const float* A_lds_ptr = (const float*)(Mtile
      + (warp_id / N_WARP_SIZE) * M_THREAD * 8 // start of m_frag
      + (lane_id / N_THREAD) * 4
    );
    /*! \todo Zig-zag to avoid bank conflict */
    const float* B_lds_ptr = (const float*)(Ntile 
      + (warp_id % N_WARP_SIZE) * N_THREAD * 8 
      + (lane_id % N_THREAD) * 4
    );    

    {
      /*! \note load data into shared memory */

      // load A to Mtile
      /*! \todo does float4 matters*/
      float4 tmp4;
      //TODO 看看编译器又没有死码消除
      float4* sts_tmp4_ptr = (float4*)A_sts_ptr;
      {
        //* version = 2
        // tmp4.x = A_ldg_ptr[0];
        // tmp4.y = A_ldg_ptr[K];
        // tmp4.z = A_ldg_ptr[K * 2];
        // tmp4.w = A_ldg_ptr[K * 3];        
        // sts_tmp4_ptr[0] = tmp4;
      }
      tmp4 = ((const float4*)A_ldg_ptr)[0];
      A_sts_ptr[0] = tmp4.x;
      A_sts_ptr[M_TILE_SIZE] = tmp4.y;
      A_sts_ptr[M_TILE_SIZE * 2] = tmp4.z;
      A_sts_ptr[M_TILE_SIZE * 3] = tmp4.w;
      // load B to Mtile
      /*! \todo does comppiler use ldgsts?
       *  No. checked PTX with .target sm_89
       */
      //TODO 可以更改访存的方法
      {
        //* ldgsts 失败的方法
        //* version = 0
        // B_sts_ptr[0]  = B_ldg_ptr[0];
        // B_sts_ptr[32] = B_ldg_ptr[32];
        // B_sts_ptr[64] = B_ldg_ptr[64];
        // B_sts_ptr[96] = B_ldg_ptr[96];  
        //* 这样优化确实有效果, 可能是128-Byte   
      }
      sts_tmp4_ptr = (float4*)B_sts_ptr;
      {
        //* version = 1
        //* useful1.ncu 里的高Cache-hit rate是因为分四次读了这个cacheline
        /*! \todo 这样做并不能触发coalescing */
        // tmp4.x = B_ldg_ptr[0];
        // tmp4.y = B_ldg_ptr[1];
        // tmp4.z = B_ldg_ptr[2];
        // tmp4.w = B_ldg_ptr[3];
      }
      tmp4 = ((const float4*)B_ldg_ptr)[0];
      sts_tmp4_ptr[0] = tmp4;
      __syncthreads();
    }

    #ifdef DEBUG_M_TILE
    {
      if (D != nullptr) {
        for (int i = 0; i < 4; ++i)
          D[blockDim.x * i + threadIdx.x] = Mtile[blockDim.x * i + threadIdx.x];
          __syncthreads();
      }
    }
    #endif

    /*! \note Warp tile */
    for (int k = 0; k < K_TILE_SIZE; ++k) {

    #ifdef DEBUG_SHARED_BOUND
      __syncthreads();
      printf("%llu:%d\n", threadIdx.x, (A_lds_ptr - Mtile));
      printf("%llu:%d\n", threadIdx.x, (B_lds_ptr - Ntile));
      if (threadIdx.x == 0) printf("--------------------------------------\n");
      __syncthreads();
    #endif

      // load shmem to reg
      {
        /*! \note checked PTX, no need to manually use asm */
        float4 tmp4;
        tmp4 = ((float4*)(A_lds_ptr))[0];
        A_frag[0] = tmp4.x;
        A_frag[1] = tmp4.y;
        A_frag[2] = tmp4.z;
        A_frag[3] = tmp4.w;
        tmp4 = ((float4*)(B_lds_ptr))[0];
        B_frag[0] = tmp4.x;
        B_frag[1] = tmp4.y;
        B_frag[2] = tmp4.z;
        B_frag[3] = tmp4.w;
      }

      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
          C_frag[m][n] += A_frag[m] * B_frag[n];
        }
      } 

      {
        float4 tmp4;
        tmp4 = ((float4*)(B_lds_ptr + N_THREAD * 4))[0];
        B_frag[4] = tmp4.x;
        B_frag[5] = tmp4.y;
        B_frag[6] = tmp4.z;
        B_frag[7] = tmp4.w;        
      }

      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int n = 4; n < 8; ++n) {
          C_frag[m][n] += A_frag[m] * B_frag[n];
        }
      } 

      {
        float4 tmp4;
        tmp4 = ((float4*)(A_lds_ptr + M_THREAD * 4))[0];
        A_frag[4] = tmp4.x;
        A_frag[5] = tmp4.y;
        A_frag[6] = tmp4.z;
        A_frag[7] = tmp4.w;
      }

      A_lds_ptr += M_TILE_SIZE;
      B_lds_ptr += N_TILE_SIZE;

      // compute
      #pragma unroll
      for (int m = 4; m < 8; ++m) {
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          C_frag[m][n] += A_frag[m] * B_frag[n];
        }
      }
    }

    A_ldg_ptr += K_TILE_SIZE;
    B_ldg_ptr += K_TILE_SIZE * N;
    __syncthreads();
  }

  /*! \note Write Back */
  /*! \todo Naive implementation: reg to global memory*/
  /*! \todo Write to shared memory first, write to global memory second */
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    stg128(
      C_frag[m][0],
      C_frag[m][1],
      C_frag[m][2],
      C_frag[m][3],
      C_stg_ptr
    );
    stg128(
      C_frag[m][4],
      C_frag[m][5],
      C_frag[m][6],
      C_frag[m][7],
      C_stg_ptr + 32
    );    
    /*! \note next line of output matrix */
    C_stg_ptr += N;
  }

  C_stg_ptr += (M_THREAD - 1) * 4 * N;

  #pragma unroll
  for (int m = 4; m < 8; ++m) {
    stg128(
      C_frag[m][0],
      C_frag[m][1],
      C_frag[m][2],
      C_frag[m][3],
      C_stg_ptr
    );  
    stg128(
      C_frag[m][4],
      C_frag[m][5],
      C_frag[m][6],
      C_frag[m][7],
      C_stg_ptr + 32
    );      
    /*! \note next line of output matrix */  
    C_stg_ptr += N;
  }
}

#ifndef TO_PTX
float* gemm0() {
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  fill_mat_1(A, M, K);
  fill_mat_1(B, K, N);

  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * K * sizeof(float));
  cudaMalloc((void **)&cuB, K * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(32, 32);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  gemm0_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuB, cuC, M, N, K);
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

float* gemm1() {
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  fill_mat_1(A, M, K);
  fill_mat_1(B, K, N);

  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * K * sizeof(float));
  cudaMalloc((void **)&cuB, K * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  gemm1_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuB, cuC, M, N, K);
  cudaEventRecord(end, stream);

  cu_err = cudaEventSynchronize(end);
  cudaEventElapsedTime(&t1, begin, end);

  cudaMemcpy(C, cuC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(cuA);
  cudaFree(cuB);
  cudaFree(cuC);
  free(A);
  free(B);
  return C;
}
#endif // TO_PTX

float* gemm2() {
  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  #ifndef TEST
  fill_mat_rng(A, M, K);
  fill_mat_rng(B, K, N);
  #else
  fill_mat_1(A, M, K);
  fill_mat_1(B, K, N);
  #endif

  float *cuA, *cuB, *cuC;
  cudaMalloc((void **)&cuA, M * K * sizeof(float));
  cudaMalloc((void **)&cuB, K * N * sizeof(float));
  cudaMalloc((void **)&cuC, M * N * sizeof(float));

  cudaMemcpy(cuA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 gridDim(N / N_TILE_SIZE, M / M_TILE_SIZE);
  dim3 blockDim(256);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, stream);
  gemm2_kernel<<<gridDim, blockDim, 0, stream>>>(cuA, cuB, cuC, M, N, K);
  cudaEventRecord(end, stream);

  cu_err = cudaEventSynchronize(end);
  cudaEventElapsedTime(&t2, begin, end);

  if (cu_err != cudaSuccess) {
    std::cerr << "CUDA error in kernel2"
              << ": " << cudaGetErrorString(cu_err) << std::endl;
  }

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
  constexpr size_t TFLOP = 2 * (size_t)M * (size_t)N * (size_t)K;
  std::cout << "start..." << std::endl;
  #ifndef TO_PTX
  float *mat0 = gemm0();
  // check_mat(mat0, M, N, K);
  free(mat0);
  std::cout << "kernel0: " << t0 << "ms  TFLOPs: "
            << TFLOP / t0 / 1e9
            << std::endl;  
  float *mat1 = gemm1();
  // check_mat(mat1, M, N, K);
  free(mat1);
  std::cout << "kernel1: " << t1 << "ms  TFLOPs: "
            << TFLOP / t1 / 1e9
            << std::endl; 
  #endif // TO_PTX
  float* mat2 = gemm2();
  #ifdef TEST
  check_mat(mat2, M, N, K);
  #endif
  free(mat2);
  std::cout << "kernel2: " << t2 << "ms  TFLOPs: "
            << TFLOP / t2 / 1e9
            << std::endl; 
  return 0;
}