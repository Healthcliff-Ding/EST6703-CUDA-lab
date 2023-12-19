# EST6703-CUDA-lab

```
张鼎言 023037910010
袁钰淇 
```

## GEMM1

### 测试
- 4090平台: 82.58 TFLOPs fp32
    - gemm0: 136.685ms  TFLOPs: 5.0276
    - gemm1: 112.009ms  TFLOPs: 6.13516
    - gemm2: 22.108ms   TFLOPs: 31.0836
- 3090平台: 35.58 TFLOPs fp32

### 修改
1. Version 0 => Version 1
    修改了 B_ldg 的方法, 提升了10%
2. Version 1 => Version 2
    修改了 B_sts 的方法, 提升了4%
3. Version 2 => Version 3
    修改了 A_ldg 的方法, 提升了3%, 避免了STS的 Bank conflict
4. Version 3 => Version 4
    将FMA和FMA之前的 LDS 并行处理

## Add Bias

## Row Reduce Sub Mean

## Kernel Fusion

## GEMM FP16 Tensor Core

## SoftMax FP32

## LayerNorm