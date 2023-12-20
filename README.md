# EST6703-CUDA-lab

```
张鼎言 023037910010
袁钰淇 
```

## GEMM1

### 测试
- 4090平台: 82.58 TFLOPs fp32
    - gemm2: 62% fp32 peak performance
- 3090平台: 35.58 TFLOPs fp32
    - gemm2: 62% fp32 peak performance

```Bash
# 拉下来仓库以后, 修改 Makefile 中 sm89 为 sm86
# 使用 ncu 进行性能测试
chmod a+x nncu.sh
make ncu
# 使用 Nsight Compute 软件打开 ncu-rep 文件
```

### 修改
1. Version 0 => Version 1
    修改了 B_ldg 的方法, 提升了10%
2. Version 1 => Version 2
    修改了 B_sts 的方法, 提升了4%
3. Version 2 => Version 3
    修改了 A_ldg 的方法, 提升了3%, 避免了STS的 Bank conflict
4. Version 3 => Version 4
    将FMA和FMA之前的 LDS 交错处理

## Add Bias

Memory bound. 80%+

## Row Reduce Sub Mean



## Kernel Fusion

## GEMM FP16 Tensor Core

## SoftMax FP32

## LayerNorm
