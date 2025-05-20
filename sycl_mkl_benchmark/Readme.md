# sycl_mkl_bf16gemm_benchmark
```
export KMP_AFFINITY=granularity=fine,compact,1,0
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
icpx -fsycl sycl_mkl_bf16gemm_benchmark.cpp -qmkl; ./a.out 10000 10000 10000
```

# mkl_gemm_bf16bf16f32_benchmark
```
export KMP_AFFINITY=granularity=fine,compact,1,0
icpx mkl_gemm_bf16bf16f32_benchmark.cpp -qmkl -o mkl_bf16bf16f32; ./mkl_bf16bf16f32 10000 10000 10000
```

# mkl_gemm_f16f16f32_benchmark
```
export KMP_AFFINITY=granularity=fine,compact,1,0
icpx mkl_gemm_f16f16f32_benchmark.cpp -qmkl -o mkl_f16f16f32; ./mkl_f16f16f32 10000 10000 10000
```
