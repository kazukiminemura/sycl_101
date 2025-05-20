# mkl_gemm_bf16bf16f32_benchmark
```
export KMP_AFFINITY=granularity=fine,compact,1,0
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
icpx -fsycl mkl_gemm_bf16bf16f32_benchmark.cpp -qmkl; ./a.out 10000 10000 10000
```

# mkl_gemm_bf16bf16f32_benchmark
```
export KMP_AFFINITY=granularity=fine,compact,1,0
icpx -fsycl mkl_bf16bf16f32_benchmark.cpp -qmkl; ./a.out 10000 10000 10000
```
