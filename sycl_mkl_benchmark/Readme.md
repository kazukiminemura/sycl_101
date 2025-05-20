# Usage
```
export KMP_AFFINITY=granularity=fine,compact,1,0
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
icpx -fsycl sycl_mkl_bf16_benchmark.cpp -qmkl; ./a.out 10000 10000 10000
```
