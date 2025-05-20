# Usage
```
export KMP_AFFINITY=granularity=fine,compact,1,0
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
icpx -fsycl test.cpp -qmkl; ./a.out
```
