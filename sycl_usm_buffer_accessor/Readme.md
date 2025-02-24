# Usage
icpx -fsycl -fsycl-targets=spir64,nvptx64-nvidia-
cuda sycl_usm_bufferaccessor_benchmark.cpp

## Results
```
ONEAPI_DEVICE_SELECTOR=cuda:gpu ./a.out
Selected device: NVIDIA GeForce RTX 3060 Ti
USM duration: 0.00202008 seconds
Buffer accessor duration: 0.00325168 seconds

ONEAPI_DEVICE_SELECTOR=opencl:cpu ./a.out
Selected device: 12th Gen Intel(R) Core(TM) i5-12400
USM duration: 0.0926782 seconds
Buffer accessor duration: 0.00160408 seconds
```
