#　使い方
```
OpenCL 
icpx test_opencl.cpp -lOpenCL -o test_opencl; ./test_opencl

SYCL
icpx -fsycl test_sycl.cpp -o test_sycl; ./test_sycl 

DPL
icpx -fsycl test_dpl.cpp -o test_dpl; ./test_dpl
```
