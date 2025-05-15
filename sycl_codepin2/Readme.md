![image](https://github.com/user-attachments/assets/44f6fa05-4a55-4683-9d2f-23524fe1e790)![image](https://github.com/user-attachments/assets/da923d00-08ae-4b6c-98e9-2c7d8c7469db)# 使い方
```
dpct example.cu --enable-codepin
```

dpct_output_codepin_sycl_codepin_sycl  
dpct_output_codepin_sycl_codepin_cuda  
が生成される

# SYCL code
```
cd dpct_output_codepin_sycl_codepin_sycl
icpx -fsycl example.dp.cpp

./a.out
get_memory_info: ext_intel_free_memory is not supported.
CodePin data sampling is enabled for data dump. As follow list 3 configs for data sampling:
CODEPIN_RAND_SEED: 0
CODEPIN_SAMPLING_THRESHOLD: 20
CODEPIN_SAMPLING_PERCENT: 1
get_memory_info: ext_intel_free_memory is not supported.
Result[0]: (2, 3, 4)
Result[1]: (2, 3, 4)
Result[2]: (2, 3, 4)
Result[3]: (1, 1, 1) <-- wrong![image](https://github.com/user-attachments/assets/96fc441a-3718-451c-a220-8fb734292d02)
```

# CUDA code
```
cd dpct_output_codepin_sycl_codepin_cuda
nvcc -std=c++17 example.cu

a.exe
Result[0]: (2, 3, 4)
Result[1]: (2, 3, 4)
Result[2]: (2, 3, 4)
Result[3]: (2, 3, 4)
```

# Codepin
```
python "C:\Program Files (x86)\Intel\oneAPI\dpcpp-ct\latest\bin\codepin-report.py"  --instrumented-cuda-log <path/to/CodePin_CUDA.json> --instrumented-sycl-log <path/to/CodePin_SYCL.json>


CodePin Summary
Totally APIs count, 2
Consistently APIs count, 0
Most Time-consuming Kernel(CUDA), vectorAdd:C:/Users/MTL/source/codepin/example.cu:25:5:epilog, time:0.827392
Most Time-consuming Kernel(SYCL), vectorAdd:C:/Users/MTL/source/codepin/example.cu:25:5:prolog, time:0.0
Peak Device Memory Used(CUDA), 1143472128
Peak Device Memory Used(SYCL), 17662062592
CUDA Meta Data ID, SYCL Meta Data ID, Type, Detail
vectorAdd:C:/Users/MTL/source/codepin/example.cu:25:5:epilog,vectorAdd:C:/Users/MTL/source/codepin/example.cu:25:5:epilog,Data value,The location of failed ID Errors occurred during comparison: d_a->"Data"->[3]->"Data"->[0]->"x"->"Data"->[0] and [ERROR: DATA VALUE MISMATCH] the CUDA value 1 differs from the SYCL value 0.; d_result->"Data"->[3]->"Data"->[0]->"x"->"Data"->[0] and [ERROR: DATA VALUE MISMATCH] the CUDA value 2 differs from the SYCL value 1.
vectorAdd:C:/Users/MTL/source/codepin/example.cu:25:5:prolog,vectorAdd:C:/Users/MTL/source/codepin/example.cu:25:5:prolog,Data value,[WARNING: METADATA MISMATCH] The pair of prolog data vectorAdd:C:/Users/MTL/source/codepin/example.cu:25:5:prolog are mismatched, and the corresponding pair of epilog data matches. This mismatch may be caused by the initialized memory or argument used in the API vectorAdd.![Uploading image.png…]()

```
