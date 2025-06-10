# Usage
問題をデバッグするために、CodePin を有効にした状態で CUDA プログラムを移行します:  
```dpct example.cu --enable-codepin```


移行後、次の 2 つのファイルが生成されます:  
```
dpct_output_codepin_sycl/example.dp.cpp
dpct_output_codepin_cuda/example.cu
```

## ワークスペース構成:
workspace  
├── example.cu  
├── dpct_output_codepin_sycl  
│   ├── example.dp.cpp  
│   ├── generated_schema.hpp  
│   └── MainSourceFiles.yaml  
├── dpct_output_codepin_cuda  
│   ├── example.cu  
│   └── generated_schema.hpp  
    

# Log file will be generation
## CUDA
```
nvcc -std=c++17 example.cu
a.exe

CodePin data sampling is enabled for data dump. As follow list 3 configs for data sampling:
CODEPIN_RAND_SEED: 0
CODEPIN_SAMPLING_THRESHOLD: 20
CODEPIN_SAMPLING_PERCENT: 1
Result[0]: (2, 3, 4)
Result[1]: (2, 3, 4)
Result[2]: (2, 3, 4)
Result[3]: (2, 3, 4)
```

## SYCL
```
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -std=c++17 example.dp.cpp
a.exe

get_memory_info: ext_intel_free_memory is not supported.
CodePin data sampling is enabled for data dump. As follow list 3 configs for data sampling:
CODEPIN_RAND_SEED: 0
CODEPIN_SAMPLING_THRESHOLD: 20
CODEPIN_SAMPLING_PERCENT: 1
get_memory_info: ext_intel_free_memory is not supported.
Result[0]: (2, 3, 4)
Result[1]: (2, 3, 4)
Result[2]: (2, 3, 4)
Result[3]: (1, 1, 1)
```

# CodePin レポートの生成:
```codepin-report.py [-h] --instrumented-cuda-log <file path> --instrumented-sycl-log <file path>```    

```
python3 "C:\Program Files (x86)\Intel\oneAPI\2025.2\bin\codepin-report.py" --instrumented-cuda-log dpct_output_codepin_sycl_codepin_cuda\CodePin_CUDA_2025-06-10_10-31-38.json --instrumented-sycl-log dpct_output_codepin_sycl_codepin_sycl\CodePin_SYCL_2025-06-10_10-31-03.json --generate-data-flow-graph
Finished comparison of the two files and found differences. Please check 'CodePin_Report.csv' file located in your project directory.
```
このスクリプトを実行すると、CSV ファイルが生成されます。

# レポート
The location of failed ID Errors occurred during comparison:    
d_a->"Data"->[3]->"Data"->[0]->"x"->"Data"->[0] and [ERROR: DATA VALUE MISMATCH] the CUDA value 1 differs from the SYCL value 0.;    

d_result->"Data"->[3]->"Data"->[0]->"x"->"Data"->[0] and [ERROR: DATA VALUE MISMATCH] the CUDA value 2 differs from the SYCL value 1.    

