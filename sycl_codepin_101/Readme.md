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
```

## SYCL
```
icpx -fsycl -std=c++17 example.dp.cpp
a.exe
```

# CodePin レポートの生成:
```codepin-report.py [-h] --instrumented-cuda-log <file path> --instrumented-sycl-log <file path>```    
このスクリプトを実行すると、CSV ファイルが生成されます。
