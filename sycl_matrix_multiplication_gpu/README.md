# 実行手順
## コンパイル
```
icpx -fsycl -O2 -o sycl_matrix_multiply_gpu sycl_matrix_multiply_gpu.cpp
```

## 実行
``` ./matrix_multiply ```

# 出力例
```
M: 1024, N: 1024, K: 1024
sequential matrix multiplication: 2.50958 seconds
Running on: NVIDIA GeForce RTX 3060 Ti
somewhat parallel matrix multiplication: 0.746615 seconds
Even more parallel matrix multiplication: 0.0056592 seconds
single workitem per workgroup parallel matrix multiplication: 0.0835435 seconds
column parallel matrix multiplication: 0.0563295 seconds
Tiled matrix mulpilication: 0.0629744 seconds
```

