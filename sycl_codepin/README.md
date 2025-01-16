![image](https://github.com/user-attachments/assets/8adfb1bc-85b5-4e80-aabf-64b9243d15ae)dpctは、CUDAコードをSYCLコードに変換するためのツールです。--enable-codepinオプションを使用すると、CodeplayのCodeplay ComputeCppを利用してSYCLコードを生成することができます。以下に、CUDAコードをSYCLコードに変換するサンプルを示します。
  
まず、CUDAコードの例を示します。このコードは、配列の各要素を2倍にするものです。  
```
#include <cuda_runtime.h>
#include <iostream>

__global__ void multiply_by_two(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2;
    }
}

int main() {
    const int N = 16;
    int data[N];
    for (int i = 0; i < N; i++) data[i] = i;

    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

    multiply_by_two<<<1, N>>>(d_data, N);

    cudaMemcpy(data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    for (int i = 0; i < N; i++) std::cout << data[i] << " ";
    std::cout << "\n";

    return 0;
}
```
次に、このCUDAコードをSYCLコードに変換するためにdpctを使用します。以下のコマンドを実行します。  
```dpct --enable-codepin cuda_example.cu```
変換後のSYCLコードは以下のようになります。  
```
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

void multiply_by_two(sycl::queue &q, int* data, int N) {
    q.parallel_for(sycl::range<1>(N), = {
        if (idx[0] < N) {
            data[idx[0]] *= 2;
        }
    });
}

int main() {
    sycl::queue q;
    const int N = 16;
    int data[N];
    for (int i = 0; i < N; i++) data[i] = i;

    int* d_data = sycl::malloc_device<int>(N, q);
    q.memcpy(d_data, data, N * sizeof(int)).wait();

    multiply_by_two(q, d_data, N);

    q.memcpy(data, d_data, N * sizeof(int)).wait();
    sycl::free(d_data, q);

    for (int i = 0; i < N; i++) std::cout << data[i] << " ";
    std::cout << "\n";

    return 0;
}
```

dpct_output_codepin_cuda/sycl_codepin.cuとdpct_output_codepin_sycl/sycl_codepin.dp.cpp と  をそれぞれ
```
nvcc test.cu
icpx -fsycl sycl_codepin.dp.cpp
```
ビルドし、生成されたバイナリを実行すると、次の実行ログファイルが生成されます

| [
  {
    "CodePin Random Seed": "0",
    "CodePin Sampling Threshold": "20",
    "CodePin Sampling Percent": "1"
  },
  {
    "ID": "multiply_by_two:/nfs/site/home/kminemur/projects/codepin/test.cu:20:5:prolog",
    "Device Name": "NVIDIA A100 80GB PCIe",
    "Device ID": "0",
    "Stream Address": "0",
    "Free Device Memory": "84528594944",
    "Total Device Memory": "84974239744",
    "Elapse Time(ms)": "0",
    "CheckPoint": {
      "d_data": {
        "Type": "Pointer",
        "Address": "0x6e217200000",
        "Index": "0",
        "Data": [
          {
            "Type": "int",
            "Data": [
              0
            ]
          }
        ]
      },
      "N": {
        "Type": "int",
        "Address": "0x7ffedd23a2bc",
        "Index": "1",
        "Data": [
          16
        ]
      }
    }
  },
  {
    "ID": "multiply_by_two:/nfs/site/home/kminemur/projects/codepin/test.cu:20:5:epilog",
    "Device Name": "NVIDIA A100 80GB PCIe",
    "Device ID": "0",
    "Stream Address": "0",
    "Free Device Memory": "84528594944",
    "Total Device Memory": "84974239744",
    "Elapse Time(ms)": "131.161",
    "CheckPoint": {
      "d_data": {
        "Type": "Pointer",
        "Address": "0x6e217200000",
        "Index": "0",
        "Data": [
          {
            "Type": "int",
            "Data": [
              0
            ]
          }
        ]
      },
      "N": {
        "Type": "int",
        "Address": "0x7ffedd23a2ac",
        "Index": "1",
        "Data": [
          16
        ]
      }
    }
  }
] | TD3 |
