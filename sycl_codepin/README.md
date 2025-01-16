# 使い方
dpctは、CUDAコードをSYCLコードに変換するためのツールです。--enable-codepinオプションを使用すると、oneAPIのdpctなどを利用してSYCLコードを生成することができます。以下に、CUDAコードをSYCLコードに変換するサンプルを示します。
  
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
```dpct --enable-codepin sycl_codepin.cu```
変換後のSYCLコードは以下のようになります。  
```
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <dpct/codepin/codepin.hpp>
#include "codepin_autogen_util.hpp"
#include <iostream>

void multiply_by_two(int* data, int N, const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < N) {
        data[idx] *= 2;
    }
}

int main() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    const int N = 16;
    int data[N];
    for (int i = 0; i < N; i++) data[i] = i;

    int* d_data;
    d_data = sycl::malloc_device<int>(N, q_ct1);
dpctexp::codepin::get_ptr_size_map()[d_data] = N * sizeof(int);
    q_ct1.memcpy(d_data, data, N * sizeof(int));

    dpctexp::codepin::gen_prolog_API_CP(
        "multiply_by_two:C:/Users/kminemur/source/sycl_101/sycl_codepin/"
        "sycl_codepin.cu:20:5",
        &q_ct1, "d_data", d_data, "N", N);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, N), sycl::range<3>(1, 1, N)),
        [=](sycl::nd_item<3> item_ct1) {
            multiply_by_two(d_data, N, item_ct1);
        });

    dpctexp::codepin::gen_epilog_API_CP(
        "multiply_by_two:C:/Users/kminemur/source/sycl_101/sycl_codepin/"
        "sycl_codepin.cu:20:5",
        &q_ct1, "d_data", d_data, "N", N);
q_ct1.memcpy(data, d_data, N * sizeof(int)).wait();
    dpct::dpct_free(d_data, q_ct1);

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
ビルドし、生成されたバイナリを実行すると、実行ログファイルが生成されます


