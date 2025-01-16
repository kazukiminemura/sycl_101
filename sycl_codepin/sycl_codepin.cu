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
