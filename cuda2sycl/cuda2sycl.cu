#include <cuda.h>
#include <iostream>

// ベクトル加算のCUDAカーネル
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA devices available." << std::endl;
        return 0;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
    }

    int N = 1024;
    size_t bytes = N * sizeof(float);

    // ホストメモリの割り当て
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // データの初期化
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // デバイスメモリの割り当て
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // データをデバイスに転送
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // カーネルの起動
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 結果をホストにコピー
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 結果の検証
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("エラー: インデックス %d\n", i);
            break;
        }
    }
    printf("計算完了\n");

    // メモリの解放
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
