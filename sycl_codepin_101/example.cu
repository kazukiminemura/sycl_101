//example.cu
#include <iostream>
__global__ void vectorAdd(int3 *a, int3 *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    result[tid].x = a[tid].x + 1;
    result[tid].y = a[tid].y + 1;
    result[tid].z = a[tid].z + 1;
}

int main() {
    const int vectorSize = 4;
    int3 h_a[vectorSize], h_result[vectorSize];
    int3 *d_a, *d_result;
    for (int i = 0; i < vectorSize; ++i)
        h_a[i] = make_int3(1, 2, 3);

    cudaMalloc((void **)&d_a, vectorSize * sizeof(int3));
    cudaMalloc((void **)&d_result, vectorSize * sizeof(int3));

    // Copy host vectors to device
    // !! Using 12 instead of "sizeof(int3)"
    cudaMemcpy(d_a, h_a, vectorSize * 12, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    vectorAdd<<<1, 4>>>(d_a, d_result);

    // Copy result from device to host
    cudaMemcpy(h_result, d_result, vectorSize * sizeof(int3),
        cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < vectorSize; ++i) {
        std::cout << "Result[" << i << "]: (" << h_result[i].x << ", "
                << h_result[i].y << ", " << h_result[i].z << ")\n";
    }
}

/*
Execution Result:
Result[0]: (2, 3, 4)
Result[1]: (2, 3, 4)
Result[2]: (2, 3, 4)
Result[3]: (2, 3, 4)
*/
