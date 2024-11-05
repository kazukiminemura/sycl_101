#include <sycl/sycl.hpp>
#include <syclcompat.hpp>
#include <iostream>

namespace sc = syclcompat;

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);

    // Host memory allocation
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Device memory allocation using syclcompat
    float *d_a, *d_b, *d_c;
    sc::malloc((void **)&d_a, bytes);
    sc::malloc((void **)&d_b, bytes);
    sc::malloc((void **)&d_c, bytes);

    // Copy data from host to device
    sc::memcpy(d_a, h_a, bytes);
    sc::memcpy(d_b, h_b, bytes);

    // Kernel execution configuration
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch kernel using syclcompat
    sc::launch_kernel(vector_add, blocks_per_grid, threads_per_block, d_a, d_b, d_c, N);

    // Copy result back to host
    sc::memcpy(h_c, d_c, bytes);

    // Wait for device to finish all operations
    sc::device_synchronize();

    // Check results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector addition was successful!" << std::endl;
    } else {
        std::cout << "Vector addition failed." << std::endl;
    }

    // Free device and host memory
    sc::free(d_a);
    sc::free(d_b);
    sc::free(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

