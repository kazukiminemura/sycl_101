#include <sycl/sycl.hpp>
#include <syclcompat.hpp>
#include <iostream>

namespace sc = syclcompat;

// Kernel function for vector addition
void vector_add(const float *a, const float *b, float *c, int n, sycl::queue& q) {
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        });
    });
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
    float *d_a = sycl::malloc_device<float>(N, sc::get_default_queue());
    float *d_b = sycl::malloc_device<float>(N, sc::get_default_queue());
    float *d_c = sycl::malloc_device<float>(N, sc::get_default_queue());

    sycl::queue q;

    // Copy data from host to device
    q.memcpy(d_a, h_a, bytes);
    q.memcpy(d_b, h_b, bytes);

    // Launch kernel using syclcompat
    vector_add(d_a, d_b, d_c, N, q);

    // Copy result back to host
    q.memcpy(h_c, d_c, bytes);

    // Wait for device to finish all operations
    sc::get_default_queue().wait();

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
    sc::free(d_a, sc::get_default_queue());
    sc::free(d_b, sc::get_default_queue());
    sc::free(d_c, sc::get_default_queue());
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
