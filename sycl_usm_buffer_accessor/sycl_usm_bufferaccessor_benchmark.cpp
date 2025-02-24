#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace sycl;
constexpr size_t N = 1000000;

void usm_example(queue &q, float *a, float *b, float *c) {
    q.submit([&](handler &h) {
        h.parallel_for(range<1>(N), [=](id<1> i) {
            c[i] = a[i] + b[i];
        });
    }).wait();
}

void buffer_example(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, buffer<float, 1> &c) {
    q.submit([&](handler &h) {
        auto a_acc = a.get_access<access::mode::read>(h);
        auto b_acc = b.get_access<access::mode::read>(h);
        auto c_acc = c.get_access<access::mode::write>(h);
        h.parallel_for(range<1>(N), [=](id<1> i) {
            c_acc[i] = a_acc[i] + b_acc[i];
        });
    }).wait();
}

int main() {
    queue q;
    std::cout << "Selected device: " << q.get_device().get_info<info::device::name>() << std::endl;

    float *usm_a = malloc_shared<float>(N, q);
    float *usm_b = malloc_shared<float>(N, q);
    float *usm_c = malloc_shared<float>(N, q);

    for (size_t i = 0; i < N; i++) {
        usm_a[i] = i;
        usm_b[i] = i * 2;
    }

    auto start = std::chrono::high_resolution_clock::now();
    usm_example(q, usm_a, usm_b, usm_c);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> usm_duration = end - start;
    std::cout << "USM duration: " << usm_duration.count() << " seconds" << std::endl;

    std::array<float, N> array_a, array_b;
    for (size_t i = 0; i < N; i++){
	    array_a[i] = i;
	    array_b[i] = i * 2;
    }

    buffer<float, 1> buf_a(array_a);
    buffer<float, 1> buf_b(array_b);
    buffer<float, 1> buf_c{range<1>(N)};

    start = std::chrono::high_resolution_clock::now();
    buffer_example(q, buf_a, buf_b, buf_c);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buffer_duration = end - start;
    std::cout << "Buffer accessor duration: " << buffer_duration.count() << " seconds" << std::endl;

    free(usm_a, q);
    free(usm_b, q);
    free(usm_c, q);

    return 0;
}
