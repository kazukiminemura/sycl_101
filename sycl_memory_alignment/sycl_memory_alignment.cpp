#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>

using namespace sycl;

template <typename T>
void test_alignment(queue& q, size_t N, size_t alignment) {
    std::cout << "Testing type: " << typeid(T).name() << ", alignment: " << alignment << std::endl;

    T* data = static_cast<T*>(aligned_alloc_device(alignment, N * sizeof(T), q));
    q.fill(data, T{0}, N).wait();

    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](handler &h) {
        h.parallel_for(range<1>(N), [=](id<1> i) {
            data[i] = data[i] + T{1};
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Elapsed time: " << duration.count() << " ms\n";

    free(data, q);
}

int main() {
    constexpr size_t N = 1 << 20;

    try {
        queue intel_q{gpu_selector_v};
        std::cout << "Running on: " << intel_q.get_device().get_info<info::device::name>() << "\n";

        test_alignment<float>(intel_q, N, alignof(float));
        test_alignment<float4>(intel_q, N / 4, alignof(float4));
        test_alignment<float8>(intel_q, N / 8, alignof(float8));
    } catch (exception const& e) {
        std::cerr << "Intel GPU test failed: " << e.what() << "\n";
    }

    return 0;
}
