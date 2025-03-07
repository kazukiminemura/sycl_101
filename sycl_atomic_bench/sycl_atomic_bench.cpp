#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;
constexpr size_t N = 1024;

int main(){
    queue q;
    std::cout << q.get_device().get_info<info::device::name>() << std::endl;

    // normal kernel
    {
        auto start = std::chrono::high_resolution_clock::now();
        buffer<int> buf(N);
        int M = 16;
        q.submit([&](handler& h){
            accessor acc{buf, h};
            h.parallel_for(range(N), [=](id<1> i){
                int j = i % M;
                acc[j] = j;
            });
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Normal kernel execution time: " << diff.count() << " seconds" << std::endl;
    }

    // kernel with atomic_ref and buffer
    {
        auto start = std::chrono::high_resolution_clock::now();
        buffer<int> buf(N);
        int M = 16;
        q.submit([&](handler& h){
            accessor acc{buf, h};
            h.parallel_for(range(N), [=](id<1> i){
                int j = i % M;
                atomic_ref<int, memory_order::relaxed, memory_scope::system, 
                    access::address_space::global_space> atomic_acc(acc[j]);
                atomic_acc += 1;
            });
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Atomic kernel execution with buffer time: " << diff.count() << " seconds" << std::endl;
    }

    // kernel with atomic_ref and USM
    {
        auto start = std::chrono::high_resolution_clock::now();
        int* data = malloc_shared<int>(N, q);
        int M = 16;
        q.submit([&](handler& h){
            h.parallel_for(range(N), [=](id<1> i){
                int j = i % M;
                atomic_ref<int, memory_order::relaxed, memory_scope::system, 
                    access::address_space::global_space> atomic_acc(data[j]);
                atomic_acc += 1;
            });
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Atomic kernel execution with USM time: " << diff.count() << " seconds" << std::endl;
    }



    return 0;
}