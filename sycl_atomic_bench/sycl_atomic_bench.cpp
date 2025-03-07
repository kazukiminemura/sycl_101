#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;
constexpr int N = 1024;
constexpr int M = 16;

int main(){
    queue q;
    std::cout << q.get_device().get_info<info::device::name>() << std::endl;
    
    // normal kernel
    {
        auto start = std::chrono::high_resolution_clock::now();
        // buffer<int> buf{range{N}};
        buffer<int> buf(range{N});
        q.submit([&](handler& h){
            accessor acc{buf, h};
            h.parallel_for(N, [=](id<1> idx){
                int j = idx % M;
                acc[idx] = j;
            });
        });

        host_accessor host_acc{buf};
        for(int i = 0; i < N; i++){
            int j = i % M;
            // std::cout << i << "-th :" << host_acc[i] << std::endl;
            assert(host_acc[i] == j);
            // if(host_acc[j] != i){
            //     std::cout << "Error at index " << i << std::endl;
            //     break;
            // }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Normal kernel execution time: " << diff.count() << " seconds" << std::endl;
    }

    // kernel with atomic_ref and buffer
    {
        auto start = std::chrono::high_resolution_clock::now();
        buffer<int> buf(N);
        q.submit([&](handler& h){
            accessor acc{buf, h};
            h.parallel_for(range(N), [=](id<1> i){
                int j = i % M;
                atomic_ref<int, memory_order::relaxed, memory_scope::system, 
                    access::address_space::global_space> atomic_acc(acc[i]);
                atomic_acc = j;
            });
        });

        host_accessor host_acc{buf};
        for(int i = 0; i < N; i++){
            int j = i % M;
            assert(host_acc[i] == j);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Atomic kernel execution with buffer time: " << diff.count() << " seconds" << std::endl;
    }

    // kernel with atomic_ref and USM
    {
        auto start = std::chrono::high_resolution_clock::now();
        int* data = malloc_shared<int>(N, q);
        q.submit([&](handler& h){
            h.parallel_for(range(N), [=](id<1> i){
                int j = i % M;
                atomic_ref<int, memory_order::relaxed, memory_scope::system, 
                    access::address_space::global_space> atomic_acc(data[i]);
                atomic_acc = j;
            });
        });
        q.wait();

        for(int i = 0; i < N; i++){
            int j = i % M;
            assert(data[i] == j);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Atomic kernel execution with USM time: " << diff.count() << " seconds" << std::endl;

        free(data, q);
    }



    return 0;
}