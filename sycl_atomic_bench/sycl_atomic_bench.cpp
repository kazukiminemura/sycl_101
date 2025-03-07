#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;
constexpr int N = 1024;
constexpr int M = 16;

int main(){
    queue q;
    std::cout << q.get_device().get_info<info::device::name>() << std::endl;
    std::vector<int> vec_data(N);

    // normal kernel - can not behave as expected with normal kernel
    // {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     buffer<int> buf(N);
    //     q.submit([&](handler& h){
    //         accessor acc{buf, h};
    //         h.parallel_for(range(N), [=](id<1> i){
    //             acc[i] = 0;
    //         });
    //     });

    //     q.submit([&](handler& h){
    //         accessor acc{buf, h};
    //         h.parallel_for(range(N), [=](id<1> i){
    //             int j = i % M;
    //             acc[j] += 1;
    //         });
    //     });

    //     host_accessor host_acc{buf};
    //     for(int i = 0; i < M; i++){
    //         std::cout << i << "-th :" << host_acc[i] << std::endl;
    //         // assert(host_acc[i] == N/M);
    //     }
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> diff = end - start;
    //     std::cout << "Normal kernel execution time: " << diff.count() << " seconds" << std::endl;
    // }

    // kernel with atomic_ref and buffer
    {
        auto start = std::chrono::high_resolution_clock::now();
        // std::fill(vec_data.begin(), vec_data.end(), 0);
        buffer<int> buf(vec_data);
        q.submit([&](handler& h){
            accessor acc{buf, h};
            h.parallel_for(range(N), [=](id<1> i){
                acc[i] = 0;
            });
        });

        q.submit([&](handler& h){
            accessor acc{buf, h};
            stream out(1024, 1024, h);
            h.parallel_for(range(N), [=](id<1> i){
                int j = i % M;
                atomic_ref<int, memory_order::relaxed, memory_scope::system, 
                    access::address_space::global_space> atomic_acc(acc[j]);
                atomic_acc += 1;
            });
        });

        host_accessor host_acc{buf};
        for(int i = 0; i < M; i++){
            // std::cout << host_acc[i] << std::endl;
            assert(host_acc[i] == N/M);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Atomic kernel execution with buffer time: " << diff.count() << " seconds" << std::endl;
    }

    // kernel with atomic_ref and USM
    {
        auto start = std::chrono::high_resolution_clock::now();
        int* data = malloc_shared<int>(N, q);
        q.fill(data, 0, N);
        q.submit([&](handler& h){
            h.parallel_for(range(N), [=](id<1> i){
                int j = i % M;
                atomic_ref<int, memory_order::relaxed, memory_scope::system, 
                    access::address_space::global_space> atomic_acc(data[j]);
                atomic_acc += 1;
            });
        });
        q.wait();

        for(int i = 0; i < M; i++){
            // std::cout << data[i] << std::endl;
            assert(data[i] == N/M);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Atomic kernel execution with USM time: " << diff.count() << " seconds" << std::endl;

        free(data, q);
    }

    return 0;
}