#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

constexpr int N = 42;

int main(){
    // in-order kernel
    {
    auto start = std::chrono::high_resolution_clock::now();
    queue Q{property::queue::in_order()};
    int *data = malloc_shared<int>(N, Q);
    std::cout << Q.get_device().get_info<info::device::name>() << std::endl;
    Q.parallel_for(N, [=](id<1> i){ data[i] = 1; });
    Q.single_task([=](){ 
        for (int i = 1; i < N; ++i)
        data[0] += data[i]; 
    });

    Q.wait();
    assert(data[0] == N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "In-order kernel: " << duration.count() << " seconds" << std::endl;
    }

    // out-of-order, event-based kernel
    {
    auto start = std::chrono::high_resolution_clock::now();
    queue Q;
    int *data = malloc_shared<int>(N, Q);
    auto e = Q.parallel_for(N, [=](id<1> i){ data[i] = 1; });
    Q.submit([&](handler &h){
        h.depends_on(e);
        h.single_task([=](){ 
            for (int i = 1; i < N; ++i)
            data[0] += data[i]; 
        });
    });

    Q.wait();
    assert(data[0] == N);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "out-of-order, event-based kernel: " << duration.count() << " seconds" << std::endl;
    }

    // buffer accessor
    {
    auto start = std::chrono::high_resolution_clock::now();
    queue Q;
    buffer<int> data{range(N)};
    
    Q.submit([&](handler &h){
        accessor a{data, h};
        h.parallel_for(N, [=](id<1> i){ a[i] = 1; });
    });

    Q.submit([&](handler &h){
        accessor a{data, h};
        h.single_task([=](){
            for (int i = 1; i < N; ++i)
            a[0] += a[i];
        });
    });

    host_accessor h_a{data};
    assert(h_a[0] == N);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "buffer-accessor kernel: " << duration.count() << " seconds" << std::endl;
    }

}