#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;

int main() {
    queue q;
    constexpr int GLOBAL = 64;
    constexpr int LOCAL = 64;
    int *data = sycl::malloc_shared<int>(GLOBAL, q);

    std::cout << "Local Memory Size for each work-group: "
            << q.get_device().get_info<sycl::info::device::local_mem_size>()
            << " Bytes" << std::endl;


    for (int bank_count = 0; bank_count <= 64; bank_count++) {
        auto start = std::chrono::high_resolution_clock::now();

        q.submit([&](handler &h) {
            sycl::local_accessor<int, 1> slm(sycl::range(64 * 64), h);

            h.parallel_for(sycl::nd_range(sycl::range{GLOBAL}, sycl::range{LOCAL}),
                   [=](sycl::nd_item<1> it) {
                     int i = it.get_global_linear_id();
                     int j = it.get_local_linear_id();


                     slm[j*bank_count] = data[i];
                     it.barrier(sycl::access::fence_space::local_space);

                     for (int m = 0; m < 1024 * 1024; m++) {
                       slm[j*bank_count] += i * m;
                       it.barrier(sycl::access::fence_space::local_space);
                     }


                     data[i] += slm[j*bank_count];

            });


            /*
            // no-conflicts
            h.parallel_for(sycl::nd_range(sycl::range{N}, sycl::range{32}),
                   [=](sycl::nd_item<1> it) {
                     int i = it.get_global_linear_id();
                     int j = it.get_local_linear_id();

                     slm[j] = 0;
                     it.barrier(sycl::access::fence_space::local_space);

                     for (int m = 0; m < 1024 * 1024; m++) {
                       slm[j] += i * m;
                       it.barrier(sycl::access::fence_space::local_space);
                     }

                     data[i] = slm[j];
           });
           */


        }).wait();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Banks: " << bank_count << " | Execution Time: " << duration.count() << " ms" << std::endl;
    }

    return 0;
}
