#include <iostream>
#include <chrono>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <thread>
 
 
int main() {
    const int N = 10;
 
    std::cout << "=== OpenMP Test ===" << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();
        #pragma omp critical
        std::cout << "OpenMP thread " << tid << " processing index " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
 
 
    std::cout << "\n=== oneTBB Test ===" << std::endl;
    tbb::parallel_for(0, N, 1, [](int i) {
        std::cout << "TBB thread " << std::this_thread::get_id() << " processing index " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });
 
    return 0;
}
