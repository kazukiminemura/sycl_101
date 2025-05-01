#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

constexpr size_t N = 1024 * 1024 * 1024;
constexpr size_t iterations = 1000;

void stack_memory_example() {
    int data[N];
    for (size_t i = 0; i < N; ++i) {
        data[i] = i;
    }
}

void heap_memory_example() {
    int *data = new int[N];
    for (size_t i = 0; i < N; ++i) {
        data[i] = i;
    }
    delete[] data;
}

int main() {
    std::vector<long long> stack_times;
    std::vector<long long> heap_times;

    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        stack_memory_example();
        auto end = std::chrono::high_resolution_clock::now();
        stack_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        start = std::chrono::high_resolution_clock::now();
        heap_memory_example();
        end = std::chrono::high_resolution_clock::now();
        heap_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    long long average_stack_time = std::accumulate(stack_times.begin(), stack_times.end(), 0LL) / iterations;
    long long average_heap_time = std::accumulate(heap_times.begin(), heap_times.end(), 0LL) / iterations;

    std::cout << "Average stack memory time: " << average_stack_time << " ns\n";
    std::cout << "Average heap memory time: " << average_heap_time << " ns\n";

    return 0;
}
