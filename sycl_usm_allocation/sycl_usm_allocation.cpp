#include <sycl/sycl.hpp>

using namespace sycl;

constexpr int N = 42;

int main(){
    queue Q;
    // Allocate N float

    // C-style
    float *f1 = static_cast<float*>(malloc_shared(N*sizeof(float), Q));

    // C++-style
    float *f2 = malloc_shared<float>(N, Q);

    // C++-allocator-style
    usm_allocator<float, usm::alloc::shared> alloc(Q);
    float *f3 = alloc.allocate(N);

    // Free our allocations
    free(f1, Q.get_context());
    free(f2, Q);
    alloc.deallocate(f3, N);

    return 0;
}
