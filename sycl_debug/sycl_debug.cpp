#include <sycl/sycl.hpp>

using namespace sycl;

int main(){
    sycl::queue q;
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    q.submit([&](sycl::handler& h){
        stream out(1024, 256, h);
        h.parallel_for(range{8}, [=](id<1> idx){
            out << "Testing my sycl stream (this is work-item ID:)" <<  idx << ")\n";
        });
    }).wait();

}