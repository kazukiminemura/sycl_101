#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    queue q;
    device dev = q.get_device();

    std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

    // サポートされている sub-group サイズのリストを取得
    auto sub_group_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();

    std::cout << "Supported sub-group sizes: ";
    for (size_t size : sub_group_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    return 0;
}
