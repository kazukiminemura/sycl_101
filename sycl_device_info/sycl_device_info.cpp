#include <sycl/sycl.hpp>

using namespace sycl;

int main(){
    // Loop through avaiable platforms
    for(auto const& this_platform: platform::get_platforms()){
        std::cout << "Platform: " << this_platform.get_info<info::platform::name>() << std::endl;
        // Loop through devices
        for(auto const& this_device: this_platform.get_devices()){
            std::cout << "Device: " << this_device.get_info<info::device::name>() << std::endl;
        }
    }
    std::cout << "\n";

}