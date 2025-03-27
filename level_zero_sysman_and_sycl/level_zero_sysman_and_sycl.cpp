#include <iostream>
#include <level_zero/zes_api.h>
#include <sycl/sycl.hpp>

void sysman_op1(){
    // Proceed with Sysman operations
    std::uint32_t num_drivers = 0U;
    std::cout << "DEBUG: Calling zesDriverGet(&num_drivers, nullptr)" << std::endl;
    ze_result_t result2 = zesDriverGet(&num_drivers, nullptr);
    if (result2 == ZE_RESULT_SUCCESS) {
        std::cout << "Sysman driver initialization successful." << std::endl;
    } else if (result2 == ZE_RESULT_ERROR_UNINITIALIZED) {
        std::cout << "Sysman driver initialization failed: ZE_RESULT_ERROR_UNINITIALIZED" << std::endl;
    } else if (result2 == ZE_RESULT_ERROR_DEVICE_LOST) {
        std::cout << "Sysman driver initialization failed: ZE_RESULT_ERROR_DEVICE_LOST" << std::endl;
    } else if (result2 == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
        std::cout << "Sysman driver initialization failed: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY" << std::endl;
    } else if (result2 == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
        std::cout << "Sysman driver initialization failed: ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY" << std::endl;
    } else if (result2 == ZE_RESULT_ERROR_INVALID_NULL_POINTER) {
        std::cout << "Sysman driver initialization failed: ZE_RESULT_ERROR_INVALID_NULL_POINTER" << std::endl;
    }
}

int main() {
    // Sysman initialization
    ze_result_t result = zesInit(0);
    if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
        // Handle error
        std::cout << "Sysman initialization failed: ZE_RESULT_ERROR_UNINITIALIZED" << std::endl;
        return -1;
    }

    // Proceed with Sysman operations
    sysman_op1();

    // SYCL Initialization
    try {
        auto device = sycl::device{sycl::gpu_selector_v};
        std::cout << "SYCL device initialized successfully." << std::endl;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL initialization failed: " << e.what() << std::endl;
        return -1;
    }

    // Proceed with Sysman operations
    sysman_op1();

    return 0;
}
