#include <level_zero/ze_api.h>
#include <iostream>

// LevelZero ドライバ初期化
void initLevelZero() {
    ze_result_t result = zeInit(0);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "Failed to initialize Level Zero\n";
        exit(1);
    }
}

// driver ハンドラ取得
ze_driver_handle_t getDriver() {
    uint32_t driverCount = 0;
    ze_result_t result = zeDriverGet(&driverCount, nullptr);
    if (result != ZE_RESULT_SUCCESS || driverCount == 0) {
        std::cerr << "Failed to get driver count\n";
        exit(1);
    }

    ze_driver_handle_t hDriver;
    result = zeDriverGet(&driverCount, &hDriver);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "Failed to get driver\n";
        exit(1);
    }

    return hDriver;
}

// device ハンドラ取得
ze_device_handle_t getDevice(ze_driver_handle_t hDriver) {
    uint32_t deviceCount = 0;
    ze_result_t result = zeDeviceGet(hDriver, &deviceCount, nullptr);
    if (result != ZE_RESULT_SUCCESS || deviceCount == 0) {
        std::cerr << "Failed to get device count\n";
        exit(1);
    }

    ze_device_handle_t hDevice;
    result = zeDeviceGet(hDriver, &deviceCount, &hDevice);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "Failed to get device\n";
        exit(1);
    }

    return hDevice;
}

int main() {
    // LevelZero ドライバ初期化
    initLevelZero();

    // driver ハンドラ取得
    ze_driver_handle_t hDriver = getDriver();
    std::cout << "Driver handle: " << hDriver << "\n";

    // device ハンドラ取得
    ze_device_handle_t hDevice = getDevice(hDriver);
    std::cout << "Device handle: " << hDevice << "\n";

    return 0;
}
