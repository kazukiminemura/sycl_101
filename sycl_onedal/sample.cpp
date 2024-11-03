#include <sycl/sycl.hpp>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include <oneapi/dal/table/column_accessor.hpp>
#include <oneapi/dal/table/homogen.hpp>

namespace dal = oneapi::dal;


void try_add_device(std::vector<sycl::device>& devices, int (*selector)(const sycl::device&)) {
    try {
        devices.push_back(sycl::ext::oneapi::detail::select_device(selector));
    }
    catch (...) {
    }
}

std::vector<sycl::device> list_devices() {
    std::vector<sycl::device> devices;
    try_add_device(devices, &sycl::cpu_selector_v);
    try_add_device(devices, &sycl::gpu_selector_v);
    return devices;
}

void run(sycl::queue &q) {
    constexpr std::int64_t row_count = 6;
    constexpr std::int64_t column_count = 2;
    const float data_host[] = {
        0.f, 6.f, 1.f, 7.f, 2.f, 8.f, 3.f, 9.f, 4.f, 10.f, 5.f, 11.f,
    };

    auto data = sycl::malloc_shared<float>(row_count * column_count, q);
    q.memcpy(data, data_host, sizeof(float) * row_count * column_count).wait();

    auto table = dal::v1::homogen_table{q,
                                     data,
                                     row_count,
                                     column_count,
                                     dal::detail::make_default_delete<const float>(q) };

//     dal::column_accessor<const float> acc{ table };

//     for (std::int64_t col = 0; col < table.get_column_count(); col++) {
//         std::cout << "column " << col << " values: ";

//         const auto col_values = acc.pull(q, col);
//         for (std::int64_t i = 0; i < col_values.get_count(); i++) {
//             std::cout << col_values[i] << ", ";
//         }
//         std::cout << std::endl;
//     }
}

int main(int argc, char const *argv[]) {
    for (auto d : list_devices()) {
    std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                << ", " << d.get_info<sycl::info::device::name>() << "\n"
                << std::endl;
    auto q = sycl::queue{ d };
    // run(q);
    }
    return 0;
}