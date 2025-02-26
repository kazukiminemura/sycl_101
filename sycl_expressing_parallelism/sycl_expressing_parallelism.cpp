#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace sycl;

int main() {
    const size_t N = 1024;
    const size_t B = 4;
    std::vector<int> data1(N, 1);
    std::vector<int> data2(N, 1);

    // デバイスとキューの設定
    queue q;
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;

    // バッファの作成
    buffer<int, 1> buf1(data1.data(), range<1>(N));
    buffer<int, 1> buf2(data2.data(), range<1>(N));

    // parallel_forを使ったカーネル
    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i<10000; ++i){
      q.submit([&](handler& h) {
          auto acc = buf1.get_access<access::mode::read_write>(h);
          h.parallel_for(range<1>(N), [=](auto& idx) {
              acc[idx] = 1;
          });
      }).wait();
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;

    // parallel_for_work_groupを使ったカーネル
    auto start2 = std::chrono::high_resolution_clock::now();
    range num_group{N / B};
    range group_size{B};
    for (size_t i=0; i<10000; ++i){
      q.submit([&](handler& h) {
          auto acc = buf2.get_access<access::mode::read_write>(h);
          h.parallel_for_work_group(num_group, group_size, [=](group<1> grp){
            int jb = grp.get_id(0);
              grp.parallel_for_work_item([&](h_item<1> item) {
                int j = jb * B + item.get_local_id(0);
                  acc[j] = 1;
              });
          });
          // logical mapping
          // h.parallel_for_work_group(num_group, [=](group<1> grp){
          //   int jb = grp.get_id(0);
          //     grp.parallel_for_work_item(group_size, [&](h_item<1> item) {
          //       int j = jb * B + item.get_local_id(0);
          //         acc[j] = 1;
          //     });
          // });
      }).wait();
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end2 - start2;

    // 実行時間の表示
    std::cout << "parallel_for duration: " << duration1.count() << " seconds\n";
    std::cout << "parallel_for_work_group duration: " << duration2.count() << " seconds\n";

    // Check resutls
    host_accessor buf1_hacc(buf1, read_only);
    host_accessor buf2_hacc(buf2, read_only);
    for (size_t i=0; i<N; ++i){
      if (buf1_hacc[i] != 1){
          std::cout << "something wrong in parallel_for " << std::endl;
          break;
        }
    }
    for (size_t i=0; i<N; ++i){
      if (buf2_hacc[i] != 1){
          std::cout << "something wrong in parallel_for_work_group" << std::endl;
          break;
        }
    }

    return 0;
}