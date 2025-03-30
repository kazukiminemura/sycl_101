#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;
constexpr int N = 256;
constexpr int LOCAL_SIZE = 16;

int main() {
    queue q;
    std::vector<int> data(N, 1);
    std::vector<int> conflict_counts(LOCAL_SIZE, 0);
    buffer<int, 1> buf(data.data(), range<1>(N));
    buffer<int, 1> conflict_buf(conflict_counts.data(), range<1>(LOCAL_SIZE));

    q.submit([&](handler &h) {
        accessor acc(buf, h, read_write);
        accessor conflict_acc(conflict_buf, h, write_only);
        local_accessor<int, 1> local_mem(range<1>(LOCAL_SIZE), h);

        h.parallel_for(nd_range<1>(range<1>(N), range<1>(LOCAL_SIZE)), [=](nd_item<1> item) {
            int local_id = item.get_local_id(0);
            int group_id = item.get_group(0);
            int bank_conflict_count = 0;

            // バンクコンフリクトが発生しやすいアクセスパターン
            for (int offset = 1; offset <= LOCAL_SIZE; ++offset) {
                local_mem[(local_id * offset) % LOCAL_SIZE] = acc[group_id * LOCAL_SIZE + local_id];
                item.barrier(access::fence_space::local_space);
                acc[group_id * LOCAL_SIZE + local_id] = local_mem[(local_id * offset) % LOCAL_SIZE];
                item.barrier(access::fence_space::local_space);

                // 推定のためのバンクコンフリクトカウント
                if ((local_id * offset) % LOCAL_SIZE == (local_id * (offset - 1)) % LOCAL_SIZE) {
                    bank_conflict_count++;
                }
            }

            // ローカルグループごとのバンクコンフリクトを記録
            if (local_id == 0) {
                conflict_acc[group_id] = bank_conflict_count;
            }
        });
    }).wait();

    // ホスト側で結果を取得
    host_accessor host_conflict_acc(conflict_buf, read_only);
    int total_conflicts = 0;
    for (int i = 0; i < LOCAL_SIZE; ++i) {
        total_conflicts += host_conflict_acc[i];
    }
    std::cout << "Estimated memory bank conflicts per work-group: " << total_conflicts / LOCAL_SIZE << std::endl;

    return 0;
}
