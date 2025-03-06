#include <sycl/sycl.hpp>

using namespace sycl;
constexpr size_t N = 1024;

int main(){
    // 利用可能なデバイス上でキューを設定する
    queue q;

    // ホストコンテナを作成して、ホスト上で初期化
    std::vector<int> in_vec(N), out_vec(N);

    // 最初の例
    {
        // 入力ベクトルと出力ベクトルを初期化
        for(int i = 0; i < N; i++) in_vec[i] = i;
        std::fill(out_vec.begin(), out_vec.end(), 0);

        // バッファがスコープを外れて破棄されるように、新しいスコープを作成
        {
            // バッファを作成
            buffer in_buf(in_vec);
            buffer out_buf(out_vec);

            // コマンドグループをキューに送信
            q.submit([&](handler& h){
                // バッファにアクセス
                accessor in{in_buf, h};
                accessor out{out_buf, h};

                // 出力バッファに書き込むカーネル
                h.parallel_for(range{N}, [=](id<1> idx){
                    out[idx] = in[idx] * 2;
                });
            });

        // バッファが生きているスコープを閉じる！バッファ破棄を引き起こし、カーネルが
        // バッファへの書き込みを完了するまで待機し、書き込まれたバッファのデータをホストの割り当てにコピーする
        // （この場合は std::vector）。スコープの閉じによりバッファデストラクタが実行され、
        // 元の in_vec および out_vec に再びアクセスするのが安全になる。
        }

        for(int i = 0; i < N; i++)
            std::cout << "out_vec[" << i << "]=" << out_vec[i] << std::endl;
    }

    // 二つ目の例
    {
        // 入力ベクトルと出力ベクトルを初期化
        for(int i = 0; i < N; i++) in_vec[i] = i;
        std::fill(out_vec.begin(), out_vec.end(), 0);

        // バッファがスコープを外れて破棄されるように、新しいスコープを作成
        {
            // バッファを作成
            buffer in_buf(in_vec);
            buffer out_buf(out_vec);

            // コマンドグループをキューに送信
            q.submit([&](handler& h){
                // バッファにアクセス
                accessor in{in_buf, h};
                accessor out{out_buf, h};

                // 出力バッファに書き込むカーネル
                h.parallel_for(range{N}, [=](id<1> idx){
                    out[idx] = in[idx] * 2;
                });
            });

        // 全ての出力が期待される値と一致するか確認する。ホストアクセサーを使用！
        // バッファはまだスコープ内にある。
        host_accessor A{out_buf};
        for(int i = 0; i < N; i++)
            std::cout << "A[" << i << "]=" << A[i] << std::endl;
        }
    }

}