#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

constexpr int M = 512;
constexpr int N = 256;
constexpr int K = 1024;


int main(){
    float(*c_back)[N] = new float[M][N];

    // Intialize c_back
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) c_back[i][j] = 0.0f; 
    
    buffer<float, 2> bufA(range(M, K));
    buffer<float, 2> bufB(range(K, N));
    buffer bufC(reinterpret_cast<float*>(c_back), range(M, N));
    
    // Traditional matrix multiplication
    {
    auto start = std::chrono::high_resolution_clock::now();    
    queue Q;

    Q.submit([&](handler &h){
        accessor matrixA(bufA, h, write_only);
        h.parallel_for(range(M, K), [=](auto index) {
            matrixA[index] = 1.0f;
        });
    });

    Q.submit([&](handler &h){
        accessor matrixB(bufB, h, write_only);
        h.parallel_for(range(K, N), [=](auto index) {
            matrixB[index] = 1.0f;
        });
    });

    Q.submit([&](handler &h){
        // Traditional accessors, representing matrices in gloabl memory:
        accessor matrixA{bufA, h, read_only};
        accessor matrixB{bufB, h, read_only};
        accessor matrixC{bufC, h};

        int width_a = bufA.get_range()[1];

        // // Execute kernel.
        h.parallel_for(
            nd_range<2>{{M, N},{32,32}}, [=](nd_item<2> item){
                // Indices in the global index space:
                int m = item.get_global_id()[0];
                int n = item.get_global_id()[1];
        
                float sum = 0.0f;
                // Compute the result of one element of c
                for (int i = 0; i < width_a; i++) {
                  sum += matrixA[m][i] * matrixB[i][n];
                }
        
                matrixC[m][n] = sum;
        });
    });

    host_accessor h_a{bufC};
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            assert(h_a[i][j] == K);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Traditional matrix multiplication: " << duration.count() << " seconds" << std::endl;
    }

    // Tiled matrix multiplication
    {
    auto start = std::chrono::high_resolution_clock::now();
 
    queue Q;

    Q.submit([&](handler &h){
        accessor matrixA(bufA, h, write_only);
        h.parallel_for(range(M, K), [=](auto index) {
            matrixA[index] = 1.0f;
        });
    });

    Q.submit([&](handler &h){
        accessor matrixB(bufB, h, write_only);
        h.parallel_for(range(K, N), [=](auto index) {
            matrixB[index] = 1.0f;
        });
    });
    
    Q.submit([&](handler &h){
        // Traditional accessors, representing matrices in gloabl memory:
        accessor matrixA{bufA, h};
        accessor matrixB{bufB, h};
        accessor matrixC{bufC, h};

        // Local accessor, for one matrix tile:
        constexpr int tile_size = 16;
        local_accessor<float> tileA{tile_size, h};

        h.parallel_for(
            nd_range<2>{{M, N}, {1, tile_size}}, [=](nd_item<2> item){
                // Indices in the global index space:
                int m = item.get_global_id()[0];
                int n = item.get_global_id()[1];

                // Index in the local index space:
                int i = item.get_local_id()[1];

                float sum = 0.0f;
                for (int kk = 0; kk < K; kk += tile_size){
                    // Load the matrix tiel from amtrixA, and synchronize
                    // to ensure all work-items have a consistent view of the matrix tile in local memory
                    tileA[i] = matrixA[m][kk + i];
                    item.barrier();

                    // Perform computation using the local memor tile, and matrix B in local memory.
                    for (int k = 0; k < tile_size; ++k){
                        sum += tileA[k] * matrixB[kk + k][n];
                    }

                    // After computation, synchronize to again, to ensure all reads from the local memory tile are complete.
                    item.barrier();
                }
                //
                matrixC[m][n] = sum;
        });
    });

    host_accessor h_a{bufC};
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            assert(h_a[i][j] == K);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Tiled matrix multiplication: " << duration.count() << " seconds" << std::endl;
    }


}