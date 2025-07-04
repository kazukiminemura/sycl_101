#include <sycl/sycl.hpp>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace sycl;

int main()
{
  queue q;

  const size_t N = 16;
  const size_t M = 16;
  range<2> stencil_range(N, M);
  range<2> alloc_range(N + 2, M + 2);
  std::vector<float> input(alloc_range.size()), output(alloc_range.size());
  std::iota(input.begin(), input.end(), 1);
  std::fill(output.begin(), output.end(), 0);

  // Simple stencil
  {
    auto start = std::chrono::high_resolution_clock::now();
    buffer<float, 2> input_buf(input.data(), alloc_range);
    buffer<float, 2> output_buf(output.data(), alloc_range);

    // BEGIN CODE SNIP
    q.submit([&](handler &h)
             {
      accessor input{input_buf, h};
      accessor output{output_buf, h};

      // Compute the average of each cell and its immediate neighbors
      h.parallel_for(stencil_range, [=](id<2> idx) {
        int i = idx[0] + 1;
        int j = idx[1] + 1;

        float self = input[i][j];
        float north = input[i - 1][j];
        float east = input[i][j + 1];
        float south = input[i + 1][j];
        float west = input[i][j - 1];
        output[i][j] =
            (self + north + east + south + west) / 5.0f;
      }); });
    // END CODE SNIP

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Simple stencil duration: " << duration.count() << " seconds" << std::endl;

    // Check that all outputs match serial execution.
    bool passed = true;
    for (int i = 1; i < N + 1; ++i)
    {
      for (int j = 1; j < M + 1; ++j)
      {
        float self = input[i * (M + 2) + j];
        float north = input[(i - 1) * (M + 2) + j];
        float east = input[i * (M + 2) + (j + 1)];
        float south = input[(i + 1) * (M + 2) + j];
        float west = input[i * (M + 2) + (j - 1)];
        float gold =
            (self + north + east + south + west) / 5.0f;
        if (std::abs(output[i * (M + 2) + j] - gold) >=
            1.0E-06)
        {
          passed = false;
        }
      }
    }
    std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";
  }

  // local stencil
  {
    auto start = std::chrono::high_resolution_clock::now();
    // Create SYCL buffers associated with input/output
    buffer<float, 2> input_buf(input.data(), alloc_range);
    buffer<float, 2> output_buf(output.data(), alloc_range);

    // BEGIN CODE SNIP
    q.submit([&](handler &h)
             {
      accessor input{input_buf, h};
      accessor output{output_buf, h};

      constexpr size_t B = 4;
      range<2> local_range(B, B);
      range<2> tile_size = local_range + range<2>(2, 2);  // Includes boundary cells
      auto tile = local_accessor<float, 2>(tile_size, h);

      stream out(1024, 256, h);

      // Compute the average of each cell and its immediate
      // neighbors
      h.parallel_for(nd_range<2>(stencil_range, local_range), [=](nd_item<2> it) {
            // Load this tile into work-group local memory
            id<2> lid = it.get_local_id();
            range<2> lrange = it.get_local_range();
            for (int ti = lid[0]; ti < B + 2; ti += lrange[0]) { // lrange[0] -> 4
              int gi = ti + B * it.get_group(0);
              for (int tj = lid[1]; tj < B + 2; tj += lrange[1]) { // lrange[1] -> 4
                int gj = tj + B * it.get_group(1);
                tile[ti][tj] = input[gi][gj];
              }
            }
            group_barrier(it.get_group());

            // Compute the stencil using values from local memory
            int gi = it.get_global_id(0) + 1;
            int gj = it.get_global_id(1) + 1;

            int ti = it.get_local_id(0) + 1;
            int tj = it.get_local_id(1) + 1;

            float self = tile[ti][tj];
            float north = tile[ti - 1][tj];
            float east = tile[ti][tj + 1];
            float south = tile[ti + 1][tj];
            float west = tile[ti][tj - 1];
            output[gi][gj] = (self + north + east + south + west) / 5.0f;
      }); });
    // END CODE SNIP

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Local stencil duration: " << duration.count() << " seconds" << std::endl;

    // Check that all outputs match serial execution
    bool passed = true;
    for (int i = 1; i < N + 1; ++i)
    {
      for (int j = 1; j < M + 1; ++j)
      {
        float self = input[i * (M + 2) + j];
        float north = input[(i - 1) * (M + 2) + j];
        float east = input[i * (M + 2) + (j + 1)];
        float south = input[(i + 1) * (M + 2) + j];
        float west = input[i * (M + 2) + (j - 1)];
        float gold =
            (self + north + east + south + west) / 5.0f;
        if (std::abs(output[i * (M + 2) + j] - gold) >=
            1.0E-06)
        {
          passed = false;
        }
      }
    }
    std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";
  }
}
