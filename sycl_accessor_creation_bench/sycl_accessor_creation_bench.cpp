#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

int main(){
  constexpr int N = 42;
  queue Q;
  std::cout << Q.get_device().get_info<info::device::name>() << std::endl;

  // バッファーを三つ作成
  buffer<int> A{range(N)};
  buffer<int> B{range(N)};
  buffer<int> C{range(N)};
  accessor pC{C};

  buffer<int> A2{range(N)};
  buffer<int> B2{range(N)};
  buffer<int> C2{range(N)};
  accessor pC2{C2};

  // Simple Accessor
  auto start1 = std::chrono::high_resolution_clock::now();
  Q.submit([&](handler &h){
    accessor aA{A, h};
    accessor aB{B, h};
    accessor aC{C, h};
    h.parallel_for(N, [=](id<1> i){
      aA[i] = 1;
      aB[i] = 40;
      aC[i] = 0;
    });
  });

  Q.submit([&](handler &h){
    accessor aA{A, h};
    accessor aB{B, h};
    accessor aC{C, h};
    h.parallel_for(N, [=](id<1> i){
      aC[i] = aA[i] + aB[i];
    });
  });

  Q.submit([&](handler &h){
    h.require(pC);
    h.parallel_for(N, [=](id<1> i){
      pC[i]++;
    });
  });

  host_accessor result1{C};
  for(int i=0; i<N; ++i)
    assert(result1[i] == N);

  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> simple_accessor_duration = end1 - start1;
  std::cout << "Simple accessor duration: " << simple_accessor_duration.count() << " seconds" << std::endl;

  // Specified Accessor
  auto start2 = std::chrono::high_resolution_clock::now();
  Q.submit([&](handler &h){
    accessor aA{A2, h, write_only, no_init};
    accessor aB{B2, h, write_only, no_init};
    accessor aC{C2, h, write_only, no_init};
    h.parallel_for(N, [=](id<1> i){
      aA[i] = 1;
      aB[i] = 40;
      aC[i] = 0;
    });
  });

  Q.submit([&](handler &h){
    accessor aA{A2, h, read_only};
    accessor aB{B2, h, read_only};
    accessor aC{C2, h, read_write};
    h.parallel_for(N, [=](id<1> i){
      aC[i] = aA[i] + aB[i];
    });
  });

  Q.submit([&](handler &h){
    h.require(pC2);
    h.parallel_for(N, [=](id<1> i){
      pC2[i]++;
    });
  });

  host_accessor result2{C2, read_only};
  for(int i=0; i<N; ++i)
    assert(result2[i] == N);

  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> specified_accessor_duration = end2 - start2;
  std::cout << "Specified accessor duration: " << specified_accessor_duration.count() << " seconds" << std::endl;

}