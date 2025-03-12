#include <sycl/sycl.hpp>
#include <cfloat>
#include <chrono> 

using namespace sycl;

constexpr int num_runs = 10;
constexpr size_t scalar = 3;

float triad_parallel_for(const std::vector<float>& vecA, 
  const std::vector<float>& vecB, 
  std::vector<float>& vecC){

  assert(vecA.size() == vecB.size() &&
    vecB.size() == vecC.size());
  const size_t array_size = vecA.size();
  float min_time_ns = DBL_MAX;

  queue q{ property::queue::enable_profiling() };
  std::cout  << "Running on device: " << 
    q.get_device().get_info<info::device::name>() << std::endl;

  buffer<float> bufA(vecA);
  buffer<float> bufB(vecB);
  buffer<float> bufC(vecC);

  for (int i = 0; i < num_runs; i++){
    auto q_event = q.submit([&](handler& h){
      accessor A{ bufA, h };
      accessor B{ bufB, h };
      accessor C{ bufC, h };

      h.parallel_for(array_size, [=](id<1> idx){
        C[idx] = A[idx] + scalar * B[idx];
      });
    });

    float exec_time_ns = 
    q_event.get_profiling_info<info::event_profiling::command_end>() -
    q_event.get_profiling_info<info::event_profiling::command_start>();

    std::cout << "Execution time (iteration " << i << ") [sec]: " << 
      (float)exec_time_ns * 1.0E-9 << std::endl;
    min_time_ns = std::min(min_time_ns, exec_time_ns);
  }

  return min_time_ns;
}

float triad_simd(float* __restrict VA, float* __restrict VB,
  float* __restrict VC, size_t array_size, const float scalar) {

  float min_time_sec = DBL_MAX;

  for (int i = 0; i < num_runs; i++){
    auto ts = std::chrono::high_resolution_clock::now();
    for (size_t id = 0; id < array_size; id++) {
      VC[id] = VA[id] + scalar * VB[id];
    }
    
    auto te = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> exec_time_sec = te - ts;
    min_time_sec = std::min(min_time_sec, exec_time_sec.count());
  }
  return min_time_sec;
}


int main(){
  const size_t array_size = 1024;
  std::vector<float> vecA(array_size, 1.0);
  std::vector<float> vecB(array_size, 2.0);
  std::vector<float> vecC(array_size, 0.0);

  std::cout << "Running with stream size of " << array_size
            << " elements ("
            << (array_size * sizeof(float)) / (float)1024 / 1024
            << "MB)\n";


  float time_sec = triad_simd(vecA.data(), vecB.data(), vecC.data(), array_size, scalar);
  std::cout << "Minimum execution time (triad_simd) [sec]: " << 
    (float)time_sec << std::endl;

  float min_time_ns = triad_parallel_for(vecA, vecB, vecC);
  std::cout << "Minimum execution time (triad_parallel_for) [sec]: " << 
    (float)min_time_ns * 1.0E-9 << std::endl;

  return 0;
}
