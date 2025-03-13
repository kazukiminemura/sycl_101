# Compile for avx512
icpx -O3 -xCORE-AVX512 -qopt-zmm-usage=high sycl_stream_triad.cpp
