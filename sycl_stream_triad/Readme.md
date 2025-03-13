# Compile for avx512
icpx -fsycl -O3 -xCORE-AVX512 -qopt-zmm-usage=high sycl_stream_triad.cpp


# ASM code example
```
.LBB0_5:
        vmovupd zmm1, zmmword ptr [rsi + 8*r9]
        vfmadd213pd     zmm1, zmm0, zmmword ptr [rdi + 8*r9]
        vmovupd zmmword ptr [rdx + 8*r9], zmm1
```
