# Usage
```
icpx -fsycl sycl_memory_banks.cpp
```

```
$ ./a.out
Banks: 2 | Estimated memory bank conflicts per work-group: 0 | Execution Time: 35.2722 ms
Banks: 4 | Estimated memory bank conflicts per work-group: 0 | Execution Time: 0.130867 ms
Banks: 8 | Estimated memory bank conflicts per work-group: 0 | Execution Time: 0.117965 ms
Banks: 16 | Estimated memory bank conflicts per work-group: 0 | Execution Time: 0.111411 ms
Banks: 32 | Estimated memory bank conflicts per work-group: 0 | Execution Time: 0.130754 ms
Banks: 64 | Estimated memory bank conflicts per work-group: 1 | Execution Time: 0.135496 ms
Banks: 128 | Estimated memory bank conflicts per work-group: 2 | Execution Time: 0.222036 ms
```
