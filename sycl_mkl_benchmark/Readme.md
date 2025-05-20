# Usage
```
export KMP_AFFINITY=granularity=fine,compact,1,0
icpx -fsycl test.cpp -qmkl; ./a.out
```
