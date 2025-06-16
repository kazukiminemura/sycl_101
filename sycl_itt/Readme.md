# 使い方 (Windows)
```
git clone https://github.com/intel/ittapi.git
cd ittapi
python buildall.py --vs 17

set LIB=<path/to/sycl_itt>\ittapi\build_win\64\bin\Release;%LIB%
set INCLUDE=<path/to/sycl_itt>\ittapi\include;%INCLUDE%
icpx -fsycl sycl_itt.cpp -o sycl_itt.exe -llibittnotify
```

# プロファイリング
```
vtune --collect hotspots -- sycl_itt
```
