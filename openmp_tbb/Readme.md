# 使い方
```
icpx openmp_tbb_multithreads.cpp -ltbb -qopenmp -o openmp_tbb_multithreads
./openmp_tbb_multithreads
```

# 
```
=== OpenMP Test ===
OpenMP thread 0 processing index 0
OpenMP thread 2 processing index 4
OpenMP thread 1 processing index 2
OpenMP thread 7 processing index 9
OpenMP thread 4 processing index 6
OpenMP thread 6 processing index 8
OpenMP thread 5 processing index 7
OpenMP thread 3 processing index 5
OpenMP thread 0 processing index 1
OpenMP thread 1 processing index 3

=== oneTBB Test ===
TBB thread 139957172262784 processing index 0
TBB thread 139956848658112 processing index 5
TBB thread 139956852856512 processing index 7
TBB thread 139956844459712 processing index 6
TBB thread 139956840261312 processing index 2
TBB thread 139956836062912 processing index 8
TBB thread 139956827666112 processing index 1
TBB thread 139956831864512 processing index 3
TBB thread 139957172262784 processing index 4
TBB thread 139956848658112 processing index 9
```
