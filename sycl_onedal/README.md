# Compile command (need to specify libonedal files)
`icpx -fsycl -I./ -lonedal_dpc -lonedal_parameters_dpc -lonedal_core -lonedal_thread sample.cpp`

```
$ ./a.out 
Running on Intel(R) OpenCL, 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz

Number of rows in table: 4
Number of columns in table: 10
Is CSR table: 1
Slice of elements (compressed):      9.000 8.000 7.000 6.000 5.000 
Slice of indices (compressed):          9 10 8 9 10 
Slice of offsets:          1 3 6 
Running on Intel(R) Level-Zero, Intel(R) Graphics [0x9a49]

Number of rows in table: 4
Number of columns in table: 10
Is CSR table: 1
Slice of elements (compressed):      9.000 8.000 7.000 6.000 5.000 
Slice of indices (compressed):          9 10 8 9 10 
Slice of offsets:          1 3 6
```
