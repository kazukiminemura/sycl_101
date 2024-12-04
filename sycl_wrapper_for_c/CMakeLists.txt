cmake_minimum_required(VERSION 3.10)
project(SYCLProject)

# Set the compiler to use Intel's DPC++/SYCL compiler
set(CMAKE_CXX_COMPILER icx-cl)
set(CMAKE_C_COMPILER icx-cl)

# Add the SYCL source files and create static libraries
add_library(sycl_vector_add STATIC sycl_vector_add.cpp)
target_compile_options(sycl_vector_add PRIVATE -fsycl)

add_library(sycl_wrapper STATIC sycl_wrapper.c)

# Compile the main source file into an object file
add_library(main_obj OBJECT main.c)

# Create the main executable and link with the -fsycl flag
add_executable(main $<TARGET_OBJECTS:main_obj>)
target_link_options(main PRIVATE -fsycl)
target_link_libraries(main PRIVATE sycl_wrapper sycl_vector_add)
