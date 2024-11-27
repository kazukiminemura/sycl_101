# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

# Project name and languages
project(SyclWrapperSample LANGUAGES C CXX)

# Set the SYCL compiler (Intel DPC++)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)

# Enable SYCL support for the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

# Add the SYCL shared library
add_library(sycl_vector_add SHARED sycl_vector_add.cpp)

# Add the main C program and link the SYCL library
add_executable(main_exec main.c)
target_link_libraries(main_exec PRIVATE sycl_vector_add)


Below is an updated CMakeLists.txt file that works for both Linux and Windows. It adjusts the library and compiler settings depending on the platform.

Cross-Platform CMakeLists.txt
cmake
コードをコピーする
# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

# Project name and languages
project(SyclVectorAdd LANGUAGES C CXX)

# Set the SYCL compiler (Intel DPC++)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)

# Enable SYCL support for the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

# Define the shared library name based on the platform
if(WIN32)
    set(LIBRARY_NAME sycl_vector_add.dll)
else()
    set(LIBRARY_NAME libsycl_vector_add.so)
endif()

# Add the SYCL shared library
add_library(sycl_vector_add SHARED sycl_vector_add.cpp)

# Add the main C program and link the SYCL library
add_executable(main_exec main.c)
target_link_libraries(main_exec PRIVATE sycl_vector_add)

# Set the library's output name
set_target_properties(sycl_vector_add PROPERTIES OUTPUT_NAME "sycl_vector_add")

# Specify runtime library path for Linux and DLL directory for Windows
if(WIN32)
    # For Windows, set the DLL directory for runtime linking
    add_custom_command(
        TARGET main_exec POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:sycl_vector_add> $<TARGET_FILE_DIR:main_exec>
    )
else()
    # For Linux, set the runtime path to the current directory
    set_target_properties(main_exec PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
