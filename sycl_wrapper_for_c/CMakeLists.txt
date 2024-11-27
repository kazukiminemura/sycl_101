# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

# Project name and languages
project(SyclWrapperSample LANGUAGES C CXX)

# Set the SYCL compiler (Intel DPC++)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)
if(MSVC)
    set(CMAKE_CXX_COMPILER icx-cl)
    set(CMAKE_C_COMPILER icx-cl)
endif()

# Enable SYCL support for the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

# Define the shared library name based on the platform
#if(WIN32)
#    set(LIBRARY_NAME sycl_vector_add.dll)
#else()
#    set(LIBRARY_NAME libsycl_vector_add.so)
#endif()

# Add the SYCL shared library
add_library(sycl_vector_add SHARED sycl_vector_add.cpp)

# Add the main C program and link the SYCL library
add_executable(main_exec main.c)
target_link_libraries(main_exec PRIVATE sycl_vector_add)

# Set output directory for executables and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Handle runtime library paths for Windows
if(WIN32)
    add_custom_command(
        TARGET main_exec POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:sycl_vector_add> $<TARGET_FILE_DIR:main_exec>
    )
else()
    set_target_properties(main_exec PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
