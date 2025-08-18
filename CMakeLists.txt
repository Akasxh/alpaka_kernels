cmake_minimum_required(VERSION 3.20)
project(ALPAKA_KERNELS LANGUAGES CXX CUDA)

# Pass Alpaka install if not auto-found:
#   cmake -Dalpaka_ROOT=/path/to/alpaka-install ..
find_package(alpaka REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library for kernels
add_library(alpaka_kernels INTERFACE)
target_include_directories(alpaka_kernels INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_link_libraries(alpaka_kernels INTERFACE alpaka)

# Options to select backend
option(ALPAKA_USE_CUDA "Use CUDA backend in tests" ON)
option(ALPAKA_USE_CPU_OMP "Use CPU OpenMP backend in tests" OFF)
option(ALPAKA_USE_CPU_THREADS "Use CPU Threads backend in tests" OFF)

# Add tests
add_subdirectory(tests)
