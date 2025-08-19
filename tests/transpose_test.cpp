#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>
#include <transpose_kernel.hpp>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

#if defined(USE_CPU_OMP)
  using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
#elif defined(USE_CPU_THREADS)
  using Acc = alpaka::AccCpuThreads<Dim, Idx>;
#else
  using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;  // default: CUDA
#endif

int main() {
    using namespace alpaka_kernels;
    using T = float;

    // Hardcode input matrix (rows x cols)
    const std::size_t rows = 2;
    const std::size_t cols = 3;
    std::vector<T> INPUT = {
        1, 2, 3,
        4, 5, 6
    };

    std::cout << "Input (" << rows << "x" << cols << "):\n";
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j)
            std::cout << INPUT[i * cols + j] << " ";
        std::cout << "\n";
    }

    auto extent = alpaka::Vec<Dim, Idx>::all(rows * cols);

    // Platforms & devices
    auto platAcc  = alpaka::Platform<Acc>{};
    auto platHost = alpaka::PlatformCpu{};
    auto devAcc   = alpaka::getDevByIdx(platAcc, 0);
    auto devHost  = alpaka::getDevByIdx(platHost, 0);
    alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

    // Buffers
    auto dIn  = alpaka::allocBuf<T, Idx>(devAcc, extent);
    auto dOut = alpaka::allocBuf<T, Idx>(devAcc, extent);
    auto hIn  = alpaka::allocBuf<T, Idx>(devHost, extent);
    auto hOut = alpaka::allocBuf<T, Idx>(devHost, extent);

    // Copy INPUT -> hIn
    {
        T* p = alpaka::getPtrNative(hIn);
        for (Idx i = 0; i < rows * cols; ++i) p[i] = INPUT[i];
    }

    // H2D
    alpaka::memcpy(queue, dIn, hIn, extent);
    alpaka::wait(queue);

    // Work division
    auto workDiv = alpaka::WorkDivMembers<Dim, Idx>(
        alpaka::Vec<Dim, Idx>::all(1),  // grid
        alpaka::Vec<Dim, Idx>::all(1),  // block
        alpaka::Vec<Dim, Idx>::all(1)   // elems/thread
    );

    // Launch transpose
    TransposeKernel kernel;
    alpaka::exec<Acc>(queue, workDiv, kernel,
                      alpaka::getPtrNative(dIn),
                      alpaka::getPtrNative(dOut),
                      rows, cols);
    alpaka::wait(queue);

    // D2H
    alpaka::memcpy(queue, hOut, dOut, extent);
    alpaka::wait(queue);

    // Print result
    std::cout << "Output (" << cols << "x" << rows << "):\n";
    {
        T* p = alpaka::getPtrNative(hOut);
        for (std::size_t i = 0; i < cols; ++i) {
            for (std::size_t j = 0; j < rows; ++j)
                std::cout << p[i * rows + j] << " ";
            std::cout << "\n";
        }
    }

    return 0;
}
