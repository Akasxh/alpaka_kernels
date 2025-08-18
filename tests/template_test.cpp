// Template test file for a new kernel
// Save as tests/<name>_test.cpp
// Register it in tests/CMakeLists.txt with: add_kernel_test(<name>)

#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>
#include <cstddef>
#include <template_kernel.hpp>   // include your kernel header like relu/sigmoid

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

#if defined(USE_CPU_OMP)
  using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
#elif defined(USE_CPU_THREADS)
  using Acc = alpaka::AccCpuThreads<Dim, Idx>;
#else
  using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;  // default to CUDA
#endif

int main()  
{
    using namespace alpaka_kernels;
    using T = float;  // change type if needed

    // === Hardcoded input data for testing ===
    std::vector<T> INPUT = { -2, -1, 0, 1, 2 };
    // ========================================

    const Idx N = static_cast<Idx>(INPUT.size());
    auto extent = alpaka::Vec<Dim, Idx>::all(N);

    // Platforms & devices
    auto platAcc  = alpaka::Platform<Acc>{};
    auto platHost = alpaka::PlatformCpu{};
    auto devAcc   = alpaka::getDevByIdx(platAcc, 0);
    auto devHost  = alpaka::getDevByIdx(platHost, 0);
    alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

    // Allocate buffers
    auto dData = alpaka::allocBuf<T, Idx>(devAcc, extent);
    auto hData = alpaka::allocBuf<T, Idx>(devHost, extent);

    // Copy INPUT into host buffer
    {
        T* p = alpaka::getPtrNative(hData);
        for (Idx i = 0; i < N; ++i) p[i] = INPUT[i];
    }

    std::cout << "Input: ";
    for (auto v : INPUT) std::cout << v << " ";
    std::cout << "\n";

    // H2D
    alpaka::memcpy(queue, dData, hData, extent);
    alpaka::wait(queue);

    // Work division
    constexpr Idx blockSize = 256;
    Idx gridSize = (N + blockSize - 1) / blockSize;
    if (gridSize == 0) gridSize = 1;
    auto workDiv = alpaka::WorkDivMembers<Dim, Idx>(
        alpaka::Vec<Dim, Idx>::all(gridSize),
        alpaka::Vec<Dim, Idx>::all(blockSize),
        alpaka::Vec<Dim, Idx>::all(1)
    );

    // Run the kernel
    TemplateKernel kernel;  // replace with your kernel name
    auto dPtr = alpaka::getPtrNative(dData);
    alpaka::exec<Acc>(queue, workDiv, kernel, dPtr, static_cast<std::size_t>(N));
    alpaka::wait(queue);

    // D2H
    alpaka::memcpy(queue, hData, dData, extent);
    alpaka::wait(queue);

    // Print output
    std::cout << "Output: ";
    {
        T* p = alpaka::getPtrNative(hData);
        for (Idx i = 0; i < N; ++i) std::cout << p[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
