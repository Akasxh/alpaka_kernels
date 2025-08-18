#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>
#include <cstddef>
#include <tanh_kernel.hpp>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

#if defined(USE_CPU_OMP)
  using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
#elif defined(USE_CPU_THREADS)
  using Acc = alpaka::AccCpuThreads<Dim, Idx>;
#else
  using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;  // default to CUDA
#endif

int main() {
    using namespace alpaka_kernels;
    using T = float;

    // Hardcode inputs here
    std::vector<T> INPUT = { -3.f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 3.f };

    const Idx N = static_cast<Idx>(INPUT.size());
    auto extent = alpaka::Vec<Dim, Idx>::all(N);

    // Platforms & devices
    auto platAcc  = alpaka::Platform<Acc>{};
    auto platHost = alpaka::PlatformCpu{};
    auto devAcc   = alpaka::getDevByIdx(platAcc, 0);
    auto devHost  = alpaka::getDevByIdx(platHost, 0);
    alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

    // Buffers
    auto dData = alpaka::allocBuf<T, Idx>(devAcc, extent);
    auto hData = alpaka::allocBuf<T, Idx>(devHost, extent);

    // Copy INPUT to host buffer
    {
        T* p = alpaka::getPtrNative(hData);
        for (Idx i = 0; i < N; ++i) p[i] = INPUT[i];
    }

    std::cout << "Input:  ";
    for (auto v : INPUT) std::cout << v << ' ';
    std::cout << '\n';

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

    // Launch in-place tanh
    TanhKernel kernel;
    auto dPtr = alpaka::getPtrNative(dData);
    alpaka::exec<Acc>(queue, workDiv, kernel, dPtr, static_cast<std::size_t>(N));
    alpaka::wait(queue);

    // D2H
    alpaka::memcpy(queue, hData, dData, extent);
    alpaka::wait(queue);

    // Print output
    std::vector<T> OUT(N);
    {
        T* p = alpaka::getPtrNative(hData);
        for (Idx i = 0; i < N; ++i) OUT[i] = p[i];
    }

    std::cout << "Output: ";
    for (auto v : OUT) std::cout << v << ' ';
    std::cout << '\n';

    return 0;
}
