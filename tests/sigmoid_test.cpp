#include <alpaka/alpaka.hpp>
#include <vector>
#include <iostream>
#include <cstddef>
#include <sigmoid_kernel.hpp>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

#if defined(USE_CPU_OMP)
  using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
#elif defined(USE_CPU_THREADS)
  using Acc = alpaka::AccCpuThreads<Dim, Idx>;
#else
  using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
#endif

int main() {
  using namespace alpaka_kernels;
  using T = float;

  std::vector<T> INPUT = { -2, -1, 0, 1, 2 };
  auto platAcc  = alpaka::Platform<Acc>{};
  auto platHost = alpaka::PlatformCpu{};
  auto devAcc   = alpaka::getDevByIdx(platAcc, 0);
  auto devHost  = alpaka::getDevByIdx(platHost, 0);
  alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

  const Idx N = INPUT.size();
  auto extent = alpaka::Vec<Dim, Idx>::all(N);
  auto dData = alpaka::allocBuf<T, Idx>(devAcc, extent);
  auto hData = alpaka::allocBuf<T, Idx>(devHost, extent);

  { T* p = alpaka::getPtrNative(hData); for(Idx i=0;i<N;++i) p[i]=INPUT[i]; }
  alpaka::memcpy(queue, dData, hData, extent); alpaka::wait(queue);

  constexpr Idx blockSize = 256;
  Idx grid = (N + blockSize - 1)/blockSize; if(!grid) grid=1;
  auto wd = alpaka::WorkDivMembers<Dim, Idx>(
    alpaka::Vec<Dim, Idx>::all(grid),
    alpaka::Vec<Dim, Idx>::all(blockSize),
    alpaka::Vec<Dim, Idx>::all(1)
  );

  SigmoidKernel k;
  auto dPtr = alpaka::getPtrNative(dData);
  alpaka::exec<Acc>(queue, wd, k, dPtr, (std::size_t)N);
  alpaka::wait(queue);

  alpaka::memcpy(queue, hData, dData, extent); alpaka::wait(queue);
  T* out = alpaka::getPtrNative(hData);
  std::cout << "Sigmoid: ";
  for(Idx i=0;i<N;++i) std::cout << out[i] << ' ';
  std::cout << '\n';
}
