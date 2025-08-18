#pragma once
#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {
struct SigmoidKernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T* data, std::size_t n) const {
    for (auto i : alpaka::uniformElements(acc, n)) {
      T x = data[i];
      data[i] = T(1) / (T(1) + std::exp(-x));
    }
  }
};
} // namespace alpaka_kernels
