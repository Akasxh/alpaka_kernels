#pragma once
#include <alpaka/alpaka.hpp>
#include <cmath>

namespace alpaka_kernels {

struct TanhKernel
{
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* data, std::size_t numElements) const
    {
        for (auto i : alpaka::uniformElements(acc, numElements))
        {
            data[i] = std::tanh(data[i]);
        }
    }
};

} // namespace alpaka_kernels
