#pragma once
#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

struct ReluKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* data, std::size_t numElements) const
    {
        for (auto i : alpaka::uniformElements(acc, numElements))
        {
            T const val = data[i];
            data[i] = val > T{0} ? val : T{0};
        }
    }
};

} // namespace alpaka_kernels
