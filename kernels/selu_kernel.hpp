#pragma once
#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

struct SeluKernel
{
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* data, std::size_t numElements) const
    {
        // SELU constants from the paper
        constexpr T lambda = T(1.0507009873554805);
        constexpr T alpha  = T(1.6732632423543772);

        for (auto i : alpaka::uniformElements(acc, numElements))
        {
            const T x = data[i];
            // SELU: λ * (max(0, x) + min(0, α*(exp(x)-1)))
            const T pos = alpaka::math::max(acc, T{0}, x);
            const T neg = alpaka::math::min(acc, T{0}, alpha * (alpaka::math::exp(acc, x) - T{1}));
            data[i] = lambda * (pos + neg);
        }
    }
};

} // namespace alpaka_kernels
