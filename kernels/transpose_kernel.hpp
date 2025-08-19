#pragma once
#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

// 2D transpose: y[j, i] = x[i, j]
struct TransposeKernel
{
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* in,   
        T* out,        
        std::size_t rows,
        std::size_t cols) const
    {
        auto elems = alpaka::uniformElements(acc, rows * cols);
        for (auto idx : elems) {
            std::size_t i = idx / cols; // row
            std::size_t j = idx % cols; // col
            out[j * rows + i] = in[i * cols + j]; // simple transpose happening to T out
        }
    }
};

} // namespace alpaka_kernels
