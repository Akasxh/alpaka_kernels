#pragma once
#include <alpaka/alpaka.hpp>

namespace alpaka_kernels
{

// Template for a new kernel
// Replace `TemplateKernel` with your kernel name
struct TemplateKernel
{
    // Operator that Alpaka calls on device
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* data, std::size_t numElements) const
    {
        // Loop over elements this thread is responsible for
        for (auto i : alpaka::uniformElements(acc, numElements))
        {
            // Example operation (identity: data[i] = data[i])
            data[i] = data[i];
            // Replace this with your kernel logic
        }
    }
};

} // namespace alpaka_kernels


// currently takes T* data which is in-place, for outplace replace with smthn like T const* in, T* out.
