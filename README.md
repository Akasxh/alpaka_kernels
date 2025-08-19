# Alpaka Kernels 

currently working on  : softmax

### Currently all existing operators tested for CUDA

### To build and run :

cmake -S . -B build-cuda \
  -Dalpaka_ROOT=$HOME/alpaka-install \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
  -Dalpaka_ACC_GPU_CUDA_ONLY_MODE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DALPAKA_USE_CUDA=ON


#### Make the test files you want to test and then run the build
  
cmake --build build-cuda --target sigmoid_test -j $(nproc)

./build-cuda/tests/sigmoid_test
