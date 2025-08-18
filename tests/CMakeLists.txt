function(add_kernel_test NAME)
  set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_test.cpp")
  if(NOT EXISTS "${SRC}")
    message(FATAL_ERROR "Missing test file: ${SRC}")
  endif()

  add_executable(${NAME}_test "${SRC}")
  target_link_libraries(${NAME}_test PRIVATE alpaka_kernels)

  # Select backend via macro
  if(ALPAKA_USE_CUDA)
    target_compile_definitions(${NAME}_test PRIVATE USE_CUDA)
    # Force this .cpp to be compiled with NVCC
    set_source_files_properties("${SRC}" PROPERTIES LANGUAGE CUDA)
  elseif(ALPAKA_USE_CPU_OMP)
    target_compile_definitions(${NAME}_test PRIVATE USE_CPU_OMP)
  elseif(ALPAKA_USE_CPU_THREADS)
    target_compile_definitions(${NAME}_test PRIVATE USE_CPU_THREADS)
  endif()
endfunction()

# do a add_kernel_test and inside the brackets put the kernel name. thhe name should be saved like NAME}_test.cpp
add_kernel_test(relu)
add_kernel_test(sigmoid)
add_kernel_test(tanh)
add_kernel_test(selu)

