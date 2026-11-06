cmake_minimum_required(VERSION 3.20)
project(spider_kernels LANGUAGES CXX)
option(SPIDER_ENABLE_CUDA "Build CUDA kernels (requires CUDA Toolkit)" OFF)
option(SPIDER_BUILD_TESTS "Build native parity/smoke tests" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SPIDER_COMMON
  kernels/scalar/scalar.cpp
  kernels/scalar/scalar_phase1_1.cpp
  kernels/scalar/scalar_phase1_2.cpp
  kernels/scalar/scalar_phase1_3.cpp
  kernels/scalar/scalar_phase2_attn.cpp
  kernels/scalar/scalar_phase2_kv.cpp
  kernels/scalar/scalar_phase2_rope.cpp
  kernels/src/phase1_api_ext.cpp
)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SPIDER_ARCH)
if(SPIDER_ARCH MATCHES "aarch64|arm64|armv8|arm")
  add_library(spider_kernels SHARED ${SPIDER_COMMON} kernels/arm/arm_neon.cpp)
  target_compile_options(spider_kernels PRIVATE -O3)
  if(SPIDER_ARCH MATCHES "aarch64|arm64|armv8")
    target_compile_options(spider_kernels PRIVATE -march=armv8-a+simd)
  endif()
else()
  add_library(spider_kernels SHARED ${SPIDER_COMMON} kernels/src/dispatch.cpp)
  target_compile_options(spider_kernels PRIVATE -O3)
endif()
target_include_directories(spider_kernels PUBLIC kernels/include)
set_target_properties(spider_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(SPIDER_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(spider_cuda SHARED
    kernels/cuda/spider_kernels.cu
    kernels/cuda/spider_expert_v2.cu
    kernels/cuda/spider_attn_cuda.cu
    kernels/cuda/spider_rope_cuda.cu
  )
  target_include_directories(spider_cuda PUBLIC kernels/include)
  set_target_properties(spider_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "70;75;80;86;89")
  target_compile_options(spider_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

if(SPIDER_BUILD_TESTS)
  enable_testing()
  add_executable(spider_kernel_smoke kernels/tests/test_kernels.cpp)
  target_link_libraries(spider_kernel_smoke PRIVATE spider_kernels)
  add_test(NAME spider_kernel_smoke COMMAND spider_kernel_smoke)
  add_executable(spider_kernel_phase1_1 kernels/tests/test_phase1_1.cpp)
  target_link_libraries(spider_kernel_phase1_1 PRIVATE spider_kernels)
  add_test(NAME spider_kernel_phase1_1 COMMAND spider_kernel_phase1_1)
  add_executable(spider_kernel_phase1_2 kernels/tests/test_phase1_2.cpp)
  target_link_libraries(spider_kernel_phase1_2 PRIVATE spider_kernels)
  add_test(NAME spider_kernel_phase1_2 COMMAND spider_kernel_phase1_2)
  add_executable(spider_kernel_phase1_3 kernels/tests/test_phase1_3.cpp)
  target_link_libraries(spider_kernel_phase1_3 PRIVATE spider_kernels)
  add_test(NAME spider_kernel_phase1_3 COMMAND spider_kernel_phase1_3)
  add_executable(spider_kernel_phase1_5 kernels/tests/test_phase1_5.cpp)
  target_link_libraries(spider_kernel_phase1_5 PRIVATE spider_kernels)
  add_test(NAME spider_kernel_phase1_5 COMMAND spider_kernel_phase1_5)
  add_executable(spider_kernel_phase2 kernels/tests/test_phase2.cpp)
  target_link_libraries(spider_kernel_phase2 PRIVATE spider_kernels)
  add_test(NAME spider_kernel_phase2 COMMAND spider_kernel_phase2)
  add_executable(spider_kernel_phase2_1 kernels/tests/test_phase2_1.cpp)
  target_link_libraries(spider_kernel_phase2_1 PRIVATE spider_kernels)
  add_test(NAME spider_kernel_phase2_1 COMMAND spider_kernel_phase2_1)
  add_executable(spider_kernel_phase2_rope kernels/tests/test_phase2_rope.cpp)
  target_link_libraries(spider_kernel_phase2_rope PRIVATE spider_kernels)
  add_test(NAME spider_kernel_phase2_rope COMMAND spider_kernel_phase2_rope)
endif()

install(DIRECTORY kernels/include/ DESTINATION include)
install(TARGETS spider_kernels LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
