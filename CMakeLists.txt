# 3.17+ for CMAKE_CUDA_KNOWN_FEATURES/cuda_std_11
# 3.18 for CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(tc LANGUAGES CXX CUDA VERSION 0.1.0.0)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to OFF")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "")
endif()


set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

add_executable(tc 
main.cu 
numeric.cpp
benchmark_cpu_float_rcr.cpp
benchmark_cpu_float_rrr.cpp
benchmark_cublas.cu
benchmark_gpu_float_rrr_shmem.cu
benchmark_gpu_float_rrr.cu
benchmark_tc_half_rrr_shmem.cu
benchmark_tc_half_rrr.cu
)

target_link_libraries(tc cublas)

set_property(TARGET tc PROPERTY CXX_STANDARD 11)
set_property(TARGET tc PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET tc PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tc PROPERTY CUDA_STANDARD 11)
set_property(TARGET tc PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET tc PROPERTY ENABLE_EXPORTS ON)

target_compile_options(tc PRIVATE
$<$<COMPILE_LANGUAGE:CXX>:
 -ffast-math;
 -Wall
 -Wextra
 -Wcast-align;
 -Wdisabled-optimization;
 -Wformat=2;
 -Winit-self;
 -Wlogical-op;
 -Wmissing-include-dirs;
 -Woverloaded-virtual;
 -Wpointer-arith;
 -Wshadow;
 -Wstrict-aliasing;
 -Wswitch-enum;
 -Wvla;
 -ffast-math;
 >
)

target_compile_options(tc PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:
--Wno-deprecated-gpu-targets;
--expt-extended-lambda;
-Xcompiler=-Wall;
-Xcompiler=-Wextra;
-Xcompiler=-Wcast-align;
-Xcompiler=-Wdisabled-optimization;
-Xcompiler=-Wformat=2;
-Xcompiler=-Winit-self;
-Xcompiler=-Wlogical-op;
-Xcompiler=-Wmissing-include-dirs;
-Xcompiler=-Woverloaded-virtual;
-Xcompiler=-Wpointer-arith;
-Xcompiler=-Wshadow;
-Xcompiler=-Wstrict-aliasing;
-Xcompiler=-Wswitch-enum;
-Xcompiler=-Wvla;
-Xptxas=-v;
>
)

include(CheckCXXCompilerFlag)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(
    tc
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
     -G
     -Xcompiler=-fno-omit-frame-pointer
     -Xcompiler=-fasynchronous-unwind-tables
    >
    $<$<COMPILE_LANGUAGE:CXX>:
     -fno-omit-frame-pointer
     -g
     -fasynchronous-unwind-tables
    >
  )
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(
    tc
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo;
    -Xcompiler=-ffast-math;
    >
    $<$<COMPILE_LANGUAGE:CXX>:
    -ffast-math;
    >
    
  )
  check_cxx_compiler_flag(-march=native CXX_HAS_MARCH)
  check_cxx_compiler_flag(-mcpu=native CXX_HAS_MCPU)
  # prefer march=native, but fallback to mcpu=native if needed
  if (CXX_HAS_MARCH)
    message(STATUS "-march=native detected")
    target_compile_options(
      tc
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
  elseif (CXX_HAS_MCPU)
    message(STATUS "-mcpu=native -mtune=native instead of -march=native")
    target_compile_options(
      tc
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mcpu=native -Xcompiler=-mtune=native>
      $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native -mtune=native>
    )
  endif()
endif()