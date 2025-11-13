cmake_minimum_required(VERSION 3.20)


set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -march=native -mtune=native" CACHE STRING "" FORCE)

project(MountainRange CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()


find_package(OpenMP REQUIRED)


find_package(MPI REQUIRED)
find_package(mpl REQUIRED)




add_executable(optimize optimize.cpp)
target_link_libraries(optimize PRIVATE OpenMP::OpenMP_CXX)


add_executable(wavesolve_openmp wavesolve_openmp.cpp)
target_link_libraries(wavesolve_openmp PRIVATE OpenMP::OpenMP_CXX)

# wavesolve_mpi (MPI version)
add_executable(wavesolve_mpi wavesolve_mpi.cpp)
target_link_libraries(wavesolve_mpi PRIVATE MPI::MPI_CXX mpl::mpl)

