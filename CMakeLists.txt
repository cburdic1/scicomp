cmake_minimum_required(VERSION 3.20)

# FIX: removed -march=native -mtune=native (causes Illegal instruction on m9)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG" CACHE STRING "" FORCE)

project(MountainRange CXX)

# === Primary: Place binaries in build/ ===
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

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

#
# ==== optimize ====
#
add_executable(optimize optimize.cpp)
target_link_libraries(optimize PRIVATE OpenMP::OpenMP_CXX)

# Copy to project root too
add_custom_command(TARGET optimize POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:optimize>
            ${CMAKE_SOURCE_DIR}/optimize)

#
# ==== wavesolve_openmp ====
#
add_executable(wavesolve_openmp wavesolve_openmp.cpp)
target_link_libraries(wavesolve_openmp PRIVATE OpenMP::OpenMP_CXX)

# Copy to project root too
add_custom_command(TARGET wavesolve_openmp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:wavesolve_openmp>
            ${CMAKE_SOURCE_DIR}/wavesolve_openmp)


