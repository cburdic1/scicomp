cmake_minimum_required(VERSION 3.20)


set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG" CACHE STRING "" FORCE)

project(MountainRange CXX)

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

find_package(Threads REQUIRED)


add_executable(wavesolve_thread wavesolve_thread.cpp WaveOrthotope.hpp)
target_link_libraries(wavesolve_thread PRIVATE Threads::Threads)

add_custom_command(TARGET wavesolve_thread POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:wavesolve_thread>
            ${CMAKE_SOURCE_DIR}/wavesolve_thread)

add_executable(optimize optimize.cpp)
