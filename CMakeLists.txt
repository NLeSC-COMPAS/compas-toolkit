cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "compas_toolkit")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

file(GLOB sources "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(${PROJECT_NAME} STATIC ${sources})

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/src")

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wconversion -Werror -Wno-unused-parameter)

set(PROJECT_CLANG_TIDY clang-tidy -checks=-*,readability-*,bugprone-*,-readability-magic-numbers,-readability-use-anyofallof,-readability-else-after-return)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${PROJECT_CLANG_TIDY}")

# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)
# target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver)
# target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::nvrtc)

