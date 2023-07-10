cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "compas-toolkit")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

SET(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")

file(GLOB sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.cu")
add_library(${PROJECT_NAME} STATIC ${sources})

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/src")

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wconversion -Werror -Wno-unused-parameter")
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(PROJECT_CLANG_TIDY clang-tidy -checks=-*,readability-*,bugprone-*,-readability-magic-numbers,-readability-use-anyofallof,-readability-else-after-return)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${PROJECT_CLANG_TIDY}")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::nvrtc)

add_subdirectory(julia-bindings)
