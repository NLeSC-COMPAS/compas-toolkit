cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "compas-toolkit")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

file(GLOB_RECURSE sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.cu")
add_library(${PROJECT_NAME} STATIC ${sources} src/operators/epg.cuh)

# Directory where libraries will be placed after building
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/src")

# The CUDA compiler struggles with the "Werror" options, so we need to explicitly forward it to the host compiler
# using `-Xcompiler=-Werror` if we are compiling CUDA code.
set(CXXFLAGS
        $<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
        -Wall -Wextra -Wconversion -Wno-unused-parameter
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v">
)
target_compile_options(${PROJECT_NAME} PRIVATE ${CXXFLAGS})

# Enable C++17 support
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Enable PIC
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(PROJECT_CLANG_TIDY clang-tidy -checks=-*,readability-*,bugprone-*,-readability-magic-numbers,-readability-use-anyofallof,-readability-else-after-return)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${PROJECT_CLANG_TIDY}")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart_static)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cublas)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvrtc)

# Installation of library files
set(COMPAS_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib")
configure_file(
    ${PROJECT_SOURCE_DIR}/CompasToolkit.jl/src/CompasToolkit.input
    ${PROJECT_SOURCE_DIR}/CompasToolkit.jl/src/CompasToolkit.jl)


add_subdirectory(thirdparty/Catch2)
add_subdirectory(julia-bindings)
add_subdirectory(tests)
add_subdirectory(benchmarks)
