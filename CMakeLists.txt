cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "compas-toolkit")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# Enable C++17 support
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Enable PIC
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(PROJECT_CLANG_TIDY clang-tidy)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${PROJECT_CLANG_TIDY}")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart_static)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cublas)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvrtc)

# Installation of library files
set(COMPAS_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib")

string(RANDOM LENGTH 32 COMPAS_VERSION_HASH)
string(TIMESTAMP COMPAS_VERSION_DATE "%Y%m%d")
set(COMPAS_VERSION "${COMPAS_VERSION_DATE}-${COMPAS_VERSION_HASH}")

configure_file(
    ${PROJECT_SOURCE_DIR}/CompasToolkit.jl/src/Constants.jl.input
    ${PROJECT_SOURCE_DIR}/CompasToolkit.jl/src/Constants.jl)
configure_file(
    ${PROJECT_SOURCE_DIR}/julia-bindings/src/constants.h.input
    ${PROJECT_SOURCE_DIR}/julia-bindings/src/constants.h)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(thirdparty/Catch2)
set(KMM_USE_CUDA ON)
set(KMM_STATIC ON)
add_subdirectory(thirdparty/kmm)
target_link_libraries(${PROJECT_NAME} PUBLIC kmm)


add_subdirectory(julia-bindings)
add_subdirectory(tests)
add_subdirectory(benchmarks)
