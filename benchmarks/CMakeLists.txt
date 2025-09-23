cmake_minimum_required(VERSION 3.16)

project(benchmarks)
set(CMAKE_CXX_STANDARD 17)

if(COMPAS_USE_CUDA)
    find_package(CUDA REQUIRED)
endif()

function (add_benchmark exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE compas-toolkit)
    target_link_libraries(${exe} PRIVATE kernel_float)

    if(COMPAS_USE_CUDA)
        set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "70;80")
        target_include_directories(${exe} PRIVATE ${CUDA_TOOLKIT_INCLUDE})
    elseif(COMPAS_USE_HIP)
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    endif()
endfunction()

add_benchmark(simulate_signal simulate_signal.cpp)
add_benchmark(compute_gemm compute_gemm.cu)
add_benchmark(compute_jacobian compute_jacobian.cpp)
add_benchmark(compute_jacobian_hermitian compute_jacobian_hermitian.cpp)
#add_benchmark(simulate_sequence_fisp)
