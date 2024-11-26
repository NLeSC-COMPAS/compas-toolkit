cmake_minimum_required(VERSION 3.16)

project(benchmarks)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

#add_executable(simulate_signal simulate_signal.cpp)
#target_link_libraries(simulate_signal PRIVATE compas-toolkit)
#set_target_properties(simulate_signal PROPERTIES CUDA_ARCHITECTURES "70;80")
#target_include_directories(simulate_signal PRIVATE ${CUDA_TOOLKIT_INCLUDE})

#add_executable(simulate_sequence_fisp simulate_sequence_fisp.cpp)
#target_link_libraries(simulate_sequence_fisp PRIVATE compas-toolkit)
#set_target_properties(simulate_sequence_fisp PROPERTIES CUDA_ARCHITECTURES "70;80")
#target_include_directories(simulate_signal PRIVATE ${CUDA_TOOLKIT_INCLUDE})
