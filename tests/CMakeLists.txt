cmake_minimum_required(VERSION 3.16)

project(tests)
set(CMAKE_CXX_STANDARD 17)

file(GLOB FILES *.cpp)
add_executable(compas-toolkit-tests ${FILES})
target_link_libraries(compas-toolkit-tests PRIVATE compas-toolkit)
set_target_properties(compas-toolkit-tests PROPERTIES CUDA_ARCHITECTURES "70;80")
target_link_libraries(compas-toolkit-tests PRIVATE Catch2::Catch2WithMain)

find_package(CUDA REQUIRED)
target_include_directories(compas-toolkit-tests PRIVATE ${CUDA_TOOLKIT_INCLUDE})
