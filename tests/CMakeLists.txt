cmake_minimum_required(VERSION 3.16)

project(tests)
set(CMAKE_CXX_STANDARD 17)

file(GLOB FILES *.cpp)
add_executable(compas-toolkit-tests ${FILES})
target_link_libraries(compas-toolkit-tests PRIVATE compas-toolkit)
target_link_libraries(compas-toolkit-tests PRIVATE Catch2::Catch2WithMain)

if(COMPAS_USE_CUDA)
    find_package(CUDA REQUIRED)
    set_target_properties(compas-toolkit-tests PROPERTIES CUDA_ARCHITECTURES "70;80")
    target_include_directories(compas-toolkit-tests PRIVATE ${CUDA_TOOLKIT_INCLUDE})
elseif(COMPAS_USE_HIP)
    set_source_files_properties(${FILES} PROPERTIES LANGUAGE HIP)
endif()
