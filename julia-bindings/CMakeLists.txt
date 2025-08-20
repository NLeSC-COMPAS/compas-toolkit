file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
add_library(compas-julia STATIC ${sources})

target_compile_features(compas-julia PRIVATE cxx_std_17)
install(TARGETS compas-julia LIBRARY DESTINATION lib)
if(COMPAS_USE_CUDA)
    set_target_properties(compas-julia PROPERTIES CUDA_ARCHITECTURES "80")
elseif(COMPAS_USE_HIP)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(compas-julia compas-toolkit)
