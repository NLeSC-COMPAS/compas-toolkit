file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(compas-julia SHARED ${sources})

target_compile_features(compas-julia PRIVATE cxx_std_17)
install(TARGETS compas-julia LIBRARY DESTINATION lib)

if(COMPAS_USE_CUDA)
    set_target_properties(compas-julia PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
elseif(COMPAS_USE_HIP)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(compas-julia compas-toolkit)
