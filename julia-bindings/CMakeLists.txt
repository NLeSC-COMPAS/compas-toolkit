file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
add_library(compas-julia SHARED ${sources})

target_compile_features(compas-julia PRIVATE cxx_std_17)
install(TARGETS compas-julia LIBRARY DESTINATION lib)

target_link_libraries(compas-julia compas-toolkit)
