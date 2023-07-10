file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
add_library(compas-julia SHARED ${sources})

install(TARGETS compas-julia LIBRARY DESTINATION lib)

target_link_libraries(compas-julia compas-toolkit)

execute_process(
        COMMAND julia --eval "using CxxWrap; print(CxxWrap.prefix_path())"
        OUTPUT_VARIABLE CXX_WRAP_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${CXX_WRAP_PATH})

find_package(JlCxx REQUIRED)
target_link_libraries(compas-julia JlCxx::cxxwrap_julia)
