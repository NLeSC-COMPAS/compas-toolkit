pretty:
	clang-format --verbose -i include/compas/*/*.h src/*/*.cu src/*/*.cpp src/*/*.cuh tests/*cpp benchmarks/*cpp
	clang-format --verbose -i julia-bindings/src/*cpp

all: pretty

.PHONY : pretty
