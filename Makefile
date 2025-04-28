pretty:
	clang-format --verbose -i src/*/*.cpp src/*/*.cu src/*/*.cuh
	clang-format --verbose -i include/compas/*/*.cuh include/compas/*/*.h
	clang-format --verbose -i julia-bindings/src/*cpp tests/*cpp benchmarks/*cpp

all: pretty

.PHONY : pretty
