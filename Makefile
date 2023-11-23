pretty:
	clang-format --verbose -i src/*/*.h src/*/*.cu src/*/*.cpp src/*/*.cuh tests/*cpp
	clang-format --verbose -i julia-bindings/src/*cpp

all: pretty

.PHONY : pretty
