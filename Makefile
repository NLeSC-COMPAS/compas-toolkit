pretty:
	clang-format --verbose -i src/*/*.cpp src/*/*.cu src/*/*.cuh
	clang-format --verbose -i include/compas/*/*.cuh include/compas/*/*.h

all: pretty

.PHONY : pretty
