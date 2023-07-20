pretty:
	clang-format --verbose -i src/*/*.h src/*/*.cu src/*/*.cpp src/*/*.cuh

all: pretty

.PHONY : pretty
