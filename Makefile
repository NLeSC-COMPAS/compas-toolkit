pretty:
	clang-format --verbose -i src/*h src/*cu src/*/*.h julia-bindings/src/*.cpp

all: pretty

.PHONY : pretty
