pretty:
	clang-format --verbose -i src/*h src/*cu src/*/*.h

all: pretty

.PHONY : pretty
