ALL = $(shell find opt/*.cpp | cut -d'.' -f 1 | sed 's/$$/.o/')

inference: $(ALL) to_build.cpp
	g++ -Wall -O3 -ffast-math -march=native to_build.cpp $(ALL)

opt/%.o: opt/%.cpp opt/%_data.h opt/%.h
	g++ -Wall -O3 -ffast-math -march=native $< -c -o $@

.PHONY: clean

clean:
	rm -f a.out opt/*.o
