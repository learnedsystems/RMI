ALL = $(shell find opt/*.cpp | cut -d'.' -f 1 | sed 's/$$/.o/')
ALL_SHARED = $(shell find opt/*.cpp | cut -d'.' -f 1 | sed 's/$$/.so/')


inference: $(ALL) to_build.cpp
	g++ -Wall -O3 -ffast-math -march=native to_build.cpp $(ALL)

opt/%.o: opt/%.cpp opt/%_data.h opt/%.h
	g++ -Wall -O3 -ffast-math -march=native $< -c -o $@

opt/%.so: opt/%.cpp opt/%_data.h opt/%.h
	g++ -DEXTERN_RMI_LOOKUP -fPIC -Wall -O3 -ffast-math -march=native -shared -o $@ $<

.PHONY: clean objects

objects: $(ALL_SHARED)

clean:
	rm -f a.out opt/*.o opt/*.so
