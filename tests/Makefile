
ALL_RESULTS=results/cache_fix_result
ALL_TESTS=cache_fix

.PHONY: all
all: results $(ALL_RESULTS)
	@echo "============== TEST RESULTS ==============="
	python3 report.py

results:
	mkdir results

wiki_ts_200M_uint64:
	wget  https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/SVN8PI -O - | zstd -d > $@

rmi: $(shell find ../src/) $(shell find ../rmi_lib/src/)
	cd .. && cargo build --release
	cp ../target/release/rmi .

results/cache_fix_result: results rmi wiki_ts_200M_uint64 cache_fix
	cd cache_fix && make
	cp cache_fix/result results/cache_fix_result

.PHONY: clean
clean:
	rm -rf results rmi
	for test in $(ALL_TESTS) ; do make -C $$test clean ; done
