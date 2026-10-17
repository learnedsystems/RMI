CASES=dense dense32 cubic_line cubic_turn radix_oob radix_marker radix_prefix radix32

result: test
	rm -f stdout
	status=0; for c in $(CASES); do ./test $$c >> stdout 2>&1 || status=1; done; \
	echo $$status > result
	cat stdout >> result

data.stamp: gen_data.py
	python3 gen_data.py
	touch data.stamp

rmi_dense.cpp: ../rmi data.stamp
	../rmi data_dense_uint64 rmi_dense linear_spline,linear 1024

rmi_dense32.cpp: ../rmi data.stamp
	../rmi data_dense_uint32 rmi_dense32 linear_spline,linear 1024

rmi_cubic_line.cpp: ../rmi data.stamp
	../rmi data_line_uint64 rmi_cubic_line cubic,linear 1024

rmi_cubic_turn.cpp: ../rmi data.stamp
	../rmi data_turn_uint64 rmi_cubic_turn cubic,linear 1024

rmi_radix_oob.cpp: ../rmi data.stamp
	../rmi data_radix_uint64 rmi_radix_oob radix18,linear 1024

rmi_radix_marker.cpp: ../rmi data.stamp
	../rmi data_radix_uint64 rmi_radix_marker radix8,linear 1000

rmi_radix_prefix.cpp: ../rmi data.stamp
	../rmi data_prefix_uint64 rmi_radix_prefix radix18,linear 1024

rmi_radix32.cpp: ../rmi data.stamp
	../rmi data_split_uint32 rmi_radix32 radix,linear 1024

RMI_SOURCES=$(patsubst %, rmi_%.cpp, $(CASES))

# AddressSanitizer makes the test fail on out-of-bounds reads that land
# just outside the parameter arrays or in unmapped memory, even when they
# happen to return a usable bound.
test: main.cpp $(RMI_SOURCES)
	# -lstdc++fs is required for ancient G++s
	g++ -std=c++17 -Wall -O3 -ffast-math -march=native -fsanitize=address main.cpp $(RMI_SOURCES) -o test -lstdc++fs

.PHONY: clean
clean:
	rm -rf test result stdout rmi* data*
