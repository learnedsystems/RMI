
result: test
	$(shell ./test > stdout) 
	echo $(.SHELLSTATUS) > result
	cat stdout >> result

rmi.cpp:
	../rmi ../wiki_ts_200M_uint64 rmi linear_spline,linear 4194304 --bounded 8

test: main.cpp rmi.cpp
	g++ -std=c++17 -Wall -O3 -ffast-math -march=native main.cpp rmi.cpp -o test

.PHONY: clean
clean:
	rm -rf test result rmi*
