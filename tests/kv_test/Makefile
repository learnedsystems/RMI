# Using bash syntax

CXXFLAGS = -std=c++17 -Wall -O3 -ffast-math
CXXBACKFLAGS = 

GCC_CXXFLAGS = -march=native
GCC_CXXBACKFLAGS = -lstdc++fs
CLANG_CXXFLAGS = 
CLANG_CXXFLAGS = -stdlib=libc++

ifeq ($(CXX),g++)
  CXXFLAGS += $(GCC_CXXFLAGS)
  CXXBACKFLAGS += $(GCC_CXXBACKFLAGS)
else ifeq ($(CXX),clang++)
  CXXFLAGS += $(CLANG_CXXFLAGS)
  CXXBACKFLAGS += $(CLANG_CXXBACKFLAGS)
else
	CXX = $(error Specify C compiler with CXX (g++,clang++))
endif

test: test_rmi_whole test_rmi_mmap

test_rmi_whole: ../rmi
	echo "Running rmi whole"

	rm -rf storage/osm_cellids_200M_uint64_rmi_whole
	mkdir storage/osm_cellids_200M_uint64_rmi_whole
	../rmi data/osm_cellids_200M_uint64 rmi robust_linear,linear 16777216
	mv rmi.cpp rmi.h rmi_data.h rmi_data storage/osm_cellids_200M_uint64_rmi_whole

	$(CXX) $(CXXFLAGS) main.cpp storage/osm_cellids_200M_uint64_rmi_whole/rmi.cpp -I storage/osm_cellids_200M_uint64_rmi_whole -I . -o main_osm_cellids_200M_uint64_rmi_whole $(CXXBACKFLAGS)

	bash ~/reload_nfs.sh
	./main_osm_cellids_200M_uint64_rmi_whole --data_path=data/osm_cellids_200M_uint64 --key_path=data/osm_cellids_200M_uint64_keyset --rmi_data_path=storage/osm_cellids_200M_uint64_rmi_whole/rmi_data --out_path=out_main_osm_cellids_200M_uint64_rmi_whole.txt 2>& 1 | tee log.txt

test_rmi_mmap: ../rmi
	echo "Running rmi mmap"

	rm -rf storage/osm_cellids_200M_uint64_rmi_mmap
	mkdir storage/osm_cellids_200M_uint64_rmi_mmap
	../rmi data/osm_cellids_200M_uint64 rmi robust_linear,linear 16777216 --use-mmap
	mv rmi.cpp rmi.h rmi_data.h rmi_data storage/osm_cellids_200M_uint64_rmi_mmap

	$(CXX) $(CXXFLAGS) main.cpp storage/osm_cellids_200M_uint64_rmi_mmap/rmi.cpp -I storage/osm_cellids_200M_uint64_rmi_mmap -I . -o main_osm_cellids_200M_uint64_rmi_mmap $(CXXBACKFLAGS)
	bash ~/reload_nfs.sh

	./main_osm_cellids_200M_uint64_rmi_mmap --data_path=data/osm_cellids_200M_uint64 --key_path=data/osm_cellids_200M_uint64_keyset --rmi_data_path=storage/osm_cellids_200M_uint64_rmi_mmap/rmi_data --out_path=out_main_osm_cellids_200M_uint64_rmi_mmap.txt 2>& 1 | tee log.txt


result_all: train_all
	(bash result_all.sh) 2>& 1 | tee log.txt

train_all: ../rmi
	(export CXX='$(CXX)'; export CXXFLAGS='$(CXXFLAGS)'; export CXXBACKFLAGS='$(CXXBACKFLAGS)'; bash train_all.sh) 2>& 1 | tee log.txt

.PHONY: clean
clean:
	rm -rf result rmi* stdout
