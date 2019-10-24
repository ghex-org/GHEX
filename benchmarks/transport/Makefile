OMPI_DIR ?= /home/angainor/work/institution/usit/prace_6ip/software/openmpi
PMIX_DIR ?= /home/angainor/work/institution/usit/prace_6ip/software/pmix
UCX_DIR ?= /home/angainor/work/institution/usit/prace_6ip/software/ucx
BOOST_DIR ?= /usr/include

CC=mpicc
CXX=mpicxx

FLAGS=-O3 -g -std=c++17 -I../../include

all: test

clean:
	rm *.o test

pmi.o: pmi.c
	$(CC) $(FLAGS) -I$(PMIX_DIR)/include pmi.c -c

test: ghex_futures.cpp pmi.o
	$(CXX) $(UDEF) $(FLAGS) -I$(BOOST_DIR) -I$(UCX_DIR)/include -o $@ $^ -Wl,-rpath -Wl,$(PMIX_DIR)/lib -L$(PMIX_DIR)/lib -lpmix -L$(UCX_DIR)/lib -lucp -Wl,-rpath -Wl,$(UCX_DIR)/lib -Wl,-rpath -Wl,$(PMIX_DIR)/lib -fopenmp

mpitest: mpi_avail_iter.cpp
	$(CXX) $(UDEF) $(FLAGS) -I$(UCX_DIR)/include -o $@ $^ -Wl,-rpath -Wl,$(PMIX_DIR)/lib -L$(PMIX_DIR)/lib -lpmix -L$(UCX_DIR)/lib -lucp -Wl,-rpath -Wl,$(UCX_DIR)/lib -Wl,-rpath -Wl,$(PMIX_DIR)/lib -fopenmp

pmixtest: pmixtest.c
	$(CC) $(FLAGS) -I$(PMIX_DIR)/include -I$(UCX_DIR)/include -o $@ $^ -Wl,-rpath -Wl,$(PMIX_DIR)/lib -L$(PMIX_DIR)/lib -lpmix -L$(UCX_DIR)/lib

