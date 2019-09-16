CC=mpicc
CXX=mpicxx
#PMIX_DIR=/cluster/projects/nn9999k/marcink/software/pmix/2.2.3/
#PMIX_DIR=/cluster/projects/nn9999k/marcink/software/openmpi/master
PMIX_DIR ?= /usr
UCX_DIR=$(HPCX_UCX_DIR)

FLAGS=-O3 -g -std=c++11

all: test

clean:
	rm *.o test

pmi.o: pmi.c pmi.h
	$(CC) $(FLAGS) -I$(PMIX_DIR)/include pmi.c -c

test: ghex_msg_cb_resubmit.cpp pmi.o
	$(CXX) $(UDEF) $(FLAGS) -I$(UCX_DIR)/include -o $@ $^ -Wl,-rpath -Wl,$(PMIX_DIR)/lib -L$(PMIX_DIR)/lib -lpmix -L$(UCX_DIR)/lib -lucp -Wl,-rpath -Wl,$(UCX_DIR)/lib -Wl,-rpath -Wl,$(PMIX_DIR)/lib -fopenmp

mpitest: mpi_avail_iter.cpp
	$(CXX) $(UDEF) $(FLAGS) -I$(UCX_DIR)/include -o $@ $^ -Wl,-rpath -Wl,$(PMIX_DIR)/lib -L$(PMIX_DIR)/lib -lpmix -L$(UCX_DIR)/lib -lucp -Wl,-rpath -Wl,$(UCX_DIR)/lib -Wl,-rpath -Wl,$(PMIX_DIR)/lib -fopenmp

pmixtest: pmixtest.c
	$(CC) $(FLAGS) -I$(PMIX_DIR)/include -I$(UCX_DIR)/include -o $@ $^ -Wl,-rpath -Wl,$(PMIX_DIR)/lib -L$(PMIX_DIR)/lib -lpmix -L$(UCX_DIR)/lib

