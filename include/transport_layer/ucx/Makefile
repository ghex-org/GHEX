CC=/home/angainor/work/institution/usit/prace_6ip/install/bin/mpicc
CXX=/home/angainor/work/institution/usit/prace_6ip/install/bin/mpicxx
PMIX_DIR=/home/angainor/work/pmi/install-pmix/
UCX_DIR=/home/angainor/work/institution/usit/prace_6ip/ghex/ucx-1.6.0/install/

FLAGS=-g -DGHEX_DEBUG_LEVEL=2

all: test

clean:
	rm *.o test

pmi.o: pmi.c pmi.h
	$(CC) $(FLAGS) -I$(PMIX_DIR)/include pmi.c -c

test: ucx_thread_multiple.cpp pmi.o communicator.hpp
	$(CXX) $(FLAGS) -I$(UCX_DIR)/include -o $@ $^  -L$(PMIX_DIR)/lib -lpmix -L$(UCX_DIR)/lib -lucp -Wl,-rpath -Wl,$(UCX_DIR)/lib -Wl,-rpath -Wl,$(PMIX_DIR)/lib -fopenmp
