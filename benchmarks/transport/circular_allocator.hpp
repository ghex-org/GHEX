#ifndef INCLUDED_CIRCULAR_ALLOCATOR_HPP
#define INCLUDED_CIRCULAR_ALLOCATOR_HPP

#include <mpi.h>
#include <set>
#include <algorithm>
#include <string.h>

extern int grank;

namespace ghex {

    namespace allocator {

        template <typename T, typename BaseAllocator>
        struct circular_allocator {

	    typedef T value_type;

	    BaseAllocator ba;
	    
	    static int  n_buffers;
	    static int  position;
	    static int *available;
	    static T  **buffers;

            circular_allocator() = default;

	    void initialize(int nb, int size)
	    {
		if(n_buffers == 0){
		    position = 0;
		    n_buffers = nb;
		    available = new int[nb];
		    buffers = new T*[nb];
		    for(int i=0; i<nb; i++){
			available[i] = 1;
			buffers[i] = new T[size];
		    }
		}
	    }

            [[nodiscard]] T* allocate(std::size_t n)
            {
		/* size threashold: this allocator is only useful for large messages */
		/* that are not sent in-flight by the backend */
		// if(n < 8192) return ba.allocate(n);

		int ip = position;
		int ep = (position-1)%n_buffers;
		
		while(ip != ep){
		    if(available[ip]){
			available[ip] = 0;
			position = (ip+1)%n_buffers;
			// fprintf(stderr, "%d: found free buffer %d\n", grank, ip);
			return buffers[ip];
		    }
		    ip = (ip+1)%n_buffers;
		}
		fprintf(stderr, "ooops? no free buffers..\n");
		return NULL;
            }

            void deallocate(T* p, std::size_t n)
            {
		/* size threashold: this allocator is only useful for large messages */
		/* that are not sent in-flight by the backend */
		// if(n < 8192) { 
		//     ba.deallocate(p, n);
		//     return;
		// }

		/* TODO: sort pointers, use bisection to locate */
		for(int i=0; i<n_buffers; i++){
		    if(buffers[i] == p){
			available[i] = 1;
			position = i;
			return;
		    }
		}
            }
        };

	template <typename T, typename BA>
	int   circular_allocator<T, BA>::n_buffers = 0;
	template <typename T, typename BA>
	int   circular_allocator<T, BA>::position = -1;
	template <typename T, typename BA>
	int * circular_allocator<T, BA>::available = nullptr;
	template <typename T, typename BA>
	T ** circular_allocator<T, BA>::buffers = nullptr;
    } // namespace allocator
} // namespace ghex

#endif /* INCLUDED_CIRCULAR_ALLOCATOR_HPP */
