#ifndef INCLUDED_POOL_ALLOCATOR_HPP
#define INCLUDED_POOL_ALLOCATOR_HPP

#include <mpi.h>
#include <set>
#include <algorithm>
#include <string.h>

extern int grank;

namespace ghex {

    namespace allocator {

	template <typename T>
	struct buffer_ptr {
	    T *buffer;
	    buffer_ptr *next;
	    buffer_ptr(): buffer{nullptr}, next{nullptr}
	    {}
	};

        template <typename T, typename BaseAllocator>
        struct pool_allocator {

	    typedef T value_type;

	    BaseAllocator ba;
	    
	    static buffer_ptr<T> *buffers;
	    static buffer_ptr<T> *empty;

            pool_allocator() = default;

	    void initialize(int nb, int size)
	    {
		buffer_ptr<T> *next = nullptr, *temp;
		if(buffers == nullptr){
		    for(int i=0; i<nb; i++){
			temp = new buffer_ptr<T>;
			temp->buffer = new T[size];
			temp->next = next;
			next = temp;
		    }
		    buffers = next;
		}
	    }

            [[nodiscard]] T* allocate(std::size_t n)
            {
		if(buffers){
		    buffer_ptr<T> *temp = buffers;
		    buffers = temp->next;
		    temp->next = empty;
		    empty = temp;
		    return temp->buffer;
		}

		fprintf(stderr, "%s:%d ooops? no free buffers..\n", __FILE__, __LINE__);
		return NULL;
            }

            void deallocate(T* p, std::size_t n)
            {
		if(empty){
		    buffer_ptr<T> *temp = empty;
		    empty = empty->next;
		    temp->next = buffers;
		    temp->buffer = p;
		    buffers = temp;
		    return;
		}

		fprintf(stderr, "%s:%d ooops? no free buffer containers..\n", __FILE__, __LINE__);
		return;
            }
        };
	    
	template <typename T, typename BA>
	buffer_ptr<T> *pool_allocator<T, BA>::buffers = nullptr;

	template <typename T, typename BA>
	buffer_ptr<T> *pool_allocator<T, BA>::empty = nullptr;

    } // namespace allocator
} // namespace ghex

#endif /* INCLUDED_POOL_ALLOCATOR_HPP */
