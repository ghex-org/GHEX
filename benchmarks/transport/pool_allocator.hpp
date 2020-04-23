/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_POOL_ALLOCATOR_HPP
#define INCLUDED_POOL_ALLOCATOR_HPP

#include <mpi.h>
#include <set>
#include <algorithm>
#include <deque>
#include <string.h>

#include <ghex/transport_layer/ucx/threads.hpp>

namespace ghex {

    namespace allocator {

	template <typename T>
	struct buffer_ptr {
	    T *m_buffer;
	    std::size_t m_size;

	    buffer_ptr() = delete;
	    buffer_ptr(T *p, std::size_t size): 
		m_buffer{p}, m_size{size}
	    {}
	};

	template <typename T>
	static std::vector<std::vector<buffer_ptr<T>>> buffers;

	int thrid;
	DECLARE_THREAD_PRIVATE(thrid)

        template <typename T, typename BaseAllocator>
        struct pool_allocator {

	    typedef T value_type;

	    BaseAllocator m_ba;
	    
            pool_allocator(){
		thrid = GET_THREAD_NUM();
		THREAD_MASTER (){
		    thread_rank_type nthr = GET_NUM_THREADS();
		    if(buffers<T>.size() != nthr){
			buffers<T>.resize(nthr);
		    }
		}
		THREAD_BARRIER();
	    }

            pool_allocator(const pool_allocator &other) :
		m_ba{other.m_ba}
	    {}

	    void initialize(int nb, int size)
	    {
		for(int i=0; i<nb; i++){
		    buffer_ptr<T> container(m_ba.allocate(size), size);
		    memset(container.m_buffer, 0, size);
		    buffers<T>[thrid].push_back(container);
		}
	    }

            [[nodiscard]] T* allocate(std::size_t size)
            {
		if(0 == buffers<T>[thrid].size()){
		    return m_ba.allocate(size);
		} else {
		    buffer_ptr<T> &container = buffers<T>[thrid].back();
		    T *data = container.m_buffer;
		    buffers<T>[thrid].pop_back();
		    return data;
		}
            }

            void deallocate(T* p, std::size_t size)
            {
		buffers<T>[thrid].emplace_back(p, size);
            }

            void release(){
		int size = buffers<T>[thrid].size();
		for(int i=0; i<size; i++){
		    buffer_ptr<T> &container = buffers<T>[thrid].back();
		    m_ba.deallocate(container.m_buffer, container.m_size);
		    buffers<T>[thrid].pop_back();
		}
	    }

        };
    } // namespace allocator
} // namespace ghex

#endif /* INCLUDED_POOL_ALLOCATOR_HPP */
