/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_GHEX_TL_SHARED_MESSAGE_BUFFER_HPP
#define INCLUDED_GHEX_TL_SHARED_MESSAGE_BUFFER_HPP

#include <memory>
#include "./message_buffer.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

	    /* A struct that keeps the message with the refcount.
	       The shared message buffer keeps a pointer to a dynamically
	       allocated refcounted_message. That pointer can be shared
	       between many messages. 
	     */
	    template <typename Allocator>
	    struct refcounted_message {

                using message_type      = message_buffer<Allocator>;

		message_type m_message;
		int refcount;

		refcounted_message(size_t capacity, Allocator allc):
		    m_message{std::move(message_type(capacity, allc))}, refcount{1}
		{}
    
		refcounted_message(size_t capacity, size_t size, Allocator allc):
		    m_message{std::move(message_type(capacity, size, allc))}, refcount{1}
		{}

		refcounted_message(message_type&& m):
		    m_message{std::move(m)}, refcount{1}
		{}		
	    };
	    
            /** The raw shared_message_buffer is a copyable version of message_buffer which internally holds a shared ptr to
              * a message_buffer instance and forwards all calls to this shared instance (the interface is identical 
              * to message_buffer. 
	      *
	      * Compared to the standard shared_message_buffer, here we do not use the shared_ptr, but instead we use
	      * a raw C pointer to refcounted_message.
	      */
            template<typename Allocator = std::allocator<unsigned char>>
            class shared_message_buffer
            {
            public: // member types

                using message_type      = message_buffer<Allocator>;
                using byte              = typename message_type::byte;
                using value_type        = typename message_type::value_type;
                using allocation_type   = typename message_type::allocation_type;
                using allocator_type    = typename message_type::allocator_type;
                using pointer           = typename message_type::pointer;
                using const_pointer     = typename message_type::const_pointer;
                using raw_pointer       = typename message_type::raw_pointer;
                using raw_const_pointer = typename message_type::raw_const_pointer;
    
                static constexpr bool can_be_shared = true;

            public: // members

                // std::shared_ptr<message_type> m_message;
		refcounted_message<Allocator> *m_sptr;

            public: // ctors

                template<
                    typename Alloc = Allocator, 
                    typename std::enable_if< std::is_default_constructible<Alloc>::value, int>::type = 0>
                shared_message_buffer(size_t size_, Alloc alloc = Alloc{}){
		    m_sptr = new refcounted_message<Allocator>(size_,alloc);
		}

                template<
                    typename Alloc,
                    typename std::enable_if<!std::is_default_constructible<Alloc>::value, int>::type = 0>
                shared_message_buffer(size_t size_, Alloc alloc) {
		    m_sptr = new refcounted_message<Allocator>(size_,alloc);
		}

		shared_message_buffer(message_type&& m){
		    m_sptr = new refcounted_message(m);
		}

                shared_message_buffer(const shared_message_buffer& other){
		    m_sptr = other.m_sptr;
		    m_sptr->refcount++;
		}
		
                shared_message_buffer(shared_message_buffer&& other){
		    m_sptr = other.m_sptr;
		    other.m_sptr = NULL;	
		}
		
                shared_message_buffer& operator=(const shared_message_buffer& other){
		    m_sptr = other.m_sptr;
		    m_sptr->refcount++;
		    return *this;
		}
		
		shared_message_buffer& operator=(shared_message_buffer&& other){
		    m_sptr = other.m_sptr;
		    other.m_sptr = NULL;
		    return *this;
		}

		~shared_message_buffer(){
		    if(m_sptr){
			if(m_sptr->refcount==0) ERR("inconsistent refcount in shared message");
			m_sptr->refcount--;
			if(m_sptr->refcount==0) {
			    delete m_sptr;
			    m_sptr = NULL;
			}
		    }
		}

            public: // member functions
    
                bool is_shared() const { return use_count() > 1; }
                auto use_count() const { return m_sptr->refcount; }

                std::size_t size() const noexcept { return m_sptr->m_message.size(); }
                std::size_t capacity() const noexcept { return m_sptr->m_message.capacity(); }

                raw_const_pointer data() const noexcept { return m_sptr->m_message.data(); }
                raw_pointer data() noexcept { return m_sptr->m_message.data(); }

                // template <typename T>
                // T* data() { return m_message->template data<T>(); }
                // template <typename T>
                // const T* data() const { return m_message->template data<T>(); }

                raw_const_pointer begin() const noexcept { return m_sptr->m_message.begin(); }
                raw_const_pointer end() const noexcept { return m_sptr->m_message.end(); }
                raw_pointer begin() noexcept { return m_sptr->m_message.begin(); }
                raw_pointer end() noexcept { return m_sptr->m_message.end(); }

                void reserve(std::size_t n) { m_sptr->m_message.reserve(n); }
                void resize(std::size_t n) { m_sptr->m_message.resize(n); }
                void clear() { m_sptr->m_message.clear(); }
                void swap(shared_message_buffer& other) { std::swap(m_sptr, other.m_sptr); }

		/* manually decrease the use count. Needed in the UCX communicator */
		void release(){
		    if(nullptr == m_sptr) return;
		    m_sptr->refcount--;
		    if(m_sptr->refcount==0) delete m_sptr;
		    m_sptr = nullptr;
		}
            };

            template<typename A>
            void swap(shared_message_buffer<A>& a, shared_message_buffer<A>& b)
            {
                a.swap(b);
            }

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_SHARED_MESSAGE_BUFFER_HPP */
