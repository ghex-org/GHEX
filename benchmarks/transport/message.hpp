/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GHEX_MPI_MESSAGE_HPP
#define GHEX_MPI_MESSAGE_HPP

#include <cassert>
#include <memory>

extern int grank;

namespace gridtools
{
namespace ghex
{
namespace mpi
{

/** message is a class that represents a buffer of bytes. Each transport
     *  layer will need (potentially) a different type of message.
     *
     * A message can be resized.
     *
     * A message is a move-only object.
     *
     * The capacity indicates the size of the allocated storage, while the
     * size indicates the amnount bytes used in the message
     *
     * The intended use is to fill the message with data using enqueue or at<>
     * and then send it, or receive it and then access it.
     *
     * @tparam Allocator Allocator used by the message
     */
template <typename Allocator = std::allocator<unsigned char>>
struct message
{
    using byte = unsigned char;
    Allocator m_alloc;
    size_t m_capacity;
    byte *m_payload;
    size_t m_size;

    static constexpr bool can_be_shared = false;

    /** Constructor that take capacity and allocator. Size is kept to 0
         *
         * @param capacity Capacity
         * @param alloc Allocator instance
         */
    message(size_t capacity = 0, Allocator alloc = Allocator{})
        : m_alloc{alloc}, m_capacity{capacity}, m_payload(nullptr), m_size{0}
    {
        if (m_capacity > 0)
            m_payload = std::allocator_traits<Allocator>::allocate(m_alloc, m_capacity);
    }

    /** Constructor that take capacity size and allocator.
         * The size elements are un-initialzed. Requires size<=capacity.
         *
         * @param capacity Capacity
         * @param size
         * @param alloc Allocator instance
         */
    message(size_t capacity, size_t size, Allocator alloc = Allocator{})
        : message(capacity, alloc)
    {
        m_size = size;
        assert(m_size <= m_capacity);
    }

    /** Copy constructor only does shallo copy, and it should only be used
     * to put messages in a container, like std::vector
     */
    message(message const& other)
        : m_alloc{other.m_alloc}, m_capacity{other.m_capacity}, m_payload{other.m_payload}, m_size(other.m_size)
    {}

    message(message &&other)
        : m_alloc{std::move(other.m_alloc)}, m_capacity{other.m_capacity}, m_payload{other.m_payload}, m_size(other.m_size)
    {
        other.m_capacity = 0;
        other.m_payload = nullptr;
        other.m_size = 0;
    }

    ~message()
    {
	if (m_payload)
	    std::allocator_traits<Allocator>::deallocate(m_alloc, m_payload, m_capacity);
	m_payload = nullptr;
    }

    constexpr bool is_shared() { return can_be_shared; }
    size_t use_count() const { return 1; }

    /** This is the main function used by the communicator to access the
         * message to send or receive data. This is done so that a std::vector
         * could be used as message.
         *
         * @return Pointer to the beginning of the message
         */
    unsigned char *data() const
    {
        return m_payload;
    }

    /** This is the main function used by the communicator to access the
         * message to send or receive data as a C-array of type T
         *
         * @tparam T Type of the value that should the pointer returned point to
         * @return Pointer to the beginning of the message as a T*
         */
    template <typename T>
    T *data() const
    {
        assert(reinterpret_cast<std::uintptr_t>(m_payload) % alignof(T) == 0);
        return reinterpret_cast<T *>(m_payload);
    }

    /** This is the main function used by the communicator to access the
         * size of the message to send or receive. This is done so that a std::vector
         * could be used as message.
         *
         * @return current size
         */
    size_t size() const
    {
        return m_size;
    }

    /** Function to set the size. Condition is that  the new size must be
         * smaller than the capacity (no resize). The main use of this is when the
         * user uses memcpy from .data() pointer and then set the size for sending.
         *
         * @param s New size
        */
    void resize(size_t s)
    {
        assert(s <= m_capacity);
        m_size = s;
    }

    /** Reset the size of the message to 0 */
    void empty()
    {
        m_size = 0;
    }

    /** This function returns the capacity of the message
         *
         * @return current capacity
         */
    size_t capacity() const { return m_capacity; }

    /** Simple iterator facility to read the bytes out of the message */
    unsigned char *begin() { return m_payload; }
    unsigned char *end() const { return m_payload + m_size; }

    /** Function to set a message to a new capacity. Size is unchanged */
    void reserve(size_t new_capacity)
    {
        if (new_capacity <= m_capacity)
            return;

        if (m_payload)
            std::allocator_traits<Allocator>::deallocate(m_alloc, m_payload, m_capacity);
        byte *new_storage = std::allocator_traits<Allocator>::allocate(m_alloc, new_capacity);
        m_payload = new_storage;
        m_capacity = new_capacity;
        m_size = 0;
    }
};

/** shared_message is a class that represents a buffer of bytes. Each transport
     *  layer will need (potentially) a different type of message.
     *
     * A shared_message can be resized.
     *
     * A shared_message can exists in multiple instances that are in fact
     * shared (it's a shared_ptr). The different copies of the shared_message
     * are reference counted, so that there are no lifetime concerns.
     *
     * The capacity indicates the size of the allocated storage, while the
     * size indicates the amnount bytes used in the message
     *
     * The intended use is to fill the message with data using enqueue or at<>
     * and then send it, or receive it and then access it.
     *
     * @tparam Allocator Allocator used by the message
     */
template <typename Allocator = std::allocator<unsigned char>>
struct shared_message
{
    std::shared_ptr<message<Allocator>> m_s_message;

    static constexpr bool can_be_shared = true;

    shared_message() = default;

    /** Constructor that take capacity and allocator. Size is kept to 0
         *
         * @param capacity Capacity
         * @param alloc Allocator instance
         */
    shared_message(size_t capacity, Allocator allc = Allocator{})
        : m_s_message{std::make_shared<message<Allocator>>(capacity, allc)}
    {}

    /** Constructor that take capacity size and allocator.
         * The size elements are un-initialzed. Requires size>=capacity.
         *
         * @param capacity Capacity
         * @param size
         * @param alloc Allocator instance
         */
    shared_message(size_t capacity, size_t size, Allocator allc = Allocator{})
        : m_s_message{std::make_shared<message<Allocator>>(capacity, size, allc)}
    {}

    /* Showing these to highlight the semantics */
    shared_message(shared_message const &) = default;
    shared_message(shared_message &&) = default;

    void operator=(shared_message const &other){
	m_s_message = other.m_s_message;
    }

    void release(){
	m_s_message.reset();
    }

    /** This is the main function used by the communicator to access the
         * message to send or receive data. This is done so that a std::vector
         * could be used as message.
         *
         * @return Pointer to the beginning of the message
         */
    unsigned char *data() const
    {
        return m_s_message->data();
    }

    /** This is the main function used by the communicator to access the
         * message to send or receive data as a C-array of type T
         *
         * @tparam T Type of the value that should the pointer returned point to
         * @return Pointer to the beginning of the message as a T*
         */
    template <typename T>
    T *data() const
    {
        return m_s_message->template data<T>();
    }

    bool is_shared() { return use_count() > 1; }

    /** Returns the number of owners of this shared_message */
    long use_count() const { return m_s_message.use_count(); }

    /** Checks if the message contains a message or if it has beed deallocated */
    bool is_valid() const { return m_s_message; }

    /** This is the main function used by the communicator to access the
         * size of the message to send or receive. This is done so that a std::vector
         * could be used as message.
         *
         * @return current size
         */
    size_t size() const
    {
        return m_s_message->size();
    }

    /** Function to set the size. Condition is that  the new size must be
         * smaller than the capacity (no resize). The main use of this is when the
         * user uses memcpy from .data() pointer and then set the size for sending.
         *
         * @param s New size
        */
    void resize(size_t s)
    {
        m_s_message->resize(s);
    }

    /** Function to set a message to a new capacity. Size is unchanged */
    void reserve(size_t new_capacity)
    {
        m_s_message->reserve(new_capacity);
    }

    /** Reset the size of the message to 0 */
    void empty()
    {
        m_s_message->empty();
    }

    size_t capacity() const { return m_s_message->capacity(); }

    /** Simple iterator facility to read the bytes out of the message */
    unsigned char *begin() { return m_s_message->begin(); }
    unsigned char *end() const { return m_s_message->end(); }
};


template <typename Allocator>
struct refcount_message {
    message<Allocator> msg;
    int refcount;
    
    refcount_message(size_t capacity, Allocator allc):
	msg(std::move(message<Allocator>(capacity, allc))), refcount{1}
    {}
    
    refcount_message(size_t capacity, size_t size, Allocator allc):
	msg(std::move(message<Allocator>(capacity, size, allc))), refcount{1}
    {}

    // ~refcount_message(){
    // 	fprintf(stderr, "refcount_message deleted\n");
    // }
};

template <typename Allocator = std::allocator<unsigned char>>
struct raw_shared_message
{
    refcount_message<Allocator> *m_sptr;

    static constexpr bool can_be_shared = true;

    raw_shared_message() = default;

    raw_shared_message(size_t capacity, Allocator allc = Allocator{})
    {
	m_sptr = new refcount_message<Allocator>(capacity, allc);
    }

    raw_shared_message(size_t capacity, size_t size, Allocator allc = Allocator{})
    {
	m_sptr = new refcount_message<Allocator>(capacity, size, allc);
    }

    raw_shared_message(raw_shared_message &&other){
    	m_sptr = other.m_sptr;
    	other.m_sptr = NULL;	
    }

    void operator=(raw_shared_message const &other){
	m_sptr = other.m_sptr;
	m_sptr->refcount++;
    }

    ~raw_shared_message(){
	if(m_sptr){
	    m_sptr->refcount--;
	    if(m_sptr->refcount==0) delete m_sptr;
	}
    }

    void release(){
	m_sptr->refcount--;
	if(m_sptr->refcount==0) delete m_sptr;
    }

    unsigned char *data() const
    {
        return m_sptr->msg.data();
    }

    bool is_shared() { return m_sptr->refcount > 1; }

    long use_count() const { return m_sptr->refcount; }

    bool is_valid() const { return m_sptr != nullptr; }

    size_t size() const
    {
        return m_sptr->msg.size();
    }

    void resize(size_t s)
    {
	m_sptr->msg.resize(s);
    }

    void reserve(size_t new_capacity)
    {
        m_sptr->msg.reserve(new_capacity);
    }

    void empty()
    {
        m_sptr->msg.empty();
    }

    size_t capacity() const { return m_sptr->msg.capacity(); }

    unsigned char *begin() { return m_sptr->msg.begin(); }
    unsigned char *end() const { return m_sptr->msg.end(); }
};

} // namespace mpi
} // namespace ghex
} // namespace gridtools

#endif
