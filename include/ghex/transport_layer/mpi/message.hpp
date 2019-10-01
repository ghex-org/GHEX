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
#include <cstring>

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
    using byte           = unsigned char;
    using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<byte>;
    using alloc_traits   = std::allocator_traits<allocator_type>;
    using value_type     = typename alloc_traits::value_type; 
    using pointer        = typename alloc_traits::pointer; // pointer could be a fancy pointer, i.e. pointer != byte*
    
    allocator_type m_alloc;
    size_t m_capacity;
    pointer m_payload;
    size_t m_size;

    static constexpr bool can_be_shared = false;

    message(allocator_type alloc)
        : m_alloc{alloc}, m_capacity{0u}, m_payload(nullptr), m_size{0u}
    {}

    /** Constructor that take capacity and allocator. Size is kept to 0
         *
         * @param capacity Capacity
         * @param alloc allocator_type instance
         */
    message(size_t capacity = 0, allocator_type alloc = allocator_type{})
        : m_alloc{alloc}, m_capacity{capacity}, m_payload(nullptr), m_size{0}
    {
        if (m_capacity > 0)
            m_payload = alloc_traits::allocate(m_alloc, m_capacity);
    }

    /** Constructor that take capacity size and allocator.
         * The size elements are un-initialzed. Requires size<=capacity.
         *
         * @param capacity Capacity
         * @param size
         * @param alloc allocator_type instance
         */
    message(size_t capacity, size_t size, allocator_type alloc = allocator_type{})
        : message(capacity, alloc)
    {
        m_size = size;
        assert(m_size <= m_capacity);
    }
    
    ///** Copy constructor only does shallo copy, and it should only be used
    // * to put messages in a container, like std::vector
    // */
    //message(message const& other)
    //    : m_alloc{ alloc_traits::select_on_copy_construction(other.m_alloc) }
    //    , m_capacity{other.m_capacity}, m_payload{other.m_payload}, m_size(other.m_size)
    //{}

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
            alloc_traits::deallocate(m_alloc, m_payload, m_capacity);
        m_payload = nullptr;
    }

    message& operator=(message&& other)
    {
        using propagate_alloc = typename alloc_traits::propagate_on_container_move_assignment;
        if (propagate_alloc::value || m_alloc == other.m_alloc)
        {
            if (m_payload) alloc_traits::deallocate(m_alloc, m_payload, m_capacity);
            if (propagate_alloc::value)
            {
                void* ptr  = &m_alloc;
                m_alloc.~allocator_type();
                new(ptr) allocator_type{std::move(other.m_alloc)};
            }
            m_capacity = other.m_capacity;
            m_payload  = other.m_payload;
            m_size     = other.m_size;
            other.m_capacity = 0;
            other.m_payload  = nullptr;
            other.m_size     = 0;
        }
        else 
        {
            if (m_capacity < other.m_size)
            {
                if (m_payload) alloc_traits::deallocate(m_alloc, m_payload, m_capacity);
                m_payload = alloc_traits::allocate(m_alloc, other.m_size);
                m_capacity = other.m_size;
            }
            byte* dst = m_payload;
            byte* src = other.m_payload;
            std::memcpy(dst, src, other.m_size);
            m_size = other.m_size;
            other.m_size     = 0;
        }
        return *this;
    }

    allocator_type get_allocator() const noexcept { return m_alloc; }

    constexpr bool is_shared() { return can_be_shared; }
    size_t use_count() const { return 1; }

    /** This is the main function used by the communicator to access the
         * message to send or receive data. This is done so that a std::vector
         * could be used as message.
         *
         * @return Pointer to the beginning of the message
         */
    byte* data() const
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
        byte* byte_ptr = m_payload;
        assert(reinterpret_cast<std::uintptr_t>(byte_ptr) % alignof(T) == 0);
        return reinterpret_cast<T *>(byte_ptr);
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
    byte* begin() { return m_payload; }
    byte* end() const { return m_payload + m_size; }

    /** Function to set a message to a new capacity. Size is unchanged */
    void reserve(size_t new_capacity)
    {
        if (new_capacity <= m_capacity)
            return;

        if (m_payload)
            alloc_traits::deallocate(m_alloc, m_payload, m_capacity);
        pointer new_storage = alloc_traits::allocate(m_alloc, new_capacity);
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
    using message_type   = message<Allocator>;
    using byte           = typename message_type::byte;
    using allocator_type = typename message_type::allocator_type;
    using value_type     = typename message_type::value_type; 
    using pointer        = typename message_type::pointer;
    
    //using allocator_type = Allocator;

    std::shared_ptr<message_type> m_s_message;

    static constexpr bool can_be_shared = true;

    shared_message(allocator_type allc = allocator_type{})
        : m_s_message{std::make_shared<message_type>(0u, allc)}
    {
    }

    /** Constructor that take capacity and allocator. Size is kept to 0
         *
         * @param capacity Capacity
         * @param alloc allocator_type instance
         */
    shared_message(size_t capacity, allocator_type allc = allocator_type{})
        : m_s_message{std::make_shared<message_type>(capacity, allc)}
    {
    }

    /** Constructor that take capacity size and allocator.
         * The size elements are un-initialzed. Requires size>=capacity.
         *
         * @param capacity Capacity
         * @param size
         * @param alloc allocator_type instance
         */
    shared_message(size_t capacity, size_t size, allocator_type allc = allocator_type{})
        : m_s_message{std::make_shared<message_type>(capacity, size, allc)}
    {
    }

    /* Showing these to highlight the semantics */
    shared_message(shared_message const &) = default;
    shared_message(shared_message &&) = default;

    shared_message& operator=(shared_message&&) = default;
    shared_message& operator=(const shared_message&) = default;

    void reset() noexcept { m_s_message.reset(); }

    /** This is the main function used by the communicator to access the
         * message to send or receive data. This is done so that a std::vector
         * could be used as message.
         *
         * @return Pointer to the beginning of the message
         */
    auto data() const
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
    auto data() const
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
    auto begin() { return m_s_message->begin(); }
    auto end() const { return m_s_message->end(); }
};
} // namespace mpi
} // namespace ghex
} // namespace gridtools

#endif
