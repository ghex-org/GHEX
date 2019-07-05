#include <cassert>
#include <cstring>
#include <memory>

namespace mpi {

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
    template <typename Allocator = std::allocator<unsigned char> >
    struct message {
        using byte = unsigned char;
        Allocator m_alloc;
        size_t m_capacity;
        byte* m_payload;
        size_t m_size;

        /** Constructor that take capacity and allocator. Size is kept to 0
         *
         * @param capacity Capacity
         * @param alloc Allocator instance
         */
        message(size_t capacity, Allocator alloc = Allocator{})
            : m_alloc{alloc}
            , m_capacity{capacity}
            , m_payload(nullptr)
            , m_size{0}
        {
        if (m_capacity > 0)
                m_payload = m_alloc.allocate(m_capacity);
        }

        /** Constructor that take capacity size and allocator.
         * The size elements are un-initialzed. Requires size>=capacity.
         *
         * @param capacity Capacity
         * @param size
         * @param alloc Allocator instance
         */
        message(size_t capacity, size_t size, Allocator alloc = Allocator{})
            : m_alloc{alloc}
            , m_capacity{capacity}
            , m_payload(nullptr)
            , m_size{size}
        {
            assert(m_size <= m_capacity);
            if (m_capacity > 0)
                m_payload = m_alloc.allocate(m_capacity);
        }

        message(message const&) = delete;
        message(message&& other) {
            m_alloc = std::move(other.m_alloc);
            m_capacity = other.m_capacity;
            m_payload = other.m_payload;
            m_size = other.m_size;
            other.m_capacity = 0;
            other.m_payload = nullptr;
            other.m_size = 0;
        }

        ~message() {
            if (m_payload) m_alloc.deallocate(m_payload, m_capacity);
        }

        /** This is the main function used by the communicator to access the
         * message to send or receive data. This is done so that a std::vector
         * could be used as message.
         *
         * @return Pointer to the beginning of the message
         */
        unsigned char* data() const {
            return m_payload;
        }

        /** This is the main function used by the communicator to access the
         * size of the message to send or receive. This is done so that a std::vector
         * could be used as message.
         *
         * @return current size
         */
        size_t size() const {
            return m_size;
        }


        /** Reset the size of the message to 0 */
        void empty() {
            m_size == 0;
        }

        size_t capacity() const { return m_capacity; }

        /** Simple iterator facility to read the bytes out of the message */
        unsigned char* begin() { return m_payload; }
        unsigned char* end() const { return m_payload + m_size; }

        /** Function to resize a message to a new capacity. Size is unchanged */
        void resize(size_t new_capacity) {
            assert(new_capacity >= m_size);

            byte* new_storage = m_alloc.allocate(new_capacity);
            std::memcpy((void*)new_storage, (void*)m_payload, m_size);

            m_alloc.deallocate(m_payload, m_capacity);
            m_payload = new_storage;
            m_capacity = new_capacity;
        }

        /** Function to add an element of type T at the end of the message.
         * Size will be updated. In debug mode a check is performed to ensure the
         * address where the insertion is done is aligned with for T.
         *
         * @tparam T Type of the value to be added (deduced)
         *
         * @param x Value to be added
         */
        template <typename T>
        void enqueue(T x) {
            if (m_size + sizeof(T) > m_capacity) {
                resize((m_capacity+1)*1.2);
            }
            unsigned char* payload_T = m_payload + m_size;
            *reinterpret_cast<T*>(payload_T) = x;
            m_size += sizeof(T);
        }

        /** Function to access an element of type T at position pos in the message.
         * Size will be updated. In debug mode a check is performed to ensure the
         * in-bound access, anf to check that the address is aligned properly
         * for type T.
         *
         * @tparam T Type of the value to be added (not deduced)
         *
         * @param pos Position (in bytes) in the message
         *
         * @return Reference to an element of type T
         */
        template <typename T>
        T& at(size_t pos /* in bytes */ ) const {
            assert(pos < m_size);
            assert(reinterpret_cast<std::uintptr_t>(m_payload+pos) % alignof(T) == 0);
            return *(reinterpret_cast<T*>(m_payload + pos));
        }
    };

} //namespace mpi
