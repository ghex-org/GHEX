#include <cassert>
#include <cstring>
#include <memory>

namespace mpi {

/** message is a class that represents a buffer of bytes. Each transport
 *  layer will need (potentially) a different type of message
 */
template <typename Allocator = std::allocator<unsigned char> >
struct message {
    using byte = unsigned char;
    Allocator m_alloc;
    size_t m_capacity;
    byte* m_payload;
    size_t m_size;

    message(size_t capacity, Allocator alloc = Allocator{})
        : m_alloc{alloc}
        , m_capacity{capacity}
        , m_payload(nullptr)
        , m_size{0}
    {
       if (m_capacity > 0)
            m_payload = m_alloc.allocate(m_capacity);
    }

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
    message(message&& other) = default;

    ~message() {
        if (m_payload) m_alloc.deallocate(m_payload, m_capacity);
    }

    unsigned char* data() const {
        return m_payload;
    }

    size_t size() const {
        return m_size;
    }

    unsigned char* begin() { return m_payload; }
    unsigned char* end() const { return m_payload + m_size; }

    void resize(size_t new_capacity) {
        assert(new_capacity >= m_size);

        byte* new_storage = m_alloc.allocate(new_capacity);
        std::memcpy((void*)new_storage, (void*)m_payload, m_size);

        m_alloc.deallocate(m_payload, m_capacity);
        m_payload = new_storage;
        m_capacity = new_capacity;
    }

    template <typename T>
    void enqueue(T x) {
        if (m_size + sizeof(T) > m_capacity) {
            resize((m_capacity+1)*1.2);
        }
        unsigned char* payload_T = m_payload + m_size;
        *reinterpret_cast<T*>(payload_T) = x;
        m_size += sizeof(T);
    }

    template <typename T>
    T& at(size_t pos /* in bytes */ ) const {
        assert(pos < m_size);
        assert(reinterpret_cast<std::uintptr_t>(m_payload+pos) % alignof(T) == 0);
        return *(reinterpret_cast<T*>(m_payload + pos));
    }

    void empty() {
        m_size == 0;
    }
};

} //namespace mpi
