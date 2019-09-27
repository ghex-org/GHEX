

#include <utility>

/** @brief creates a class whic manages a C struct and lifetime is managed through init and destroy C functions. */
#define GHEX_C_MANAGED_STRUCT(name, c_type, init, destroy)                              \
struct name                                                                             \
{                                                                                       \
    c_type m_struct;                                                                    \
    bool m_moved = false;                                                               \
                                                                                        \
    template<typename... Args>                                                          \
    name(Args&&... args) noexcept                                                       \
    {                                                                                   \
        init(&m_struct, args...);                                                       \
    }                                                                                   \
                                                                                        \
    name(const name&) = delete;                                                         \
                                                                                        \
    name& operator=(const name&) = delete;                                              \
                                                                                        \
    name(name&& other) noexcept                                                         \
    : m_struct{std::move(other.m_struct)}                                               \
    , m_moved{other.m_moved}                                                            \
    {                                                                                   \
        other.m_moved = true;                                                           \
    }                                                                                   \
                                                                                        \
    name& operator=(name&& other) noexcept                                              \
    {                                                                                   \
        if (!m_moved)                                                                   \
            destroy(m_struct);                                                          \
        m_struct.~c_type();                                                             \
        ::new((void*)(&m_struct)) c_type{std::move(other.m_struct)};                    \
        m_moved = other.m_moved;                                                        \
        other.m_moved = true;                                                           \
        return *this;                                                                   \
    }                                                                                   \
                                                                                        \
    ~name()                                                                             \
    {                                                                                   \
        if (!m_moved)                                                                   \
            destroy(m_struct);                                                          \
    }                                                                                   \
                                                                                        \
    operator bool() const noexcept {return !m_moved;}                                   \
                                                                                        \
    operator       c_type&()       noexcept {return m_struct;}                          \
    operator const c_type&() const noexcept {return m_struct;}                          \
                                                                                        \
          c_type& get()       noexcept {return m_struct;}                               \
    const c_type& get() const noexcept {return m_struct;}                               \
                                                                                        \
};
