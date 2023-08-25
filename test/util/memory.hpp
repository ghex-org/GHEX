/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ghex/config.hpp>
#include <hwmalloc/device.hpp>

namespace ghex
{
namespace test
{
namespace util
{
template<typename T>
struct memory
{
    using value_type = T;
    unsigned int         m_size;
    std::unique_ptr<T[]> m_host_memory;
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    struct deleter
    {
        bool no_device_delete;
        void operator()(T* ptr) const
        {
            if (!no_device_delete) hwmalloc::device_free(ptr);
        }
    };
    std::unique_ptr<T[], deleter> m_device_memory;
#endif

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    memory(unsigned int size_, const T& value = T{}, bool no_device_delete = false)
#else
    memory(unsigned int size_, const T& value = T{}, bool /*no_device_delete*/ = false)
#endif
    : m_size{size_}
    , m_host_memory
    {
        new T[m_size]
    }
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    , m_device_memory((T*)hwmalloc::device_malloc(sizeof(T) * m_size), deleter{no_device_delete})
#endif
    {
        for (unsigned int i = 0; i < m_size; ++i) m_host_memory[i] = value;
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
        clone_to_device();
#endif
    }

    memory(const memory&) = delete;
    memory(memory&&) = default;

    T* data() const { return m_host_memory.get(); }
    T* host_data() const { return m_host_memory.get(); }
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    T* device_data() const { return m_device_memory.get(); }
#endif

    unsigned int size() const { return m_size; }

    const T& operator[](unsigned int i) const { return m_host_memory[i]; }
    T&       operator[](unsigned int i) { return m_host_memory[i]; }

    T* begin() { return m_host_memory.get(); }
    T* end() { return m_host_memory.get() + m_size; }

    const T* begin() const { return m_host_memory.get(); }
    const T* end() const { return m_host_memory.get() + m_size; }

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    void clone_to_device()
    {
        hwmalloc::memcpy_to_device(m_device_memory.get(), m_host_memory.get(), m_size * sizeof(T));
    }
    void clone_to_host()
    {
        hwmalloc::memcpy_to_host(m_host_memory.get(), m_device_memory.get(), m_size * sizeof(T));
    }
#endif
};

} // namespace util
} // namespace test
} // namespace ghex
