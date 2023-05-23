/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <map>
#include <mutex>

namespace ghex
{
namespace rma
{
namespace cuda
{
/** @brief  A cache for gpu rma pointers. Cuda IPC has a restriction that a memory region can only
  * be attached once per process. Therefore, we need a cache to look this up. */
struct resource_cache
{
    struct impl_t
    {
        std::map<void*, void*> m_ptr_map;
        auto                   find(void* key) { return m_ptr_map.find(key); }
        decltype(auto)         operator[](void* key) { return m_ptr_map[key]; }
        auto                   end() { return m_ptr_map.end(); }
    };
    using lock_type = std::lock_guard<std::mutex>;
    std::mutex            m_mtx;
    std::map<int, impl_t> m_rank_map;
    auto                  find(int key) { return m_rank_map.find(key); }
    decltype(auto)        operator[](int key) { return m_rank_map[key]; }
    auto                  end() { return m_rank_map.end(); }
    auto&                 mtx() { return m_mtx; }
};

// singleton pattern
static inline resource_cache&
get_cache()
{
    static resource_cache inst{};
    return inst;
}

} // namespace cuda
} // namespace rma
} // namespace ghex
