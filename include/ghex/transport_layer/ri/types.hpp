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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_TYPES_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_TYPES_HPP

#include <stdint.h>

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {

using byte = unsigned char;
using size_type = int_fast64_t;

struct remote_host_ {};
struct remote_device_ {};
struct target_ {};
static constexpr target_        target;
static constexpr remote_host_   host;
static constexpr remote_device_ device;

enum class locality
{
    thread,
    process,
    remote
};

struct chunk
{
    using pointer = byte*;

    pointer const   m_ptr;
    size_type const m_size;

    pointer   data() const noexcept { return m_ptr; }
    size_type size() const noexcept { return m_size; }

    byte& operator[](size_type i) const { return m_ptr[i]; }
};

template<typename Reference>
struct arrow_proxy
{
    Reference  r;
    Reference* operator->() { return &r; }
};


} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_TYPES_HPP */
