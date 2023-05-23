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

#include <ghex/device/attributes.hpp>

namespace ghex
{
namespace rma
{
// chunk of contiguous memory of type T
// used in range iterators for example
template<typename T>
struct chunk
{
    using value_type = T;
    using size_type = unsigned long;

    T*        m_ptr;
    size_type m_size;

    GHEX_FUNCTION
    T* data() const noexcept { return m_ptr; }
    GHEX_FUNCTION
    T operator[](unsigned int i) const noexcept { return m_ptr[i]; }
    GHEX_FUNCTION
    T& operator[](unsigned int i) noexcept { return m_ptr[i]; }
    GHEX_FUNCTION
    size_type size() const noexcept { return m_size; }
    GHEX_FUNCTION
    size_type bytes() const noexcept { return m_size * sizeof(T); }
};

} // namespace rma
} // namespace ghex
