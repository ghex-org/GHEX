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

#include <ghex/config.hpp>
#include <ghex/context.hpp>
#include <ghex/device/id.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

namespace ghex
{
template<typename Arch>
struct arch_traits;

template<>
struct arch_traits<cpu>
{
    static constexpr const char* name = "CPU";

    using device_id_type = int;
    using message_type = context::message_type;

    static device_id_type default_id() { return 0; }

    static device_id_type current_id() { return 0; }

    static message_type make_message(oomph::communicator& c, std::size_t size,
        device_id_type = current_id())
    {
        return c.make_buffer<unsigned char>(size);
    }

    static message_type make_message(oomph::communicator& c, void* ptr, std::size_t size,
        device_id_type = current_id())
    {
        return c.make_buffer<unsigned char>((unsigned char*)ptr, size);
    }
};

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
template<>
struct arch_traits<gpu>
{
    static constexpr const char* name = "GPU";

    using device_id_type = int;
    using message_type = context::message_type;

    static device_id_type default_id() { return 0; }

    static device_id_type current_id() { return device::get_id(); }

    static message_type make_message(oomph::communicator& c, std::size_t size,
        device_id_type index = current_id())
    {
        static_assert(std::is_same<decltype(index), device_id_type>::value,
            "trick to prevent warnings");
        return c.make_device_buffer<unsigned char>(size, index);
    }

    static message_type make_message(oomph::communicator& c, void* device_ptr, std::size_t size,
        device_id_type index = current_id())
    {
        static_assert(std::is_same<decltype(index), device_id_type>::value,
            "trick to prevent warnings");
        return c.make_device_buffer<unsigned char>((unsigned char*)device_ptr, size, index);
    }
};
#endif

} // namespace ghex
