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
#ifndef INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP
#define INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP

#include <memory>
#include <functional>

namespace gridtools {
namespace ghex {

// type erased bulk communication object
struct bulk_communication_object
{
private:
    struct handle
    {
        std::function<void()> m_wait;
        void wait() { m_wait(); }
    };

    struct bulk_co_iface
    {
        virtual ~bulk_co_iface() {}
        virtual handle exchange() = 0;
    };

    template<typename CO>
    struct bulk_co_impl : public bulk_co_iface
    {
        CO m;
        bulk_co_impl(CO&& co) : m{std::move(co)} {}
        handle exchange() override final { return {std::move(m.exchange().m_wait_fct)}; }
    };

    std::unique_ptr<bulk_co_iface> m_impl;

public:
    bulk_communication_object() = default;
    template<typename CO>
    bulk_communication_object(CO&& co) : m_impl{ std::make_unique<bulk_co_impl<CO>>(std::move(co)) } {}
    bulk_communication_object(bulk_communication_object&&) = default;
    bulk_communication_object& operator=(bulk_communication_object&&) = default;

    handle exchange() { return m_impl->exchange(); }
};

} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP */
