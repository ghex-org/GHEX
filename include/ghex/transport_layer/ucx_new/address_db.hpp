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
#ifndef INCLUDED_GHEX_TL_UCX_ADDRESS_DB_HPP
#define INCLUDED_GHEX_TL_UCX_ADDRESS_DB_HPP

#include <memory>
#include "./endpoint.hpp"

namespace gridtools {
namespace ghex {
namespace tl {
namespace ucx {

struct address_db_t
{
    using rank_type = typename endpoint_t::rank_type;
    using tag_type = typename endpoint_t::tag_type;

    struct iface
    {
        virtual rank_type rank() = 0;
        virtual rank_type size() = 0;
        virtual int est_size() = 0;
        virtual void init(const ucx::address_t&) = 0;
        virtual address_t find(rank_type) = 0;
        virtual ~iface() {}
    };

    template<typename Impl>
    struct impl_t final : public iface
    {
        Impl m_impl;
        impl_t(const Impl& impl) : m_impl{impl} {}
        impl_t(Impl&& impl) : m_impl{std::move(impl)} {}
        rank_type rank() override { return m_impl.rank(); }
        rank_type size() override { return m_impl.size(); }
        int est_size() override { return m_impl.est_size(); }
        void init(const address_t& addr) override { m_impl.init(addr); }
        address_t find(rank_type rank) override { return m_impl.find(rank); }
    };

    std::unique_ptr<iface> m_impl;

    template<typename Impl>
    address_db_t(Impl&& impl)
        : m_impl{std::make_unique<impl_t<std::remove_cv_t<std::remove_reference_t<Impl>>>>(std::forward<Impl>(impl))}{}

    inline rank_type rank() const { return m_impl->rank(); }
    inline rank_type size() const { return m_impl->size(); }
    inline int est_size() const { return m_impl->est_size(); }
    inline void init(const address_t& addr) { m_impl->init(addr); }
    inline address_t find(rank_type rank) { return m_impl->find(rank); }
};

} // namespace ucx
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_ADDRESS_DB_HPP */
