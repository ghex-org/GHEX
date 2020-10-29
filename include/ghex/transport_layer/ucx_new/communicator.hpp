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
#pragma once

#include "./request.hpp"

namespace gridtools{
namespace ghex {
namespace tl {
namespace ucx {

class communicator
{
public:
    using rank_type = typename state::rank_type;
    using address_type = rank_type;
    using tag_type = typename state::tag_type;
    using message_type = typename state::message_type; 
                    
    template<typename T>
    using future = future_t<T>;
    using request_cb_type = typename state::request_cb;

private: // members
    state* m_state;

public:
    communicator(state* st) noexcept
    : m_state{st}
    {}

    communicator(const communicator&) noexcept = default;
    communicator(communicator&&) noexcept = default;
    communicator& operator=(const communicator&) noexcept = default;
    communicator& operator=(communicator&&) noexcept = default;
    
    rank_type rank() const noexcept { return m_state->rank(); }
    rank_type size() const noexcept { return m_state->size(); }
    address_type address() const noexcept { return rank(); }
    const auto& rank_topology() const noexcept { return m_state->rank_topology(); }
    bool is_local(rank_type r) const noexcept { return m_state->rank_topology().is_local(r); }
    rank_type local_rank() const noexcept { return m_state->rank_topology().local_rank(); }

    template <typename Message>
    [[nodiscard]] future<void> send(const Message &msg, rank_type dst, tag_type tag)
    {
        return {m_state->send(msg,dst,tag)};
    }
    
    template<typename CallBack>
    request_cb_type send(message_type&& msg, rank_type dst, tag_type tag, CallBack&& callback)
    {
        return m_state->send(std::move(msg), dst, tag, std::forward<CallBack>(callback));
    }
                    
    template <typename Message>
    [[nodiscard]] future<void> recv(Message &msg, rank_type src, tag_type tag)
    {
        return {m_state->recv(msg,src,tag)};
    }

    template<typename CallBack>
    request_cb_type recv(message_type&& msg, rank_type src, tag_type tag, CallBack&& callback)
    {
        return m_state->recv(std::move(msg), src, tag, std::forward<CallBack>(callback));
    }
                    
    auto progress() { return m_state->progress(); }
};

} // namespace ucx
} // namespace tl
} // namespace ghex
} // namespace gridtools
