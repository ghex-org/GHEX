/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#ifndef INCLUDED_GHEX_TL_UCX_ENDPOINT_HPP
#define INCLUDED_GHEX_TL_UCX_ENDPOINT_HPP

#include "./error.hpp"
#include "./address.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace ucx {

                struct endpoint_t
                {
                    using rank_type = int;

                    rank_type m_rank;
                    ucp_ep_h  m_ep;
                    ucp_worker_h m_worker;
                    bool      m_moved = false;

                    endpoint_t() noexcept : m_moved(true) {}
                    endpoint_t(rank_type rank, ucp_worker_h local_worker, const address_t& remote_worker_address)
                        : m_rank(rank)
                        , m_worker{local_worker}
                    {
                        ucp_ep_params_t ep_params;
                        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
                        ep_params.address    = remote_worker_address.get();
                        GHEX_CHECK_UCX_RESULT(
                            ucp_ep_create(local_worker, &ep_params, &(m_ep))
                        );
                    }

                    endpoint_t(const endpoint_t&) = delete;
                    endpoint_t& operator=(const endpoint_t&) = delete;

                    endpoint_t(endpoint_t&& other) noexcept
                    : m_rank(other.m_rank)
                    , m_ep(other.m_ep)
                    , m_worker(other.m_worker)
                    , m_moved(other.m_moved)
                    {
                        other.m_moved = true;
                    }

                    endpoint_t& operator=(endpoint_t&& other) noexcept
                    {
                        destroy();
                        m_ep.~ucp_ep_h();
                        ::new((void*)(&m_ep)) ucp_ep_h{other.m_ep};
                        m_rank = other.m_rank;
                        m_worker= other.m_worker;
                        m_moved = other.m_moved;
                        other.m_moved = true;
                        return *this;
                    }

                    ~endpoint_t() { destroy(); }

                    void destroy()
                    {
                        if (!m_moved){
                            ucs_status_ptr_t ret = ucp_ep_close_nb(m_ep, UCP_EP_CLOSE_MODE_FLUSH);
                            if(UCS_OK == reinterpret_cast<std::uintptr_t>(ret)) return;
                            if(UCS_PTR_IS_ERR(ret)) return;
                            
                            // wait untile the ep is destroyed, free the request
                            while(UCS_OK != ucp_request_check_status(ret))
                                ucp_worker_progress(m_worker);
                            ucp_request_free(ret);
                        }
                    }

                    operator bool() const noexcept { return m_moved; }
                    operator ucp_ep_h() const noexcept { return m_ep; }

                    rank_type rank() const noexcept { return m_rank; }
                    ucp_ep_h& get()       noexcept { return m_ep; }
                    const ucp_ep_h& get() const noexcept { return m_ep; }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_ENDPOINT_HPP */
