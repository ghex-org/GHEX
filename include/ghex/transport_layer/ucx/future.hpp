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
#ifndef INCLUDED_GHEX_TL_UCX_FUTURE_HPP
#define INCLUDED_GHEX_TL_UCX_FUTURE_HPP

#include "./request.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace ucx {

                /** @brief future template for non-blocking communication */
                template<typename T>
                struct future
                {
                    using value_type  = T;
                    using handle_type = request;

                    value_type m_data;
                    handle_type m_handle;

                    future(value_type&& data, handle_type&& h) 
                    :   m_data(std::move(data))
                    ,   m_handle(std::move(h))
                    {}
                    future(const future&) = delete;
                    future(future&&) = default;
                    future& operator=(const future&) = delete;
                    future& operator=(future&&) = default;

                    void wait() noexcept
                    {
                        m_handle.wait();
                    }

                    bool test() noexcept
                    {
                        return m_handle.test();
                    }

                    bool ready() noexcept
                    {
                        return m_handle.test();
                    }

                    [[nodiscard]] value_type get()
                    {
                        wait(); 
                        return std::move(m_data); 
                    }

                    /** Cancel the future.
                      * @return True if the request was successfully canceled */
                    bool cancel()
                    {
			return m_handle.cancel();
                    }
                };

                template<>
                struct future<void>
                {
                    using handle_type = request;

                    handle_type m_handle;

                    future() noexcept = default; 
                    future(handle_type&& h) 
                    :   m_handle(std::move(h))
                    {}
                    future(const future&) = delete;
                    future(future&&) = default;
                    future& operator=(const future&) = delete;
                    future& operator=(future&&) = default;

                    void wait() noexcept
                    {
                        m_handle.wait();
                    }

                    bool test() noexcept
                    {
                        return m_handle.test();
                    }

                    bool ready() noexcept
                    {
                        return m_handle.test();
                    }

                    void get()
                    {
                        wait(); 
                    }

                    bool cancel()
                    {
			return m_handle.cancel();
                    }
                };

		template<typename Allocator>
                struct future_cb
                {
                    using handle_type = request_cb<Allocator>;

                    handle_type m_handle;

                    future_cb() {}
                    future_cb(handle_type&& h):
			m_handle(std::move(h))
                    {
			fprintf(stderr, "%s %d m_req %p\n", __FUNCTION__, __LINE__, m_handle.m_req);
		    }

                    future_cb(const future_cb&) = delete;
                    future_cb(future_cb&& other) = default;
                    future_cb& operator=(const future_cb&) = delete;
                    future_cb& operator=(future_cb&& other)
		    {
		    	fprintf(stderr, "%s %d m_req %p\n", __FUNCTION__, __LINE__, other.m_handle.m_req);
		    	m_handle = std::move(other.m_handle);
		    	fprintf(stderr, "%s %d m_req %p\n", __FUNCTION__, __LINE__, m_handle.m_req);
		    }

                    void wait() noexcept
                    {
                        m_handle.wait();
                    }

                    bool test() noexcept
                    {
                        return m_handle.test();
                    }

                    bool ready() noexcept
                    {
                        return m_handle.test();
                    }

                    void get()
                    {
                        wait(); 
                    }

                    bool cancel()
                    {
			return m_handle.cancel();
                    }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_FUTURE_HPP */

