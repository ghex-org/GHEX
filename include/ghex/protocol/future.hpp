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
#ifndef INCLUDED_FUTURE_HPP
#define INCLUDED_FUTURE_HPP

#include <utility>

namespace gridtools {

    namespace ghex {

        namespace protocol {

            /** @brief future template for non-blocking communication
             * @tparam H handle tyoe
             * @tparam T data type*/
            template<typename H, typename T>
            struct future_base
            {
                using handle_type = H;
                using value_type  = T;

                future_base(value_type&& data, handle_type&& h) 
                :   m_data(std::move(data)),
                    m_handle(std::move(h))
                {}
                future_base(const future_base&) = delete;
                future_base(future_base&&) = default;
                future_base& operator=(const future_base&) = delete;
                future_base& operator=(future_base&&) = default;

                void wait()
                {
                    m_handle.wait();
                }

                bool test()
                {
                    return (bool)m_handle.test();
                }

                [[nodiscard]] value_type get() noexcept
                {
                    wait(); 
                    return std::move(m_data); 
                }

                value_type m_data;
                handle_type m_handle;
            };

            /** @brief specialization for void */
            template<typename H>
            struct future_base<H,void>
            {
                using handle_type = H;

                future_base(handle_type&& h) 
                :   m_handle(std::move(h))
                {}
                future_base(const future_base&) = delete;
                future_base(future_base&&) = default;
                future_base& operator=(const future_base&) = delete;
                future_base& operator=(future_base&&) = default;

                void wait()
                {
                    m_handle.wait();
                }

                bool test()
                {
                    return (bool)m_handle.test();
                }

                void get() noexcept 
                {
                    wait(); 
                }

                handle_type m_handle;
            };

        } // namespace protocol

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_FUTURE_HPP */

