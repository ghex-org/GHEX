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
#ifndef INCLUDED_GHEX_THREADS_NONE_PRIMITIVES_HPP
#define INCLUDED_GHEX_THREADS_NONE_PRIMITIVES_HPP

#include <boost/callable_traits.hpp>

namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace none {
           
                template<typename F>
                using void_return_type = typename std::enable_if<
                    std::is_same<boost::callable_traits::return_type_t<F>,void>::value, 
                    void>::type;

                template<typename F>
                using return_type = typename std::enable_if<
                    !std::is_same<boost::callable_traits::return_type_t<F>,void>::value, 
                    boost::callable_traits::return_type_t<F>>::type;

                struct primitives
                {
                public: // member types
                    using id_type = int;

                    class token
                    {
                    private: // members
                        id_type m_id = 0;
                        
                        friend primitives;

                        token(id_type id) noexcept
                        : m_id(id)
                        {}

                    public: // ctors
                        token() = default;
                        token(const token&) = default;
                        token(token&&) = default;
                        
                        token& operator=(const token&) = default;
                        token& operator=(token&&) = default;

                    public: // member functions
                        id_type id() const noexcept { return m_id; }
                    };

                private: // members

                public: // ctors
                    primitives(int=1) noexcept {} 
                    primitives(const primitives&) = delete;
                    primitives(primitives&&) = delete;

                public: // public member functions
                    inline token get_token() noexcept { return {0}; }
                    
                    int size() const noexcept { return 1; }

                    inline void barrier(token&) const {}

                    template <typename F>
                    inline void single(token&, F && f) const { f(); }

                    template <typename F>
                    inline void master(token&, F && f) const { f(); }

                    template <typename F>
                    inline void_return_type<F> critical(F && f) const { f(); }

                    template <typename F>
                    inline return_type<F> critical(F && f) const { return f(); }
                };

            } // namespace none
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_NONE_PRIMITIVES_HPP */

