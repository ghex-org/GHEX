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
#ifndef INCLUDED_GHEX_COMMON_TEST_EQ_HPP
#define INCLUDED_GHEX_COMMON_TEST_EQ_HPP

namespace gridtools {
    
    namespace ghex {

        namespace detail {

            // helper template metafunction to test equality of a type with respect to all element types of a tuple
            template<typename Test, typename... Ts>
            struct test_eq_t {};

            template<typename Test, typename T0, typename T1, typename... Ts>
            struct test_eq_t<Test,T0,T1,Ts...> : public 
                std::integral_constant<
                    bool, 
                    std::is_same<Test,T0>::value && test_eq_t<Test,T1,Ts...>::value
                > {};

            template<typename Test, typename T0>
            struct test_eq_t<Test,T0> : public 
                std::integral_constant<bool, std::is_same<Test,T0>::value> {};

        } // namespace detail

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_COMMON_TEST_EQ_HPP */

