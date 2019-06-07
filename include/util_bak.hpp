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
#ifndef INCLUDED_UTIL_HPP
#define INCLUDED_UTIL_HPP

#include <utility>

namespace gridtools {

    namespace detail {

        template<typename Func>
        void invoke_with_arg(Func&&)
        {}

        template<typename Func, typename Arg0, typename... Args>
        void invoke_with_arg(Func&& f, Arg0&& a0, Args&&... as)
        {
            f(std::forward<Arg0>(a0));
            invoke_with_arg(std::forward<Func>(f), std::forward<Args>(as)...);
        }

        template<typename Tuple, typename Func, std::size_t... Is>
        void for_each_impl(Tuple&& t, Func&& f, std::index_sequence<Is...>)
        {
            using std::get;
            invoke_with_arg(std::forward<Func>(f), get<Is>(std::forward<Tuple>(t))...);
        }

        template<typename Tuple, typename Func>
        void for_each(Tuple&& t, Func&& f)
        {
            using size = std::tuple_size<std::remove_reference_t<Tuple>>;
            for_each_impl(
                std::forward<Tuple>(t),
                std::forward<Func>(f),
                std::make_index_sequence<size::value>()
            );
        }

        template<typename Func>
        void invoke_with_2_args(Func&&)
        {}

        template<typename Func, typename Arg0, typename... Args>
        void invoke_with_2_args(Func&& f, Arg0&& a0, Args&&... as)
        {
            //f(std::forward<decltype(a0.first)>(a0.first), std::forward<decltype(a0.second)>(a0.second));
            f(a0.first, a0.second); //, std::forward<decltype(a0.second)>(a0.second));
            invoke_with_2_args(std::forward<Func>(f), std::forward<Args>(as)...);
        }

        template<typename Tuple1, typename Tuple2, typename Func, std::size_t... Is>
        void for_each_impl(Tuple1&& t1, Tuple2&& t2, Func&& f, std::index_sequence<Is...>)
        {
            using std::get;
            invoke_with_2_args(std::forward<Func>(f), //get<Is>(std::forward<Tuple1>(t1), get<Is>(std::forward<Tuple2>(t2))...);
                    std::make_pair<decltype(get<Is>(t1)), decltype(get<Is>(t2))>(get<Is>(t1), get<Is>(t2))...);
        }

        template<typename Tuple1, typename Tuple2, typename Func>
        void for_each(Tuple1&& t1, Tuple2&& t2, Func&& f)
        {
            using size = std::tuple_size<std::remove_reference_t<Tuple1>>;
            for_each_impl(
                std::forward<Tuple1>(t1),
                std::forward<Tuple2>(t2),
                std::forward<Func>(f),
                std::make_index_sequence<size::value>()
            );
        }



        /*template<typename T0, typename T1>
        struct ct_max
        {
            using type = std::integral_constant<std::size_t, ((T0::value) > (T1::value) ? (T0::value) : (T1::value))>;
        };

        template<typename T0, typename T1>
        struct ct_min
        {
            using type = std::integral_constant<std::size_t, ((T0::value) < (T1::value) ? (T0::value) : (T1::value))>;
        };

        template<template<typename,typename> typename Op, typename... Ts> 
        struct ct_reduce {};

        template<template<typename,typename> typename Op, typename Ta, typename Tb>
        struct ct_reduce<Op, Ta, Tb>
        {
            using type = typename Op<Ta,Tb>::type;
        };

        template<template<typename,typename> typename Op, typename Ta, typename Tb, typename Tc, typename... Ts>
        struct ct_reduce<Op,Ta,Tb,Tc,Ts...>
        {
            using type = typename ct_reduce<Op, typename ct_reduce<Op,Ta,Tb>::type, Tc, Ts...>::type;
        };*/


    }   // namespace detail

} // namespace gridtools

#endif /* INCLUDED_UTIL_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

