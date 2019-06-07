/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef INCLUDED_UTILS_HPP
#define INCLUDED_UTILS_HPP

#include <utility>
#include <tuple>
#include <gridtools/common/layout_map.hpp>


namespace gridtools {

    namespace detail {

        template<std::size_t I = 0, typename Func, typename ...Args>
        typename std::enable_if<I == sizeof...(Args)>::type for_each(std::tuple<Args...>&, Func) {}

        template<std::size_t I = 0, typename Func, typename ...Args>
        typename std::enable_if<(I < sizeof...(Args))>::type for_each(std::tuple<Args...>& t, Func f) {
                f(std::get<I>(t));
                for_each<I+1, Func, Args...>(t, f);
        }

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

        template<typename T0, typename T1>
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
        };

        template<int D, int I, typename Layout=void>
        struct for_loop
        {
            using idx = std::integral_constant<int,D-I>;

            template<typename Func, typename Array>
            inline static void apply(Func&& f, Array&& first, Array&& last) noexcept
            {
                for(auto i=first[idx::value]; i<=last[idx::value]; ++i)
                {
                    std::remove_const_t<std::remove_reference_t<Array>> x{};
                    x[idx::value] = i;
                    for_loop<D,I-1,Layout>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), x);
                }
            }

            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& y) noexcept
            {
                for(auto i=first[idx::value]; i<=last[idx::value]; ++i)
                {
                    std::remove_const_t<std::remove_reference_t<Array2>> x{y};
                    x[idx::value] = i;
                    for_loop<D,I-1,Layout>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), x);
                }
            }
        };

        template<int D,typename Layout>
        struct for_loop<D,0,Layout>
        {
            
            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&&, Array&&, Array2&& x) noexcept
            {
                apply_impl(std::forward<Func>(f), std::forward<Array2>(x), std::make_index_sequence<D>{});
            }

            template<typename Func, typename Array, std::size_t... Is>
            inline static void apply_impl(Func&& f, Array&& x, std::index_sequence<Is...>)
            {
                f(x[Is]...);
            }
        };

        template<int D, int I, int... Args>
        struct for_loop<D,I,gridtools::layout_map<Args...>>
        {
            using layout_t = gridtools::layout_map<Args...>;
            using idx = std::integral_constant<int, layout_t::template find<D-I>()>;

            template<typename Func, typename Array>
            inline static void apply(Func&& f, Array&& first, Array&& last) noexcept
            {
                for(auto i=first[idx::value]; i<=last[idx::value]; ++i)
                {
                    std::remove_const_t<std::remove_reference_t<Array>> x{};
                    x[idx::value] = i;
                    for_loop<D,I-1,layout_t>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), x);
                }
            }

            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& y) noexcept
            {
                for(auto i=first[idx::value]; i<=last[idx::value]; ++i)
                {
                    std::remove_const_t<std::remove_reference_t<Array2>> x{y};
                    x[idx::value] = i;
                    for_loop<D,I-1,layout_t>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), x);
                }
            }

        };

        template<int D, int... Args>
        struct for_loop<D,0,gridtools::layout_map<Args...>> : for_loop<D,0,void> {};

        template<int D, int I, typename Layout>
        struct for_loop_simple {};

        template<int D, int I, int... Args>
        struct for_loop_simple<D,I,gridtools::layout_map<Args...>>
        {
            using layout_t = gridtools::layout_map<Args...>;
            using idx = std::integral_constant<int, layout_t::template find<D-I>()>;

            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& extent) noexcept
            {
                std::size_t offset = 0;
                std::size_t iter = 0;
                for(auto i=first[idx::value]; i<=last[idx::value]; ++i, ++iter)
                {
                    for_loop_simple<D,I-1,layout_t>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), std::forward<Array2>(extent), offset+i, iter);
                }
            }

            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& extent, std::size_t offset, std::size_t iter) noexcept
            {
                offset *= extent[idx::value];
                iter   *= last[idx::value]-first[idx::value]+1;
                for(auto i=first[idx::value]; i<=last[idx::value]; ++i, ++iter)
                {
                    for_loop_simple<D,I-1,layout_t>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), std::forward<Array2>(extent), offset+i, iter);
                }

            }
        };

        template<int D, int... Args>
        struct for_loop_simple<D,0,gridtools::layout_map<Args...>>
        {
            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&&, Array&&, Array2&&, std::size_t offset, std::size_t iter) noexcept
            {
                f(offset, iter);
            }
        };

    }   // namespace detail

} // namespace gridtools

#endif /* INCLUDED_UTILS_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix:
