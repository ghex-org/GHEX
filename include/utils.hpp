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

        constexpr int ct_pow(int base, int exp)
        {
            return exp == 0 ? 
                1 :
                base*ct_pow(base, exp-1);
        }

        /*template<std::size_t I = 0, typename Func, typename ...Args>
        typename std::enable_if<I == sizeof...(Args)>::type for_each(std::tuple<Args...>&, Func) {}

        template<std::size_t I = 0, typename Func, typename ...Args>
        typename std::enable_if<(I < sizeof...(Args))>::type for_each(std::tuple<Args...>& t, Func f) {
                f(std::get<I>(t));
                for_each<I+1, Func, Args...>(t, f);
        }*/

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

        /**
         * @brief iterate over tuple type using a functor
         * @tparam Tuple tuple-like type
         * @tparam Func functor with signature void(auto& x), where x is tuple element
         * @param t tuple instance
         * @param f functor instance
         */
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

        /**
         * @brief iterate over two tuple types using a functor
         * @tparam Tuple1 tuple-like type
         * @tparam Tuple2 tuple-like type
         * @tparam Func functor with signature void(auto& x1, auto& x2), where x1, x2 are elements of t1 and t2
         * @param t1 tuple instance
         * @param t2 tuple instance
         * @param f functor instance
         */
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

        /** @brief compile time recursive generation of loop nest
         * @tparam D dimensionality of loop nest
         * @tparam I I==D to start recursion
         * @tparam Layout template meta function which determines order of nesting */
        template<int D, int I, typename Layout=void>
        struct for_loop
        {
            using idx = std::integral_constant<int,D-I>;

            /**
             * @brief generate loop nest
             * @tparam Func functor with signature void(x_0,x_1,...) where x_i are coordinates
             * @tparam Array coordinate vector type 
             * @param f instance of functor
             * @param first start coordinate
             * @param last end coordinate (inclusive)
             */
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

            // implementation details
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
    
        // implementation details
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
                // functor is called with expanded coordinates
                f(x[Is]...);
            }
        };

        // specialization when Layout==gridtools::layout_map
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



        // simpler for loop nest
        template<int D, int I, typename Layout>
        struct for_loop_simple {};

        /** @brief generation of loop nest assuming contiguous memory 
         * @tparam D dimensionality of loop nest
         * @tparam I I==D to start recursion
         * @tparam Args compile time list of integral constants indicating order of loop nest*/
        template<int D, int I, int... Args>
        struct for_loop_simple<D,I,gridtools::layout_map<Args...>>
        {
            using layout_t = gridtools::layout_map<Args...>;
            using idx = std::integral_constant<int, layout_t::template find<D-I>()>;

            /**
             * @brief generate loop nest
             * @tparam Func functor type with signature void(std::size_t, std::size_t)
             * @tparam Array coordinate vector type
             * @tparam Array2 coordinate difference vector type
             * @param f functor instance
             * @param first first coordinate in loop nest
             * @param last last coordinate in loop nest
             * @param extent extent of multi-dimensional array of which [first, last] is a sub-region
             */
            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& extent, Array2&& coordinate_offset) noexcept
            {
                std::size_t offset = 0;
                std::size_t iter = 0;
                for(auto i=first[idx::value]; i<=last[idx::value]; ++i, ++iter)
                {
                    for_loop_simple<D,I-1,layout_t>::apply(
                        std::forward<Func>(f), 
                        std::forward<Array>(first), 
                        std::forward<Array>(last), 
                        std::forward<Array2>(extent), 
                        std::forward<Array2>(coordinate_offset), 
                        offset+i+coordinate_offset[idx::value], 
                        iter);
                }
            }

            // implementation details
            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& extent, Array2&& coordinate_offset, 
                                     std::size_t offset, std::size_t iter) noexcept
            {
                offset *= extent[idx::value];
                iter   *= last[idx::value]-first[idx::value]+1;
                for(auto i=first[idx::value]; i<=last[idx::value]; ++i, ++iter)
                {
                    for_loop_simple<D,I-1,layout_t>::apply(
                        std::forward<Func>(f), 
                        std::forward<Array>(first), 
                        std::forward<Array>(last), 
                        std::forward<Array2>(extent), 
                        std::forward<Array2>(coordinate_offset), 
                        offset+i+coordinate_offset[idx::value], 
                        iter);
                }
            }
        };

        //// implementation details
        //template<int D, int... Args>
        //struct for_loop_simple<D,1,gridtools::layout_map<Args...>>
        //{
        //    using layout_t = gridtools::layout_map<Args...>;
        //    using idx = std::integral_constant<int, layout_t::template find<D-1>()>;

        //    template<typename Func, typename Array, typename Array2>
        //    inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& extent, Array2&& coordinate_offset, 
        //                             std::size_t offset, std::size_t iter) noexcept
        //    {
        //        offset *= extent[idx::value];
        //        iter   *= last[idx::value]-first[idx::value]+1;
        //        f(offset + first[idx::value] + coordinate_offset[idx::value], iter, last[idx::value]-first[idx::value]+1);
        //    }
        //};

        // implementation details
        template<int D, int... Args>
        struct for_loop_simple<D,0,gridtools::layout_map<Args...>>
        {
            template<typename Func, typename Array, typename Array2>
            inline static void apply(Func&& f, Array&&, Array&&, Array2&&, Array2&&, 
                                     std::size_t offset, std::size_t iter) noexcept
            {
                // functor call with two arguments
                // argument 1: offset in global multi-dimensional array
                // argument 2: offset within region defined by [first, last]
                f(offset, iter);
            }
        };

    }   // namespace detail

} // namespace gridtools

#endif /* INCLUDED_UTILS_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix:
