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
#ifndef INCLUDED_GHEX_STRUCTURED_FIELD_UTILS_HPP
#define INCLUDED_GHEX_STRUCTURED_FIELD_UTILS_HPP

#include <gridtools/common/host_device.hpp>
#include <cstddef> 
#include <gridtools/common/array.hpp>

namespace gridtools {

    template <typename T, size_t D>
    GT_FUNCTION
    array<T,D> operator+(array<T,D> a, const array<T,D>& b)
    {
        for (std::size_t i=0u; i<D; ++i) a[i] += b[i];
        return a;
    }
    template <typename T, size_t D, typename U>
    GT_FUNCTION
    array<T,D> operator+(array<T,D> a, const U& scalar)
    {
        for (std::size_t i=0u; i<D; ++i) a[i] += scalar;
        return a;
    }
    template <typename T, size_t D, typename U>
    GT_FUNCTION
    array<T,D> operator+(const U& scalar, array<T,D> a)
    {
        for (std::size_t i=0u; i<D; ++i) a[i] += scalar;
        return a;
    }
    template <typename T, size_t D>
    GT_FUNCTION
    array<T,D> operator-(array<T,D> a, const array<T,D>& b)
    {
        for (std::size_t i=0u; i<D; ++i) a[i] -= b[i];
        return a;
    }
    template <typename T, typename U, size_t D>
    GT_FUNCTION
    typename std::common_type<T, U>::type dot(const array<T,D>& a, const array<U,D>& b)
    {
        typename std::common_type<T, U>::type res = a[0]*b[0];
        for (std::size_t i=1u; i<D; ++i) res+=a[i]*b[i];
        return res;
    }
    template<template<typename, typename> class OutputStream, class CharT, class Traits, typename T, size_t D>
    OutputStream<CharT, Traits>& operator<<(OutputStream<CharT, Traits>& os, const array<T,D>& a)
    {
        os << "{" << a[0];
        for (size_t i = 1; i < D; ++i) os << ", " << a[i];
        os << "}";
        return os;
    }    

    namespace ghex {
    namespace structured {

    namespace detail {

        template<int D, int I>
        struct compute_strides_impl
        {
            template<typename Layout, typename Coordinate, typename Strides>
            GT_FUNCTION
            static void apply(const Coordinate& extents, Strides& strides)
            {
                const auto last_idx = Layout::template find<I>();
                const auto idx      = Layout::template find<I-1>();
                strides[idx]        = strides[last_idx]*extents[last_idx];
                compute_strides_impl<D,I-1>::template apply<Layout>(extents,strides);
            }
        };
        template<int D>
        struct compute_strides_impl<D,0>
        {
            template<typename Layout, typename Coordinate, typename Strides>
            GT_FUNCTION
            static void apply(const Coordinate&, Strides&)
            {
            }
        };
        template<int D>
        struct compute_strides
        {
            template<typename Layout, typename Coordinate, typename Strides>
            GT_FUNCTION
            static void apply(const Coordinate& extents, Strides& strides)
            {
                const auto idx      = Layout::template find<D-1>();
                strides[idx]        = 1;
                compute_strides_impl<D,D-1>::template apply<Layout>(extents,strides);
            }
            template<typename Layout, typename T, typename Coordinate, typename Strides>
            GT_FUNCTION
            static void apply(const Coordinate& extents, Strides& strides, std::size_t padding)
            {
                const auto idx      = Layout::template find<D-1>();
                strides[idx]        = sizeof(T);
                const auto idx2     = Layout::template find<D-2>();
                strides[idx2]       = sizeof(T)*extents[idx] + padding; 
                compute_strides_impl<D,D-2>::template apply<Layout>(extents,strides);
            }
        };
        template<>
        struct compute_strides<1>
        {
            template<typename Layout, typename Coordinate, typename Strides>
            GT_FUNCTION
            static void apply(const Coordinate&, Strides& strides)
            {
                strides[0]        = 1;
            }
            template<typename Layout, typename T, typename Coordinate, typename Strides>
            GT_FUNCTION
            static void apply(const Coordinate&, Strides& strides, std::size_t)
            {
                strides[0]        = sizeof(T);
            }
        };

        template<int D, int K>
        struct compute_coordinate_impl
        {
            template<typename Layout, typename Strides, typename Coordinate, typename I>
            GT_FUNCTION
            static void apply(const Strides& strides, Coordinate& coord, I i)
            {
                const auto idx = Layout::template find<D-(K)>();
                coord[idx]     = i/strides[idx];
                compute_coordinate_impl<D,K-1>::template apply<Layout>(strides, coord, i - coord[idx]*strides[idx]);
            }
        };
        template<int D>
        struct compute_coordinate_impl<D,0>
        {
            template<typename Layout, typename Strides, typename Coordinate, typename I>
            GT_FUNCTION
            static void apply(const Strides&, Coordinate&, I)
            {
            }
        };
        template<int D>
        struct compute_coordinate
        {
            template<typename Layout, typename Strides, typename Coordinate, typename I>
            GT_FUNCTION
            static void apply(const Strides& strides, Coordinate& coord, I i)
            {
                const auto idx = Layout::template find<0>();
                coord[idx]     = i/strides[idx];
                compute_coordinate_impl<D,D-1>::template apply<Layout>(strides, coord, i - coord[idx]*strides[idx]);
            }
        };

    } // namespace detail
    } // namespace structured
    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_FIELD_UTILS_HPP */

