/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file We describe here the way of representing the regular grids
    (multi-dimensional arrays) could be decsribed in a general way, so
    that a back-end can take this information to pack-unpack or do
    whatever with this data.

    Concepts and Notation:
    ======================

    - A processing-grid (for regural grids) is a $D$-dimensional mesh. A
      node of the mesh is identified by the usual tuple of indices
      $(p_0, ..., p_{D-1})$. The mesh may or not be toroidal in some
      dimension.

    - A gird (in this context) is a $d$-dimensional array of values,
      where $d \ge D$.

    (Lower dimensionalities can be simulated by setting the sizes of
    the extra dimensions to 1)

    - $\eta \in \{-1, 0, 1\}^D$ identifies a neighbor of a mesh-node
      in the computing grid. Given ane identifier of a node $x = (x_0,
      x_1, \ldots, x_{D-1}$, the neighbor identifier is simply
      $x+\eta$.

    - The number od neighors in a D-dimensional processing-grid is
      $3^D-1$ (easy to proove considering the number of possible
      neighbors).

    - The number of principal neighbors (the ones along main
      directions) is $2D$ (same way of proofing as before).

    - The grid has a layout map (we consider only affine layouts so
      far). If $S \in N^d$ is the tuple of strides, then the offset of
      an element is at $\sum_{i=0}^{d-1} S_ix_i$, where $x_i$ are the
      coordinate indices of the elements and the strides are listed in
      the same order as the indices of the coordinates. It is always
      possible to re-order the coordinates and strides to be so that
      the first $D$ indices correspond to the dimensions that are
      partitioned along the processing-grid. We use this convention in
      what follows, but the code will make sure the permutations are
      accounted in the proper way.

    - A dimension descriptor is a tuple of 4 integer values indicating, in
      a specific dimension, the halo in the minus direction, the halo
      in the plus direction, the begin and end of the core region.

    - A grid is then partitioned into $3^D$ regions corresponding to
      the dimension descriptors in each dimension that is
      partitioned. There are $D$ dimensions that are partitioned. The
      regions can be identified using a tuple $\eta \in \{-1, 0,
      1\}^D$, exactly as the neighbors. In fact they identify the
      elements to be shipped to the neighbors. They correspond to the/
      coordinate indices of the neighbors, if we take the central part
      to be $\eta = (0, \ldots, 0) \in Z^D$. The dimensions being
      partitioned are the first $D$ dimensions of the $d$ dimensional
      data.

    - Let us introduce the range concept. A range concept is simply an
      integer interval (we can assume it is semi-open like the typical
      C++ pair of iterators). A range $R = [a, b)$, where $b>=a$. If
      $b=a$ then the range is empty.

    - To indicate a region of a grid we can use ranges, in the same
      way it is done in Matlab or Python. If $A$ is the data of a
      grid, then $A(R_0, R_1, \ldtos, R_{D-1})$. A single index $i$
      can be interpreted ad a range as $[i, i+1)$. A range that takes
      all the elements in one dimension is indicated as $:$.

    - A dimension descriptor $H$ can provide information about the ranges
      used for data to be sent and data to be received. We will focus
      on the data to be sent, which is the interior, but it will
      probably not be noticed in the discussion in this comment.

    - We can index the range of a dimension descriptor using the same
      indices used in the $\eta$ tuples. So that given a dimension
      descriptor $H$, $H^{-1}$, $H^0$, and $H^1$ should have an
      obvious meaning (this is abuse of notation since a dimension
      descriptor is told to be a tuple, but the indices there actually
      are not accessing the elements of the tuple but rather the
      ranges associated with them - in this case for data to be sent).

    - A iteration space is the Cartesian product of ranges.

    - Given the identifier of a neighbor $\eta$, the iteration space
      containing the elements to be sent to that neighbor is specified
      as:

      $$(H_0^{\eta_0}, H_1^{\eta_1}, \ldtos, H_{D-1}^{\eta_{D-1}}, :, \ldots)$$

      Where $H_i$ is the dimension descriptor for dimension $i$. The $:$
      will take all the other dimensions in full until we cover all
      the $d$ dimensions of the grid elements.
 */

#pragma once

#include <gridtools/common/layout_map.hpp>
#include "./dimension_descriptor.hpp"
#include <tuple>
#include <gridtools/meta/filter.hpp>
#include "./halo_range.hpp"
#include <gridtools/meta/utility.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/common/tuple_util.hpp>
#include "./range_loops.hpp"
#include <iostream>

namespace gridtools {

    template <size_t N, class NewVal>
    struct replace_f {
        NewVal const & m_new_val;

        template <class Val, class I, enable_if_t<I::value == N, int> = 0>
        constexpr NewVal operator()(Val const&, I) const { return m_new_val;   }

        template <class Val, class I, enable_if_t<I::value != N, int> = 0>
        constexpr Val operator()(Val const& val, I) const { return val;   }
    };

    template <size_t N,
              class NewVal,
              class Src,
              class Indices = GT_META_CALL(meta::make_indices, (tuple_util::size<Src>, std::tuple))>
    constexpr auto replace(NewVal const &new_val, Src const &src)
        GT_AUTO_RETURN(tuple_util::transform(replace_f<N, NewVal>{new_val}, src, Indices{}));

    struct halo_sizes {
        int m_minus;
        int m_plus;

        constexpr halo_sizes(int m, int p)
            : m_minus(m)
            , m_plus(p)
        {}

        constexpr int minus() const {return m_minus;}
        constexpr int plus() const {return m_plus;}

        template <typename DataRange>
        dimension_descriptor get_dimension_descriptor(DataRange const& data_range) const {
            return {m_minus, m_plus, data_range.begin()+m_minus, data_range.end()-m_plus};
        }
    };

    template <int D>
    struct direction {
        std::array<int, D> m_data;

        constexpr direction(std::array<int, D> data) : m_data(data) {}

        constexpr int operator[](int i) const {return m_data[i];}
    };



    /** List of dimensions of a data that are partitioned */
    template <int ...D>
    struct partitioned {
        std::array<int, sizeof...(D)> m_values = {D...};
        //        using ids = std::tuple<std::integer_constant<int, D>...>;

        constexpr bool contains(int v) const {
            for (int i = 0; i < sizeof...(D); ++i) {
                if (m_values[i] == v) {
                    return true;
                }
            }
            return false;
        }

        constexpr int index_of(int v) const {
            for (int i = 0; i < sizeof...(D); ++i) {
                if (m_values[i] == v) {
                    return i;
                }
            }
            return -1;
        }
    };

    /** The idea here is that the grid-descriptor is a tuple where $D$
        entries are dimension descriptors, while the others are integers
        desctibing uninterrupted dimensions.
    */
    template < unsigned NDims >
    struct regular_grid_descriptor {

        std::array<halo_sizes, NDims> m_halos;

        template <typename Halos>
        regular_grid_descriptor(Halos && halos) : m_halos{std::forward<Halos>(halos)} {}

        template <int Len, typename ...Ts, int I>
        typename std::enable_if<Len!=I, void>::type
        print_ranges(std::tuple<Ts...> const& ranges, std::integral_constant<int, I>) const {
            auto const& range = std::get<I>(ranges);
            std::cout << I << ": " << range.begin() << " -> " << range.end() << "\n";
            print_ranges<Len>(ranges, std::integral_constant<int, I+1>{});
        }

        template <int Len, typename ...Ts, int I>
        typename std::enable_if<Len==I, void>::type
        print_ranges(std::tuple<Ts...> const&, std::integral_constant<int, I>) const {
            std::cout << "\n";
        }

        template < typename Partitioned, typename DataDescriptor>
        auto inner_iteration_space(DataDescriptor const& datadsc, direction<NDims> const& dir) {
            // First create iteration space: iteration space is an array of elements with a begin and an end.
            auto ranges_of_data = make_range_of_data(datadsc, meta::make_integer_sequence<int, DataDescriptor::rank>{});
            // then we substitute the partitioned dimensions with the proper halo ranges.
            auto iteration_space = make_tuple_of_inner_ranges(ranges_of_data, m_halos, Partitioned{}, dir, std::integral_constant<int,0>{});

            print_ranges<std::tuple_size<decltype(iteration_space)>::value>(iteration_space, std::integral_constant<int, 0>{});

            return iteration_space;
        }


        template < typename Partitioned, typename DataDescriptor>
        auto outer_iteration_space(DataDescriptor const& datadsc, direction<NDims> const& dir) {
            // First create iteration space: iteration space is an array of elements with a begin and an end.
            auto ranges_of_data = make_range_of_data(datadsc, meta::make_integer_sequence<int, DataDescriptor::rank>{});
            // then we substitute the partitioned dimensions with the proper halo ranges.
            auto iteration_space = make_tuple_of_outer_ranges(ranges_of_data, m_halos, Partitioned{}, dir, std::integral_constant<int,0>{});

            print_ranges<std::tuple_size<decltype(iteration_space)>::value>(iteration_space, std::integral_constant<int, 0>{});

            return iteration_space;
        }


        template < typename Partitioned, typename Data, typename Function >
        void pack(Data const& data, Function fun, direction<NDims> const& dir) {
            //auto iteration_space = inner_iteration_space<Partitioned>(data, std::forward<direction<NDims>>(dir));
            auto iteration_space = inner_iteration_space<Partitioned>(data, dir);

            // now iteration_space is a tuple of ranges. Now we need to iterate on them and execute the functor.
            range_loop(iteration_space, [&fun,&data](auto const& indices) {fun(data(indices));});

        }

        template < typename Partitioned, typename Data, typename Function >
        void unpack(Data const& data, Function fun, direction<NDims> const& dir) {
            //auto iteration_space = outer_iteration_space<Partitioned>(data, std::forward<direction<NDims>>(dir));
            auto iteration_space = outer_iteration_space<Partitioned>(data, dir);

            // now iteration_space is a tuple of ranges. Now we need to iterate on them and execute the functor.
            range_loop(iteration_space, [&fun,&data](auto const& indices) {fun(data(indices));});

        }

    private:
        template <typename Data, int... Inds>
        auto make_range_of_data(Data const& data, meta::integer_sequence<int, Inds...>) {
            return std::make_tuple(data.template range_of<Inds>()...);
        }

        template <typename DataRanges, typename Halos, int FirstHInd, int... HInds, typename  Direction, int I>
        auto make_tuple_of_inner_ranges(DataRanges const& data_ranges, Halos const& halos, partitioned<FirstHInd, HInds...>, Direction const& dir, std::integral_constant<int, I> ) {
            //std::cout << "make_tuple_of_inner_range: " << std::endl;
            //std::cout << "  inner range = "
            //<< halos[I].get_dimension_descriptor(std::get<FirstHInd>(data_ranges)).inner_range(dir[I]).begin() << ", "
            //<< halos[I].get_dimension_descriptor(std::get<FirstHInd>(data_ranges)).inner_range(dir[I]).end();
            //std::cout << std::endl;
            //std::cout << "  I = " << I << std::endl; 
            return make_tuple_of_inner_ranges
                (replace<FirstHInd>
                 //(halos[I].get_dimension_descriptor(std::get<FirstHInd>(data_ranges)).inner_range(dir[FirstHInd]),
                 (halos[I].get_dimension_descriptor(std::get<FirstHInd>(data_ranges)).inner_range(dir[I]),
                  data_ranges),
                 halos, partitioned<HInds...>{}, dir, std::integral_constant<int, I+1>{});
        }

        template <typename DataRanges, typename Halos, int ...Inds, typename Direction, int I>
        auto make_tuple_of_inner_ranges(DataRanges const& data_ranges, Halos const& halos, partitioned<>, Direction const&, std::integral_constant<int, I> ) {
            return data_ranges;

        }

        template <typename DataRanges, typename Halos, int FirstHInd, int... HInds, typename  Direction, int I>
        auto make_tuple_of_outer_ranges(DataRanges const& data_ranges, Halos const& halos, partitioned<FirstHInd, HInds...>, Direction const& dir, std::integral_constant<int, I> ) {

            return make_tuple_of_outer_ranges
                (replace<FirstHInd>
                 (halos[I].get_dimension_descriptor(std::get<FirstHInd>(data_ranges)).outer_range(dir[FirstHInd]),
                  data_ranges),
                 halos, partitioned<HInds...>{}, dir, std::integral_constant<int, I+1>{});
        }

        template <typename DataRanges, typename Halos, int ...Inds, typename Direction, int I>
        auto make_tuple_of_outer_ranges(DataRanges const& data_ranges, Halos const& halos, partitioned<>, Direction const&, std::integral_constant<int, I> ) {
            return data_ranges;

        }

    };

}
