/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <prototype/regular_grid_descriptors.hpp>
#include <iostream>
#include <iomanip>
#include <gridtools/common/layout_map.hpp>
#include <vector>

namespace gt = gridtools;

template <typename ValueType, typename Layout>
class data_t {
public:

    struct range {
        int m_begin;
        int m_end;

        range(int b, int e)
            : m_begin(b)
            , m_end(e)
        {}

        int begin() const {
            return m_begin;
        }

        int end() const {
            return m_end;
        }
    };

    using value_type = ValueType;
    using layout = Layout;

    static constexpr int rank = Layout::masked_length;

    std::array<int, rank> m_sizes;

    template <typename ...Sizes>
    data_t(Sizes... s) : m_sizes{s...} {}

    template <int I>
    int begin() const {
        return 0;
    }

    template <int I>
    int end() const {
        return m_sizes[I];
    }

    template <int I>
    range range_of() const {
        return range(begin<I>(), end<I>());
    }

    template <int I>
    int stride() const {
        int stride = 1;
        for (int i = 0; i < I; ++i) {
            stride *= m_sizes[i];
        }
        return stride;
    }

    template <typename Array>
    ValueType operator()(Array const& indices) const {
        static_assert(std::tuple_size<Array>::value == rank);

        int offset = indices[rank-1];
        int current_stride = 1;
        for (int i = rank-2; i>=0; --i) {
            current_stride *= m_sizes[i+1];
            offset += indices[i] * current_stride;
        }

        return offset;
    }

    template <typename ...Inds>
    ValueType operator()(Inds const... inds) const {
        auto indices = std::array<int, rank>{inds...};
        return this->operator()(indices);
    }
};

int main() {

    using data_type = data_t<int, gt::layout_map<0,1> >;

    data_type data{10,10};

    for (int i0 = 0; i0 < 10; ++i0) {
        for (int i1 = 0; i1 < 10; ++i1) {
            std::cout << std::setw(4) << data(i0,i1) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";


    {   // 1D example
        std::array<gt::halo_sizes, 1> halos = { gt::halo_sizes{ 2, 3 } };

        gt::regular_grid_descriptor< 1 /* number of partitioned dimensions */ > x(halos);

        std::vector<int> container;


        x.pack< gt::partitioned<1> >(data,
                                     [&container](data_type::value_type x) {container.push_back(x);},
                                     gt::direction<1>({-1}));

        std::cout << "\n";
        std::cout << "\n";
        std::for_each(container.begin(), container.end(), [](int x) {std::cout << std::setw(5) << x << " ";});
        std::cout << "\n";
    }

    {
        std::array<gt::halo_sizes, 2> halos = { gt::halo_sizes{ 2, 3 }, { 1, 2 } };

        gt::regular_grid_descriptor< 2 /* number of partitioned dimensions */ > x(halos);

        std::vector<int> container;


        x.pack< gt::partitioned<0, 1> >(data,
                                     [&container](data_type::value_type x) {container.push_back(x);},
                                     gt::direction<2>({-1,-1}));

        x.pack< gt::partitioned<0, 1> >(data,
                                     [&container](data_type::value_type x) {container.push_back(x);},
                                     gt::direction<2>({-1,0}));

        x.pack< gt::partitioned<0, 1> >(data,
                                     [&container](data_type::value_type x) {container.push_back(x);},
                                     gt::direction<2>({-1,1}));


        std::cout << "\n";
        std::cout << "\n";
        std::for_each(container.begin(), container.end(), [](int x) {std::cout << std::setw(5) << x << " ";});
        std::cout << "\n";
    }

    { // unit test for partitioned
        constexpr gt::partitioned<2,3> p;

        static_assert(p.contains(3));
        static_assert(!p.contains(4));
    }

}
