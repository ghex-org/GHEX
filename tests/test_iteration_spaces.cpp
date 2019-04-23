/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>
#include <type_traits>
#include <prototype/dimension_descriptor.hpp>
#include <prototype/is_multi_arg.hpp>
#include <gridtools/common/numerics.hpp>
#include <gridtools/common/array.hpp>

namespace gridtools {

    namespace _impl {
        template <typename U, int D>
        struct make_array {

            template <typename T, int I>
            struct append_dims {
                using type = typename append_dims<T,I-1>::type[3];
            };

            template <typename T>
            struct append_dims<T, 1> {
                using type = T[3];
            };

            using type = typename append_dims<U, D>::type;
        };

        template <typename T>
        T& access_multi_dim(T& x) {
            return x;
        }

        template <typename T, typename ...Ints>
        typename std::remove_all_extents<T>::type& access_multi_dim(T x[3], int i, Ints ...is) {
            return access_multi_dim(x[i], is...);
        }


        template <int Dims>
        struct neighbor_loop {

            template <typename F, typename T, typename ...Ints>
            static void for_each(F f, T x, Ints ...is) {
                for (int i = 0; i < 3; ++i) {
                    neighbor_loop<Dims-1>::for_each(f, x, is..., i);
                }
            }
        };

        template <>
        struct neighbor_loop<0> {

            template <typename F, typename Arg, bool>
            struct execute;

            template <typename F, typename Arg>
            struct execute< F, Arg, false > {
                template < typename ...Ints >
                static void go(F f, Arg& x, Ints...) {
                    f(x);
                }
            };

            template <typename F, typename Arg>
            struct execute< F, Arg, true > {
                template < typename ...Ints >
                static void go(F f, Arg& x, Ints... is) {
                    f(x, is...);
                }
            };


            template <typename F, typename T, typename ...Ints>
            static void for_each(F f, T x, Ints ...is) {
                using value_type = typename std::remove_all_extents<typename std::remove_pointer<typename std::remove_reference<T>::type>::type>::type&;
                value_type y = access_multi_dim(x, is...);
                execute<F, value_type, is_multi_arg<F, value_type>::type::value>::go(f, y, is...);
            }
        };

    } // namespace _impl


    /** an iteration space is the Cartesian product of ranges

        given a sequence of dimension_descriptors this class provides
        the itaration spaces associated with each neighbor identifier
     */
    template <int Dims>
    class iteration_spaces {
        using m_bounds_type = typename _impl::make_array<array<halo_range, Dims>, Dims >::type;

        m_bounds_type m_inner_bounds;
        m_bounds_type m_outer_bounds;

        struct helper_inner {
            array<dimension_descriptor, Dims> const& m_halos;

            helper_inner(array<dimension_descriptor, Dims> const& halos)
                : m_halos(halos)
            {}

            template <typename ...Ints>
            void operator()(array<halo_range, Dims>& iteration_space, Ints ...ints) const {
                _impl::access_multi_dim(iteration_space, ints...) = {m_halos[ints].inner_range(ints-1) ...};
            }
        };

        void setup_inner(array<dimension_descriptor, Dims> const& halos, m_bounds_type& inner_bounds) {
            _impl::neighbor_loop<3>::for_each(helper_inner(halos), inner_bounds);

        }


        struct helper_outer {
            array<dimension_descriptor, Dims> const& m_halos;

            helper_outer(array<dimension_descriptor, Dims> const& halos)
                : m_halos(halos)
            {}

            template <typename ...Ints>
            void operator()(array<halo_range, Dims>& iteration_space, Ints ...ints) const {
                _impl::access_multi_dim(iteration_space, ints...) = {m_halos[ints].outer_range(ints-1) ...};
            }
        };

        void setup_outer(array<dimension_descriptor, Dims> const& halos, m_bounds_type& outer_bounds) {
            _impl::neighbor_loop<3>::for_each(helper_outer(halos), outer_bounds);

        }

    public:

        iteration_spaces(array<dimension_descriptor, Dims> const& halos)
        {
            setup_inner(halos, m_inner_bounds);
            setup_outer(halos, m_outer_bounds);
            for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
            for (int k=0; k<3; ++k)
                std::cout
                << "["
                << "(" << m_inner_bounds[i][j][k][0].begin() << ", " << m_inner_bounds[i][j][k][0].end() << "), "
                << "(" << m_inner_bounds[i][j][k][1].begin() << ", " << m_inner_bounds[i][j][k][1].end() << "), "
                << "(" << m_inner_bounds[i][j][k][2].begin() << ", " << m_inner_bounds[i][j][k][2].end() << ")] "
                << std::endl;
            std::cout << std::endl;
        }

        template <typename ...Ints>
        halo_range operator()(Ints ...i) const {
            return _impl::access_multi_dim(m_inner_bounds, (i+1)...)[0];
        }
    };
}

namespace gt = gridtools;

int main() {

    constexpr gt::array<gt::dimension_descriptor, 3> halos{
        gt::dimension_descriptor{3,2,5,9},
        {1,3,2,6},
        {2,1,2,8},
    };

    int x[3][3][3];

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                x[i][j][k] = i*i*i+j*j+k;
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                std::cout << x[i][j][k] << " == " << gt::_impl::access_multi_dim(x, i, j ,k) << "\n";
            }
        }
    }

    gt::_impl::neighbor_loop<3>::for_each([](int x) { std::cout << x << " ";}, x);
    std::cout << "\n";

    gt::_impl::neighbor_loop<3>::for_each([](int x, int i, int j, int k) { std::cout << "{("
                                                                                     << i << ", "
                                                                                     << j << ", "
                                                                                     << k << ") -> "
                                                                                     << x << "} ";}, x);
    std::cout << "\n";

    int y[3][3][3];

    gt::_impl::neighbor_loop<3>::for_each([](int& x, int i, int j, int k) { x = i*i*i+j*j+k;}, y);

    bool is_equal = true;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                is_equal && (x[i][j][k] == y[i][j][k]);
            }
        }
    }

    std::cout << "is equal is " << std::boolalpha << is_equal << "\n";

    /*constexpr gt::array<gt::dimension_descriptor, 4> halos4{
        gt::dimension_descriptor{3,2,5,9},
        {1,3,2,6},
        {2,1,2,8},
        {2,1,2,8}
    };*/

    // this is most probably wrong and not used anywhere else
    // do not consider for now!
    gt::iteration_spaces<3> is(halos);

}
