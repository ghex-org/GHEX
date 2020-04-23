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
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <utility>
#include <thread>
#include <cstring>
#include <gridtools/common/layout_map.hpp>
#include <ghex/arch_list.hpp>
#include <ghex/communication_object.hpp>
#include <ghex/pattern.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/domain_descriptor.hpp>
#include <ghex/transport_layer/mpi/context.hpp>
#include <ghex/threads/atomic/primitives.hpp>
#include "../utils/triplet.hpp"

using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::atomic::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;


/* CPU data descriptor */
template <typename T, typename DomainDescriptor>
class my_data_desc {

    using coordinate_t = typename DomainDescriptor::coordinate_type;
    using layout_map_t = gridtools::layout_map<2, 1, 0>;
    using Byte = unsigned char;

    const DomainDescriptor& m_domain;
    coordinate_t m_halos_offset;
    array<T, layout_map_t> m_values;

public:

    my_data_desc(const DomainDescriptor& domain,
                 const coordinate_t& halos_offset,
                 const array<T, layout_map_t>& values) :
        m_domain{domain},
        m_halos_offset{halos_offset},
        m_values{values} {}

    /** @brief data type size, mandatory*/
    std::size_t data_type_size() const {
        return sizeof (T);
    }

    /** @brief single access set function, not mandatory but used by the corresponding multiple access operator*/
    void set(const T& value, const coordinate_t& coords) {
        m_values(coords[0] + m_halos_offset[0], coords[1] + m_halos_offset[1], coords[2] + m_halos_offset[2]) = value;
    }

    /** @brief single access get function, not mandatory but used by the corresponding multiple access operator*/
    const T& get(const coordinate_t& coords) const {
        return m_values(coords[0] + m_halos_offset[0], coords[1] + m_halos_offset[1], coords[2] + m_halos_offset[2]);
    }

    /** @brief multiple access set function, needed by GHEX in order to perform the unpacking
     * @tparam IterationSpace iteration space type
     * @param is iteration space which to loop through in order to retrieve the coordinates at which to set back the buffer values
     * @param buffer buffer with the data to be set back*/
    template <typename IterationSpace>
    void set(const IterationSpace& is, const Byte* buffer) {
        gridtools::ghex::detail::for_loop<3, 3, layout_map_t>::apply([this, &buffer](auto... indices){
            coordinate_t coords{indices...};
            set(*(reinterpret_cast<const T*>(buffer)), coords);
            buffer += sizeof(T);
        }, is.local().first(), is.local().last());
    }

    /** @brief multiple access get function, needed by GHEX in order to perform the packing
     * @tparam IterationSpace iteration space type
     * @param is iteration space which to loop through in order to retrieve the coordinates at which to get the data
     * @param buffer buffer to be filled*/
    template <typename IterationSpace>
    void get(const IterationSpace& is, Byte* buffer) const {
        gridtools::ghex::detail::for_loop<3, 3, layout_map_t>::apply([this, &buffer](auto... indices){
            coordinate_t coords{indices...};
            const T* tmp_ptr{&get(coords)};
            std::memcpy(buffer, tmp_ptr, sizeof(T));
            buffer += sizeof(T);
        }, is.local().first(), is.local().last());
    }

};


TEST(communication_object, constructor) {

    using domain_descriptor_t = gridtools::ghex::structured::domain_descriptor<int,3>;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = domain_descriptor_t::halo_generator_type; //gridtools::structured::halo_generator<domain_id_t, 3>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    /* Problem sizes */
    const int d1 = 2;
    const int d2 = 2;
    const int d3 = 1;
    const int DIM1 = 10;
    const int DIM2 = 15;
    const int DIM3 = 20;
    const int H1m = 1;
    const int H1p = 1;
    const int H2m = 1;
    const int H2p = 1;
    const int H3m = 1;
    const int H3p = 1;
    const std::array<int, 3> g_first{0, 0, 0};
    const std::array<int, 3> g_last{d1*DIM1-1, d2*DIM2-1, d3*DIM3-1};
    const std::array<int, 6> halos{H1m, H1p, H2m, H2p, H3m, H3p};
    const std::array<bool, 3> periodic{true, true, true};

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        context.world().rank(),
        coordinate_t{(context.world().rank() % d1    ) * DIM1    , (context.world().rank() / d1)     * DIM2    , 0},
        coordinate_t{(context.world().rank() % d1 + 1) * DIM1 - 1, (context.world().rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    using communication_object_t = gridtools::ghex::communication_object<decltype(patterns)::value_type, gridtools::ghex::cpu>;

    auto comm = context.get_communicator(context.get_token());
    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        EXPECT_NO_THROW(
        cos.push_back(communication_object_t{p,comm});
        );
    }

}


TEST(communication_object, exchange) {

    using domain_descriptor_t = gridtools::ghex::structured::domain_descriptor<int, 3>;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = domain_descriptor_t::halo_generator_type; //gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    /* Problem sizes */
    const int d1 = 2;
    const int d2 = 2;
    const int d3 = 1;
    const int DIM1 = 5;
    const int DIM2 = 5;
    const int DIM3 = 5;
    const int H1m = 1;
    const int H1p = 1;
    const int H2m = 1;
    const int H2p = 1;
    const int H3m = 1;
    const int H3p = 1;
    const std::array<int, 3> g_first{0, 0, 0};
    const std::array<int, 3> g_last{d1*DIM1-1, d2*DIM2-1, d3*DIM3-1};
    const std::array<int, 6> halos{H1m, H1p, H2m, H2p, H3m, H3p};
    const std::array<bool, 3> periodic{true, true, true};
    int coords[3]{context.world().rank() % d1, context.world().rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        context.world().rank(),
        coordinate_t{(context.world().rank() % d1    ) * DIM1    , (context.world().rank() / d1)     * DIM2    , 0},
        coordinate_t{(context.world().rank() % d1 + 1) * DIM1 - 1, (context.world().rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    using communication_object_t = gridtools::ghex::communication_object<decltype(patterns)::value_type, gridtools::ghex::cpu>;

    auto comm = context.get_communicator(context.get_token());
    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p,comm});
    }

    triple_t<USE_DOUBLE, double>* _values_1 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_1(_values_1, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, domain_descriptor_t> data_1{
        local_domains[0],
        coordinate_t{H1m, H2m, H3m},
        values_1
    };

    /* Just an initialization */
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * coords[0],
                            jj - H2m + DIM2 * coords[1],
                            kk - H3m + DIM3 * coords[2]
                        );
            }

    auto h = cos[0].exchange(data_1);
    h.wait();

    int passed = true;

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, double> t1;
                int t1x, t1y, t1z;

                t1x = modulus(ii - H1m + DIM1 * coords[0], DIM1 * 2);
                t1y = modulus(jj - H2m + DIM2 * coords[1], DIM2 * 2);
                t1z = modulus(kk - H3m + DIM3 * coords[2], DIM3);

                t1 = triple_t<USE_DOUBLE, double>(t1x, t1y, t1z).floor();

                if (values_1(ii, jj, kk) != t1) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1 " << values_1(ii, jj, kk) << " != " << t1 << "\n";

                }

            }

    EXPECT_TRUE(passed);

}


TEST(communication_object, exchange_asymmetric_halos) {

    using domain_descriptor_t = gridtools::ghex::structured::domain_descriptor<int, 3>;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = domain_descriptor_t::halo_generator_type; //gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    /* Problem sizes */
    const int d1 = 2;
    const int d2 = 2;
    const int d3 = 1;
    const int DIM1 = 5;
    const int DIM2 = 10;
    const int DIM3 = 15;
    const int H1m = 0;
    const int H1p = 1;
    const int H2m = 2;
    const int H2p = 3;
    const int H3m = 2;
    const int H3p = 1;
    const std::array<int, 3> g_first{0, 0, 0};
    const std::array<int, 3> g_last{d1*DIM1-1, d2*DIM2-1, d3*DIM3-1};
    const std::array<int, 6> halos{H1m, H1p, H2m, H2p, H3m, H3p};
    const std::array<bool, 3> periodic{true, true, true};
    int coords[3]{context.world().rank() % d1, context.world().rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        context.world().rank(),
        coordinate_t{(context.world().rank() % d1    ) * DIM1    , (context.world().rank() / d1)     * DIM2    , 0},
        coordinate_t{(context.world().rank() % d1 + 1) * DIM1 - 1, (context.world().rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    using communication_object_t = gridtools::ghex::communication_object<decltype(patterns)::value_type, gridtools::ghex::cpu>;

    auto comm = context.get_communicator(context.get_token());
    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p,comm});
    }

    triple_t<USE_DOUBLE, double>* _values_1 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_1(_values_1, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, domain_descriptor_t> data_1{
        local_domains[0],
        coordinate_t{H1m, H2m, H3m},
        values_1
    };

    /* Just an initialization */
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * coords[0],
                            jj - H2m + DIM2 * coords[1],
                            kk - H3m + DIM3 * coords[2]
                        );
            }

    auto h = cos[0].exchange(data_1);
    h.wait();

    int passed = true;

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, double> t1;
                int t1x, t1y, t1z;

                t1x = modulus(ii - H1m + DIM1 * coords[0], DIM1 * 2);
                t1y = modulus(jj - H2m + DIM2 * coords[1], DIM2 * 2);
                t1z = modulus(kk - H3m + DIM3 * coords[2], DIM3);

                t1 = triple_t<USE_DOUBLE, double>(t1x, t1y, t1z).floor();

                if (values_1(ii, jj, kk) != t1) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1 " << values_1(ii, jj, kk) << " != " << t1 << "\n";

                }

            }

    EXPECT_TRUE(passed);

}


TEST(communication_object, exchange_multiple_fields) {

    using domain_descriptor_t = gridtools::ghex::structured::domain_descriptor<int, 3>;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = domain_descriptor_t::halo_generator_type; //gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    /* Problem sizes */
    const int d1 = 2;
    const int d2 = 2;
    const int d3 = 1;
    const int DIM1 = 5;
    const int DIM2 = 5;
    const int DIM3 = 5;
    const int H1m = 1;
    const int H1p = 1;
    const int H2m = 1;
    const int H2p = 1;
    const int H3m = 1;
    const int H3p = 1;
    const std::array<int, 3> g_first{0, 0, 0};
    const std::array<int, 3> g_last{d1*DIM1-1, d2*DIM2-1, d3*DIM3-1};
    const std::array<int, 6> halos{H1m, H1p, H2m, H2p, H3m, H3p};
    const std::array<bool, 3> periodic{true, true, true};
    const int add = 1;
    int coords[3]{context.world().rank() % d1, context.world().rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        context.world().rank(),
        coordinate_t{(context.world().rank() % d1    ) * DIM1    , (context.world().rank() / d1)     * DIM2    , 0},
        coordinate_t{(context.world().rank() % d1 + 1) * DIM1 - 1, (context.world().rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    using communication_object_t = gridtools::ghex::communication_object<decltype(patterns)::value_type, gridtools::ghex::cpu>;

    auto comm = context.get_communicator(context.get_token());
    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p,comm});
    }

    triple_t<USE_DOUBLE, int>* _values_1 = new triple_t<USE_DOUBLE, int>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, int>, layout_map_type> values_1(_values_1, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, int>, domain_descriptor_t> data_1{
        local_domains[0],
        coordinate_t{H1m, H2m, H3m},
        values_1
    };

    triple_t<USE_DOUBLE, double>* _values_2 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_2(_values_2, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, domain_descriptor_t> data_2{
        local_domains[0],
        coordinate_t{H1m, H2m, H3m},
        values_2
    };

    /* Just an initialization */
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>();
                values_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>(
                            ii - H1m + DIM1 * coords[0],
                            jj - H2m + DIM2 * coords[1],
                            kk - H3m + DIM3 * coords[2]
                        );
                values_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * coords[0] + add,
                            jj - H2m + DIM2 * coords[1] + add,
                            kk - H3m + DIM3 * coords[2] + add
                        );
            }

    auto h = cos[0].exchange(data_1, data_2);
    h.wait();

    int passed = true;

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, int> t1;
                int tx1, ty1, tz1;

                tx1 = modulus(ii - H1m + DIM1 * coords[0], DIM1 * 2);
                ty1 = modulus(jj - H2m + DIM2 * coords[1], DIM2 * 2);
                tz1 = modulus(kk - H3m + DIM3 * coords[2], DIM3);

                t1 = triple_t<USE_DOUBLE, int>(tx1, ty1, tz1).floor();

                if (values_1(ii, jj, kk) != t1) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1 " << values_1(ii, jj, kk) << " != " << t1 << "\n";

                }

                triple_t<USE_DOUBLE, double> t2;
                int tx2, ty2, tz2;

                tx2 = modulus(ii - H1m + DIM1 * coords[0], DIM1 * 2) + add;
                ty2 = modulus(jj - H2m + DIM2 * coords[1], DIM2 * 2) + add;
                tz2 = modulus(kk - H3m + DIM3 * coords[2], DIM3)     + add;

                t2 = triple_t<USE_DOUBLE, double>(tx2, ty2, tz2).floor();

                if (values_2(ii, jj, kk) != t2) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_2 " << values_2(ii, jj, kk) << " != " << t2 << "\n";

                }

            }

    EXPECT_TRUE(passed);

}


TEST(communication_object, multithreading) {

    using domain_descriptor_t = gridtools::ghex::structured::domain_descriptor<int, 3>;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = domain_descriptor_t::halo_generator_type; //gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(2, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    /* Problem sizes */
    const int d1 = 2;
    const int d2 = 2;
    const int d3 = 1;
    const int DIM1 = 5;
    const int DIM2 = 5;
    const int DIM3 = 5;
    const int H1m = 1;
    const int H1p = 1;
    const int H2m = 1;
    const int H2p = 1;
    const int H3m = 1;
    const int H3p = 1;
    const std::array<int, 3> g_first{0, 0, 0};
    const std::array<int, 3> g_last{d1 * DIM1 * 2 - 1, d2 * DIM2 - 1, d3 * DIM3 - 1};
    const std::array<int, 6> halos{H1m, H1p, H2m, H2p, H3m, H3p};
    const std::array<bool, 3> periodic{true, true, true};
    int coords[3]{context.world().rank() % d1, context.world().rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        context.world().rank() * 2,
        coordinate_t{(context.world().rank() % d1) * DIM1 * 2               , (context.world().rank() / d1    ) * DIM2    , 0     },
        coordinate_t{(context.world().rank() % d1) * DIM1 * 2 + DIM1 - 1    , (context.world().rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    domain_descriptor_t my_domain_2{
        context.world().rank() * 2 + 1,
        coordinate_t{(context.world().rank() % d1) * DIM1 * 2 + DIM1        , (context.world().rank() / d1    ) * DIM2    , 0     },
        coordinate_t{(context.world().rank() % d1) * DIM1 * 2 + DIM1 * 2 - 1, (context.world().rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_2);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    using communication_object_t = gridtools::ghex::communication_object<decltype(patterns)::value_type, gridtools::ghex::cpu>;

    //auto comm = context.get_communicator(context.get_token());
    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        //cos.push_back(communication_object_t{p,comm});
        cos.push_back(communication_object_t{p,context.get_communicator(context.get_token())});
    }

    triple_t<USE_DOUBLE, double>* _values_1 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_1(_values_1, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, domain_descriptor_t> data_1{
        local_domains[0],
        coordinate_t{H1m, H2m, H3m},
        values_1
    };

    triple_t<USE_DOUBLE, double>* _values_2 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_2(_values_2, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, domain_descriptor_t> data_2{
        local_domains[1],
        coordinate_t{H1m, H2m, H3m},
        values_2
    };

    /* Just an initialization */
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
                values_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * 2 * coords[0],
                            jj - H2m + DIM2     * coords[1],
                            kk - H3m + DIM3     * coords[2]
                        );
                values_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * 2 * coords[0] + DIM1,
                            jj - H2m + DIM2     * coords[1]       ,
                            kk - H3m + DIM3     * coords[2]
                        );
            }

    std::vector<std::thread> threads;

    threads.push_back(std::thread([&cos, &data_1](){
        auto h1 = cos[0].exchange(data_1);
        h1.wait();
    }));

    threads.push_back(std::thread([&cos, &data_2](){
        auto h2 = cos[1].exchange(data_2);
        h2.wait();
    }));

    for (auto& t : threads) {
        t.join();
    }

    int passed = true;

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, double> t1;
                int tx1, ty1, tz1;

                tx1 = modulus(ii - H1m + DIM1 * 2 * coords[0], DIM1 * 2 * 2);
                ty1 = modulus(jj - H2m + DIM2     * coords[1], DIM2     * 2);
                tz1 = modulus(kk - H3m + DIM3     * coords[2], DIM3        );

                t1 = triple_t<USE_DOUBLE, double>(tx1, ty1, tz1).floor();

                if (values_1(ii, jj, kk) != t1) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1 " << values_1(ii, jj, kk) << " != " << t1 << "\n";

                }

                triple_t<USE_DOUBLE, double> t2;
                int tx2, ty2, tz2;

                tx2 = modulus(ii - H1m + DIM1 * 2 * coords[0] + DIM1, DIM1 * 2 * 2);
                ty2 = modulus(jj - H2m + DIM2     * coords[1]       , DIM2     * 2);
                tz2 = modulus(kk - H3m + DIM3     * coords[2]       , DIM3        );

                t2 = triple_t<USE_DOUBLE, double>(tx2, ty2, tz2).floor();

                if (values_2(ii, jj, kk) != t2) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_2 " << values_2(ii, jj, kk) << " != " << t2 << "\n";

                }


            }

    EXPECT_TRUE(passed);

}


TEST(communication_object, multithreading_multiple_fileds) {

    using domain_descriptor_t = gridtools::ghex::structured::domain_descriptor<int, 3>;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = domain_descriptor_t::halo_generator_type; //gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(2, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    /* Problem sizes */
    const int d1 = 2;
    const int d2 = 2;
    const int d3 = 1;
    const int DIM1 = 5;
    const int DIM2 = 5;
    const int DIM3 = 5;
    const int H1m = 1;
    const int H1p = 1;
    const int H2m = 1;
    const int H2p = 1;
    const int H3m = 1;
    const int H3p = 1;
    const std::array<int, 3> g_first{0, 0, 0};
    const std::array<int, 3> g_last{d1 * DIM1 * 2 - 1, d2 * DIM2 - 1, d3 * DIM3 - 1};
    const std::array<int, 6> halos{H1m, H1p, H2m, H2p, H3m, H3p};
    const std::array<bool, 3> periodic{true, true, true};
    int coords[3]{context.world().rank() % d1, context.world().rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        context.world().rank() * 2,
        coordinate_t{(context.world().rank() % d1) * DIM1 * 2               , (context.world().rank() / d1    ) * DIM2    , 0     },
        coordinate_t{(context.world().rank() % d1) * DIM1 * 2 + DIM1 - 1    , (context.world().rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    domain_descriptor_t my_domain_2{
        context.world().rank() * 2 + 1,
        coordinate_t{(context.world().rank() % d1) * DIM1 * 2 + DIM1        , (context.world().rank() / d1    ) * DIM2    , 0     },
        coordinate_t{(context.world().rank() % d1) * DIM1 * 2 + DIM1 * 2 - 1, (context.world().rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_2);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    using communication_object_t = gridtools::ghex::communication_object<decltype(patterns)::value_type, gridtools::ghex::cpu>;

    //auto comm = context.get_communicator(context.get_token());
    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        //cos.push_back(communication_object_t{p,comm});
        cos.push_back(communication_object_t{p,context.get_communicator(context.get_token())});
    }

    triple_t<USE_DOUBLE, int>* _values_1_1 = new triple_t<USE_DOUBLE, int>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, int>, layout_map_type> values_1_1(_values_1_1, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, int>, domain_descriptor_t> data_1_1{
        local_domains[0],
        coordinate_t{H1m, H2m, H3m},
        values_1_1
    };

    triple_t<USE_DOUBLE, double>* _values_1_2 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_1_2(_values_1_2, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, domain_descriptor_t> data_1_2{
        local_domains[0],
        coordinate_t{H1m, H2m, H3m},
        values_1_2
    };

    triple_t<USE_DOUBLE, int>* _values_2_1 = new triple_t<USE_DOUBLE, int>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, int>, layout_map_type> values_2_1(_values_2_1, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, int>, domain_descriptor_t> data_2_1{
        local_domains[1],
        coordinate_t{H1m, H2m, H3m},
        values_2_1
    };

    triple_t<USE_DOUBLE, double>* _values_2_2 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_2_2(_values_2_2, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, domain_descriptor_t> data_2_2{
        local_domains[1],
        coordinate_t{H1m, H2m, H3m},
        values_2_2
    };

    /* Just an initialization */
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_1_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>();
                values_1_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
                values_2_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>();
                values_2_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>(
                            ii - H1m + DIM1 * 2 * coords[0],
                            jj - H2m + DIM2     * coords[1],
                            kk - H3m + DIM3     * coords[2]
                        );
                values_1_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * 2 * coords[0],
                            jj - H2m + DIM2     * coords[1],
                            kk - H3m + DIM3     * coords[2]
                        );
                values_2_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>(
                            ii - H1m + DIM1 * 2 * coords[0] + DIM1,
                            jj - H2m + DIM2     * coords[1]       ,
                            kk - H3m + DIM3     * coords[2]
                        );
                values_2_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * 2 * coords[0] + DIM1,
                            jj - H2m + DIM2     * coords[1]       ,
                            kk - H3m + DIM3     * coords[2]
                        );
            }

    std::vector<std::thread> threads;

    threads.push_back(std::thread([&cos, &data_1_1, &data_1_2](){
        auto h1 = cos[0].exchange(data_1_1, data_1_2);
        h1.wait();
    }));

    threads.push_back(std::thread([&cos, &data_2_1, &data_2_2](){
        auto h2 = cos[1].exchange(data_2_1, data_2_2);
        h2.wait();
    }));

    for (auto& t : threads) {
        t.join();
    }

    int passed = true;

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, int> t11;
                int tx11, ty11, tz11;

                tx11 = modulus(ii - H1m + DIM1 * 2 * coords[0], DIM1 * 2 * 2);
                ty11 = modulus(jj - H2m + DIM2     * coords[1], DIM2     * 2);
                tz11 = modulus(kk - H3m + DIM3     * coords[2], DIM3        );

                t11 = triple_t<USE_DOUBLE, int>(tx11, ty11, tz11).floor();

                if (values_1_1(ii, jj, kk) != t11) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1_1 " << values_1_1(ii, jj, kk) << " != " << t11 << "\n";

                }

                triple_t<USE_DOUBLE, double> t12;
                int tx12, ty12, tz12;

                tx12 = modulus(ii - H1m + DIM1 * 2 * coords[0], DIM1 * 2 * 2);
                ty12 = modulus(jj - H2m + DIM2     * coords[1], DIM2     * 2);
                tz12 = modulus(kk - H3m + DIM3     * coords[2], DIM3        );

                t12 = triple_t<USE_DOUBLE, double>(tx12, ty12, tz12).floor();

                if (values_1_2(ii, jj, kk) != t12) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1_2 " << values_1_2(ii, jj, kk) << " != " << t12 << "\n";

                }

                triple_t<USE_DOUBLE, int> t21;
                int tx21, ty21, tz21;

                tx21 = modulus(ii - H1m + DIM1 * 2 * coords[0] + DIM1, DIM1 * 2 * 2);
                ty21 = modulus(jj - H2m + DIM2     * coords[1]       , DIM2     * 2);
                tz21 = modulus(kk - H3m + DIM3     * coords[2]       , DIM3        );

                t21 = triple_t<USE_DOUBLE, int>(tx21, ty21, tz21).floor();

                if (values_2_1(ii, jj, kk) != t21) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_2_1 " << values_2_1(ii, jj, kk) << " != " << t21 << "\n";

                }

                triple_t<USE_DOUBLE, double> t22;
                int tx22, ty22, tz22;

                tx22 = modulus(ii - H1m + DIM1 * 2 * coords[0] + DIM1, DIM1 * 2 * 2);
                ty22 = modulus(jj - H2m + DIM2     * coords[1]       , DIM2     * 2);
                tz22 = modulus(kk - H3m + DIM3     * coords[2]       , DIM3        );

                t22 = triple_t<USE_DOUBLE, double>(tx22, ty22, tz22).floor();

                if (values_2_2(ii, jj, kk) != t22) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_2 " << values_2_2(ii, jj, kk) << " != " << t22 << "\n";

                }

            }

    EXPECT_TRUE(passed);

}
