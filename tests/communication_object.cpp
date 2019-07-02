/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>
#include "gtest_main_boost.cpp"
#include <vector>
#include <array>
#include <utility>
#include <thread>
#include <boost/mpi/communicator.hpp>
#include <gridtools/common/layout_map.hpp>
#include "../include/gridtools_arch.hpp"
#include "../include/communication_object.hpp"
#include "../include/protocol/communicator_base.hpp"
#include "../include/protocol/mpi.hpp"
#include "../include/pattern.hpp"
#include "../include/structured_pattern.hpp"
#include "../include/structured_domain_descriptor.hpp"
#include "triplet.hpp"
#include "../include/utils.hpp"


//struct my_domain_desc {

//    using coordinate_type = std::array<int, 3>;
//    using domain_id_type = int;

//    struct local_domain {

//        coordinate_type m_first;
//        coordinate_type m_last;

//        local_domain(const coordinate_type& first,
//                     const coordinate_type& last) :
//            m_first{first},
//            m_last{last} {}

//        const coordinate_type& first() const {return m_first;}
//        const coordinate_type& last() const {return m_last;}

//    };

//    struct halo {

//        struct local_halo {

//            coordinate_type m_first;
//            coordinate_type m_last;

//            local_halo(const coordinate_type& first,
//                       const coordinate_type& last) :
//                m_first{first},
//                m_last{last} {}

//            const coordinate_type& first() const {return m_first;}
//            const coordinate_type& last() const {return m_last;}

//        };

//        coordinate_type m_first;
//        coordinate_type m_last;
//        local_halo m_local_halo;

//        halo(const coordinate_type& first,
//             const coordinate_type& last,
//             const coordinate_type& local_first,
//             const coordinate_type& local_last) :
//            m_first{first},
//            m_last{last},
//            m_local_halo{local_first, local_last} {}

//        const coordinate_type& first() const {return m_first;}
//        const coordinate_type& last() const {return m_last;}
//        const local_halo& local() const {return m_local_halo;}
//        const halo& global() const {return *this;}

//    };

//    domain_id_type m_id;
//    coordinate_type m_first;
//    coordinate_type m_last;
//    local_domain m_local_domain;

//    my_domain_desc(const domain_id_type id,
//                   const coordinate_type& first,
//                   const coordinate_type& last) :
//        m_id{id},
//        m_first{first},
//        m_last{last},
//        m_local_domain{coordinate_type{0, 0, 0}, coordinate_type{last[0]-first[0], last[1]-first[1], last[2]-first[2]}} {}

//    domain_id_type domain_id() const {return m_id;}
//    const coordinate_type& first() const {return m_first;}
//    const coordinate_type& last() const {return m_last;}
//    const local_domain& local() const {return m_local_domain;}
//    coordinate_type size() const {return coordinate_type{m_local_domain.last()[0]+1, m_local_domain.last()[1]+1, m_local_domain.last()[2]+1};}

//};


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

    std::size_t data_type_size() const {
        return sizeof (T);
    }

    void set(const T& value, const coordinate_t& coords) {
        m_values(coords[0] + m_halos_offset[0], coords[1] + m_halos_offset[1], coords[2] + m_halos_offset[2]) = value;
    }

    const T& get(const coordinate_t& coords) const {
        return m_values(coords[0] + m_halos_offset[0], coords[1] + m_halos_offset[1], coords[2] + m_halos_offset[2]);
    }

    template <typename IterationSpace>
    void set(const IterationSpace& is, const Byte* buffer) {
        //std::cout << "DEBUG: is.first()[2] = " << is.local().first()[2] << "\n";
        //std::cout << "DEBUG: is.last()[2] = " << is.local().last()[2] << "\n";
        //std::cout.flush();
        gridtools::detail::for_loop<3, 3, layout_map_t>::apply([this, &buffer](auto... indices){
            coordinate_t coords{indices...};
            //std::cout << "DEBUG: coords = " << coords[0] << ", " << coords[1] << ", " << coords[2] << "\n";
            //std::cout.flush();
            set(*(reinterpret_cast<const T*>(buffer)), coords);
            //std::cout << "DEBUG: just set value " << get(coords) << "\n";
            //std::cout.flush();
            buffer += sizeof(T);
        }, is.local().first(), is.local().last());
    }

    template <typename IterationSpace>
    void get(const IterationSpace& is, Byte* buffer) const {
        gridtools::detail::for_loop<3, 3, layout_map_t>::apply([this, &buffer](auto... indices){
            coordinate_t coords{indices...};
            //std::cout << "DEBUG: coords = " << coords[0] << ", " << coords[1] << ", " << coords[2] << "\n";
            //std::cout.flush();
            const T* tmp_ptr{&get(coords)};
            std::memcpy(buffer, tmp_ptr, sizeof(T));
            //std::cout << "DEBUG: just got value " << *(reinterpret_cast<T*>(buffer)) << "\n";
            //std::cout.flush();
            buffer += sizeof(T);
        }, is.local().first(), is.local().last());
    }

};


TEST(communication_object, constructor) {

    using domain_descriptor_t = gridtools::structured_domain_descriptor<int,3>;
    using domain_id_t = domain_descriptor_t::domain_id_type;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = gridtools::structured_halo_generator<domain_id_t, 3>;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

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
        comm.rank(),
        coordinate_t{(comm.rank() % d1    ) * DIM1    , (comm.rank() / d1)     * DIM2    , 0},
        coordinate_t{(comm.rank() % d1 + 1) * DIM1 - 1, (comm.rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen, local_domains);

    using communication_object_t = gridtools::communication_object<std::remove_reference_t<decltype(*(patterns.begin()))>, gridtools::cpu>;

    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        EXPECT_NO_THROW(
        cos.push_back(communication_object_t{p});
        );
    }

}


TEST(communication_object, exchange) {

    using domain_descriptor_t = gridtools::structured_domain_descriptor<int, 3>;
    using domain_id_t = domain_descriptor_t::domain_id_type;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

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
    int coords[3]{comm.rank() % d1, comm.rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        comm.rank(),
        coordinate_t{(comm.rank() % d1    ) * DIM1    , (comm.rank() / d1)     * DIM2    , 0},
        coordinate_t{(comm.rank() % d1 + 1) * DIM1 - 1, (comm.rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen, local_domains);

    using communication_object_t = gridtools::communication_object<std::remove_reference_t<decltype(*(patterns.begin()))>, gridtools::cpu>;

    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p});
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

    using domain_descriptor_t = gridtools::structured_domain_descriptor<int, 3>;
    using domain_id_t = domain_descriptor_t::domain_id_type;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

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
    int coords[3]{comm.rank() % d1, comm.rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        comm.rank(),
        coordinate_t{(comm.rank() % d1    ) * DIM1    , (comm.rank() / d1)     * DIM2    , 0},
        coordinate_t{(comm.rank() % d1 + 1) * DIM1 - 1, (comm.rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen, local_domains);

    using communication_object_t = gridtools::communication_object<std::remove_reference_t<decltype(*(patterns.begin()))>, gridtools::cpu>;

    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p});
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
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>(
                            ii - H1m + DIM1 * coords[0],
                            jj - H2m + DIM2 * coords[1],
                            kk - H3m + DIM3 * coords[2]
                        );
            }
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
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
                int tx, ty, tz;

                tx = modulus(ii - H1m + DIM1 * coords[0], DIM1 * 2);
                ty = modulus(jj - H2m + DIM2 * coords[1], DIM2 * 2);
                tz = modulus(kk - H3m + DIM3 * coords[2], DIM3);

                t1 = triple_t<USE_DOUBLE, int>(tx, ty, tz).floor();

                if (values_1(ii, jj, kk) != t1) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1 " << values_1(ii, jj, kk) << " != " << t1 << "\n";

                }

            }

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, double> t2;
                int tx, ty, tz;

                tx = modulus(ii - H1m + DIM1 * coords[0], DIM1 * 2) + add;
                ty = modulus(jj - H2m + DIM2 * coords[1], DIM2 * 2) + add;
                tz = modulus(kk - H3m + DIM3 * coords[2], DIM3)     + add;

                t2 = triple_t<USE_DOUBLE, double>(tx, ty, tz).floor();

                if (values_2(ii, jj, kk) != t2) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_2 " << values_2(ii, jj, kk) << " != " << t2 << "\n";

                }

            }

    EXPECT_TRUE(passed);

}


TEST(communication_object, multithreading) {

    using domain_descriptor_t = gridtools::structured_domain_descriptor<int, 3>;
    using domain_id_t = domain_descriptor_t::domain_id_type;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

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
    int coords[3]{comm.rank() % d1, comm.rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        comm.rank() * 2,
        coordinate_t{(comm.rank() % d1) * DIM1 * 2               , (comm.rank() / d1    ) * DIM2    , 0     },
        coordinate_t{(comm.rank() % d1) * DIM1 * 2 + DIM1 - 1    , (comm.rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    domain_descriptor_t my_domain_2{
        comm.rank() * 2 + 1,
        coordinate_t{(comm.rank() % d1) * DIM1 * 2 + DIM1        , (comm.rank() / d1    ) * DIM2    , 0     },
        coordinate_t{(comm.rank() % d1) * DIM1 * 2 + DIM1 * 2 - 1, (comm.rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_2);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen, local_domains);

    using communication_object_t = gridtools::communication_object<std::remove_reference_t<decltype(*(patterns.begin()))>, gridtools::cpu>;

    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p});
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
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * 2 * coords[0],
                            jj - H2m + DIM2     * coords[1],
                            kk - H3m + DIM3     * coords[2]
                        );
            }
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
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
                int tx, ty, tz;

                tx = modulus(ii - H1m + DIM1 * 2 * coords[0], DIM1 * 2 * 2);
                ty = modulus(jj - H2m + DIM2     * coords[1], DIM2     * 2);
                tz = modulus(kk - H3m + DIM3     * coords[2], DIM3        );

                t1 = triple_t<USE_DOUBLE, double>(tx, ty, tz).floor();

                if (values_1(ii, jj, kk) != t1) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1 " << values_1(ii, jj, kk) << " != " << t1 << "\n";

                }

            }

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, double> t2;
                int tx, ty, tz;

                tx = modulus(ii - H1m + DIM1 * 2 * coords[0] + DIM1, DIM1 * 2 * 2);
                ty = modulus(jj - H2m + DIM2     * coords[1]       , DIM2     * 2);
                tz = modulus(kk - H3m + DIM3     * coords[2]       , DIM3        );

                t2 = triple_t<USE_DOUBLE, double>(tx, ty, tz).floor();

                if (values_2(ii, jj, kk) != t2) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_2 " << values_2(ii, jj, kk) << " != " << t2 << "\n";

                }

            }

    EXPECT_TRUE(passed);

}


TEST(communication_object, multithreading_multiple_fileds) {

    using domain_descriptor_t = gridtools::structured_domain_descriptor<int, 3>;
    using domain_id_t = domain_descriptor_t::domain_id_type;
    using coordinate_t = domain_descriptor_t::coordinate_type;
    using halo_generator_t = gridtools::structured_halo_generator<domain_id_t, 3>;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

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
    int coords[3]{comm.rank() % d1, comm.rank() / d1, 0}; // rank in cartesian coordinates

    std::vector<domain_descriptor_t> local_domains;

    domain_descriptor_t my_domain_1{
        comm.rank() * 2,
        coordinate_t{(comm.rank() % d1) * DIM1 * 2               , (comm.rank() / d1    ) * DIM2    , 0     },
        coordinate_t{(comm.rank() % d1) * DIM1 * 2 + DIM1 - 1    , (comm.rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    domain_descriptor_t my_domain_2{
        comm.rank() * 2 + 1,
        coordinate_t{(comm.rank() % d1) * DIM1 * 2 + DIM1        , (comm.rank() / d1    ) * DIM2    , 0     },
        coordinate_t{(comm.rank() % d1) * DIM1 * 2 + DIM1 * 2 - 1, (comm.rank() / d1 + 1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_2);

    auto halo_gen = halo_generator_t{g_first, g_last, halos, periodic};

    auto patterns = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen, local_domains);

    using communication_object_t = gridtools::communication_object<std::remove_reference_t<decltype(*(patterns.begin()))>, gridtools::cpu>;

    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p});
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
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>(
                            ii - H1m + DIM1 * 2 * coords[0],
                            jj - H2m + DIM2     * coords[1],
                            kk - H3m + DIM3     * coords[2]
                        );
            }
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_1_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_1_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>(
                            ii - H1m + DIM1 * 2 * coords[0],
                            jj - H2m + DIM2     * coords[1],
                            kk - H3m + DIM3     * coords[2]
                        );
            }
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_2_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                values_2_1(ii, jj, kk) = triple_t<USE_DOUBLE, int>(
                            ii - H1m + DIM1 * 2 * coords[0] + DIM1,
                            jj - H2m + DIM2     * coords[1]       ,
                            kk - H3m + DIM3     * coords[2]
                        );
            }
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                values_2_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>();
            }
    for (int ii = H1m; ii < DIM1 + H1m; ++ii)
        for (int jj = H2m; jj < DIM2 + H2m; ++jj)
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
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

                triple_t<USE_DOUBLE, int> t1;
                int tx, ty, tz;

                tx = modulus(ii - H1m + DIM1 * 2 * coords[0], DIM1 * 2 * 2);
                ty = modulus(jj - H2m + DIM2     * coords[1], DIM2     * 2);
                tz = modulus(kk - H3m + DIM3     * coords[2], DIM3        );

                t1 = triple_t<USE_DOUBLE, int>(tx, ty, tz).floor();

                if (values_1_1(ii, jj, kk) != t1) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1_1 " << values_1_1(ii, jj, kk) << " != " << t1 << "\n";

                }

            }

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, double> t1;
                int tx, ty, tz;

                tx = modulus(ii - H1m + DIM1 * 2 * coords[0], DIM1 * 2 * 2);
                ty = modulus(jj - H2m + DIM2     * coords[1], DIM2     * 2);
                tz = modulus(kk - H3m + DIM3     * coords[2], DIM3        );

                t1 = triple_t<USE_DOUBLE, double>(tx, ty, tz).floor();

                if (values_1_2(ii, jj, kk) != t1) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_1_2 " << values_1_2(ii, jj, kk) << " != " << t1 << "\n";

                }

            }

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, int> t2;
                int tx, ty, tz;

                tx = modulus(ii - H1m + DIM1 * 2 * coords[0] + DIM1, DIM1 * 2 * 2);
                ty = modulus(jj - H2m + DIM2     * coords[1]       , DIM2     * 2);
                tz = modulus(kk - H3m + DIM3     * coords[2]       , DIM3        );

                t2 = triple_t<USE_DOUBLE, int>(tx, ty, tz).floor();

                if (values_2_1(ii, jj, kk) != t2) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_2_1 " << values_2_1(ii, jj, kk) << " != " << t2 << "\n";

                }

            }

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, double> t2;
                int tx, ty, tz;

                tx = modulus(ii - H1m + DIM1 * 2 * coords[0] + DIM1, DIM1 * 2 * 2);
                ty = modulus(jj - H2m + DIM2     * coords[1]       , DIM2     * 2);
                tz = modulus(kk - H3m + DIM3     * coords[2]       , DIM3        );

                t2 = triple_t<USE_DOUBLE, double>(tx, ty, tz).floor();

                if (values_2_2(ii, jj, kk) != t2) {
                    passed = false;
                    std::cout << ii << ", " << jj << ", " << kk << " values found != expected: "
                              << "values_2 " << values_2_2(ii, jj, kk) << " != " << t2 << "\n";

                }

            }

    EXPECT_TRUE(passed);

}
