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
#include "gridtools_arch.hpp"
#include "communication_object.hpp"
#include "protocol/communicator_base.hpp"
#include "protocol/mpi.hpp"
#include "pattern.hpp"
#include "structured_pattern.hpp"
#include "triplet.hpp"
#include "utils.hpp"


struct my_domain_desc {

    using coordinate_type = std::array<int, 3>;
    using domain_id_type = int;

    struct local_domain {

        coordinate_type m_first;
        coordinate_type m_last;

        local_domain(const coordinate_type& first,
                     const coordinate_type& last) :
            m_first{first},
            m_last{last} {}

        const coordinate_type& first() const {return m_first;}
        const coordinate_type& last() const {return m_last;}

    };

    struct halo {

        struct local_halo {

            coordinate_type m_first;
            coordinate_type m_last;

            local_halo(const coordinate_type& first,
                       const coordinate_type& last) :
                m_first{first},
                m_last{last} {}

            const coordinate_type& first() const {return m_first;}
            const coordinate_type& last() const {return m_last;}

        };

        coordinate_type m_first;
        coordinate_type m_last;
        local_halo m_local_halo;

        halo(const coordinate_type& first,
             const coordinate_type& last,
             const coordinate_type& local_first,
             const coordinate_type& local_last) :
            m_first{first},
            m_last{last},
            m_local_halo{local_first, local_last} {}

        const coordinate_type& first() const {return m_first;}
        const coordinate_type& last() const {return m_last;}
        const local_halo& local() const {return m_local_halo;}

    };

    domain_id_type m_id;
    coordinate_type m_first;
    coordinate_type m_last;
    local_domain m_local_domain;

    my_domain_desc(const domain_id_type id,
                   const coordinate_type& first,
                   const coordinate_type& last) :
        m_id{id},
        m_first{first},
        m_last{last},
        m_local_domain{coordinate_type{0, 0, 0}, coordinate_type{last[0]-first[0], last[1]-first[1], last[2]-first[2]}} {}

    domain_id_type domain_id() const {return m_id;}
    const coordinate_type& first() const {return m_first;}
    const coordinate_type& last() const {return m_last;}
    const local_domain& local() const {return m_local_domain;}
    coordinate_type size() const {return coordinate_type{m_last[0]+1, m_last[1]+1, m_last[2]+1};}

};


/* CPU data descriptor */
template <typename T, typename DomainDescriptor>
class my_data_desc {

    using coordinate_type = typename DomainDescriptor::coordinate_type;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;
    using Byte = unsigned char;

    const DomainDescriptor& m_domain;
    coordinate_type m_halos_offset;
    array<triple_t<USE_DOUBLE, double>, layout_map_type> m_values;

public:

    my_data_desc(const DomainDescriptor& domain,
                 const coordinate_type& halos_offset,
                 const array<triple_t<USE_DOUBLE, double>, layout_map_type>& values) :
        m_domain{domain},
        m_halos_offset{halos_offset},
        m_values{values} {}

    std::size_t data_type_size() const {
        return sizeof (T);
    }

    void set(const T& value, const coordinate_type& coords) {
        m_values(coords[0] + m_halos_offset[0], coords[1] + m_halos_offset[1], coords[2] + m_halos_offset[2]) = value;
    }

    const T& get(const coordinate_type& coords) const {
        return m_values(coords[0] + m_halos_offset[0], coords[1] + m_halos_offset[1], coords[2] + m_halos_offset[2]);
    }

    template <typename IterationSpace>
    void set(const IterationSpace& is, const Byte* buffer) {
        gridtools::detail::for_loop<3, 3, layout_map_type>::apply([this, &buffer](auto... indices){
            coordinate_type coords{indices...};
            std::cout << "DEBUG: coords = " << coords[0] << ", " << coords[1] << ", " << coords[2] << "\n";
            std::cout.flush();
            set(*(reinterpret_cast<const T*>(buffer)), coords);
            std::cout << "DEBUG: just set value " << get(coords) << "\n";
            std::cout.flush();
            buffer += sizeof(T);
        }, is.local().first(), is.local().last());
    }

    template <typename IterationSpace>
    void get(const IterationSpace& is, Byte* buffer) const {
        gridtools::detail::for_loop<3, 3, layout_map_type>::apply([this, &buffer](auto... indices){
            coordinate_type coords{indices...};
            std::cout << "DEBUG: coords = " << coords[0] << ", " << coords[1] << ", " << coords[2] << "\n";
            std::cout.flush();
            const T* tmp_ptr{&get(coords)};
            std::memcpy(buffer, tmp_ptr, sizeof(T));
            std::cout << "DEBUG: just got value " << get(coords) << "\n";
            std::cout.flush();
            buffer += sizeof(T);
        }, is.local().first(), is.local().last());
    }

};


/* 3D halo generator */
template<int H1m, int H1p, int H2m, int H2p, int H3m, int H3p>
auto halo_gen = [](const my_domain_desc& d) {

    using coordinate_type = my_domain_desc::coordinate_type;
    using halo_type = my_domain_desc::halo;

    std::vector<halo_type> halos;

    halo_type bottom{
        coordinate_type{d.first()[0], d.first()[1], ((d.first()[2] - H3m) + d.size()[2]) % d.size()[2]},
        coordinate_type{d.last()[0] , d.last()[1] , ((d.first()[2] - 1)   + d.size()[2]) % d.size()[2]},
        coordinate_type{d.local().first()[0], d.local().first()[1], d.local().first()[2] - H3m},
        coordinate_type{d.local().last()[0] , d.local().last()[1] , d.local().first()[2] - 1}
    };
    halos.push_back(bottom);

    halo_type top{
        coordinate_type{d.first()[0], d.first()[1], d.first()[2] + 1},
        coordinate_type{d.last()[0] , d.last()[1] , d.first()[2] + H3p},
        coordinate_type{d.local().first()[0], d.local().first()[1], d.local().last()[2] + 1},
        coordinate_type{d.local().last()[0] , d.local().last()[1] , d.local().last()[2] + H3p}
    };
    halos.push_back(top);

    halo_type left{
        coordinate_type{((d.first()[0] - H1m) + d.size()[0]*4) % (d.size()[0]*4), d.first()[1], d.first()[2]},
        coordinate_type{((d.first()[0] - 1)   + d.size()[0]*4) % (d.size()[0]*4), d.last()[1], d.last()[2]},
        coordinate_type{d.local().first()[0] - H1m, d.local().first()[1], d.local().first()[2]},
        coordinate_type{d.local().first()[0] - 1  , d.local().last()[1] , d.local().last()[2]}
    };
    halos.push_back(left);

    halo_type right{
        coordinate_type{(d.last()[0] + 1)   % (d.size()[0]*4), d.first()[1], d.first()[2]},
        coordinate_type{(d.last()[0] + H1p) % (d.size()[0]*4), d.last()[1] , d.last()[2]},
        coordinate_type{d.local().last()[0] + 1  , d.local().first()[1], d.local().first()[2]},
        coordinate_type{d.local().last()[0] + H1p, d.local().last()[1] , d.local().last()[2]}
    };
    halos.push_back(right);

    return halos;

};


TEST(communication_object, constructor) {

    using coordinate_type = my_domain_desc::coordinate_type;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

    /* Problem sizes */
    const int DIM1 = 10;
    const int DIM2 = 15;
    const int DIM3 = 20;
    const int H1m = 1;
    const int H1p = 1;
    const int H2m = 0;
    const int H2p = 0;
    const int H3m = 1;
    const int H3p = 1;

    std::vector<my_domain_desc> local_domains;

    my_domain_desc my_domain_1{
        comm.rank()*2,
        coordinate_type{(comm.rank()%2) * (DIM1*2)             , (comm.rank()/2) * DIM2      , 0},
        coordinate_type{(comm.rank()%2) * (DIM1*2) + (DIM1-1)  , (comm.rank()/2+1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    my_domain_desc my_domain_2{
        comm.rank()*2+1,
        coordinate_type{(comm.rank()%2) * (DIM1*2) + DIM1      , (comm.rank()/2) * DIM2      , 0},
        coordinate_type{(comm.rank()%2) * (DIM1*2) + (DIM1*2-1), (comm.rank()/2+1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_2);

    auto patterns = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen<H1m, H1p, H2m, H2p, H3m, H3p>, local_domains);

    using communication_object_type = gridtools::communication_object<std::remove_reference_t<decltype(*(patterns.begin()))>, gridtools::cpu>;

    std::vector<communication_object_type> cos;
    for (const auto& p : patterns) {
        EXPECT_NO_THROW(cos.push_back(communication_object_type{p}));
    }

}


TEST(communication_object, exchange) {

    std::cout << "DEBUG: exchange test \n";
    std::cout.flush();

    using coordinate_type = my_domain_desc::coordinate_type;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

    /* Problem sizes */
    int coords[3]{comm.rank()%2, comm.rank()/2, 0}; // rank in cartesian coordinates
    const int DIM1 = 10;
    const int DIM2 = 15;
    const int DIM3 = 20;
    const int H1m = 1;
    const int H1p = 1;
    const int H2m = 0;
    const int H2p = 0;
    const int H3m = 1;
    const int H3p = 1;

    std::vector<my_domain_desc> local_domains;

    my_domain_desc my_domain_1{
        comm.rank()*2,
        coordinate_type{(comm.rank()%2) * (DIM1*2)             , (comm.rank()/2) * DIM2      , 0},
        coordinate_type{(comm.rank()%2) * (DIM1*2) + (DIM1-1)  , (comm.rank()/2+1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_1);

    my_domain_desc my_domain_2{
        comm.rank()*2+1,
        coordinate_type{(comm.rank()%2) * (DIM1*2) + DIM1      , (comm.rank()/2) * DIM2      , 0},
        coordinate_type{(comm.rank()%2) * (DIM1*2) + (DIM1*2-1), (comm.rank()/2+1) * DIM2 - 1, DIM3-1}
    };
    local_domains.push_back(my_domain_2);

    auto patterns = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen<H1m, H1p, H2m, H2p, H3m, H3p>, local_domains);

    using communication_object_type = gridtools::communication_object<std::remove_reference_t<decltype(*(patterns.begin()))>, gridtools::cpu>;

    std::vector<communication_object_type> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_type{p});
    }

    triple_t<USE_DOUBLE, double>* _values_1 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_1(_values_1, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, my_domain_desc> data_1{
        local_domains[0],
        coordinate_type{H1m, H2m, H3m},
        values_1
    };

    triple_t<USE_DOUBLE, double>* _values_2 = new triple_t<USE_DOUBLE, double>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, double>, layout_map_type> values_2(_values_2, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));
    my_data_desc<triple_t<USE_DOUBLE, double>, my_domain_desc> data_2{
        local_domains[1],
        coordinate_type{H1m, H2m, H3m},
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
                values_1(ii, jj, kk) = triple_t<USE_DOUBLE, double>(ii - H1m + (DIM1)*coords[0], jj - H2m + (DIM2)*coords[1], kk - H3m + (DIM3)*coords[2]);
                values_2(ii, jj, kk) = triple_t<USE_DOUBLE, double>(ii - H1m + (DIM1)*coords[0], jj - H2m + (DIM2)*coords[1], kk - H3m + (DIM3)*coords[2]);
            }

    // CHECK IF THE USER CODE MAKES ANY SENSE FROM HERE!

    std::vector<std::thread> threads;

    threads.push_back(std::thread([&cos, &data_1](){
        auto h = cos[0].exchange(data_1);
        h.wait();
    }));

    threads.push_back(std::thread([&cos, &data_2](){
        auto h = cos[1].exchange(data_2);
        h.wait();
    }));

    for (auto& t : threads) {
        t.join();
    }

    // CHECK TESTS FROM HERE!

    int passed = true;

    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii)
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj)
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {

                triple_t<USE_DOUBLE, double> ta;
                int tax, tay, taz;

                tax = modulus(ii - H1m + (DIM1)*coords[0], DIM1 * 2);
                tay = modulus(jj - H2m + (DIM2)*coords[1], DIM2 * 2);
                taz = modulus(kk - H3m + (DIM3)*coords[2], DIM3);

                ta = triple_t<USE_DOUBLE, double>(tax, tay, taz).floor();

                if (values_1(ii, jj, kk) != ta) {
                    passed = false;
                }

            }

    EXPECT_TRUE(passed);

}
