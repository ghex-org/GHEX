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

    domain_id_type m_id;
    coordinate_type m_first;
    coordinate_type m_last;

    struct halo {

        coordinate_type m_first;
        coordinate_type m_last;

        halo(const coordinate_type& first, const coordinate_type& last) :
            m_first{first},
            m_last{last} {}

        const coordinate_type& first() const { return m_first; }
        const coordinate_type& last() const { return m_last; }

    };

    my_domain_desc(const domain_id_type id, const coordinate_type& first, const coordinate_type& last) :
        m_id{id},
        m_first{first},
        m_last{last} {}

    domain_id_type domain_id() const { return m_id; }
    const coordinate_type& first() const { return m_first; }
    const coordinate_type& last() const { return m_last; }

};


/* CPU data descriptor */
template <typename T, typename DomainDescriptor>
class my_data_desc {

    using coordinate_type = typename DomainDescriptor::coordinate_type;
    using layout_map_type = gridtools::layout_map<2, 1, 0>;

    const DomainDescriptor& m_domain;
    array<triple_t<USE_DOUBLE, double>, layout_map_type> m_values;

public:

    using Byte = unsigned char;

    my_data_desc(const DomainDescriptor& domain, const array<triple_t<USE_DOUBLE, double>, layout_map_type>& values) :
        m_domain{domain},
        m_values{values} {}

    void set(const T& value, const coordinate_type& coords) {
        m_values(coords[0], coords[1], coords[2]) = value;
    }

    const T& get(const coordinate_type& coords) const {
        return m_values(coords[0], coords[1], coords[2]);
    }

    template <typename IterationSpace>
    void set(const IterationSpace& is, const Byte* buffer) {
        gridtools::detail::for_loop<3, 3, layout_map_type>::apply([this, &buffer](auto... indices){
            coordinate_type coords{indices...};
            set(*(reinterpret_cast<const T*>(buffer)), coords);
            buffer += sizeof(T);
        }, is.first(), is.last());
    }

    template <typename IterationSpace>
    void get(const IterationSpace& is, Byte* buffer) const {
        gridtools::detail::for_loop<3, 3, layout_map_type>::apply([this, &buffer](auto... indices){
            coordinate_type coords{indices...};
            std::memcpy(reinterpret_cast<void*>(buffer), reinterpret_cast<const void*>(&get(coords)), sizeof(T));
            buffer += sizeof(T);
        }, is.first(), is.last());
    }

};


TEST(communication_object, constructor) {

    using coordinate_type = my_domain_desc::coordinate_type;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

    /* same domain setup as for pattern test */
    std::vector<my_domain_desc> local_domains;

    my_domain_desc my_domain_1{
        comm.rank()*2,
        coordinate_type{(comm.rank()%2)*20, (comm.rank()/2)*15, 0},
        coordinate_type{(comm.rank()%2)*20+9, (comm.rank()/2+1)*15-1, 19}
    };
    local_domains.push_back(my_domain_1);

    my_domain_desc my_domain_2{
        comm.rank()*2+1,
        coordinate_type{(comm.rank()%2)*20+10, (comm.rank()/2)*15, 0},
        coordinate_type{(comm.rank()%2)*20+19, (comm.rank()/2+1)*15-1, 19}
    };
    local_domains.push_back(my_domain_2);

    auto halo_gen = [](const my_domain_desc& d) {

        using halo_type = my_domain_desc::halo;

        std::vector<halo_type> halos;

        halo_type bottom{d.first(), d.last()};
        bottom.m_last[2] = bottom.m_first[2]-1;
        bottom.m_first[2] -= 2;
        bottom.m_first[2] = (bottom.m_first[2]+20)%20;
        bottom.m_last[2] = (bottom.m_last[2]+20)%20;
        halos.push_back(bottom);

        auto top{bottom};
        top.m_first[2] = 0;
        top.m_last[2] = 1;
        halos.push_back(top);

        halo_type left{d.first(), d.last()};
        left.m_last[0] = left.m_first[0]-1;
        left.m_first[0] -= 2;
        left.m_first[0] = (left.m_first[0]+40)%40;
        left.m_last[0] = (left.m_last[0]+40)%40;
        halos.push_back(left);

        halo_type right{d.first(), d.last()};
        right.m_first[0] = right.m_last[0]+1;
        right.m_last[0] += 2;
        right.m_first[0] = (right.m_first[0]+40)%40;
        right.m_last[0] = (right.m_last[0]+40)%40;

        halos.push_back( right );

        return halos;

    };

    auto patterns = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen, local_domains);

    using communication_object_type = gridtools::communication_object<std::remove_reference_t<decltype(patterns[0])>, gridtools::cpu>;

    std::vector<communication_object_type> cos;
    for (auto& p : patterns) {
        EXPECT_NO_THROW(cos.push_back(communication_object_type{p}));
    }

}
