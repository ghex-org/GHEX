/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_PROCESSOR_GRID_HPP
#define INCLUDED_PROCESSOR_GRID_HPP

#include <mpi.h>
//#include <gridtools/common/array.hpp>
//#include <gridtools/common/layout_map.hpp>
#include <array>

#include "../structured_domain_descriptor.hpp"
//#include "../protocol/setup.hpp"
#include "../protocol/mpi.hpp"

namespace gridtools {

    template<typename Protocol>
    struct gt_grid
    {
        using domain_descriptor_type = structured_domain_descriptor<int,3>;
        using domain_id_type         = typename domain_descriptor_type::domain_id_type;
        //protocol::setup_communicator m_setup_comm;
        MPI_Comm m_setup_comm;
        protocol::communicator<Protocol> m_comm;
        std::vector<domain_descriptor_type> m_domains;
        std::array<int, 3> m_global_extents;
        std::array<bool, 3> m_periodic;
    };
    
    template<typename Layout = layout_map<0,1,2>, typename Array0, typename Array1>
    gt_grid<protocol::mpi>
    make_gt_processor_grid(const Array0& local_extents, const Array1& periodicity, MPI_Comm cart_comm)
    {
        int dims[3];
        int periods[3];
        int coords[3];
        MPI_Cart_get(cart_comm, 3, dims, periods, coords);
        int rank;
        MPI_Cart_rank(cart_comm, coords, &rank);

        std::array<bool, 3> periodic;
        std::copy(periodicity.begin(), periodicity.end(), periodic.begin());
        const std::array<int, 3> global_extents = {
            local_extents[0]*dims[Layout::template at<0>()],
            local_extents[1]*dims[Layout::template at<1>()],
            local_extents[2]*dims[Layout::template at<2>()]};
        const std::array<int, 3> global_first = {
            local_extents[0]*coords[Layout::template at<0>()],
            local_extents[1]*coords[Layout::template at<1>()],
            local_extents[2]*coords[Layout::template at<2>()]};
        const std::array<int, 3> global_last = {
            local_extents[0]*(coords[Layout::template at<0>()]+1)-1,
            local_extents[1]*(coords[Layout::template at<1>()]+1)-1,
            local_extents[2]*(coords[Layout::template at<2>()]+1)-1};

        structured_domain_descriptor<int,3> local_domain{rank, global_first, global_last};

        return {cart_comm, protocol::communicator<protocol::mpi>{cart_comm}, {local_domain}, global_extents, periodic}; 

    }
}

#endif /* INCLUDED_PROCESSOR_GRID_HPP */

