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

#include <numeric>

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

        // scan algorithm: x-direction
        std::vector<int> extents_x(dims[0]);
        for (int i=0; i<dims[0]; ++i)
        {
            int coords_i[3] = {i,0,0};
            int rank_i;
            MPI_Cart_rank(cart_comm, coords_i, &rank_i);
            if (coords[0]==i && coords[1]==0 && coords[2]==0)
            {
                // broadcast
                int lext = local_extents[0];
                extents_x[i] = lext;
                MPI_Bcast(&lext, sizeof(int), MPI_BYTE, rank_i, cart_comm);
            }
            else
            {
                // recv
                MPI_Bcast(&extents_x[i], sizeof(int), MPI_BYTE, rank_i, cart_comm);
            }
        }
        std::partial_sum(extents_x.begin(), extents_x.end(), extents_x.begin());
        // scan algorithm: y-direction
        std::vector<int> extents_y(dims[1]);
        for (int i=0; i<dims[1]; ++i)
        {
            int coords_i[3] = {0,i,0};
            int rank_i;
            MPI_Cart_rank(cart_comm, coords_i, &rank_i);
            if (coords[1]==i && coords[0]==0 && coords[2]==0)
            {
                // broadcast
                int lext = local_extents[1];
                extents_y[i] = lext;
                MPI_Bcast(&lext, sizeof(int), MPI_BYTE, rank_i, cart_comm);
            }
            else
            {
                // recv
                MPI_Bcast(&extents_y[i], sizeof(int), MPI_BYTE, rank_i, cart_comm);
            }
        }
        std::partial_sum(extents_y.begin(), extents_y.end(), extents_y.begin());
        // scan algorithm: z-direction
        std::vector<int> extents_z(dims[2]);
        for (int i=0; i<dims[2]; ++i)
        {
            int coords_i[3] = {0,0,i};
            int rank_i;
            MPI_Cart_rank(cart_comm, coords_i, &rank_i);
            if (coords[2]==i && coords[0]==0 && coords[1]==0)
            {
                // broadcast
                int lext = local_extents[2];
                extents_z[i] = lext;
                MPI_Bcast(&lext, sizeof(int), MPI_BYTE, rank_i, cart_comm);
            }
            else
            {
                // recv
                MPI_Bcast(&extents_z[i], sizeof(int), MPI_BYTE, rank_i, cart_comm);
            }
        }
        std::partial_sum(extents_z.begin(), extents_z.end(), extents_z.begin());


        const std::array<int, 3> global_extents = {
            extents_x.back(),
            extents_y.back(),
            extents_z.back()};
        const std::array<int, 3> global_first = {
            coords[0]==0 ? 0 : extents_x[coords[0]-1],
            coords[1]==0 ? 0 : extents_y[coords[1]-1],
            coords[2]==0 ? 0 : extents_z[coords[2]-1]};
        const std::array<int, 3> global_last = {
            global_first[0] + local_extents[0] -1,
            global_first[1] + local_extents[1] -1,
            global_first[2] + local_extents[2] -1};

        /*const std::array<int, 3> global_extents = {
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
            local_extents[2]*(coords[Layout::template at<2>()]+1)-1};*/

        structured_domain_descriptor<int,3> local_domain{rank, global_first, global_last};

        return {cart_comm, protocol::communicator<protocol::mpi>{cart_comm}, {local_domain}, global_extents, periodic}; 

    }
}

#endif /* INCLUDED_PROCESSOR_GRID_HPP */

