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

namespace gridtools {
    namespace ghex {
        namespace tl {

            template <typename TLCommunicator>
            void barrier(MPI_Comm comm, TLCommunicator& tlcomm)
            {
                MPI_Request req = MPI_REQUEST_NULL;
                int flag;
                MPI_Ibarrier(comm, &req);
                while(true) {
                    tlcomm.progress();
                    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                    if(flag) break;
                }
            }
        }
    }
}
