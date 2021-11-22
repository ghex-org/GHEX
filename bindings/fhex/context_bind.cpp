/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <fhex/context_bind.hpp>
#include <fhex/ghex_defs.hpp>
#include <mpi.h>
#include <sched.h>
#include <sys/sysinfo.h>

namespace
{
ghex::context* ghex_context_obj = nullptr;
ghex::barrier* ghex_barrier_obj = nullptr;
int            ghex_nthreads = 0;
} // namespace

namespace fhex
{
ghex::context&
context()
{
    return *ghex_context_obj;
}

ghex::barrier&
barrier()
{
    return *ghex_barrier_obj;
}
} // namespace fhex

extern "C" void
ghex_init(int nthreads, MPI_Fint fcomm)
{
    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);
    ghex_nthreads = nthreads;
    ghex_context_obj = new ghex::context{ccomm, nthreads > 1};
    ghex_barrier_obj = new ghex::barrier(*ghex_context_obj, nthreads);
}

extern "C" void
ghex_finalize()
{
    delete ghex_context_obj;
    delete ghex_barrier_obj;
}

extern "C" int
ghex_get_current_cpu()
{
    return sched_getcpu();
}

extern "C" int
ghex_get_ncpus()
{
    return get_nprocs_conf();
}

extern "C" void
ghex_barrier(int type)
{
    switch (type)
    {
        case (GhexBarrierThread):
            ghex_barrier_obj->thread_barrier();
            break;
        case (GhexBarrierRank):
            ghex_barrier_obj->rank_barrier();
            break;
        default:
            (*ghex_barrier_obj)();
            break;
    }
}
