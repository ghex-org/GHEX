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
#include <mpi.h>
#include <sched.h>
#include <sys/sysinfo.h>

namespace
{
ghex::context* ghex_context;
}

namespace fhex
{
ghex::context&
context()
{
    return *ghex_context;
}
} // namespace fhex

extern "C" void
ghex_init(MPI_Fint fcomm)
{
    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);
    int      mpi_thread_safety;
    MPI_Query_thread(&mpi_thread_safety);
    const bool thread_safe = (mpi_thread_safety == MPI_THREAD_MULTIPLE);
    ghex_context = new ghex::context{ccomm, thread_safe};
}

extern "C" void
ghex_finalize()
{
    delete ghex_context;
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
