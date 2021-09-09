#include "context_bind.hpp"
#include <mpi.h>
#include <sched.h>
#include <sys/sysinfo.h>

//using namespace gridtools::ghex::fhex;

namespace ghex
{
namespace fhex
{
context_uptr_type ghex_context;
//            int ghex_nthreads = 1;

//            /* barrier has to be shared between the threads */
//            gridtools::ghex::tl::barrier_t *ghex_barrier = nullptr;
} // namespace fhex
} // namespace ghex

extern "C" void
ghex_init(MPI_Fint fcomm)
{
    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);
    int      mpi_thread_safety;
    MPI_Query_thread(&mpi_thread_safety);
    const bool thread_safe = (mpi_thread_safety == MPI_THREAD_MULTIPLE);
    ghex::fhex::ghex_context = std::make_unique<ghex::context>(ccomm, thread_safe);
    //ghex::fhex::ghex_nthreads = nthreads;
    //ghex::fhex::ghex_barrier = new gridtools::ghex::tl::barrier_t(nthreads);
}

extern "C" void
ghex_finalize()
{
    ghex::fhex::ghex_context.reset();
    //delete gridtools::ghex::fhex::ghex_barrier;
    //gridtools::ghex::fhex::ghex_barrier = nullptr;
    //gridtools::ghex::fhex::ghex_nthreads = 1;
}

extern "C" void
ghex_obj_free(ghex::fhex::obj_wrapper** wrapper_ref)
{
    auto wrapper = *wrapper_ref;

    /* clear the fortran-side variable */
    *wrapper_ref = nullptr;
    delete wrapper;
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
