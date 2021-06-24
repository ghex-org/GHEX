#include "context_bind.hpp"
#include <mpi.h>
#include <sched.h>
#include <sys/sysinfo.h>

using namespace gridtools::ghex::fhex;

namespace gridtools {
    namespace ghex {
        namespace fhex {
            context_uptr_type ghex_context;
            int ghex_nthreads = 1;
            
            /* barrier has to be shared between the threads */
            gridtools::ghex::tl::barrier_t *ghex_barrier = nullptr;

#ifdef GHEX_USE_UCP
            hwmalloc::ucx::context *c = nullptr;
            hwmalloc::heap<hwmalloc::ucx::context> *h = nullptr;
#else
            hwmalloc::mpi::context *c = nullptr;
            hwmalloc::heap<hwmalloc::mpi::context> *h = nullptr;
#endif
        }
    }
}

extern "C"
void ghex_init(int nthreads, MPI_Fint fcomm)
{
    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);
    gridtools::ghex::fhex::ghex_context = ghex::tl::context_factory<transport>::create(ccomm);
    gridtools::ghex::fhex::ghex_nthreads = nthreads;    
    gridtools::ghex::fhex::ghex_barrier = new gridtools::ghex::tl::barrier_t(nthreads);

#ifdef GHEX_USE_UCP
    c = new hwmalloc::ucx::context{gridtools::ghex::fhex::ghex_context->get_transport_context()};
    h = new hwmalloc::heap<hwmalloc::ucx::context>(c);
#else
    c = new hwmalloc::mpi::context{gridtools::ghex::fhex::ghex_context->get_transport_context()};
    h = new hwmalloc::heap<hwmalloc::mpi::context>(c);
#endif
}

extern "C"
void ghex_finalize()
{
    gridtools::ghex::fhex::ghex_context.reset();
    delete gridtools::ghex::fhex::ghex_barrier;
    gridtools::ghex::fhex::ghex_barrier = nullptr;
    gridtools::ghex::fhex::ghex_nthreads = 1;
}

extern "C"
void ghex_obj_free(obj_wrapper **wrapper_ref)
{
    obj_wrapper *wrapper = *wrapper_ref;

    /* clear the fortran-side variable */
    *wrapper_ref = nullptr;
    delete wrapper;
}

extern "C"
int ghex_get_current_cpu()
{
    return sched_getcpu();
}

extern "C"
int ghex_get_ncpus()
{
    return get_nprocs_conf();
}
