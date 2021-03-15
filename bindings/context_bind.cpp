#include "context_bind.hpp"
#include <mpi.h>
#include <sched.h>
#include <sys/sysinfo.h>

context_uptr_type context;
int __GHEX_nthreads = 1;

/* barrier has to be shared between the threads */
gridtools::ghex::tl::barrier_t *barrier = NULL;

extern "C"
void ghex_init(int nthreads, MPI_Fint fcomm)
{
    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);    
    context = ghex::tl::context_factory<transport>::create(ccomm);
    __GHEX_nthreads = nthreads;    
    barrier = new gridtools::ghex::tl::barrier_t(nthreads);
}

extern "C"
void ghex_finalize()
{
    context.reset();
    delete barrier;
    barrier = NULL;
    __GHEX_nthreads = 1;
}

extern "C"
void ghex_obj_free(ghex::bindings::obj_wrapper **wrapper_ref)
{
    ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

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
