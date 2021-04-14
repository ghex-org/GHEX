#ifndef INCLUDED_GHEX_FORTRAN_CONTEXT_BIND_HPP
#define INCLUDED_GHEX_FORTRAN_CONTEXT_BIND_HPP

#include "obj_wrapper.hpp"
#include <vector>
#include <ghex/transport_layer/util/barrier.hpp>

namespace ghex = gridtools::ghex;

#ifdef GHEX_USE_UCP

/* UCX backend */
#include <ghex/transport_layer/ucx/context.hpp>
using transport    = ghex::tl::ucx_tag;
#else

/* fallback MPI backend */
#include <ghex/transport_layer/mpi/context.hpp>
using transport    = ghex::tl::mpi_tag;
#endif

namespace gridtools {
    namespace ghex {
        namespace fhex {
            
            using context_type = typename gridtools::ghex::tl::context_factory<transport>::context_type;
            using context_uptr_type = std::unique_ptr<context_type>;
            using communicator_type = context_type::communicator_type;

            extern context_uptr_type ghex_context;
            extern int ghex_nthreads;
            extern gridtools::ghex::tl::barrier_t *ghex_barrier;
        }
    }
}

#endif /* INCLUDED_GHEX_FORTRAN_CONTEXT_BIND_HPP */
