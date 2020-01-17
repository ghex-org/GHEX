#include "message_bind.hpp"
#include "obj_wrapper.hpp"
#include <iostream>
#include <vector>

#include <ghex/transport_layer/shared_message_buffer.hpp>

namespace ghex = gridtools::ghex;

#ifdef GHEX_USE_OPENMP

/* OpenMP */
#include <ghex/threads/omp/primitives.hpp>
using threading    = ghex::threads::omp::primitives;
#else

/* no multithreading */
#include <ghex/threads/none/primitives.hpp>
using threading    = ghex::threads::none::primitives;
#endif


#ifdef GHEX_USE_UCP

/* UCX backend */
#include <ghex/transport_layer/ucx/context.hpp>
using transport    = ghex::tl::ucx_tag;
#else

/* fallback MPI backend */
#include <ghex/transport_layer/mpi/context.hpp>
using transport    = ghex::tl::mpi_tag;
#endif

using context_type = ghex::tl::context<transport, threading>;
using communicator_type = context_type::communicator_type;

/* fortran-side user callback */
typedef void (*f_callback)(void *mesg, int rank, int tag);

struct callback {
    f_callback cb;
    callback(f_callback pcb) : cb{pcb} {}
    void operator()(communicator_type::message_type &&message, int rank, int tag) {
        if(cb){
            // change any_message to a shared message. We may resubmit inside the callback, 
            // so we would like to keep working on a shared message, not on any_message
            using message_type = ghex::tl::shared_message_buffer<>;
            message_type *mptr = reinterpret_cast<message_type*>(&message.m_ptr2);
            printf("data passed to cb %p %p\n", mptr->data(), message.data());
            message_type smessage{std::move(*mptr)};
            ghex::bindings::obj_wrapper *wmessage = new ghex::bindings::obj_wrapper(std::move(smessage));
            cb(wmessage, rank, tag);
            delete wmessage;
        }
    }
};

extern "C"
void comm_delete(ghex::bindings::obj_wrapper **wrapper_ref)
{
    ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

    // clear the fortran-side variable
    *wrapper_ref = nullptr;
    delete wrapper;
}

extern "C"
int comm_progress(ghex::bindings::obj_wrapper *wrapper)
{
    return ghex::bindings::get_object_ptr_safe<communicator_type>(wrapper)->progress();
}

extern "C"
void* comm_send(ghex::bindings::obj_wrapper *wcomm, ghex::bindings::obj_wrapper *wmessage, int rank, int tag)
{
    using message_type = ghex::tl::shared_message_buffer<>;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    message_type msg{ghex::bindings::get_object_safe<message_type>(wmessage)};
    return new ghex::bindings::obj_wrapper(comm->send(msg, rank, tag));
}

extern "C"
void* comm_send_cb(ghex::bindings::obj_wrapper *wcomm, ghex::bindings::obj_wrapper *wmessage, int rank, int tag, f_callback cb)
{
    using message_type = ghex::tl::shared_message_buffer<>;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);

    // pass shared message to send by reference
    message_type *pmessage = ghex::bindings::get_object_ptr_safe<message_type>(wmessage);
    return new ghex::bindings::obj_wrapper(comm->send(*pmessage, rank, tag, callback{cb}));
}

extern "C"
void* comm_recv(ghex::bindings::obj_wrapper *wcomm, ghex::bindings::obj_wrapper *wmessage, int rank, int tag)
{
    using message_type = ghex::tl::shared_message_buffer<>;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);
    message_type msg{ghex::bindings::get_object_safe<message_type>(wmessage)};
    return new ghex::bindings::obj_wrapper(comm->recv(msg, rank, tag));
}

extern "C"
void* comm_recv_cb(ghex::bindings::obj_wrapper *wcomm, ghex::bindings::obj_wrapper *wmessage, int rank, int tag, f_callback cb)
{
    using message_type = ghex::tl::shared_message_buffer<>;
    communicator_type *comm = ghex::bindings::get_object_ptr_safe<communicator_type>(wcomm);

    // pass shared message to send by reference
    message_type *pmessage = ghex::bindings::get_object_ptr_safe<message_type>(wmessage);
    return new ghex::bindings::obj_wrapper(comm->recv(*pmessage, rank, tag, callback{cb}));
}
