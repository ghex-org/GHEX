#define GHEX_DEBUG_LEVEL 2

#include "message_bind.hpp"
#include "obj_wrapper.hpp"
#include <iostream>
#include <vector>
#include <transport_layer/mpi/communicator.hpp>
#include <allocator/persistent_allocator.hpp>

typedef void (**f_callback)(int rank, int tag, void *mesg);

using t_communicator = gridtools::ghex::mpi::communicator;

extern "C"
void *comm_new()
{
    return new gridtools::ghex::bindings::obj_wrapper(t_communicator());
}

extern "C"
void comm_delete(gridtools::ghex::bindings::obj_wrapper **wrapper_ref)
{
    gridtools::ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

    /* clear the fortran-side variable */
    *wrapper_ref = nullptr;
    delete wrapper;
}

extern "C"
int comm_progress(gridtools::ghex::bindings::obj_wrapper *wrapper)
{
    return gridtools::ghex::bindings::get_object_ptr<t_communicator>(wrapper)->progress();
}

template <typename T>
struct callback {
    T msg;
    f_callback cb;

    callback(T m, f_callback pcb) : msg{m}, cb{pcb} {}
    
    void operator()(int rank, int tag) {
     	if(cb) {
	    gridtools::ghex::bindings::obj_wrapper wrapper(msg);
	    (*cb)(rank, tag, &wrapper);
	}
    }
};

extern "C"
void* comm_send(gridtools::ghex::bindings::obj_wrapper *wcomm, gridtools::ghex::bindings::obj_wrapper *wmessage, int rank, int tag)
{
    SHARED_MESSAGE_CALL(wmessage, {
	    return new gridtools::ghex::bindings::obj_wrapper(gridtools::ghex::bindings::get_object_ptr<t_communicator>(wcomm)->send(msg, rank, tag));
	} );
}

extern "C"
void comm_send_cb(gridtools::ghex::bindings::obj_wrapper *wcomm, gridtools::ghex::bindings::obj_wrapper *wmessage, int rank, int tag, f_callback cb)
{
    SHARED_MESSAGE_CALL(wmessage, {
	    gridtools::ghex::bindings::get_object_ptr<t_communicator>(wcomm)->send(msg, rank, tag, callback<message_type>{msg, cb});
	} );
}

extern "C"
void comm_send_multi(gridtools::ghex::bindings::obj_wrapper *wcomm, gridtools::ghex::bindings::obj_wrapper *wmessage, const int *ranks, int n_ranks, int tag, f_callback cb)
{
    SHARED_MESSAGE_CALL(wmessage, {
	    gridtools::ghex::bindings::get_object_ptr<t_communicator>(wcomm)->send_multi(msg, std::vector<int>(ranks, ranks + n_ranks), tag, callback<message_type>{msg, cb});
	} );
}

extern "C"
void* comm_recv(gridtools::ghex::bindings::obj_wrapper *wcomm, gridtools::ghex::bindings::obj_wrapper *wmessage, int rank, int tag)
{
    SHARED_MESSAGE_CALL(wmessage, {
	    return new gridtools::ghex::bindings::obj_wrapper(gridtools::ghex::bindings::get_object_ptr<t_communicator>(wcomm)->recv(msg, rank, tag));
	} );
}

extern "C"
void comm_recv_cb(gridtools::ghex::bindings::obj_wrapper *wcomm, gridtools::ghex::bindings::obj_wrapper *wmessage, int rank, int tag, f_callback cb)
{
    SHARED_MESSAGE_CALL(wmessage, {
	    gridtools::ghex::bindings::get_object_ptr<t_communicator>(wcomm)->recv(msg, rank, tag, callback<message_type>{msg, cb});
	} );
}

extern "C"
void *comm_detach(gridtools::ghex::bindings::obj_wrapper *wcomm, int rank, int tag)
{
    t_communicator::future_type future = gridtools::ghex::bindings::get_object_ptr<t_communicator>(wcomm)->detach(rank, tag);
    return new gridtools::ghex::bindings::obj_wrapper(future);
}
