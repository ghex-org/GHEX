#include "obj_wrapper.hpp"
#include <transport_layer/mpi/communicator.hpp>

using t_communicator = gridtools::ghex::mpi::communicator;

extern "C"
void future_wait(gridtools::ghex::bindings::obj_wrapper *wfuture)
{
    gridtools::ghex::bindings::get_object<t_communicator::future_type>(wfuture).wait();
}

extern "C"
bool future_ready(gridtools::ghex::bindings::obj_wrapper *wfuture)
{
    return gridtools::ghex::bindings::get_object<t_communicator::future_type>(wfuture).ready();
}

extern "C"
bool future_cancel(gridtools::ghex::bindings::obj_wrapper *wfuture)
{
    return gridtools::ghex::bindings::get_object<t_communicator::future_type>(wfuture).cancel();
}
