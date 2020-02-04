#include "context_bind.hpp"
#include "future_bind.hpp"
#include "obj_wrapper.hpp"
#include <vector>


extern "C"
void ghex_future_wait(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->wait();
}

extern "C"
bool ghex_future_ready(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->ready();
}

extern "C"
bool ghex_future_cancel(ffuture_type *ffut)
{
    communicator_type::future<void> *future = reinterpret_cast<communicator_type::future<void>*>(ffut->data);
    return future->cancel();
}
