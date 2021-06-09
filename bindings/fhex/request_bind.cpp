#include "context_bind.hpp"
#include "obj_wrapper.hpp"
#include "request_bind.hpp"

#include <vector>

using namespace gridtools::ghex::fhex;

extern "C"
bool ghex_request_test(frequest_type *freq)
{
    communicator_type::request_cb_type *req = reinterpret_cast<communicator_type::request_cb_type*>(freq->data);
    return req->test();
}

extern "C"
bool ghex_request_cancel(frequest_type *freq)
{
    communicator_type::request_cb_type *req = reinterpret_cast<communicator_type::request_cb_type*>(freq->data);
    return req->cancel();
}
