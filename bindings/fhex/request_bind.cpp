#include "context_bind.hpp"
#include "obj_wrapper.hpp"
#include "request_bind.hpp"

#include <vector>

using namespace gridtools::ghex::fhex;

extern "C"
bool ghex_request_test_single(frequest_type *freq)
{
    communicator_type::request_cb_type *req = reinterpret_cast<communicator_type::request_cb_type*>(freq->data);
    return req->test();
}

extern "C"
bool ghex_request_test_multi(frequest_multi_type *freq)
{
    using rtype = communicator_type::request_cb_type;
    bool ret = true;
    std::vector<rtype> &requests = reinterpret_cast<std::vector<rtype>&>(freq->data);
    for(long unsigned int i=0; i<requests.size(); i++){
        ret = ret && requests[i].test();
    }
    return ret;
}

extern "C"
bool ghex_request_cancel(frequest_type *freq)
{
    communicator_type::request_cb_type *req = reinterpret_cast<communicator_type::request_cb_type*>(freq->data);
    return req->cancel();
}
