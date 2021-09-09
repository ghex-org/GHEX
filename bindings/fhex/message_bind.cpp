/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <fhex/context_bind.hpp>
#include <cstring>

//namespace gridtools {
//    namespace ghex {
//        namespace fhex {
//
//            typedef enum
//            {
//                 GhexAllocatorHost=1,
//                 GhexAllocatorDevice=2
//            } ghex_allocator_type;
//
//            using host_allocator_type = std::allocator<unsigned char>;
//        }
//    }
//}

extern "C" void*
ghex_message_new(std::size_t size)
{
    return new ghex::context::message_type{fhex::context().make_buffer(size)};
}

extern "C" void
ghex_message_free(ghex::context::message_type** wmsg)
{
    auto msg = *wmsg;
    // clear the fortran-side variable
    wmsg = nullptr;
    delete msg;
}

extern "C" void
ghex_message_zero(ghex::context::message_type* wmsg)
{
    std::memset(wmsg->data(), 0, wmsg->size());
}

extern "C" unsigned char*
ghex_message_data(ghex::context::message_type* wmsg, std::size_t* size)
{
    *size = wmsg->size();
    return wmsg->data();
}
