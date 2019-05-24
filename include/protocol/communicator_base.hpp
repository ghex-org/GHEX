// 
// GridTools
// 
// Copyright (c) 2014-2019, ETH Zurich
// All rights reserved.
// 
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
// 
#ifndef INCLUDED_COMMUNICATOR_BASE_HPP
#define INCLUDED_COMMUNICATOR_BASE_HPP

#include "future.hpp"

namespace gridtools {

namespace protocol {

template<typename P>
class communicator 
{
    // concept
    // -------
    // using protocol_type = P;
    // using handle_type   = ...;
    // using address_type  = ...;
    // template<typename T>
    // using future = future_base<handle_type,T>;
};

} // namespace protocol

} // namespace gridtools

#endif /* INCLUDED_COMMUNICATOR_BASE_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

