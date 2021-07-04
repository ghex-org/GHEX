/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#pragma once

#include <ghex/arch_traits.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/transport_layer/context.hpp>

#include <ghex/transport_layer/mpi/context.hpp>

#include <ghex/structured/pattern.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>

#include <ghex/bindings/python/type_list.hpp>
#include <ghex/bindings/python/utils/register_wrapper.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {

struct type_list {
    using architecture_types = gridtools::meta::list<gridtools::ghex::cpu>; // todo: gpu
    using domain_id_types = gridtools::meta::list<int>;
    using data_types = gridtools::meta::list<double, float>;

    using transport = gridtools::ghex::tl::mpi_tag;
    using context_type = typename gridtools::ghex::tl::context_factory<transport>::context_type;

    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<int,3>;
    using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<int,3>;
    //template<typename T, typename Arch, int... Is>
    //using field_descriptor_type  = gridtools::ghex::structured::regular::field_descriptor<T,Arch,domain_descriptor_type, Is...>;
};

}
}
}
}