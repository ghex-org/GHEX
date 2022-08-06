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
#ifdef __CUDACC__
    using architecture_types = gridtools::meta::list<gridtools::ghex::cpu, gridtools::ghex::gpu>;
#else
    using architecture_types = gridtools::meta::list<gridtools::ghex::cpu>;
#endif
    using domain_id_types = gridtools::meta::list<int>;
    using data_types = gridtools::meta::list<double, float>;
    using grid_types = gridtools::meta::list<gridtools::ghex::structured::grid>;
    using dims = gridtools::meta::list<std::integral_constant<int, 3>>;
    // TODO: layout maps per dim
    using layout_maps = gridtools::meta::list<
        gridtools::layout_map<0, 1, 2>,
        gridtools::layout_map<0, 2, 1>,
        gridtools::layout_map<1, 0, 2>,
        gridtools::layout_map<1, 2, 0>,
        gridtools::layout_map<2, 0, 1>,
        gridtools::layout_map<2, 1, 0>
    >;

    template <typename DomainDescriptor>
    using domain_range_type = std::vector<DomainDescriptor>;

    using transport = gridtools::ghex::tl::mpi_tag;
    using context_type = typename gridtools::ghex::tl::context_factory<transport>::context_type;
    using communicator_type = typename context_type::communicator_type;

    using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<int,std::integral_constant<int, 3>>;
};

}
}
}
}