/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <array>
#include <tuple>
#include <vector>
#include <cassert>
#include <sstream>
#include <numeric>
#include <algorithm>

#include <gridtools/common/for_each.hpp>

#include <ghex/buffer_info.hpp>
#include <ghex/structured/grid.hpp>
#include <ghex/structured/pattern.hpp>

#include <register_class.hpp>
#include <structured/regular/field_descriptor.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

namespace pyghex
{
namespace structured
{
namespace regular
{
namespace
{
template<int... val>
using int_tuple_constant = gridtools::meta::list<std::integral_constant<int, val>...>;

template<typename Arch>
struct ndarray_device_type;

template<>
struct ndarray_device_type<ghex::cpu>
{
    using type = nanobind::device::cpu;
};

template<>
struct ndarray_device_type<ghex::gpu>
{
#ifdef __HIP_PLATFORM_HCC__
    using type = nanobind::device::rocm;
#else
    using type = nanobind::device::cuda;
#endif
};

template<typename T>
std::vector<nanobind::ssize_t>
byte_strides(const T& b)
{
    std::vector<nanobind::ssize_t> result(b.ndim());
    const auto                     itemsize = static_cast<nanobind::ssize_t>(b.itemsize());
    for (size_t i = 0; i < b.ndim(); ++i) result[i] = b.stride(i) * itemsize;
    return result;
}

} // namespace

void
register_field_descriptor(nanobind::module_& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, field_descriptor_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace nanobind::literals;

            using field_descriptor_type = gridtools::meta::first<decltype(l)>;
            using T = typename field_descriptor_type::value_type;
            using domain_id_type = typename field_descriptor_type::domain_id_type;
            using domain_descriptor_type = typename field_descriptor_type::domain_descriptor_type;
            using arch_type = typename field_descriptor_type::arch_type;
            using layout_map = typename field_descriptor_type::layout_map;
            using dimension = typename field_descriptor_type::dimension;
            using array = std::array<int, dimension::value>;
            using grid_type = ghex::structured::grid::template type<domain_descriptor_type>;
            using pattern_type = ghex::pattern<grid_type, domain_id_type>;
            using buffer_info_type =
                ghex::buffer_info<pattern_type, arch_type, field_descriptor_type>;

            auto _field_descriptor = register_class<field_descriptor_type>(m);
            register_class<buffer_info_type>(m);

            auto make_field_descriptor =
                [](const domain_descriptor_type&                                        dom,
                    nanobind::ndarray<T, typename ndarray_device_type<arch_type>::type> b,
                    const array& offsets, const array& extents)
            {
                if (b.ndim() != dimension::value)
                {
                    std::stringstream error;
                    error << "Field has wrong dimensions. Expected " << dimension::value
                          << ", but got " << b.ndim();
                    throw nanobind::type_error(error.str().c_str());
                }

                auto strides = byte_strides(b);

                auto ordered_strides = strides;
                std::sort(ordered_strides.begin(), ordered_strides.end(),
                    [](nanobind::ssize_t a, nanobind::ssize_t b) { return a > b; });

                array b_layout_map;
                for (size_t i = 0; i < dimension::value; ++i)
                {
                    auto it = std::find(ordered_strides.begin(), ordered_strides.end(), strides[i]);
                    b_layout_map[i] = std::distance(ordered_strides.begin(), it);
                    if (b_layout_map[i] != layout_map::at(i))
                    {
                        throw nanobind::type_error("Buffer has a different layout than specified.");
                    }
                }

                return ghex::wrap_field<arch_type, layout_map>(dom, static_cast<T*>(b.data()),
                    offsets, extents, strides);
            };

            _field_descriptor.def(nanobind::new_(make_field_descriptor),
                nanobind::keep_alive<0, 3>());
        });
}

} // namespace regular
} // namespace structured
} // namespace pyghex
