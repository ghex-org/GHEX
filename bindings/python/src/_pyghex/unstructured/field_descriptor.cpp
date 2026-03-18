/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <vector>
#include <sstream>

#include <gridtools/common/for_each.hpp>

#include <ghex/buffer_info.hpp>
#include <ghex/unstructured/grid.hpp>
#include <ghex/unstructured/pattern.hpp>

#include <register_class.hpp>
#include <unstructured/field_descriptor.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace pyghex
{
namespace unstructured
{

namespace
{
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
nanobind::ssize_t
byte_stride(const T& b, const size_t dim)
{
    return b.stride(dim) * static_cast<nanobind::ssize_t>(b.itemsize());
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

            using type = gridtools::meta::first<decltype(l)>;
            using T = typename type::value_type;
            using domain_id_type = typename type::domain_id_type;
            using domain_descriptor_type = typename type::domain_descriptor_type;
            using arch_type = typename type::arch_type;
            using grid_type = ghex::unstructured::grid::template type<domain_descriptor_type>;
            using pattern_type = ghex::pattern<grid_type, domain_id_type>;
            using buffer_info_type = ghex::buffer_info<pattern_type, arch_type, type>;

            auto _field_descriptor = register_class<type>(m);
            register_class<buffer_info_type>(m);

            auto make_field_descriptor =
                [](const domain_descriptor_type&                                        dom,
                    nanobind::ndarray<T, typename ndarray_device_type<arch_type>::type> b)
            {
                if (b.ndim() > 2u)
                {
                    std::stringstream error;
                    error << "Field has too many dimensions. Expected at most 2, but got "
                          << b.ndim();
                    throw nanobind::type_error(error.str().c_str());
                }

                if (static_cast<std::size_t>(b.shape(0)) != dom.size())
                {
                    std::stringstream error;
                    error << "Field's first dimension (" << static_cast<std::size_t>(b.shape(0))
                          << ") must match the size of the domain (" << dom.size() << ")";
                    throw nanobind::type_error(error.str().c_str());
                }

                bool        levels_first = true;
                std::size_t outer_strides = 0u;
                const auto  stride_0 = byte_stride(b, 0);
                const auto  stride_1 = (b.ndim() == 2) ? byte_stride(b, 1) : nanobind::ssize_t{0};
                if (b.ndim() == 2 && stride_1 != sizeof(T))
                {
                    levels_first = false;
                    if (stride_0 != sizeof(T))
                    {
                        std::stringstream error;
                        error << "Field's strides are not compatible with GHEX. Expected that the "
                                 "(byte) stride of dimension 0 is "
                              << sizeof(T) << " but got " << (std::size_t)(stride_0) << ".";
                        throw nanobind::type_error(error.str().c_str());
                    }
                    if (((std::size_t)(stride_1) % sizeof(T)) != 0)
                    {
                        std::stringstream error;
                        error << "Field's strides are not compatible with GHEX. Expected that the "
                                 "(byte) stride of dimension 1  "
                              << (std::size_t)(stride_1) << " is a multiple of the element size "
                              << sizeof(T) << ".";
                        throw nanobind::type_error(error.str().c_str());
                    }
                    outer_strides = stride_1 / sizeof(T);
                }
                else if (b.ndim() == 2)
                {
                    // stride_1 == sizeof(T): levels are the inner (fast) dimension
                    if (((std::size_t)(stride_0) % sizeof(T)) != 0)
                    {
                        std::stringstream error;
                        error << "Field's strides are not compatible with GHEX. Expected that the "
                                 "(byte) stride of dimension 0 "
                              << (std::size_t)(stride_0) << " is a multiple of the element size of "
                              << sizeof(T) << ".";
                        throw nanobind::type_error(error.str().c_str());
                    }
                    outer_strides = stride_0 / sizeof(T);
                }
                else if (stride_0 != sizeof(T))
                {
                    std::stringstream error;
                    error
                        << "Field's strides are not compatible with GHEX. With one dimension expected "
                           "the stride to be "
                        << sizeof(T) << " but got " << stride_0 << ".";
                    throw nanobind::type_error(error.str().c_str());
                }

                const std::size_t levels = (b.ndim() == 1) ? 1u : (std::size_t)b.shape(1);
                return type{dom, static_cast<T*>(b.data()), levels, levels_first, outer_strides};
            };

            _field_descriptor.def(nanobind::new_(make_field_descriptor),
                nanobind::keep_alive<0, 3>());
        });
}

} // namespace unstructured
} // namespace pyghex
