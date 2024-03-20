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
struct buffer_info_accessor
{
};

template<>
struct buffer_info_accessor<ghex::cpu>
{
    static pybind11::buffer_info get(pybind11::object& buffer)
    {
        return buffer.cast<pybind11::buffer>().request();
    }
};

template<>
struct buffer_info_accessor<ghex::gpu>
{
    static pybind11::buffer_info get(pybind11::object& buffer)
    {
        using namespace pybind11::literals;
#ifdef __HIP_PLATFORM_HCC__
        pybind11::dict info = buffer.attr("__hip_array_interface__");
#else
        pybind11::dict info = buffer.attr("__cuda_array_interface__");
#endif

        [[maybe_unused]] bool readonly = info["data"].cast<pybind11::tuple>()[1].cast<bool>();
        assert(!readonly);

        void* ptr = reinterpret_cast<void*>(
            info["data"].cast<pybind11::tuple>()[0].cast<pybind11::ssize_t>());

        // create buffer protocol format and itemsize from typestr
        pybind11::function memory_view = pybind11::module::import("builtins").attr("memoryview");
        pybind11::function np_array = pybind11::module::import("numpy").attr("array");
        pybind11::buffer   empty_buffer =
            memory_view(np_array(pybind11::list(), "dtype"_a = info["typestr"]));
        pybind11::ssize_t itemsize = empty_buffer.request().itemsize;
        std::string       format = empty_buffer.request().format;

        std::vector<pybind11::ssize_t> shape = info["shape"].cast<std::vector<pybind11::ssize_t>>();
        pybind11::ssize_t              ndim = shape.size();

        std::vector<pybind11::ssize_t> strides(ndim);
        if (pybind11::isinstance<pybind11::none>(info["strides"]))
        {
            strides[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i)
            {
                strides[i] = (strides[i + 1] * shape[i + 1]) * itemsize;
            }
        }
        else
        {
            strides = info["strides"].cast<std::vector<pybind11::ssize_t>>();
            assert(pybind11::ssize_t(strides.size()) == ndim);
        }

        return pybind11::buffer_info(
            ptr,        /* Pointer to buffer */
            itemsize,   /* Size of one scalar */
            format,     /* Python struct-style format descriptor */
            ndim,       /* Number of dimensions */
            shape,      /* Buffer dimensions */
            strides     /* Strides (in bytes) for each index */
        );
    }
};

template<typename Arch>
pybind11::buffer_info
get_buffer_info(pybind11::object& buffer)
{
    return buffer_info_accessor<Arch>::get(buffer);
}
} // namespace

void
register_field_descriptor(pybind11::module& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, field_descriptor_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace pybind11::literals;

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
            using buffer_info_type = ghex::buffer_info<pattern_type, arch_type, field_descriptor_type>;

            auto _field_descriptor = register_class<field_descriptor_type>(m);
            /*auto _buffer_info =*/ register_class<buffer_info_type>(m);

            _field_descriptor
                .def(pybind11::init(
                         [](const domain_descriptor_type& dom, pybind11::object& b,
                             const array& offsets, const array& extents)
                         {
                             pybind11::buffer_info info = get_buffer_info<arch_type>(b);

                             if (info.format != pybind11::format_descriptor<T>::format())
                             {
                                 std::stringstream error;
                                 error << "Incompatible format: expected a " << typeid(T).name()
                                       << " buffer.";
                                 throw pybind11::type_error(error.str());
                             }

                             auto ordered_strides = info.strides;
                             std::sort(ordered_strides.begin(), ordered_strides.end(), [](int a, int b) { return a > b; });
                             array b_layout_map;
                             for (size_t i = 0; i < dimension::value; ++i) {
                                 auto it = std::find(ordered_strides.begin(), ordered_strides.end(), info.strides[i]);
                                 b_layout_map[i] = std::distance(ordered_strides.begin(), it);
                                 if (b_layout_map[i] != layout_map::at(i)) {
                                     throw pybind11::type_error("Buffer has a different layout than specified.");
                                 }
                             }

                             return ghex::wrap_field<arch_type, layout_map>(dom,
                                 static_cast<T*>(info.ptr), offsets, extents, info.strides);
                         }),
                    pybind11::keep_alive<0, 2>());
        });
}

} // namespace regular
} // namespace structured
} // namespace pyghex
