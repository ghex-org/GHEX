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

namespace pyghex
{
namespace unstructured
{
namespace
{

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

            using type = gridtools::meta::first<decltype(l)>;
            using T = typename type::value_type;
            using domain_id_type = typename type::domain_id_type;
            using domain_descriptor_type = typename type::domain_descriptor_type;
            using arch_type = typename type::arch_type;
            using grid_type = ghex::unstructured::grid::template type<domain_descriptor_type>;
            using pattern_type = ghex::pattern<grid_type, domain_id_type>;
            using buffer_info_type = ghex::buffer_info<pattern_type, arch_type, type>;

            auto _field_descriptor = register_class<type>(m);
            /*auto _buffer_info = */register_class<buffer_info_type>(m);

            _field_descriptor
                .def(pybind11::init(
                    [](const domain_descriptor_type& dom, pybind11::object& b)
                    {
                        pybind11::buffer_info info = get_buffer_info<arch_type>(b);

                        if (info.format != pybind11::format_descriptor<T>::format())
                        {
                            std::stringstream error;
                            error << "Incompatible format: expected a " << typeid(T).name()
                                  << " buffer.";
                            throw pybind11::type_error(error.str());
                        }

                        if (info.ndim > 2u)
                        {
                            throw pybind11::type_error("field has too many dimensions");
                        }

                        if (static_cast<std::size_t>(info.shape[0]) != dom.size())
                        {
                            throw pybind11::type_error(
                                "field's first dimension must match the size of the domain");
                        }

                        bool levels_first = true;
                        std::size_t outer_strides = 0u;
                        if (info.ndim == 2 && info.strides[1] != sizeof(T))
                        {
                            levels_first = false;
                            if (info.strides[0] != sizeof(T))
                                throw pybind11::type_error("field's strides are not compatible with GHEX");
                            outer_strides = info.strides[1] / sizeof(T);
                            if (outer_strides*sizeof(T) != (std::size_t)(info.strides[1]))
                                throw pybind11::type_error("field's strides are not compatible with GHEX");
                        }
                        else if (info.ndim == 2)
                        {
                            if (info.strides[1] != sizeof(T))
                                throw pybind11::type_error("field's strides are not compatible with GHEX");
                            outer_strides = info.strides[0] / sizeof(T);
                            if (outer_strides*sizeof(T) != (std::size_t)(info.strides[0]))
                                throw pybind11::type_error("field's strides are not compatible with GHEX");
                        }
                        else
                        {
                            if (info.strides[0] != sizeof(T))
                                throw pybind11::type_error("field's strides are not compatible with GHEX");
                        }
                        std::size_t levels =
                            (info.ndim == 1) ? 1u : (std::size_t)info.shape[1];

                        return type{dom, static_cast<T*>(info.ptr), levels, levels_first};
                }),
                pybind11::keep_alive<0, 2>());
        });
}

} // namespace unstructured
} // namespace pyghex
