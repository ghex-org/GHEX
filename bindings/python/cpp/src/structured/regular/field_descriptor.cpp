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

#include <cassert>
#include <sstream>

#include <gridtools/common/generic_metafunctions/for_each.hpp>
#include <gridtools/meta.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/utils/demangle.hpp>
#include <ghex/bindings/python/type_list.hpp>

#include <ghex/structured/regular/field_descriptor.hpp>

namespace py = pybind11;

template <int... val>
using int_tuple_constant = gridtools::meta::list<std::integral_constant<int, val>...>;


namespace detail {
    using args = gridtools::meta::cartesian_product<
        gridtools::ghex::bindings::python::type_list::data_types,
        gridtools::meta::list<gridtools::ghex::cpu>,
        gridtools::meta::list<gridtools::ghex::structured::regular::domain_descriptor<int, 3>>,
        gridtools::meta::list<int_tuple_constant<0, 1, 2>, int_tuple_constant<2, 1, 0>>>;

    template<typename T, typename Arch, typename DomainDescriptor, typename Order>
    using field_descriptor_type = gridtools::ghex::structured::regular::field_descriptor<
        T, Arch, DomainDescriptor, gridtools::meta::at_c<Order, 0>::value, gridtools::meta::at_c<Order, 1>::value, gridtools::meta::at_c<Order, 2>::value
    >;

    using field_descriptor_specializations = gridtools::meta::transform<gridtools::meta::rename<field_descriptor_type>::template apply, args>;
}

// todo: dim independent
template<typename T, typename Arch, typename DomainDescriptor, int I, int J, int K>
struct type_exporter<gridtools::ghex::structured::regular::field_descriptor<T, Arch, DomainDescriptor, I, J, K>> {
    using field_descriptor_type = gridtools::ghex::structured::regular::field_descriptor<T, Arch, DomainDescriptor, I, J, K>;

    void operator() (pybind11::module_&, py::class_<field_descriptor_type> cls) {
        constexpr std::size_t dim = 3;

        using array_type = std::array<int, dim>;

        auto wrapper = [] (const DomainDescriptor& dom, py::buffer b, const array_type& offsets, const array_type& extents) {
            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<T>::format()) {
                std::stringstream error;
                error << "Incompatible format: expected a " << typeid(T).name() << " buffer.";
                throw pybind11::type_error(error.str());
            }

            std::array<int, 3> order = {0, 1, 2};
            std::sort(order.begin(), order.end(), [&info](int a, int b) {
                return info.strides[a] > info.strides[b];
            });
            if (I != order[0] || J != order[1] || K != order[2]) {
                throw pybind11::type_error("Buffer has a different layout than specified.");
            }

            if (info.ndim != 3)
                throw std::runtime_error("Incompatible buffer dimension.");

            return gridtools::ghex::wrap_field<Arch, I, J, K>(dom, static_cast<T*>(info.ptr), offsets, extents);
        };

        cls.def(py::init(wrapper));
    }
};

GHEX_PYBIND11_EXPORT_TYPE(type_exporter, detail::field_descriptor_specializations)

/*
GHEX_PYBIND_EXPORT_IN_CURRENT_MODULE(field_descriptor_exporter,
    ghex4py::type_list::architecture_types,
    metal::list<domain_descriptor_type>,
    ghex4py::type_list::data_types,
    orders)

GHEX_PYBIND11_EXPORT_TYPE(gridtools::ghex::structured::regular::field_descriptor,
    ghex4py::type_list::architecture_types,
    metal::list<domain_descriptor_type>,
    ghex4py::type_list::data_types,
    orders)
*/