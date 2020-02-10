/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_GHEX_STRUCTURED_GRID_HPP
#define INCLUDED_GHEX_STRUCTURED_GRID_HPP

#include "../common/coordinate.hpp"

namespace gridtools {
    namespace ghex {
        namespace structured {

    namespace detail {

        template<typename CoordinateArrayType>
        struct grid 
        {
            using coordinate_base_type    = CoordinateArrayType;
            using coordinate_type         = coordinate<coordinate_base_type>;
            using coordinate_element_type = typename coordinate_type::element_type;
            using dimension               = typename coordinate_type::dimension;    
        };

        template<typename A>
        struct grid<coordinate<A>>
        {
            using coordinate_base_type    = A;
            using coordinate_type         = coordinate<A>;
            using coordinate_element_type = typename coordinate_type::element_type;
            using dimension               = typename coordinate_type::dimension;
        };

    } // namespace detail

    /** @brief type to indicate structured grids */
    struct grid 
    {
        template<typename Domain>
        using type = detail::grid<typename Domain::coordinate_type>;
    };
        } //namespace structured
    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_GRID_HPP */

