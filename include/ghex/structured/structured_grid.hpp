/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_STRUCTURED_GRID_HPP
#define INCLUDED_STRUCTURED_GRID_HPP

#include "../common/coordinate.hpp"

namespace gridtools {

    namespace detail {

        template<typename CoordinateArrayType>
        struct structured_grid 
        {
            using coordinate_base_type    = CoordinateArrayType;
            using coordinate_type         = coordinate<coordinate_base_type>;
            using coordinate_element_type = typename coordinate_type::element_type;
            using dimension               = typename coordinate_type::dimension;    
        };

    } // namespace detail

    /** @brief type to indicate structured grids */
    struct structured_grid 
    {
        template<typename Domain>
        using type = detail::structured_grid<typename Domain::coordinate_type>;
    };

} // namespace gridtools

#endif /* INCLUDED_STRUCTURED_GRID_HPP */

