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
#ifndef INCLUDED_UNSTRUCTURED_GRID_HPP
#define INCLUDED_UNSTRUCTURED_GRID_HPP

namespace gridtools {

    namespace detail {

        template<typename Index>
        struct unstructured_grid {
            using index_t = Index;
        };

    } // namespace detail

    /** @brief type to indicate unstructured grids */
    struct unstructured_grid {
        template<typename Domain>
        using type = detail::unstructured_grid<typename Domain::index_t>;
    };

} // namespace gridtools

#endif /* INCLUDED_UNSTRUCTURED_GRID_HPP */
