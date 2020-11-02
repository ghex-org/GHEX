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
#ifndef INCLUDED_GLUE_GRIDTOOLS_MAKE_GT_PATTERN_HPP
#define INCLUDED_GLUE_GRIDTOOLS_MAKE_GT_PATTERN_HPP

#include <array>
#include "../../structured/pattern.hpp"
#include "../../structured/regular/halo_generator.hpp"

namespace gridtools {

    namespace ghex {

        template<typename Grid, typename Halos>
        auto make_gt_pattern(Grid& grid, Halos&& halos)
        {
            const std::array<int,3> first{0,0,0};
            const std::array<int,3> last{grid.m_global_extents[0]-1, grid.m_global_extents[1]-1, grid.m_global_extents[2]-1};
            using halo_gen_type = structured::regular::halo_generator<typename Grid::domain_id_type,3>;
            auto halo_gen = halo_gen_type(first,last, std::forward<Halos>(halos), grid.m_periodic);

            return make_pattern<structured::grid>(grid.m_context, halo_gen, grid.m_domains);
        }

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GLUE_GRIDTOOLS_MAKE_GT_PATTERN_HPP */

