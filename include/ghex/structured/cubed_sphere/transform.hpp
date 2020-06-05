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
#ifndef INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_TRANSFORM_HPP
#define INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_TRANSFORM_HPP

#include <array>

namespace gridtools {
    namespace ghex {
        namespace structured {
            namespace cubed_sphere {

                // cubed sphere tiles and coordinate system
                //
                //            +----------+
                //            |       (2)|
                //            |      x ∧ |
                //            |        | |
                //            |    y<--+ |
                // +----------+----------+----------+----------+
                // | +-->y (4)|       (0)|       (1)| +-->y (3)|
                // | |        | ∧ y      | ∧ y      | |        |
                // | ∨ x      | |        | |        | ∨ x      |
                // |          | +-->x    | +-->x    |          |
                // +----------+----------+----------+----------+
                //            |       (5)|
                //            | ∧ y      |
                //            | |        |
                //            | +-->x    |
                //            +----------+

                /** @brief affine transform matrix for tile coordinate transforms */
                struct transform {
                    std::array<int,4> m_rotation;
                    std::array<int,2> m_translation;
                    std::array<int,2> m_offset;
                    constexpr bool switched_xy() const noexcept {return m_rotation[0]==0;}
                    constexpr bool reversed_x() const noexcept {
                        return switched_xy() ? (m_rotation[1]==-1) : (m_rotation[0]==-1);
                    }
                    constexpr bool reversed_y() const noexcept {
                        return switched_xy() ? (m_rotation[2]==-1) : (m_rotation[3]==-1);
                    }
                    /** @brief transform tile coordinates
                      * @param x x coordinate
                      * @param y y coordinate
                      * @param c number of cells along the cube edges
                      * @return transformed coordinate array */
                    constexpr std::array<int,2> operator()(int x, int y, int c) const noexcept {
                        return { m_rotation[0]*x + m_rotation[1]*y + m_translation[0]*c + m_offset[0],
                                 m_rotation[2]*x + m_rotation[3]*y + m_translation[1]*c + m_offset[1]};
                    }
                };
                
                // neighbor tiles, order: -x,+x,-y,+y
                static constexpr std::array<std::array<int,4>,6> tile_lu = {
                    std::array<int,4>{4,1,5,2}, // 0
                    std::array<int,4>{0,3,5,2}, // 1
                    std::array<int,4>{0,3,1,4}, // 2
                    std::array<int,4>{2,5,1,4}, // 3
                    std::array<int,4>{2,5,3,0}, // 4
                    std::array<int,4>{4,1,3,0}  // 5
                };

                static constexpr transform identity_transform = {{1,0,0,1},{0,0},{0,0}};

                // transform of tile coordinates to neighbor tile coordinates, order -x,+x,-y,+y
                static constexpr std::array<std::array<transform,4>,6> transform_lu = {
                    std::array<transform,4>{
                        transform{{ 0,-1, 1, 0},{ 1, 1},{-1, 0}},
                        transform{{ 1, 0, 0, 1},{-1, 0},{ 0, 0}},
                        transform{{ 1, 0, 0, 1},{ 0, 1},{ 0, 0}},
                        transform{{ 0, 1,-1, 0},{-1, 1},{ 0,-1}}},
                    std::array<transform,4>{
                        transform{{ 1, 0, 0, 1},{ 1, 0},{ 0, 0}},
                        transform{{ 0,-1, 1, 0},{ 1,-1},{-1, 0}},
                        transform{{ 0, 1,-1, 0},{ 1, 1},{ 0,-1}},
                        transform{{ 1, 0, 0, 1},{ 0,-1},{ 0, 0}}},
                    std::array<transform,4>{
                        transform{{ 0,-1, 1, 0},{ 1, 1},{-1, 0}},
                        transform{{ 1, 0, 0, 1},{-1, 0},{ 0, 0}},
                        transform{{ 1, 0, 0, 1},{ 0, 1},{ 0, 0}},
                        transform{{ 0, 1,-1, 0},{-1, 1},{ 0,-1}}},
                    std::array<transform,4>{
                        transform{{ 1, 0, 0, 1},{ 1, 0},{ 0, 0}},
                        transform{{ 0,-1, 1, 0},{ 1,-1},{-1, 0}},
                        transform{{ 0, 1,-1, 0},{ 1, 1},{ 0,-1}},
                        transform{{ 1, 0, 0, 1},{ 0,-1},{ 0, 0}}},
                    std::array<transform,4>{
                        transform{{ 0,-1, 1, 0},{ 1, 1},{-1, 0}},
                        transform{{ 1, 0, 0, 1},{-1, 0},{ 0, 0}},
                        transform{{ 1, 0, 0, 1},{ 0, 1},{ 0, 0}},
                        transform{{ 0, 1,-1, 0},{-1, 1},{ 0,-1}}},
                    std::array<transform,4>{
                        transform{{ 1, 0, 0, 1},{ 1, 0},{ 0, 0}},
                        transform{{ 0,-1, 1, 0},{ 1,-1},{-1, 0}},
                        transform{{ 0, 1,-1, 0},{ 1, 1},{ 0,-1}},
                        transform{{ 1, 0, 0, 1},{ 0,-1},{ 0, 0}}}
                };

                // inverse transform: neigbhor tile coordinates to this tile coordinates
                static constexpr std::array<std::array<transform,4>,6> inverse_transform_lu = {
                    std::array<transform,4>{
                        transform_lu[4][3],
                        transform_lu[1][0],
                        transform_lu[5][3],
                        transform_lu[2][0]},
                    std::array<transform,4>{
                        transform_lu[0][1],
                        transform_lu[3][2],
                        transform_lu[5][1],
                        transform_lu[2][2]},
                    std::array<transform,4>{
                        transform_lu[0][3],
                        transform_lu[3][0],
                        transform_lu[1][3],
                        transform_lu[4][0]},
                    std::array<transform,4>{
                        transform_lu[2][1],
                        transform_lu[5][2],
                        transform_lu[1][1],
                        transform_lu[4][2]},
                    std::array<transform,4>{
                        transform_lu[2][3],
                        transform_lu[5][0],
                        transform_lu[3][3],
                        transform_lu[0][0]},
                    std::array<transform,4>{
                        transform_lu[4][1],
                        transform_lu[1][2],
                        transform_lu[3][1],
                        transform_lu[0][2]}
                };

            } // namespace cubed_sphere
        } // namespace structured
    } // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_TRANSFORM_HPP
