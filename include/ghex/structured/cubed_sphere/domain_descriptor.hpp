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
#ifndef INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_DOMAIN_DESCRIPTOR_HPP
#define INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_DOMAIN_DESCRIPTOR_HPP

#include <stdexcept>
#include <cassert>
#include "../../common/coordinate.hpp"

namespace gridtools {
    namespace ghex {
        namespace structured {
            namespace cubed_sphere {

                /** @brief cube dimensions
                  * This struct describes the cube which is projected onto the sphere. The first
                  * member @p edge_size describes the number of cells along one edge of the cube. The second
                  * member @p levels describes the number of levels/cells in z/altitude direction */
                struct cube {
                    int edge_size;
                    int levels;
                };
        
                /** @brief domain id type comprised of tile id and tile-local id */
                struct domain_id_type {
                    int tile;
                    coordinate<std::array<int,2>> id;
                };

                // strict ordering of domain_id objects
                static inline bool operator>(const domain_id_type& a, const domain_id_type& b) noexcept {
                    if (a.tile > b.tile) return true;
                    if (b.tile > a.tile) return false;
                    if (a.id[1] > b.id[1]) return true;
                    if (a.id[1] < b.id[1]) return false;
                    if (a.id[0] > b.id[0]) return true;
                    if (a.id[0] < b.id[0]) return false;
                    return false;
                }
                static inline bool operator<(const domain_id_type& a, const domain_id_type& b) noexcept {
                    if (a.tile < b.tile) return true;
                    if (b.tile < a.tile) return false;
                    if (a.id[1] < b.id[1]) return true;
                    if (a.id[1] > b.id[1]) return false;
                    if (a.id[0] < b.id[0]) return true;
                    if (a.id[0] > b.id[0]) return false;
                    return false;
                }
                
                static inline bool operator==(const domain_id_type& a, const domain_id_type& b) noexcept {
                    return a.tile==b.tile && a.id==b.id;
                }

                /** @brief implements domain descriptor concept for cubed sphere grids */
                class domain_descriptor
                {
                public: // member types
                    using domain_id_type      = ::gridtools::ghex::structured::cubed_sphere::domain_id_type;
                    // 4-dimensional: tile, x, y, z
                    using dimension           = std::integral_constant<int,3>;
                    using coordinate_type     = coordinate<std::array<int,4>>;

                private: // members
                    cube m_c;
                    domain_id_type m_id;
                    coordinate_type m_first;
                    coordinate_type m_last;

                public: // ctors
                    
                    /** @brief construct a local domain 
                      * @param c cube instance
                      * @param tile tile/face id
                      * @param x_first first x coordinate
                      * @param x_last last x coordinate
                      * @param y_first first y coordinate
                      * @param y_last last y coordinate */
                    domain_descriptor(const cube& c, int tile, int x_first, int x_last, int y_first, int y_last)
                    : m_c{c}, m_id{tile,{x_first,y_first}} {
                        m_first[0] = tile;
                        m_last[0]  = tile;
                        m_first[1] = x_first;
                        m_last[1]  = x_last;
                        m_first[2] = y_first;
                        m_last[2]  = y_last;
                        m_first[3] = 0;
                        m_last[3]  = m_c.levels-1;
                    }

                public: // member functions
                    int edge_size() const noexcept { return m_c.edge_size; }
                    const domain_id_type& domain_id() const noexcept { return m_id; }
                    const coordinate_type& first() const { return m_first; }
                    const coordinate_type& last() const { return m_last; }
                };

            } // namespace cubed_sphere
        } // namespace structured
    } // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_DOMAIN_DESCRIPTOR_HPP
