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
#ifndef INCLUDED_GHEX_CUBED_SPHERE_DOMAIN_DESCRIPTOR_HPP
#define INCLUDED_GHEX_CUBED_SPHERE_DOMAIN_DESCRIPTOR_HPP

#include <stdexcept>
#include <cassert>
#include "../common/coordinate.hpp"

namespace gridtools {
    namespace ghex {
        namespace cubed_sphere {
    
            // forward declaration
            class halo_generator;

            /** @brief domain id type comprised of tile id and tile-local id */
            struct domain_id_t {
                int tile;
                int id;
            };

            // strict ordering of domain_id objects
            static inline bool operator>(const domain_id_t& a, const domain_id_t& b) noexcept {
                if (a.tile > b.tile) return true;
                if (b.tile > a.tile) return false;
                if (a.id > b.id) return true;
                if (b.id > a.id) return false;
                return false;
            }
            static inline bool operator<(const domain_id_t& a, const domain_id_t& b) noexcept {
                if (a.tile < b.tile) return true;
                if (b.tile < a.tile) return false;
                if (a.id < b.id) return true;
                if (b.id < a.id) return false;
                return false;
            }
            
            static inline bool operator==(const domain_id_t& a, const domain_id_t& b) noexcept {
                return a.tile==b.tile && a.id==b.id;
            }

            /** @brief implements domain descriptor concept for cubed sphere grids */
            class domain_descriptor
            {
            public: // member types
                using domain_id_type      = domain_id_t;
                // 4-dimensional: tile, x, y, z
                using dimension           = std::integral_constant<int,4>;
                using coordinate_type     = coordinate<std::array<int,dimension::value>>;
                using halo_generator_type = halo_generator;

            private: // members
                int m_x;
                domain_id_type m_id;
                coordinate_type m_first;
                coordinate_type m_last;

            public: // ctors
                /** @brief construct a local domain
                 * @tparam Array coordinate-like array type (4-dimensional)
                 * @param x_ number of grid cells along an edge of one of the cube's faces
                 * @param local_id tile-local id of subdomain
                 * @param first first coordinate in domain (global coordinate)
                 * @param last last coordinate in domain (including, global coordinate) */
                template<typename Array>
                domain_descriptor(int x_, int local_id, const Array& first, const Array& last)
                : m_x{x_}, m_id{first[0],local_id} {
                    std::copy(std::begin(first), std::end(first), m_first.begin());
                    std::copy(std::begin(last), std::end(last), m_last.begin());
                    assert(("first and last must be on same tile", first[0] == last[0]));
                }
                
                /** @brief construct a local domain
                 * @tparam Array coordinate-like array type (3-dimensional)
                 * @param x_ number of grid cells along an edge of one of the cube's faces
                 * @param tile_id tile id
                 * @param local_id tile-local id of subdomain
                 * @param first first coordinate in domain (3-dimensional)
                 * @param last last coordinate in domain (including, 3-dimensional) */
                template<typename Array>
                domain_descriptor(int x_, int tile_id, int local_id, const Array& first, const Array& last )
                : m_x{x_}, m_id{tile_id,local_id} {
                    m_first[0] = tile_id;
                    m_last[0] = tile_id;
                    std::copy(std::begin(first), std::end(first), m_first.begin()+1);
                    std::copy(std::begin(last), std::end(last), m_last.begin()+1);
                }

            public: // member functions
                int x() const noexcept { return m_x; }
                const domain_id_type& domain_id() const noexcept { return m_id; }
                const coordinate_type& first() const { return m_first; }
                const coordinate_type& last() const { return m_last; }
            };

        } // namespace cubed_sphere
    } // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_CUBED_SPHERE_DOMAIN_DESCRIPTOR_HPP
