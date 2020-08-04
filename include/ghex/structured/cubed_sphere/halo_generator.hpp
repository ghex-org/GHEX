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
#ifndef INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_HALO_GENERATOR_HPP
#define INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_HALO_GENERATOR_HPP

#include <algorithm>
#include <vector>
#include "./transform.hpp"
#include "./domain_descriptor.hpp"

namespace gridtools {
    namespace ghex {
        namespace structured {
            namespace cubed_sphere {
        
                /** @brief halo generator for cubed-sphere grids */
                class halo_generator {
                public: // member types
                    using domain_type = domain_descriptor;
                    using coordinate_type = domain_type::coordinate_type;

                private: // member types
                    // describes an iteration space
                    struct box {
                        const coordinate_type& first() const { return m_first; }
                        const coordinate_type& last() const { return m_last; }
                        coordinate_type& first() { return m_first; }
                        coordinate_type& last() { return m_last; }
                        coordinate_type m_first;
                        coordinate_type m_last;
                    };

                    // describes a pair of iteration spaces (local and global)
                    struct box2 {
                        const box& local() const { return m_local; }
                        const box& global() const { return m_global; }
                        box& local() { return m_local; }
                        box& global() { return m_global; }
                        box m_local;
                        box m_global;
                    };

                private: // members
                    std::array<int, 4> m_halo;

                public: // ctors
                    /** @brief add an equal halo around the domain (default case)
                      * @param halo number of halo cells */
                    halo_generator(int halo) noexcept : m_halo{halo,halo,halo,halo} {}

                    /** @brief construct with custom halo
                      * @tparam Array 4-D array type
                      * @param halos halos in +x,-x,-y,+y directions (for staggered grids) */
                    template<typename Array>
                    halo_generator(Array&& halos) noexcept {
                        std::copy(std::begin(halos), std::end(halos), m_halo.begin());
                    }
            
                public: // member functions
                    /** @brief returns a vector of generated halo regions */
                    auto operator()(const domain_type& d) const {
                        // get tile id
                        const int tile_id = d.domain_id().tile;
                        // get resolution of cube
                        const int c = d.edge_size();
                        // domain in global tile coords
                        const box domain_box{ d.first(), d.last() };
                        // domain in local tile coords
                        const box domain_box_local{
                            coordinate_type{0/*tile_id*/, 0, 0, 0},
                            coordinate_type{0/*tile_id*/, d.last()[1]-d.first()[1],
                                                     d.last()[2]-d.first()[2], 
                                                     d.last()[3]-d.first()[3]}}; 
                        // cube tile in global tile coords
                        const box tile_box{coordinate_type{tile_id,  0,  0,d.first()[3]},
                                           coordinate_type{tile_id,c-1,c-1,d.last()[3]}};
                        // create 4 neighbor regions based on neighbor tiles
                        // in global tile coords
                        auto n_m0 = tile_box;
                        n_m0.first()[1] -= c;
                        n_m0.last()[1] -= c;
                        auto n_p0 = tile_box;
                        n_p0.first()[1] += c;
                        n_p0.last()[1] += c;
                        auto n_0m = tile_box;
                        n_0m.first()[2] -= c;
                        n_0m.last()[2] -= c;
                        auto n_0p = tile_box;
                        n_0p.first()[2] += c;
                        n_0p.last()[2] += c;
                        std::vector<box> neighbor_regions { n_m0, n_p0, n_0m, n_0p };

                        // create 4 halo regions: bottom, left, right, top halo regions
                        // bottom and top regions include left and right corners
                        // save as box2: local and global tile coords
                        std::vector<box2> halo_regions;
                        halo_regions.reserve(4);
                        {
                            auto h = domain_box;
                            h.first()[1] -= m_halo[0];
                            h.first()[2] -= m_halo[2];
                            h.last()[1]   = domain_box.last()[1]+m_halo[1];
                            h.last()[2]   = domain_box.first()[2]-1;
                            auto h_loc = box {
                                domain_box_local.first() + (h.first() - domain_box.first()),
                                domain_box_local.last()  + (h.last()  - domain_box.last())};
                            halo_regions.push_back( box2{h_loc, h} );
                        }
                        {
                            auto h = domain_box;
                            h.first()[1] -= m_halo[0];
                            h.last()[1]   = domain_box.first()[1]-1;
                            auto h_loc = box {
                                domain_box_local.first() + (h.first() - domain_box.first()),
                                domain_box_local.last()  + (h.last()  - domain_box.last())};
                            halo_regions.push_back( box2{h_loc, h} );
                        }
                        {
                            auto h = domain_box;
                            h.first()[1]  = domain_box.last()[1]+1;
                            h.last()[1]   = domain_box.last()[1]+m_halo[1];
                            auto h_loc = box {
                                domain_box_local.first() + (h.first() - domain_box.first()),
                                domain_box_local.last()  + (h.last()  - domain_box.last())};
                            halo_regions.push_back( box2{h_loc, h} );
                        }
                        {
                            auto h = domain_box;
                            h.first()[1] -= m_halo[0];
                            h.first()[2]  = domain_box.last()[2]+1;
                            h.last()[1]   = domain_box.last()[1]+m_halo[1];
                            h.last()[2]   = domain_box.last()[2]+m_halo[3];
                            auto h_loc = box {
                                domain_box_local.first() + (h.first() - domain_box.first()),
                                domain_box_local.last()  + (h.last()  - domain_box.last())};
                            halo_regions.push_back( box2{h_loc, h} );
                        }

                        // loop over 4 halo regions and intersect with this tile
                        // and with neighbor tiles to find halo regions
                        std::vector<box2> result;
                        for (const auto& h_box : halo_regions) {
                            // intersect with this tile
                            const auto h_i = intersect(h_box, tile_box);
                            if ((h_i.global().first()[1] <= h_i.global().last()[1]) && 
                                (h_i.global().first()[2] <= h_i.global().last()[2])) {
                                result.push_back(h_i);
                            }
                            // intersect with the 4 neighbor tiles
                            for (int n=0; n<4; ++n) {
                                auto h_n = intersect(h_box, neighbor_regions[n]);
                                if ((h_n.global().first()[1] <= h_n.global().last()[1]) && 
                                    (h_n.global().first()[2] <= h_n.global().last()[2])) {
                                    // transform to neighbor coordinates
                                    const auto& t = transform_lu[tile_id][n];
                                    const auto f = t(h_n.global().first()[1], h_n.global().first()[2], c);
                                    const auto l = t(h_n.global().last()[1],  h_n.global().last()[2],  c);
                                    h_n.global().first()[0] = tile_lu[tile_id][n];
                                    h_n.global().last ()[0] = tile_lu[tile_id][n];
                                    if (t.reversed_x()) {
                                        h_n.global().first()[1] = l[0];
                                        h_n.global().last ()[1] = f[0];
                                    }
                                    else {
                                        h_n.global().first()[1] = f[0];
                                        h_n.global().last ()[1] = l[0];
                                    }
                                    if (t.reversed_y()) {
                                        h_n.global().first()[2] = l[1];
                                        h_n.global().last ()[2] = f[1];
                                    }
                                    else {
                                        h_n.global().first()[2] = f[1];
                                        h_n.global().last ()[2] = l[1];
                                    }
                                    result.push_back(h_n);
                                }
                            }
                        }
                        return result;
                    }
                    
                    // return intersection of 2 iteration spaces
                    box intersect(const box& a, const box& b) const {
                        return { max(a.first(), b.first()),
                                 min(a.last(), b.last()) };
                    }

                    // return intersection of a pair of iteration spaces (box2)
                    // and another iteration space in global coordinates (without transform)
                    box2 intersect(const box2& a, const box& b) const {
                        const box global = intersect(a.global(), b);
                        const box local {
                            a.local().first() + (global.first() - a.global().first()),
                            a.local().last()  + (global.last()  - a.global().last())};
                        return {local,global};
                    }

                    /** @brief returns intersection of a pair of iteration spaces (local and global) with another
                      * global iteration space, taking into account possible coordinate transforms 
                      * @param d domain descriptor: determines the coordinate system in which to perform the intersection
                      * @param first_a_local first coordinate of the iteration space a in local coordinates
                      * @param last_a_local last coordinate of the iteration space a in local coordinates
                      * @param first_a_global first coordinate of the iteration space a in global coordinates
                      * @param last_a_global last coordinate of the iteration space in global coordinates
                      * @param first_b_global first coordinate of the iteration space b in global coordinates
                      * @param last_b_global last coordinate of the iteration space b in global coordinates
                      * @return  valid intersection (as pairs of local and global iteration spaces) of a and b if
                      * b is on the same cube tile as the domain, invalid intersection otherwise */
                    box2 intersect(const domain_type& d,
                                   const coordinate_type& first_a_local,  const coordinate_type& last_a_local,
                                   const coordinate_type& first_a_global, const coordinate_type& last_a_global,
                                   const coordinate_type& first_b_global, const coordinate_type& last_b_global) 
                        const noexcept 
                    {
                        const auto b_a_local   = box{first_a_local, last_a_local};
                        const auto b_a_global  = box{first_a_global, last_a_global};
                        const auto b_b_global  = box{first_b_global, last_b_global};
                        const auto tile = first_a_global[0];
                        const auto tile_b = first_b_global[0];
                        if (tile != tile_b)
                            return {box{last_a_local,first_a_local},box{last_a_global,first_a_global}};
                        if (tile != d.domain_id().tile) {
                            // transform is needed
                            // intersect global box
                            const auto x = intersect(b_a_global, b_b_global);
                            // look up neighbor index
                            int n;
                            for (n=0; n<4; ++n)
                                if (tile_lu[d.domain_id().tile][n] == tile)
                                    break;
                            // get inverse transform
                            const auto& t = inverse_transform_lu[d.domain_id().tile][n];
                            auto f = t(x.first()[1],x.first()[2],d.edge_size());
                            auto l = t(x.last()[1],x.last()[2],d.edge_size());
                            f[0] -= d.first()[1];
                            f[1] -= d.first()[2];
                            l[0] -= d.first()[1];
                            l[1] -= d.first()[2];
                            auto first_a_local_new = first_a_local;
                            auto last_a_local_new = last_a_local;
                            first_a_local_new[3] += x.first()[3]-first_a_global[3];
                            last_a_local_new[3]  += x.last()[3]-last_a_global[3];
                            if (t.reversed_x()) {
                                first_a_local_new[1] = l[0];
                                last_a_local_new[1]  = f[0];
                            }
                            else {
                                first_a_local_new[1] = f[0];
                                last_a_local_new[1]  = l[0];
                            }
                            if (t.reversed_y()) {
                                first_a_local_new[2] = l[1];
                                last_a_local_new[2]  = f[1];
                            }
                            else {
                                first_a_local_new[2] = f[1];
                                last_a_local_new[2]  = l[1];
                            }
                            return {box{first_a_local_new, last_a_local_new}, x};
                        }
                        else {
                            return intersect(box2{b_a_local,b_a_global}, b_b_global);
                        }
                    }
                };

            } // namespace cubed_sphere
        } // namespace structured
    } // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_HALO_GENERATOR_HPP
