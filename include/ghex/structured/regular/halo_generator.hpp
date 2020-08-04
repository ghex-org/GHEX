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
#ifndef INCLUDED_GHEX_STRUCTURED_REGULAR_HALO_GENERATOR_HPP
#define INCLUDED_GHEX_STRUCTURED_REGULAR_HALO_GENERATOR_HPP

#include "./domain_descriptor.hpp"

namespace gridtools {
    namespace ghex {
    namespace structured {
    namespace regular {

    /** @brief halo generator for structured domains
     * @tparam DomainIdType domain id type
     * @tparam Dimension dimension of domain*/
    template<typename DomainIdType, int Dimension>
    class halo_generator
    {
    public: // member types
        using domain_type     = domain_descriptor<DomainIdType,Dimension>;
        using dimension       = typename domain_type::dimension;
        using coordinate_type = typename grid::template type<domain_type>::coordinate_type;

    private: // member types
        struct box
        {
            const coordinate_type& first() const { return m_first; }
            const coordinate_type& last() const { return m_last; }
            coordinate_type& first() { return m_first; }
            coordinate_type& last() { return m_last; }
            coordinate_type m_first;
            coordinate_type m_last;
        };

        struct box2
        {
            const box& local() const { return m_local; }
            const box& global() const { return m_global; }
            box& local() { return m_local; }
            box& global() { return m_global; }
            box m_local;
            box m_global;
        };

    public: // ctors
        /** @brief construct a halo generator
         * @tparam Array coordinate-like type
         * @tparam RangeHalos range type holding halos
         * @tparam RangePeriodic range type holding periodicity info
         * @param g_first first global coordinate of total domain (used for periodicity)
         * @param g_last last global coordinate of total domain (including, used for periodicity)
         * @param halos list of halo sizes (dim0_dir-, dim0_dir+, dim1_dir-, dim1_dir+, ...)
         * @param periodic list of bools indicating periodicity per dimension (true, true, false, ...) */
        template<typename Array, typename RangeHalos, typename RangePeriodic>
        halo_generator(const Array& g_first, const Array& g_last, RangeHalos&& halos, RangePeriodic&& periodic)
        {
            std::copy(std::begin(g_first), std::end(g_first), m_first.begin());
            std::copy(std::begin(g_last), std::end(g_last), m_last.begin());
            m_halos.fill(0);
            m_periodic.fill(true);
            std::copy(halos.begin(), halos.end(), m_halos.begin());
            std::copy(periodic.begin(), periodic.end(), m_periodic.begin());
        }

        // construct without periodicity
        halo_generator(std::initializer_list<int> halos)
        {
            m_halos.fill(0);
            m_periodic.fill(false);
            std::copy(halos.begin(), halos.end(), m_halos.begin());
        }

        /** @brief generate halos
         * @param dom local domain instance
         * @return vector of halos of type box2 */
        auto operator()(const domain_type& dom) const
        {
            box2 left;
            for (int d=0; d<dimension::value; ++d)
            {
                left.local().first()[d]  = -m_halos[d*2];
                left.local().last()[d]   = -1;
                left.global().first()[d] = left.local().first()[d]+dom.first()[d];
                left.global().last()[d]  = dom.first()[d]-1;
            }
            box2 middle;
            for (int d=0; d<dimension::value; ++d)
            {
                middle.local().first()[d]  = 0;
                middle.local().last()[d]   = dom.last()[d]-dom.first()[d];
                middle.global().first()[d] = dom.first()[d];
                middle.global().last()[d]  = dom.last()[d];
            }
            box2 right;
            for (int d=0; d<dimension::value; ++d)
            {
                right.local().first()[d]  = middle.local().last()[d]+1;
                right.local().last()[d]   = middle.local().last()[d]+m_halos[d*2+1];
                right.global().first()[d] = middle.global().last()[d]+1;
                right.global().last()[d]  = middle.global().last()[d]+m_halos[d*2+1];
            }
            std::array<box2,3> outer_spaces{left,middle,right};

            // generate outer halos
            auto outer_halos = compute_spaces<box2>(outer_spaces);

            // filter out unused halos
            decltype(outer_halos) halos;
            for (int j=0; j<static_cast<int>(outer_halos.size()); ++j)
            {
                if (j==(::gridtools::ghex::detail::ct_pow(3,dimension::value)/2)) continue;
                if (outer_halos[j].local().last() >= outer_halos[j].local().first())
                    halos.push_back(outer_halos[j]);
            }

            // handle periodicity
            for (unsigned int i=0; i<halos.size(); ++i)
            {
                for (int d=0; d<dimension::value; ++d)
                {
                    if (!m_periodic[d]) continue;
                    const auto ext_h = halos[i].global().last()[d] - halos[i].global().first()[d];
                    const auto ext = m_last[d]+1-m_first[d];
                    const auto offset_l = halos[i].global().first()[d] - m_first[d];
                    halos[i].global().first()[d] = (offset_l+ext)%ext + m_first[d];
                    halos[i].global().last()[d]  = halos[i].global().first()[d] + ext_h;
                }
            }

            return halos;
        }

        box2 intersect(const domain_type& /*d*/,
                       const coordinate_type& first_a_local,  const coordinate_type& /*last_a_local*/,
                       const coordinate_type& first_a_global, const coordinate_type& last_a_global,
                       const coordinate_type& first_b_global, const coordinate_type& last_b_global) 
            const noexcept 
        {
            const box global_box{
                max(first_a_global, first_b_global),
                min(last_a_global,  last_b_global)};
            const box local_box{
                first_a_local + (global_box.first() - first_a_global),
                first_a_local + (global_box.last()  - first_a_global)};
            return {local_box, global_box};
        }

    private: // member functions
        template<typename Box, typename Spaces>
        std::vector<Box> compute_spaces(const Spaces& spaces) const
        {
            std::vector<Box> x;
            x.reserve(::gridtools::ghex::detail::ct_pow(3,dimension::value));
            Box b;
            compute_spaces<Box>(0, spaces, b, x);
            return x;
        }

        template<typename Box, typename Spaces, typename Container>
        void compute_spaces(int d, const Spaces& spaces, Box& current_box, Container& c) const
        {
            if (d==dimension::value)
            {
                 c.push_back(current_box);
            }
            else
            {
                current_box.local().first()[d]  = spaces[0].local().first()[d];
                current_box.global().first()[d] = spaces[0].global().first()[d];
                current_box.local().last()[d]   = spaces[0].local().last()[d];
                current_box.global().last()[d]  = spaces[0].global().last()[d];
                compute_spaces(d+1, spaces, current_box, c);

                current_box.local().first()[d]  = spaces[1].local().first()[d];
                current_box.global().first()[d] = spaces[1].global().first()[d];
                current_box.local().last()[d]   = spaces[1].local().last()[d];
                current_box.global().last()[d]  = spaces[1].global().last()[d];
                compute_spaces(d+1, spaces, current_box, c);

                current_box.local().first()[d]  = spaces[2].local().first()[d];
                current_box.global().first()[d] = spaces[2].global().first()[d];
                current_box.local().last()[d]   = spaces[2].local().last()[d];
                current_box.global().last()[d]  = spaces[2].global().last()[d];
                compute_spaces(d+1, spaces, current_box, c);
            }
        }

    private: // members
        coordinate_type m_first;
        coordinate_type m_last;
        std::array<int,dimension::value*2> m_halos;
        std::array<bool,dimension::value> m_periodic;
    };

    } // namespace regular
    } // namespace structured
    } // namespace ghex

} // namespac gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_REGULAR_HALO_GENERATOR_HPP */

