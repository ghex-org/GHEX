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
#ifndef INCLUDED_SIMPLE_FIELD_HPP
#define INCLUDED_SIMPLE_FIELD_HPP

#include <type_traits>
#include <initializer_list>
#include <array>
#include <vector>
#include <algorithm>

#include "utils.hpp"
#include "structured_grid.hpp"

namespace gridtools {

    //template<int Dimension>
    template<typename DomainIdType, int Dimension>
    class simple_halo_generator;

    template<typename DomainIdType, int Dimension>
    class simple_domain_descriptor
    {
    public: // member types
        using domain_id_type      = DomainIdType;
        using dimension           = std::integral_constant<int,Dimension>;
        using coordinate_type     = std::array<int,dimension::value>;
        using halo_generator_type = simple_halo_generator<DomainIdType,Dimension>;

    public: // ctors
        template<typename Array>
        simple_domain_descriptor(domain_id_type id, const Array& first, const Array& last)
        : m_id{id}
        {
            std::copy(first.begin(), first.end(), m_first.begin());
            std::copy(last.begin(), last.end(), m_last.begin());
        }

    public: // member functions
        domain_id_type domain_id() const { return m_id; }
        const coordinate_type& first() const { return m_first; }
        const coordinate_type& last() const { return m_last; }

    private: // members
        domain_id_type  m_id;
        coordinate_type m_first;
        coordinate_type m_last;
    };

    //template<typename DomainDescriptor>
    template<typename DomainIdType, int Dimension>
    class simple_halo_generator
    {
    public: // member types
        using domain_type     = simple_domain_descriptor<DomainIdType,Dimension>;
        using dimension       = typename domain_type::dimension;
        using coordinate_type = typename structured_grid::template type<domain_type>::coordinate_type;

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
        
        template<typename GlobalDomain>
        simple_halo_generator(const GlobalDomain& gd, std::initializer_list<int> halos,std::initializer_list<bool> periodic)
        : simple_halo_generator(gd.first(), gd.last(), halos, periodic)
        {}

        template<typename Array>
        simple_halo_generator(const Array& g_first, const Array& g_last, std::initializer_list<int> halos,std::initializer_list<bool> periodic)
        {
            std::copy(g_first.begin(), g_first.end(), m_first.begin());
            std::copy(g_last.begin(), g_last.end(), m_last.begin());
            m_halos.fill(0);
            m_periodic.fill(true);
            std::copy(halos.begin(), halos.end(), m_halos.begin());
            std::copy(periodic.begin(), periodic.end(), m_periodic.begin());
        }

        simple_halo_generator(std::initializer_list<int> halos)
        {
            m_halos.fill(0);
            m_periodic.fill(false);
            std::copy(halos.begin(), halos.end(), m_halos.begin());
        }

        auto operator()(const domain_type& d) const
        {
            //using size = std::integral_constant<int, detail::ct_pow(3,dimension::value)-1>;
            box2 left;
            for (int d=0; d<dimension::value; ++d)
            {
                left.local().first()[d]  = -m_halos[d*2];
                left.local().last()[d]   = -1;
                left.global().first()[d] = left.local().first()[d]+m_first[d];
                left.global().last()[d]  = m_first[d]-1;
            }
            box2 middle;
            for (int d=0; d<dimension::value; ++d)
            {
                middle.local().first()[d]  = 0;
                middle.local().last()[d]   = m_last[d]-m_first[d];
                middle.global().first()[d] = m_first[d];
                middle.global().last()[d]  = m_last[d];
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
            for (int j=0; j<static_cast<int>(halos.size()); ++j)
            {
                if (j==(detail::ct_pow(3,dimension::value)/2)) continue;
                if (outer_halos[j].local().last() >= outer_halos[j].local().first())
                    halos.push_back(outer_halos[j]);
                /*bool empty = false;
                for (int i=0; i<dimension::value; ++i)
                {
                    if (outer_halos[j].local().last()[i] < outer_halos[j].local().first()[i])
                    {
                        empty = true;
                        break;
                    }
                }
                if (empty) continue;
                halos.push_back(outer_halos[j]);*/
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

            return std::move(halos);
        }

    private: // member functions
        
        template<typename Box, typename Spaces>
        std::vector<Box> compute_spaces(const Spaces& spaces) const
        {
            std::vector<Box> x;
            x.reserve(detail::ct_pow(3,dimension::value));
            Box b;
            compute_spaces<Box>(0, spaces, b, x);
            return std::move(x);
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

    template<typename T, typename Device, typename DomainDescriptor, int... Order>
    class simple_field_wrapper {};

    template<typename T, typename Device, typename DomainIdType, int Dimension, int... Order>
    class simple_field_wrapper<T,Device,simple_domain_descriptor<DomainIdType,Dimension>, Order...>
    {
    public:
        using value_type             = T;
        using device_type            = Device;
        using domain_descriptor_type = simple_domain_descriptor<DomainIdType,Dimension>;
        using dimension              = typename domain_descriptor_type::dimension;
        using layout_map             = gridtools::layout_map<Order...>;
        using domain_id_type         = DomainIdType;
        using coordinate_type        = typename domain_descriptor_type::halo_generator_type::coordinate_type;

    private:
        using halo_descriptor_type   = coordinate_type;
        using periodicity_type       = std::array<bool,dimension::value>;

    private:
        domain_id_type m_dom_id;
        value_type* m_data;
        coordinate_type m_strides;
        coordinate_type m_offsets;
        coordinate_type m_extents;

    public: // ctors

        template<typename Array>
        simple_field_wrapper(domain_id_type dom_id, value_type* data, const Array& offsets, const Array& extents)
        : m_dom_id(dom_id), m_data(data), m_strides(1)
        { 
            std::copy(offsets.begin(), offsets.end(), m_offsets.begin());
            std::copy(extents.begin(), extents.end(), m_extents.begin());
            // compute strides
        }

    private:

    public:
        typename device_type::id_type device_id() const { return 0; }

        domain_id_type domain_id() const { return m_dom_id; }

        value_type& operator()(const coordinate_type& x)
        {
            return m_data[dot(x,m_strides)];
        }

        template<typename IndexContainer>
        void pack(T* buffer, const IndexContainer& c)
        {
            std::size_t b=0;
            const coordinate_type o{m_offsets};
            for (const auto& is : c)
            {
                detail::for_loop<dimension::value,dimension::value,layout_map>::apply(
                    [this,&o,buffer,&b](auto... indices)
                    {
                        buffer[b++] = this->operator()(coordinate_type{indices...}+o);
                    }, 
                    is.local().first(), 
                    is.local().last());
            }
        }

        template<typename IndexContainer>
        void unpack(const T* buffer, const IndexContainer& c)
        {
        }
    };

} // namespace gridtools

#endif /* INCLUDED_SIMPLE_FIELD_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

