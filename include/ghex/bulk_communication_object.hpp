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
#ifndef INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP
#define INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP

#include <memory>
#include <functional>
#include <vector>
#include <map>
#include <tuple>
#include <boost/mp11.hpp>
#include "./common/moved_bit.hpp"
#include "./common/utils.hpp"
#include "./communication_object_2.hpp"
#include "./rma/range_traits.hpp"
#include "./rma/range_factory.hpp"
#include "./transport_layer/ri/types.hpp"

namespace gridtools {
namespace ghex {

// type erased bulk communication object
struct generic_bulk_communication_object
{
private:
    struct handle
    {
        std::function<void()> m_wait;
        void wait() { m_wait(); }
    };

    struct bulk_co_iface
    {
        virtual ~bulk_co_iface() {}
        virtual handle exchange() = 0;
    };

    template<typename CO>
    struct bulk_co_impl : public bulk_co_iface
    {
        CO m;
        bulk_co_impl(CO&& co) : m{std::move(co)} {}
        handle exchange() override final { return {std::move(m.exchange().m_wait_fct)}; }
    };

    std::unique_ptr<bulk_co_iface> m_impl;

public:
    generic_bulk_communication_object() = default;
    template<typename CO>
    generic_bulk_communication_object(CO&& co) : m_impl{ std::make_unique<bulk_co_impl<CO>>(std::move(co)) } {}
    generic_bulk_communication_object(generic_bulk_communication_object&&) = default;
    generic_bulk_communication_object& operator=(generic_bulk_communication_object&&) = default;

    handle exchange() { return m_impl->exchange(); }
};

template<template <typename> class RangeGen, typename Pattern, typename... Fields>
class bulk_communication_object
{
public: // member types
    using pattern_type = Pattern;
    using communicator_type = typename pattern_type::communicator_type;
    using grid_type = typename pattern_type::grid_type;
    using domain_id_type = typename pattern_type::domain_id_type;
    using co_type = communication_object<communicator_type,grid_type,domain_id_type>;
    using co_handle = typename co_type::handle_type;
    template<typename Field>
    using buffer_info_type = typename co_type::template buffer_info_type<typename Field::arch_type, Field>;

private:

    using pattern_map = std::map<const pattern_type*, pattern_type>;

private: // member types
    template<typename A>
    struct select_arch_q
    {
        template<typename Field>
        using fn = typename Field::template rebind_arch<A>;
    };

    template<typename Field>
    using select_range = typename RangeGen<Field>::range_type;
    
    using field_types = boost::mp11::mp_unique<boost::mp11::mp_list<Fields...>>;
    using buffer_info_types = boost::mp11::mp_transform<buffer_info_type, field_types>;
    using cpu_fields = boost::mp11::mp_transform_q<select_arch_q<cpu>,field_types>;
#ifdef __CUDACC__
    using gpu_fields = boost::mp11::mp_transform_q<select_arch_q<gpu>,field_types>;
    using all_fields = boost::mp11::mp_unique<boost::mp11::mp_append<cpu_fields,gpu_fields>>;
#else
    using all_fields = boost::mp11::mp_unique<cpu_fields>;
#endif
    using all_ranges = boost::mp11::mp_transform<select_range,all_fields>;
    using range_factory = rma::range_factory<all_ranges>;

    template<typename Field>
    using select_target_range = std::vector<typename RangeGen<Field>::template target_range<
        range_factory,communicator_type>>;
    template<typename Field>
    using select_source_range = std::vector<typename RangeGen<Field>::template source_range<
        range_factory,communicator_type>>;


    template<typename Field>
    struct target_ranges
    {
        using ranges_type = select_target_range<Field>;
        using range_type = typename ranges_type::value_type;
        std::vector<ranges_type> m_ranges;
    };

    template<typename Field>
    struct source_ranges
    {
        using ranges_type = select_source_range<Field>;
        using range_type = typename ranges_type::value_type;
        std::vector<ranges_type> m_ranges; 
    };

    template<typename Field>
    struct field_container
    {
        Field m_field;
        pattern_type& m_remote_pattern;
        pattern_type& m_local_pattern;

        field_container(communicator_type comm, const Field& f, const pattern_type& pattern,
            pattern_map& local_map, pattern_map& remote_map)
        : m_field{f}
        , m_remote_pattern(remote_map.insert(std::make_pair(&pattern, pattern)).first->second)
        , m_local_pattern(local_map.insert(std::make_pair(&pattern, pattern)).first->second)
        {
            // prepare local and remote patterns
            // =================================

            // loop over all subdomains in pattern
            for (int n = 0; n<pattern.size(); ++n)
            {
                // remove local fields from remote pattern
                auto& r_p = m_remote_pattern[n];
                auto r_it = r_p.send_halos().begin();
                while (r_it != r_p.send_halos().end())
                {
                    const auto local = rma::range_traits<RangeGen>::is_local(comm, r_it->first.mpi_rank);
                    if (local != tl::ri::locality::remote) r_it = r_p.send_halos().erase(r_it);
                    else ++r_it;
                }

                // remove remote fields from local pattern
                auto& l_p = m_local_pattern[n];
                auto l_it = l_p.send_halos().begin();
                while (l_it != l_p.send_halos().end())
                {
                    const auto local = rma::range_traits<RangeGen>::is_local(comm, l_it->first.mpi_rank);
                    if (local != tl::ri::locality::remote) ++l_it;
                    else l_it = l_p.send_halos().erase(l_it);
                }


                // remove local fields from remote pattern
                r_it = r_p.recv_halos().begin();
                while (r_it != r_p.recv_halos().end())
                {
                    const auto local = rma::range_traits<RangeGen>::is_local(comm, r_it->first.mpi_rank);
                    if (local != tl::ri::locality::remote) r_it = r_p.recv_halos().erase(r_it);
                    else ++r_it;
                }

                // remove remote fields from local pattern
                l_it = l_p.recv_halos().begin();
                while (l_it != l_p.recv_halos().end())
                {
                    const auto local = rma::range_traits<RangeGen>::is_local(comm, l_it->first.mpi_rank);
                    if (local != tl::ri::locality::remote) ++l_it;
                    else l_it = l_p.recv_halos().erase(l_it);
                }
            }
        }

        field_container(const field_container&) = default;
        field_container(field_container&&) = default;
    };

    template<typename Field>
    using field_vector = std::vector<field_container<Field>>;
    
    using field_container_t = boost::mp11::mp_rename<boost::mp11::mp_transform<field_vector,field_types>,std::tuple>;
    
    using buffer_info_container_t = boost::mp11::mp_rename<boost::mp11::mp_transform<std::vector,buffer_info_types>,std::tuple>;

    using target_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<target_ranges, field_types>,std::tuple>;
    using source_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<source_ranges, field_types>,std::tuple>;

private: // members

    communicator_type       m_comm;
    co_type                 m_co;
    pattern_map             m_local_pattern_map;
    pattern_map             m_remote_pattern_map;
    field_container_t       m_field_container_tuple;
    buffer_info_container_t m_buffer_info_container_tuple;
    target_ranges_t         m_target_ranges_tuple;
    source_ranges_t         m_source_ranges_tuple;
    moved_bit               m_moved;
    bool                    m_initialized = false;

public: // ctors

    bulk_communication_object(communicator_type comm)
    : m_comm(comm)
    , m_co(comm)
    {}

    bulk_communication_object(const bulk_communication_object&) = delete;
    bulk_communication_object(bulk_communication_object&&) = default;

public:

    template<typename Field>
    void add_field(buffer_info_type<Field> bi)
    {
        if (m_moved)
            throw std::runtime_error("error: trying to add a field to a CO which is moved");
        if (m_initialized)
            throw std::runtime_error("error: this CO has been initialized already");

        using f_cont_t  = std::vector<field_container<Field>>;
        using t_range_t = target_ranges<Field>;
        using s_range_t = source_ranges<Field>;

        auto& f_cont  = std::get<f_cont_t>(m_field_container_tuple);
        auto& t_range = std::get<t_range_t>(m_target_ranges_tuple);
        auto& s_range = std::get<s_range_t>(m_source_ranges_tuple);

        // store field
        f_cont.push_back(field_container<Field>(m_comm, bi.get_field(), bi.get_pattern_container(),
            m_local_pattern_map, m_remote_pattern_map));
        s_range.m_ranges.resize(s_range.m_ranges.size()+1);
        t_range.m_ranges.resize(t_range.m_ranges.size()+1);

        auto& f = f_cont.back().m_field;
        // loop over patterns 
        for (auto& p : f_cont.back().m_local_pattern)
        {
            // check if field has the right domain
            if (f.domain_id() == p.domain_id())
            {
                // loop over halos and set up source ranges
                for (auto h_it = p.send_halos().begin(); h_it != p.send_halos().end(); ++h_it)
                {
                    for (auto it = h_it->second.rbegin(); it != h_it->second.rend(); ++it)
                    {
                        const auto& c = *it;
                        s_range.m_ranges.back().emplace_back(
                            m_comm, f, c, h_it->first.mpi_rank, h_it->first.tag); 
                    }
                }
                // loop over halos and set up target
                for (auto h_it = p.recv_halos().begin(); h_it != p.recv_halos().end(); ++h_it)
                {
                    for (auto it = h_it->second.rbegin(); it != h_it->second.rend(); ++it)
                    {
                        const auto local = rma::range_traits<RangeGen>::is_local(m_comm, h_it->first.mpi_rank);
                        const auto& c = *it;
                        t_range.m_ranges.back().emplace_back(
                            m_comm, f, c, h_it->first.mpi_rank, h_it->first.tag, local); 
                    }
                }
            }
        }
    }
    
    template<typename... F>
    void add_fields(buffer_info_type<F>... bis)
    {
        auto bis_tp = std::make_tuple(bis...);
        for (std::size_t i=0; i<sizeof...(F); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(F)>(i,
            [this,&bis_tp](auto i) {
                // get the field Index 
                using I = decltype(i);
                add_field(std::get<I::value>(bis_tp));
            });
        }
    }

    void init()
    {
        // loop over Fields
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get source and target ranges
                auto& t_range = std::get<I::value>(m_target_ranges_tuple);
                auto& s_range = std::get<I::value>(m_source_ranges_tuple);
                auto& bi_cont = std::get<I::value>(m_buffer_info_container_tuple);
                auto& f_cont  = std::get<I::value>(m_field_container_tuple);
                // add remote exchange
                for (auto& f : f_cont)
                    bi_cont.push_back( f.m_remote_pattern(f.m_field) );
                // complete the handshake
                for (auto& t_vec : t_range.m_ranges)
                    for (auto& r : t_vec)
                    {
                        r.send();
                    }
                for (auto& s_vec : s_range.m_ranges)
                    for (auto& r : s_vec)
                    {
                        r.recv();
                    }
            });
        }
        
        m_initialized = true;
    }
    
private:

    co_handle exchange_remote()
    {
        return exchange_remote(std::make_index_sequence<boost::mp11::mp_size<field_types>::value>());
    }

    template<std::size_t... I>
    co_handle exchange_remote(std::index_sequence<I...>)
    {
        return m_co.exchange(std::make_pair(std::get<I>(m_buffer_info_container_tuple).begin(),
            std::get<I>(m_buffer_info_container_tuple).end())...);
    }

public:

    auto exchange()
    {
        if (!m_initialized) init();

        // start remote exchange
        auto h = exchange_remote();
        
        // loop over Fields
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get target ranges for fields and give remotes access
                auto& t_range = std::get<I::value>(m_target_ranges_tuple);
                for (auto& t_vec : t_range.m_ranges)
                    for (auto& r : t_vec)
                        r.end_target_epoch();
            });
        }
        // loop over fields for putting
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get source range
                auto& s_range = std::get<I::value>(m_source_ranges_tuple);
                // put data
                for (auto& s_vec : s_range.m_ranges)
                    for (auto& r : s_vec)
                        r.put();
            });
        }
        // loop over fields for waiting
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get target ranges and wait
                auto& t_range = std::get<I::value>(m_target_ranges_tuple);
                for (auto& t_vec : t_range.m_ranges)
                    for (auto& r : t_vec)
                        r.start_target_epoch();
            });
        }

        return h;
    }
};

} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_BULK_COMMUNICATION_OBJECT_HPP */
