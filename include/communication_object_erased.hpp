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
#ifndef INCLUDED_COMMUNICATION_OBJECT_ERASED_HPP
#define INCLUDED_COMMUNICATION_OBJECT_ERASED_HPP

#include "util.hpp"
#include "buffer_info.hpp"
#include "devices.hpp"

namespace gridtools {

    namespace detail {

        template<typename Tuple>
        struct transform {};

        template<template<typename...> typename Tuple, typename... Ts>
        struct transform<Tuple<Ts...>>
        {
            template<template<typename> typename CT>
            using with = Tuple<CT<Ts>...>;
        };

        template<typename IndexContainerType, std::size_t I, typename PtrTuple, typename MemfuncTuple>
        auto memfunc_ptr_impl(PtrTuple&& p, MemfuncTuple&& f)
        {
            auto ptr = std::get<I>(p);
            auto f_ptr = std::get<I>(f);
            return [ptr,f_ptr] (void* buffer, const IndexContainerType& c) { (ptr->*f_ptr)(buffer,c); };
        }

        template<typename IndexContainerType, typename PtrTuple, typename MemfuncTuple, std::size_t... Is>
        auto memfunc_ptr(PtrTuple&& p, MemfuncTuple&& f, std::index_sequence<Is...>)
        {
            return std::array<std::function<void(void*,const IndexContainerType&)>, sizeof...(Is)>{
                memfunc_ptr_impl<IndexContainerType,Is>(std::forward<PtrTuple>(p), std::forward<MemfuncTuple>(f))...
            };
        }

    } // namespace detail

    template<typename Pattern>
    struct communication_object {};

    template<typename P, typename GridType, typename DomainIdType>
    class communication_object<pattern<P,GridType,DomainIdType>>
    {
    public: // member types

        using pattern_type            = pattern<P,GridType,DomainIdType>;
        using index_container_type    = typename pattern_type::index_container_type;
        using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
        template<typename D, typename F>
        using buffer_info_t           = buffer_info<pattern_type,D,F>;

    private: // member types

        using pack_function_type   = std::function<void(void*,const index_container_type&)>;
        using unpack_function_type = std::function<void(void*,const index_container_type&)>;

        template<typename Device>
        struct internal_co_t
        {
            using device_type = Device;
            using id_type = typename device_type::id_type;
            using vector_type = typename device_type::template vector_type<char>;
            using memory_type = std::map<
                id_type,
                std::map<
                    extended_domain_id_type,
                    vector_type
                >
            >;
            memory_type recv_memory;
            memory_type send_memory;
        };

        using internal_co_list = detail::transform<device::device_list>::with<internal_co_t>;

    private: // members

        internal_co_list m_list;

    public: // member functions

        template<typename... Devices, typename... Fields>
        void exchange(buffer_info_t<Devices, Fields>&&... buffer_infos)
        {
            std::array<std::size_t,                      sizeof...(Fields)> alignments{alignof(typename buffer_info_t<Devices,Fields>::value_type)...};
            std::array<const std::vector<std::size_t> *, sizeof...(Fields)> sizes_recv{&buffer_infos.sizes_recv()...};
            std::array<const std::vector<std::size_t> *, sizeof...(Fields)> sizes_send{&buffer_infos.sizes_send()...};
            std::array<const pattern_type*,              sizeof...(Fields)> patterns{&buffer_infos.get_pattern()...};
            std::array<std::vector<std::size_t>,         sizeof...(Fields)> recv_offsets;
            std::array<std::vector<std::size_t>,         sizeof...(Fields)> send_offsets;


            std::tuple<typename Devices::id_type...> indices{buffer_infos.device_id()...};
            std::tuple<Fields*...>                   field_ptrs{&buffer_infos.get_field()...};
            std::tuple<void (Fields::*)(void*,const index_container_type&)...> pack_fct_ptrs{&Fields::template pack<index_container_type>...};
            std::tuple<void (Fields::*)(void*,const index_container_type&)...> unpack_fct_ptrs{&Fields::template unpack<index_container_type>...};

            auto pack_funcs   = detail::memfunc_ptr<index_container_type>(field_ptrs,   pack_fct_ptrs, std::make_index_sequence<sizeof...(Fields)>());
            auto unpack_funcs = detail::memfunc_ptr<index_container_type>(field_ptrs, unpack_fct_ptrs, std::make_index_sequence<sizeof...(Fields)>());

            std::tuple<internal_co_t<Devices>*...> list{&(std::get< internal_co_t<Devices> >(m_list))...};

            //std::cout << "size of list = " << std::tuple_size<decltype(list)>::value << std::endl;
            //std::cout << "size of m_list = " << std::tuple_size<decltype(m_list)>::value << std::endl;
            
            int i = 0;
            detail::for_each(list, indices, [&patterns,&i,&recv_offsets,&send_offsets,&alignments,&sizes_recv,&sizes_send](auto ico, auto idx)
            { 
                std::cout << "analyzing buffer info..." << std::endl;
                std::cout << "  i = " << i << std::endl;
                std::cout << "  idx = " << idx << std::endl;
                using device_type = typename std::remove_reference_t<decltype(*ico)>::device_type;
                std::cout << "  device = " <<  device_type::name << std::endl;
                {
                    auto& p = ico->recv_memory[idx];
                    std::cout << "  recv memory map size = " << p.size() << std::endl;
                    std::cout << "  ptr to ico = " << ico << std::endl;
                    std::cout << "  ptr to internal memory object = " << &p << std::endl;
                    auto& pat = *patterns[i];
                    int j = 0;
                    for (const auto& p_id_c : pat.recv_halos())
                    {
                        std::cout << "    j = " << j << std::endl;
                        std::cout << "    p_id_c.first.id = " << p_id_c.first.id  << std::endl;
                        std::cout << "    p_id_c.first.mpi_rank = " << p_id_c.first.mpi_rank  << std::endl;
                        std::cout << "    p_id_c.first.tag = " << p_id_c.first.tag  << std::endl;
                        auto it = p.find(p_id_c.first);
                        if (it == p.end())
                        {
                            std::cout << "    ext_domain_id not found, creating it" << std::endl;
                            it = p.insert( std::make_pair( p_id_c.first, device_type::template make_vector<char>(idx) ) ).first;
                        }
                        else
                        {
                            std::cout << "    ext_domain found!!!!" << std::endl;
                        }
                        std::cout << "    it->second.size() = " << it->second.size() << std::endl;
                        //recv_offset[i][p_id_c.first] = it.second.size();
                        recv_offsets[i].push_back(it->second.size());
                        it->second.resize( it->second.size() + alignments[i] + sizes_recv[i]->operator[](j) );
                        std::cout << "    it->second.size() after = " << it->second.size() << std::endl;
                        ++j;
                    }
                }
                {
                    allocate<device_type>(
                        ico->send_memory[idx],
                        patterns[i]->send_halos(),
                        idx,
                        send_offsets[i],
                        *(sizes_send[i]),
                        alignments[i]);

                    auto& p = ico->send_memory[idx];
                    std::cout << "  send memory map size = " << p.size() << std::endl;
                    std::cout << "  ptr to ico = " << ico << std::endl;
                    std::cout << "  ptr to internal memory object = " << &p << std::endl;
                    auto& pat = *patterns[i];
                    int j = 0;
                    for (const auto& p_id_c : pat.send_halos())
                    {
                        std::cout << "    j = " << j << std::endl;
                        std::cout << "    p_id_c.first.id = " << p_id_c.first.id  << std::endl;
                        std::cout << "    p_id_c.first.mpi_rank = " << p_id_c.first.mpi_rank  << std::endl;
                        std::cout << "    p_id_c.first.tag = " << p_id_c.first.tag  << std::endl;
                        auto it = p.find(p_id_c.first);
                        if (it == p.end())
                        {
                            std::cout << "    ext_domain_id not found, creating it" << std::endl;
                            it = p.insert( std::make_pair( p_id_c.first, device_type::template make_vector<char>(idx) ) ).first;
                        }
                        else
                        {
                            std::cout << "    ext_domain found!!!!" << std::endl;
                        }
                        std::cout << "    it->second.size() = " << it->second.size() << std::endl;
                        send_offsets[i].push_back(it->second.size());
                        it->second.resize( it->second.size() + alignments[i] + sizes_send[i]->operator[](j) );
                        std::cout << "    it->second.size() after = " << it->second.size() << std::endl;
                        ++j;
                    }
                }
                ++i;
            });

            std::cout << "recv offsets: " << std::endl;
            for (const auto& x : recv_offsets)
            {
                for (const auto& y : x)
                    std::cout << y << ", ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    private: // member functions

        template<typename D>
        void allocate(
            typename internal_co_t<D>::memory_type::value_type& p, 
            const typename pattern_type::map_type& halos, 
            typename D::index_type idx,
            std::vector<std::size_t>& offsets,
            const std::vector<std::size_t>& sizes,
            std::size_t alignment)
        {
            int j = 0;
            for (const auto& p_id_c : halos)
            {
                auto it = p.find(p_id_c.first);
                if (it == p.end())
                    it = p.insert( std::make_pair( p_id_c.first, D::template make_vector<char>(idx) ) ).first;
                offsets.push_back(it->second.size());
                it->second.resize( it->second.size() + alignment + sizes[j] );
                ++j;
            }
        }
    };

} // namespace gridtools

#endif /* INCLUDED_COMMUNICATION_OBJECT_ERASED_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

