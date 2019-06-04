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
        using communicator_type       = typename pattern_type::communicator_type;
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

    /*private:

        template<typename D>
        struct buffer_info_internal
        {
            typename D::id_type  idx;

            std::vector<typename internal_co_t<Devices>::vector_type*> recv_memory;
            std::vector<typename internal_co_t<Devices>::vector_type*> send_memory;
        };*/

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

            //std::tuple<std::vector<typename internal_co_t<Devices>::vector_type*>, sizeof...(Fields)> recv_memory;
            //std::tuple<std::vector<typename internal_co_t<Devices>::vector_type*>, sizeof...(Fields)> send_memory;

            std::array<communicator_type*,               sizeof...(Fields)> comms;
            auto& comm = *(comms[0]);

            std::tuple<typename Devices::id_type...> indices{buffer_infos.device_id()...};
            std::tuple<Fields*...>                   field_ptrs{&buffer_infos.get_field()...};
            std::tuple<void (Fields::*)(void*,const index_container_type&)...> pack_fct_ptrs{&Fields::template pack<index_container_type>...};
            std::tuple<void (Fields::*)(void*,const index_container_type&)...> unpack_fct_ptrs{&Fields::template unpack<index_container_type>...};

            auto pack_funcs   = detail::memfunc_ptr<index_container_type>(field_ptrs,   pack_fct_ptrs, std::make_index_sequence<sizeof...(Fields)>());
            auto unpack_funcs = detail::memfunc_ptr<index_container_type>(field_ptrs, unpack_fct_ptrs, std::make_index_sequence<sizeof...(Fields)>());

            std::tuple<internal_co_t<Devices>*...> list{&(std::get< internal_co_t<Devices> >(m_list))...};
            int i = 0;
            detail::for_each(list, indices, [this,&patterns,&i,&recv_offsets,&send_offsets,&alignments,&sizes_recv,&sizes_send](auto ico, auto idx)
            { 
                using device_type = typename std::remove_reference_t<decltype(*ico)>::device_type;
                allocate<device_type>(ico->recv_memory[idx], patterns[i]->recv_halos(), idx, 
                                      recv_offsets[i], *(sizes_recv[i]), alignments[i]);
                allocate<device_type>(ico->send_memory[idx], patterns[i]->send_halos(), idx, 
                                      send_offsets[i], *(sizes_send[i]), alignments[i]);
                ++i;
            });

            /*i = 0;
            detail::for_each(list, indices, [](auto ico, auto idx)
            { 
                using device_type = typename std::remove_reference_t<decltype(*ico)>::device_type;
                ico
            }*/
        }

    private: // member functions

        template<typename D>
        void allocate(
            typename internal_co_t<D>::memory_type::mapped_type& p, 
            const typename pattern_type::map_type& halos, 
            typename D::id_type idx,
            std::vector<std::size_t>& offsets,
            const std::vector<std::size_t>& sizes,
            std::size_t alignment)
            //std::vector<typename internal_co_t<D>::vector_type*>& memory)
        {
            int j = 0;
            for (const auto& p_id_c : halos)
            {
                auto it = p.find(p_id_c.first);
                if (it == p.end())
                    it = p.insert( std::make_pair( p_id_c.first, D::template make_vector<char>(idx) ) ).first;
                offsets.push_back(it->second.size());
                //memory.push_back( &it->second );
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

