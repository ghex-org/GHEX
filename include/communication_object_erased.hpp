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

        /*template<typename IndexContainerType, std::size_t I, typename PtrTuple, typename MemfuncTuple>
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
        }*/

    } // namespace detail

    template<typename P, typename GridType, typename DomainIdType>
    class communication_object_erased;

    template<typename P, typename GridType, typename DomainIdType>
    class communication_handle
    {
    private:
        friend class communication_object_erased<P,GridType,DomainIdType>;
        using co_t = communication_object_erased<P,GridType,DomainIdType>;

        using communicator_type       = protocol::communicator<P>;
        using pattern_type            = pattern<P,GridType,DomainIdType>;
        using index_container_type    = typename pattern_type::index_container_type;
        using extended_domain_id_type = typename pattern_type::extended_domain_id_type;

        using pack_function_type   = std::function<void(void*,const index_container_type&)>;
        using unpack_function_type = std::function<void(const void*,const index_container_type&)>;

        struct field
        {
            struct chunk_type
            {
                std::size_t m_offset;
                void* m_buffer;
            };

            const pattern_type* m_pattern;
            pack_function_type m_pack;
            unpack_function_type m_unpack;
            std::vector<chunk_type> m_recv_chunks;
            std::vector<chunk_type> m_send_chunks;
        };

    private:
        communication_handle(co_t& co, const communicator_type& comm, std::size_t size) : m_co{&co}, m_comm{comm}, m_fields(size) {}

    public:

        void post()
        {
            pack();
            detail::for_each(m_co->m_mem, [this](auto& m)
            {
                for (auto& mm : m.send_memory)
                {
                    for (auto& p : mm.second)
                    {
                        if (p.second.size()>0)
                        {
                            m_futures.push_back(m_comm.isend(p.first.address, p.first.tag, p.second));
                        }
                    }
                }
                for (auto& mm : m.recv_memory)
                {
                    for (auto& p : mm.second)
                    {
                        if (p.second.size()>0)
                        {
                            m_futures.push_back(m_comm.irecv(p.first.address, p.first.tag, p.second.data(), p.second.size()));
                        }
                    }
                }
            });
        }

        void wait()
        {
            for (auto& f : m_futures) f.wait();
            unpack();
            m_co->clear();
        }

        void pack()
        {
            for (auto& f : m_fields)
            {
                std::size_t k=0;
                for (const auto& p_id_c : f.m_pattern->send_halos())
                {
                    f.m_pack(f.m_send_chunks[k++].m_buffer, p_id_c.second);
                }
            }
        }

        void unpack()
        {
            for (auto& f : m_fields)
            {
                std::size_t k=0;
                for (const auto& p_id_c : f.m_pattern->recv_halos())
                {
                    f.m_unpack(f.m_recv_chunks[k++].m_buffer, p_id_c.second);
                }
            }
        }

    private:
        co_t* m_co;
        communicator_type m_comm;
        std::vector<field> m_fields;
        std::vector<typename communicator_type::template future<void>> m_futures;
    };

    template<typename P, typename GridType, typename DomainIdType>
    class communication_object_erased
    {
    private:

        friend class communication_handle<P,GridType,DomainIdType>;
        using handle_type             = communication_handle<P,GridType,DomainIdType>;
        using communicator_type       = typename handle_type::communicator_type;
        using pattern_type            = typename handle_type::pattern_type;
        using index_container_type    = typename handle_type::index_container_type;
        using extended_domain_id_type = typename handle_type::extended_domain_id_type;

        template<typename D, typename F>
        using buffer_info_type = buffer_info<pattern_type,D,F>;

        template<typename Device>
        struct buffer_memory
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

        using memory_type = detail::transform<device::device_list>::with<buffer_memory>;

    private:
        
        memory_type m_mem;

    public:

        template<typename... Devices, typename... Fields>
        [[nodiscard]] handle_type exchange(buffer_info_type<Devices,Fields>... buffer_infos)
        {
            using buffer_infos_ptr_t  = std::tuple<std::remove_reference_t<decltype(buffer_infos)>*...>;
            using memory_t   = std::tuple<buffer_memory<Devices>*...>;

            buffer_infos_ptr_t buffer_info_tuple{&buffer_infos...};
            handle_type h(*this,std::get<0>(buffer_info_tuple)->get_pattern().communicator(), sizeof...(Fields));

            memory_t memory_tuple{&(std::get<buffer_memory<Devices>>(m_mem))...};

            int i = 0;
            detail::for_each(memory_tuple, buffer_info_tuple, [&i,&h](auto mem, auto bi) 
            {
                using device_type = typename std::remove_reference_t<decltype(*mem)>::device_type;
                using value_type  = typename std::remove_reference_t<decltype(*bi)>::value_type;
                auto& f     = h.m_fields[i];
                f.m_pattern = &(bi->get_pattern());
                f.m_pack    = [bi](void* buffer, const index_container_type& c) {bi->get_field().pack(reinterpret_cast<value_type*>(buffer),c); };
                f.m_unpack  = [bi](const void* buffer, const index_container_type& c) {bi->get_field().unpack(reinterpret_cast<const value_type*>(buffer),c); };
                f.m_recv_chunks.resize(f.m_pattern->recv_halos().size());
                f.m_send_chunks.resize(f.m_pattern->send_halos().size());

                auto& m_recv = mem->recv_memory[bi->device_id()];
                auto& m_send = mem->send_memory[bi->device_id()];

                int j=0;
                for (const auto& p_id_c : f.m_pattern->recv_halos())
                {
                    auto it = m_recv.find(p_id_c.first);
                    if (it == m_recv.end())
                        it = m_recv.insert(std::make_pair(p_id_c.first, device_type::template make_vector<char>(bi->device_id()))).first;
                    f.m_recv_chunks[j].m_offset = it->second.size();
                    it->second.resize( 
                        it->second.size() 
                        + alignof(value_type) 
                        + static_cast<std::size_t>(pattern_type::num_elements(p_id_c.second))*sizeof(value_type));
                    ++j;
                }
                j=0;
                for (const auto& p_id_c : f.m_pattern->send_halos())
                {
                    auto it = m_send.find(p_id_c.first);
                    if (it == m_send.end())
                        it = m_send.insert(std::make_pair(p_id_c.first, device_type::template make_vector<char>(bi->device_id()))).first;
                    f.m_send_chunks[j].m_offset = it->second.size();
                    it->second.resize( 
                        it->second.size() 
                        + alignof(value_type) 
                        + static_cast<std::size_t>(pattern_type::num_elements(p_id_c.second))*sizeof(value_type));
                    ++j;
                }
                ++i;
            });

            i = 0;
            detail::for_each(memory_tuple, buffer_info_tuple, [&i,&h](auto mem, auto bi) 
            {
                using device_type = typename std::remove_reference_t<decltype(*mem)>::device_type;
                using value_type  = typename std::remove_reference_t<decltype(*bi)>::value_type;
                auto& f     = h.m_fields[i];
                auto& m_recv = mem->recv_memory[bi->device_id()];
                auto& m_send = mem->send_memory[bi->device_id()];

                int j=0;
                for (const auto& p_id_c : f.m_pattern->recv_halos())
                {
                    auto& vec = m_recv[p_id_c.first];
                    f.m_recv_chunks[j].m_buffer = device_type::template align<value_type>(
                        vec.data()+f.m_recv_chunks[j].m_offset, bi->device_id());
                    ++j;
                }
                j=0;
                for (const auto& p_id_c : f.m_pattern->send_halos())
                {
                    auto& vec = m_send[p_id_c.first];
                    f.m_send_chunks[j].m_buffer = device_type::template align<value_type>(
                        vec.data()+f.m_send_chunks[j].m_offset, bi->device_id());
                    ++j;
                }
                ++i;
            });

            return std::move(h);
        }

    private:

        template<typename D>
        void allocate()
        {
            // todo: simplify code above by calling this function
        }
        template<typename D>
        void align()
        {
            // todo: simplify code above by calling this function
        }

        void clear()
        {
            detail::for_each(m_mem, [](auto& m)
            {
                for (auto& p0 : m.recv_memory)
                    for (auto& p1 : p0.second)
                        p1.second.resize(0);
                for (auto& p0 : m.send_memory)
                    for (auto& p1 : p0.second)
                        p1.second.resize(0);
            });
        }
    };

//    template<typename Pattern>
//    struct communication_object {};
//
//    template<typename P, typename GridType, typename DomainIdType>
//    class communication_object<pattern<P,GridType,DomainIdType>>
//    {
//    public: // member types
//
//        using pattern_type            = pattern<P,GridType,DomainIdType>;
//        using index_container_type    = typename pattern_type::index_container_type;
//        using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
//        using communicator_type       = typename pattern_type::communicator_type;
//        template<typename D, typename F>
//        using buffer_info_t           = buffer_info<pattern_type,D,F>;
//
//    private: // member types
//
//        using pack_function_type   = std::function<void(void*,const index_container_type&)>;
//        using unpack_function_type = std::function<void(void*,const index_container_type&)>;
//
//        template<typename Device>
//        struct internal_co_t
//        {
//            using device_type = Device;
//            using id_type = typename device_type::id_type;
//            using vector_type = typename device_type::template vector_type<char>;
//            using memory_type = std::map<
//                id_type,
//                std::map<
//                    extended_domain_id_type,
//                    vector_type
//                >
//            >;
//            memory_type recv_memory;
//            memory_type send_memory;
//        };
//
//        using internal_co_list = detail::transform<device::device_list>::with<internal_co_t>;
//
//    private: // members
//
//        internal_co_list m_list;
//
//    private:
//
//        template<typename D>
//        struct buffer_info_internal
//        {
//            //using memory_type = std::vector<typename D::template vector_type<char> *>;
//            //typename D::id_type  idx;
//
//            std::vector<typename internal_co_t<D>::vector_type*> recv_memory;
//            std::vector<typename internal_co_t<D>::vector_type*> send_memory;
//
//            pack_function_type   m_pack;
//            unpack_function_type m_unpack;
//            //memory_type          m_memory;
//            typename D::id_type  m_idx;
//            DomainIdType         m_domain_id;
//        };
//
//        template<typename D>
//        using bi_vec = std::vector<buffer_info_internal<D>>;
//
//        using bi_list = detail::transform<device::device_list>::with<bi_vec>;
//
//        bi_list m_bi_list;
//
//    public: // member functions
//
//        template<typename... Devices, typename... Fields>
//        void exchange(buffer_info_t<Devices, Fields>&&... buffer_infos)
//        {
//            std::array<std::size_t,                      sizeof...(Fields)> alignments{alignof(typename buffer_info_t<Devices,Fields>::value_type)...};
//            std::array<const std::vector<std::size_t> *, sizeof...(Fields)> sizes_recv{&buffer_infos.sizes_recv()...};
//            std::array<const std::vector<std::size_t> *, sizeof...(Fields)> sizes_send{&buffer_infos.sizes_send()...};
//            std::array<const pattern_type*,              sizeof...(Fields)> patterns{&buffer_infos.get_pattern()...};
//            std::array<std::vector<std::size_t>,         sizeof...(Fields)> recv_offsets;
//            std::array<std::vector<std::size_t>,         sizeof...(Fields)> send_offsets;
//
//            //std::tuple<std::vector<typename internal_co_t<Devices>::vector_type*>, sizeof...(Fields)> recv_memory;
//            //std::tuple<std::vector<typename internal_co_t<Devices>::vector_type*>, sizeof...(Fields)> send_memory;
//
//            std::array<communicator_type*,               sizeof...(Fields)> comms;
//            auto& comm = *(comms[0]);
//
//            std::tuple<typename Devices::id_type...> indices{buffer_infos.device_id()...};
//            std::tuple<Fields*...>                   field_ptrs{&buffer_infos.get_field()...};
//            std::tuple<void (Fields::*)(void*,const index_container_type&)...> pack_fct_ptrs{&Fields::template pack<index_container_type>...};
//            std::tuple<void (Fields::*)(void*,const index_container_type&)...> unpack_fct_ptrs{&Fields::template unpack<index_container_type>...};
//
//            auto pack_funcs   = detail::memfunc_ptr<index_container_type>(field_ptrs,   pack_fct_ptrs, std::make_index_sequence<sizeof...(Fields)>());
//            auto unpack_funcs = detail::memfunc_ptr<index_container_type>(field_ptrs, unpack_fct_ptrs, std::make_index_sequence<sizeof...(Fields)>());
//
//            std::tuple<internal_co_t<Devices>*...> list{&(std::get< internal_co_t<Devices> >(m_list))...};
//
//            //std::tuple<buffer_info_t<Devices,Fields>...> bis{std::move(buffer_infos)...};
//
//            int i = 0;
//            detail::for_each(list, indices, [this,&patterns,&i,&recv_offsets,&send_offsets,&alignments,&sizes_recv,&sizes_send](auto ico, auto idx)
//            //detail::for_each(list, bis, [this,&patterns,&i,&recv_offsets,&send_offsets,&alignments,&sizes_recv,&sizes_send](auto ico, auto& bi)
//            { 
//                using device_type = typename std::remove_reference_t<decltype(*ico)>::device_type;
//                allocate<device_type>(ico->recv_memory[idx], patterns[i]->recv_halos(), idx, 
//                                      recv_offsets[i], *(sizes_recv[i]), alignments[i]);
//                allocate<device_type>(ico->send_memory[idx], patterns[i]->send_halos(), idx, 
//                                      send_offsets[i], *(sizes_send[i]), alignments[i]);
//                ++i;
//            });
//
//            /*i = 0;
//            detail::for_each(list, indices, [](auto ico, auto idx)
//            { 
//                using device_type = typename std::remove_reference_t<decltype(*ico)>::device_type;
//                ico
//            }*/
//        }
//
//    private: // member functions
//
//        template<typename D>
//        void allocate(
//            typename internal_co_t<D>::memory_type::mapped_type& p, 
//            const typename pattern_type::map_type& halos, 
//            typename D::id_type idx,
//            std::vector<std::size_t>& offsets,
//            const std::vector<std::size_t>& sizes,
//            std::size_t alignment)
//            //std::vector<typename internal_co_t<D>::vector_type*>& memory)
//        {
//            int j = 0;
//            for (const auto& p_id_c : halos)
//            {
//                auto it = p.find(p_id_c.first);
//                if (it == p.end())
//                    it = p.insert( std::make_pair( p_id_c.first, D::template make_vector<char>(idx) ) ).first;
//                offsets.push_back(it->second.size());
//                //memory.push_back( &it->second );
//                it->second.resize( it->second.size() + alignment + sizes[j] );
//                ++j;
//            }
//        }
//    };

} // namespace gridtools

#endif /* INCLUDED_COMMUNICATION_OBJECT_ERASED_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

