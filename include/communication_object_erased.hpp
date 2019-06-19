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

#include "utils.hpp"
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

    } // namespace detail

    // forward declaration
    template<typename P, typename GridType, typename DomainIdType>
    class communication_object_erased;

    /** @brief handle type for waiting on asynchronous communication processes
     * @tparam P message protocol type
     * @tparam GridType grid tag type
     * @tparam DomainIdType domain id type*/
    template<typename P, typename GridType, typename DomainIdType>
    class communication_handle
    {
    private: // friend class
        friend class communication_object_erased<P,GridType,DomainIdType>;

    private: // member types
        using co_t                    = communication_object_erased<P,GridType,DomainIdType>;
        using communicator_type       = protocol::communicator<P>;
        using pattern_type            = pattern<P,GridType,DomainIdType>;
        using index_container_type    = typename pattern_type::index_container_type;
        using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
        using pack_function_type      = std::function<void(void*,const index_container_type&)>;
        using unpack_function_type    = std::function<void(const void*,const index_container_type&)>;

        // holds bookkeeping info for a data field
        // - type erased function pointers to pack and unpack methods
        // - pointers to memory locations
        struct field
        {
            struct chunk_type
            {
                std::size_t m_offset;
                void* m_buffer;
            };
            using chunk_vector_type = std::vector<chunk_type>;
            const pattern_type* m_pattern;
            pack_function_type m_pack;
            unpack_function_type m_unpack;
            chunk_vector_type m_recv_chunks;
            chunk_vector_type m_send_chunks;
        };

    private: // private constructor
        communication_handle(co_t& co, const communicator_type& comm, std::size_t size) : m_co{&co}, m_comm{comm}, m_fields(size) {}

    public: // copy and move ctors
        communication_handle(communication_handle&&) = default;
        communication_handle(const communication_handle&) = delete;

    public: // member function
        /** @brief  wait for communication to be finished*/
        void wait()
        {
            for (auto& f : m_futures) f.wait();
            unpack();
            m_co->clear();
        }

    private: // member functions
        void post()
        {
            pack();
            detail::for_each(m_co->m_mem, [this](auto& m)
            {
                for (auto& mm : m.recv_memory)
                    for (auto& p : mm.second)
                        if (p.second.size()>0)
                        {
                            //std::cout << "irecv(" << p.first.address << ", " << p.first.tag << ", " << p.second.size() << ")" << std::endl;
                            m_futures.push_back(m_comm.irecv(p.first.address, p.first.tag, p.second.data(), p.second.size()));
                        }
                for (auto& mm : m.send_memory)
                    for (auto& p : mm.second)
                        if (p.second.size()>0)
                        {
                            //std::cout << "isend(" << p.first.address << ", " << p.first.tag << ", " << p.second.size() << ")" << std::endl;
                            m_futures.push_back(m_comm.isend(p.first.address, p.first.tag, p.second));
                        }
            });
        }

        void pack()
        {
            // should I try to use inverse map to pack buffer by buffer?
            for (auto& f : m_fields)
            {
                std::size_t k=0;
                for (const auto& p_id_c : f.m_pattern->send_halos())
                    f.m_pack(f.m_send_chunks[k++].m_buffer, p_id_c.second);
            }
        }

        void unpack()
        {
            for (auto& f : m_fields)
            {
                std::size_t k=0;
                for (const auto& p_id_c : f.m_pattern->recv_halos())
                    f.m_unpack(f.m_recv_chunks[k++].m_buffer, p_id_c.second);
            }
        }

    private: // members
        co_t* m_co;
        communicator_type m_comm;
        std::vector<field> m_fields;
        std::vector<typename communicator_type::template future<void>> m_futures;
    };

    /** @brief communication object responsible for exchanging halo data. Allocates storage depending on the 
     * device type and device id of involved fields.
     * @tparam P message protocol type
     * @tparam GridType grid tag type
     * @tparam DomainIdType domain id type*/
    template<typename P, typename GridType, typename DomainIdType>
    class communication_object_erased
    {
    private: // friend class
        friend class communication_handle<P,GridType,DomainIdType>;

    public: // member types
        /** @brief handle type returned by exhange operation */
        using handle_type             = communication_handle<P,GridType,DomainIdType>;

    private: // member types
        using communicator_type       = typename handle_type::communicator_type;
        using pattern_type            = typename handle_type::pattern_type;
        using index_container_type    = typename handle_type::index_container_type;
        using extended_domain_id_type = typename handle_type::extended_domain_id_type;

        template<typename D, typename F>
        using buffer_info_type = buffer_info<pattern_type,D,F>;

        // holds actual memory
        // one instance will be created per device type
        // memory is organized in a map:
        // map( device_id -> 
        //                   map( extended_domain_id ->
        //                                               vector of chars (device specific) ))
        template<typename Device>
        struct buffer_memory
        {
            using device_type = Device;
            using id_type = typename device_type::id_type;
            using vector_type = typename device_type::template vector_type<char>;
            struct extended_domain_id_comp
            {
                bool operator()(const extended_domain_id_type& l, const extended_domain_id_type& r) const noexcept
                {
                    return (l.id < r.id ? true : (l.id == r.id ? (l.tag < r.tag) : false));
                }
            };
            using memory_type = std::map<
                id_type,
                std::map<
                    extended_domain_id_type,
                    vector_type,
                    extended_domain_id_comp
                >
            >;
            memory_type recv_memory;
            memory_type send_memory;
        };

        // tuple type of buffer_memory (one element for each device in device::device_list)
        using memory_type = detail::transform<device::device_list>::with<buffer_memory>;

    private: // members
        memory_type m_mem;

    public: // member functions
        /**
         * @brief blocking variant of halo exchange
         * @tparam Devices list of device types
         * @tparam Fields list of field types
         * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern
         */
        template<typename... Devices, typename... Fields>
        void bexchange(buffer_info_type<Devices,Fields>... buffer_infos)
        {
            exchange(buffer_infos...).wait();
        }

        /**
         * @brief non-blocking exchange of halo data
         * @tparam Devices list of device types
         * @tparam Fields list of field types
         * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern
         * @return handle to await communication
         */
        template<typename... Devices, typename... Fields>
        [[nodiscard]] handle_type exchange(buffer_info_type<Devices,Fields>... buffer_infos)
        {
            using buffer_infos_ptr_t  = std::tuple<std::remove_reference_t<decltype(buffer_infos)>*...>;
            using memory_t   = std::tuple<buffer_memory<Devices>*...>;

            buffer_infos_ptr_t buffer_info_tuple{&buffer_infos...};
            handle_type h(*this,std::get<0>(buffer_info_tuple)->get_pattern().communicator(), sizeof...(Fields));
            memory_t memory_tuple{&(std::get<buffer_memory<Devices>>(m_mem))...};

            int i = 0;
            detail::for_each(memory_tuple, buffer_info_tuple, [this,&i,&h](auto mem, auto bi) 
            {
                using device_type = typename std::remove_reference_t<decltype(*mem)>::device_type;
                using value_type  = typename std::remove_reference_t<decltype(*bi)>::value_type;
                auto& f     = h.m_fields[i];
                f.m_pattern = &(bi->get_pattern());
                auto field_ptr = &(bi->get_field());
                f.m_pack    = [field_ptr](void* buffer, const index_container_type& c) 
                                  { field_ptr->pack(reinterpret_cast<value_type*>(buffer),c); };
                f.m_unpack  = [field_ptr](const void* buffer, const index_container_type& c) 
                                  { field_ptr->unpack(reinterpret_cast<const value_type*>(buffer),c); };
                f.m_recv_chunks.resize(f.m_pattern->recv_halos().size());
                f.m_send_chunks.resize(f.m_pattern->send_halos().size());
                auto& m_recv = mem->recv_memory[bi->device_id()];
                auto& m_send = mem->send_memory[bi->device_id()];
                allocate<device_type, value_type>(f.m_pattern->recv_halos(), m_recv, bi->device_id(), f.m_recv_chunks);
                allocate<device_type, value_type>(f.m_pattern->send_halos(), m_send, bi->device_id(), f.m_send_chunks);
                ++i;
            });

            i = 0;
            detail::for_each(memory_tuple, buffer_info_tuple, [this,&i,&h](auto mem, auto bi) 
            {
                using device_type = typename std::remove_reference_t<decltype(*mem)>::device_type;
                using value_type  = typename std::remove_reference_t<decltype(*bi)>::value_type;
                auto& f     = h.m_fields[i];
                auto& m_recv = mem->recv_memory[bi->device_id()];
                auto& m_send = mem->send_memory[bi->device_id()];
                align<device_type, value_type>(f.m_pattern->recv_halos(), m_recv, bi->device_id(), f.m_recv_chunks);
                align<device_type, value_type>(f.m_pattern->send_halos(), m_send, bi->device_id(), f.m_send_chunks);
                ++i;
            });

            h.post();
            return std::move(h);
        }

    private: // member functions
        // allocate memory on device
        template<typename Device, typename ValueType>
        void allocate(
            const typename pattern_type::map_type& halos, 
            typename buffer_memory<Device>::memory_type::mapped_type& memory, 
            typename Device::id_type device_id, 
            typename handle_type::field::chunk_vector_type& chunks)
        {
            std::size_t j=0;
            for (const auto& p_id_c : halos)
            {
                auto it = memory.find(p_id_c.first);
                if (it == memory.end())
                    it = memory.insert(std::make_pair(p_id_c.first, Device::template make_vector<char>(device_id))).first;
                const auto prev_size = it->second.size();
                chunks[j].m_offset = prev_size;
                it->second.resize(0);
                it->second.resize(prev_size + alignof(ValueType) +
                                  static_cast<std::size_t>(pattern_type::num_elements(p_id_c.second))*sizeof(ValueType));
                ++j;
            }
        }

        // align memory on device
        template<typename Device, typename ValueType>
        void align(
            const typename pattern_type::map_type& halos, 
            typename buffer_memory<Device>::memory_type::mapped_type& memory, 
            typename Device::id_type device_id, 
            typename handle_type::field::chunk_vector_type& chunks)
        {
            int j=0;
            for (const auto& p_id_c : halos)
            {
                auto& vec = memory[p_id_c.first];
                chunks[j].m_buffer = Device::template align<ValueType>(vec.data()+chunks[j].m_offset, device_id);
                ++j;
            }
        }

        // reset storage (but doesn't deallocate)
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

    namespace detail {

        // helper template metafunction to test equality of a type with respect to all element types of a tuple
        template<typename Test, typename... Ts>
        struct test_eq_t {};

        template<typename Test, typename T0, typename T1, typename... Ts>
        struct test_eq_t<Test,T0,T1,Ts...> : public 
            std::integral_constant<
                bool, 
                std::is_same<Test,T0>::value && test_eq_t<Test,T1,Ts...>::value
            > {};

        template<typename Test, typename T0>
        struct test_eq_t<Test,T0> : public 
            std::integral_constant<bool, std::is_same<Test,T0>::value> {};

    } // namespace detail

    /**
     * @brief creates a communication object based on the patterns involved
     * @tparam Patterns list of pattern types
     * @param ... unnamed list of pattern_holder objects
     * @return communication object
     */
    template<typename... Patterns>
    auto make_communication_object(const Patterns&...)
    {
        using ps_t = std::tuple<typename Patterns::value_type...>;
        using p_t  = std::tuple_element_t<0,ps_t>;
        using protocol_type    = typename p_t::communicator_type::protocol_type;
        using grid_type        = typename p_t::grid_type;
        using domain_id_type   = typename p_t::domain_id_type;

        using test_t = pattern_container<protocol_type,grid_type,domain_id_type>;
        static_assert(detail::test_eq_t<test_t,Patterns...>::value, "patterns are incompatible");

        return std::move(communication_object_erased<protocol_type,grid_type,domain_id_type>());
    }

} // namespace gridtools

#endif /* INCLUDED_COMMUNICATION_OBJECT_ERASED_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

