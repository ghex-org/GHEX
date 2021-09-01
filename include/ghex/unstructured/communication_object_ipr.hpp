/* 
 * GridTools
 * 
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#pragma once

#include <ghex/config.hpp>
#include <ghex/context.hpp>
#include <ghex/arch_traits.hpp>
#include <ghex/buffer_info.hpp>
#include <ghex/util/for_each.hpp>
#include <ghex/util/test_eq.hpp>
#include <ghex/communication_object_ipr.hpp>
#include <ghex/device/stream.hpp>
#include <ghex/device/guard.hpp>
#include <ghex/packer.hpp>
#include <ghex/pattern_container.hpp>
#include <ghex/unstructured/grid.hpp>

#include <iosfwd>
#include <map>
#include <unordered_map>
#include <queue>

namespace ghex
{
/** @brief handle type for waiting on asynchronous communication processes.
  * The wait function is stored in a member.
  * @tparam Index index type for domain (local indices) and iteration space
  * @tparam DomainIdType domain id type*/
template<typename Index, typename DomainIdType>
class communication_handle_ipr<unstructured::detail::grid<Index>, DomainIdType>
{
  private:
    // member types
    using index_type = Index;
    using grid_type = unstructured::detail::grid<index_type>;
    using domain_id_type = DomainIdType;

    // friend class
    friend class communication_object_ipr<grid_type, domain_id_type>;

    // members
    std::function<void()> m_wait_fct;

  public:
    // public constructor
    /** @brief construct a ready handle */
    communication_handle_ipr() = default;

  private:
    // private constructor
    /** @brief construct a handle with a wait function
      * @tparam Func function type with signature void()
      * @param wait_fct wait function */
    template<typename Func>
    communication_handle_ipr(Func&& wait_fct)
    : m_wait_fct(std::forward<Func>(wait_fct))
    {
    }

  public:
    // copy and move ctors
    communication_handle_ipr(communication_handle_ipr&&) = default;
    communication_handle_ipr(const communication_handle_ipr&) = delete;
    communication_handle_ipr& operator=(communication_handle_ipr&&) = default;
    communication_handle_ipr& operator=(const communication_handle_ipr&) = delete;

    // member functions
    /** @brief  wait for communication to be finished*/
    void wait()
    {
        if (m_wait_fct) m_wait_fct();
    }
};

/** @brief communication object responsible for exchanging halo data.
  * Allocates storage depending on the device type and device id of involved fields.
  * @tparam Index index type for domain (local indices) and iteration space
  * @tparam DomainIdType domain id type*/
template<typename Index, typename DomainIdType>
class communication_object_ipr<unstructured::detail::grid<Index>, DomainIdType>
{
  public:
    // member types

    /** @brief handle type returned by exhange operation */
    using index_type = Index;
    using grid_type = unstructured::detail::grid<index_type>;
    using domain_id_type = DomainIdType;
    using handle_type = communication_handle_ipr<grid_type, domain_id_type>;
    using pattern_type = pattern<grid_type, domain_id_type>;
    using pattern_container_type = pattern_container<grid_type, domain_id_type>;

    template<typename D, typename F>
    using buffer_info_type = buffer_info<pattern_type, D, F>;

  private:
    // member types

    using communicator_type = oomph::communicator;
    using rank_type = communicator_type::rank_type;
    using index_container_type = typename pattern_type::index_container_type;
    using pack_function_type = std::function<void(
        void*, const index_container_type&, void*)>; // last argument intended as CUDA stream
    using send_request_type = oomph::send_request;
    using recv_request_type = oomph::recv_request;

    /** @brief pair of domain ids + tag id with ordering */
    struct domain_id_pair_and_tag
    {
        domain_id_type first_id;
        domain_id_type second_id;
        int            tag;

        bool operator<(const domain_id_pair_and_tag& other) const noexcept
        {
            return (first_id < other.first_id
                        ? true
                        : (first_id > other.first_id
                                  ? false
                                  : (second_id < other.second_id
                                            ? true
                                            : (second_id > other.second_id ? false
                                                                           : (tag < other.tag)))));
        }
    };

    /** @brief Holds a pointer to a set of iteration spaces and a callback function pointer
      * which is used to store a field's pack member function.
      * This class also stores the offset in the serialized buffer in bytes.
      * The type-erased field_ptr member is only used for the gpu-vector-interface.
      * @tparam Function pack function pointer type */
    template<typename Function>
    struct field_info
    {
        using index_container_type = typename pattern_type::map_type::mapped_type;
        Function                    call_back;
        const index_container_type* index_container;
        std::size_t                 offset;
        void*                       field_ptr;
    };

    /** @brief Holds serial buffer memory and meta information associated with it.
      * @tparam Function pack function pointer type (not used for unpacking)*/
    template<class Function>
    struct buffer
    {
        using field_info_type = field_info<Function>;
        rank_type                    rank;
        int                          tag;
        context::message_type        buffer;
        std::size_t                  size;
        std::vector<field_info_type> field_infos;
        device::stream               m_stream;
    };

    /** @brief Message-like wrapper over in-place receive memory*/
    class ipr_message
    {
      public:
        using byte = unsigned char;
        using value_type = byte;

      private:
        value_type* m_ipr_ptr;
        std::size_t m_size;

      public:
        ipr_message() = default;
        ipr_message(value_type* ipr_ptr)
        : m_ipr_ptr{ipr_ptr}
        , m_size{0}
        {
        }
        ipr_message(value_type* ipr_ptr, const std::size_t size)
        : m_ipr_ptr{ipr_ptr}
        , m_size{size}
        {
        }
        const value_type* data() const { return m_ipr_ptr; }
        value_type*       data() { return m_ipr_ptr; }
        std::size_t       size() const { return m_size; }
        void              resize(const std::size_t size) { m_size = size; }
    };

    /** @brief Equivalent of field_info for in-place receive. No function needed in this case.
      * This class also stores the offset in the serialized buffer in bytes.
      * The type-erased field_ptr member is only used for the gpu-vector-interface.*/
    struct field_info_ipr
    {
        using index_container_type = typename pattern_type::map_type::mapped_type;
        const index_container_type* index_container;
        std::size_t                 offset;
        void*                       field_ptr;
    };

    /** @brief Equivalent of buffer for in-place receive
      * Vector = ipr_message, Function not needed anymore;
      * not a specialization, because it is no longer a buffer*/
    struct recv_ipr_info
    {
        using field_info_type = field_info_ipr;
        rank_type                    rank;
        int                          tag;
        ipr_message                  message;
        std::size_t                  size;
        std::vector<field_info_type> field_infos;
        device::stream               m_stream;
    };

    /** @brief Holds maps of buffers / memory regions respectively for send and in-place receive operations.
      * Maps are indexed by a device id and a domain_id_pair_and_tag.
      * With in-place receive, the tag is needed to use separate buffers / memory regions
      * for different fields when exchanging data between the same domains,
      * since receive memory will not be contiguous in that case.
      * @tparam Arch the device on which the buffer / field memory is allocated */
    template<typename Arch>
    struct buffer_memory
    {
        using arch_type = Arch;
        using device_id_type = typename arch_traits<arch_type>::device_id_type;
        using send_buffer_type = buffer<pack_function_type>;
        using send_memory_type =
            std::map<device_id_type, std::map<domain_id_pair_and_tag, send_buffer_type>>;
        using recv_memory_type =
            std::map<device_id_type, std::map<domain_id_pair_and_tag, recv_ipr_info>>;

        send_memory_type send_memory;
        recv_memory_type recv_memory;

        //using future_type = typename communicator_type::template future<void>;
        std::vector<recv_request_type> m_recv_reqs;
    };

    /** tuple type of buffer_memory (one element for each device in arch_list) */
    using memory_type = detail::transform<arch_list>::with<buffer_memory>;

    // members
    bool                           m_valid;
    communicator_type              m_comm;
    memory_type                    m_mem;
    std::vector<send_request_type> m_send_reqs;

  public:
    // ctors
    communication_object_ipr(context& ctxt)
    : m_valid(false)
    , m_comm(c.transport_context()->get_communicator())
    {
    }
    communication_object_ipr(const communication_object_ipr&) = delete;
    communication_object_ipr(communication_object_ipr&&) = default;

    // member functions

    ///** @brief blocking variant of halo exchange
    //  * @tparam Archs list of device types
    //  * @tparam Fields list of field (descriptor) types
    //  * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern */
    //template<typename... Archs, typename... Fields>
    //void bexchange(buffer_info_type<Archs, Fields>... buffer_infos)
    //{
    //    exchange(buffer_infos...).wait();
    //}

    /** @brief non-blocking exchange of halo data with in-place receive (no unpacking)
      * @tparam Archs list of device types
      * @tparam Fields list of field (descriptor) types
      * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern
      * @return handle to await communication */
    template<typename... Archs, typename... Fields>
    [[nodiscard]] handle_type exchange(buffer_info_type<Archs, Fields>... buffer_infos)
    {
        // check that arguments are compatible
        using test_t = pattern_container<communicator_type, grid_type, domain_id_type>;
        static_assert(
            detail::test_eq_t<test_t,
                typename buffer_info_type<Archs, Fields>::pattern_container_type...>::value,
            "patterns are not compatible with this communication object");

        // check for ongoing exchange operations
        if (m_valid) throw std::runtime_error("earlier exchange operation was not finished");
        m_valid = true;

        // set tag offsets
        domain_id_type domain_ids[sizeof...(Fields)] = {buffer_infos.get_field().domain_id()...};
        int max_tags[sizeof...(Fields)] = {buffer_infos.get_pattern_container().max_tag()...};
        std::unordered_map<domain_id_type, std::queue<std::pair<int, int>>>
            tag_offsets_and_max_tags_per_id{};
        for (std::size_t i = 0; i < sizeof...(Fields); ++i)
        {
            auto& q = tag_offsets_and_max_tags_per_id[domain_ids[i]];
            if (q.empty()) { q.push(std::make_pair(0, max_tags[i])); }
            else
            {
                q.push(std::make_pair(q.back().first + q.back().second + 1, max_tags[i]));
            }
        }
        int tag_offsets[sizeof...(Fields)];
        for (std::size_t i = 0; i < sizeof...(Fields); ++i)
        {
            auto& q = tag_offsets_and_max_tags_per_id[domain_ids[i]];
            tag_offsets[i] = q.front().first;
            q.pop();
        }

        // store arguments and corresponding memory in tuples
        using buffer_infos_ptr_t = std::tuple<decltype(buffer_infos)*...>;
        using memory_t = std::tuple<buffer_memory<Archs>*...>;
        buffer_infos_ptr_t buffer_info_tuple{&buffer_infos...};
        // pointers to the buffer memory for the corresponding device
        memory_t memory_tuple{&(std::get<buffer_memory<Archs>>(m_mem))...};

        // loop over memory / buffer_infos and compute required space
        int i = 0;
        for_each(memory_tuple, buffer_info_tuple, [this, &i, &tag_offsets](auto mem, auto bi) {
            using arch_type = typename std::remove_reference_t<decltype(*mem)>::arch_type;
            using value_type = typename std::remove_reference_t<decltype(*bi)>::value_type;
            auto                 field_ptr = &(bi->get_field());
            const domain_id_type my_dom_id = bi->get_field().domain_id();
            allocate<arch_type, value_type>(
                mem, bi->get_pattern(), field_ptr, my_dom_id, bi->device_id(), tag_offsets[i]);
            ++i;
        });
        handle_type h([this]() { this->wait(); });
        post_recvs();
        pack();
        return h;
    }

    void post_recvs()
    {
        detail::for_each(m_mem, [this](auto& m) {
            using arch_type = typename std::remove_reference_t<decltype(m)>::arch_type;
            for (auto& p0 : m.recv_memory)
            {
                const auto device_id = p0.first;
                for (auto& p1 : p0.second)
                {
                    if (p1.second.size > 0u)
                    {
                        p1.second.message.resize(p1.second.size) m.m_recv_reqs.emplace_back(
                            m_comm.recv(p1.second.message, p1.second.address, p1.second.tag));
                    }
                }
            }
        });
    }

    void pack()
    {
        detail::for_each(m_mem, [this](auto& m) {
            using arch_type = typename std::remove_reference_t<decltype(m)>::arch_type;
            packer<arch_type>::pack(m, m_send_reqs, m_comm);
        });
    }

  private:
    /** @brief wait function*/
    void wait()
    {
        if (!m_valid) return;
        m_comm.wait_all();
        //detail::for_each(m_mem, [this](auto& m) {
        //    for (auto& f : m.m_recv_futures) f.wait(); // no unpacking
        //});
        //for (auto& f : m_send_futures) f.wait();
        clear();
    }

    /** @brief clear the internal flags so that a new exchange can be started.
                     * Important: does not deallocate. */
    void clear()
    {
        m_valid = false;
        m_send_reqs.clear();
        detail::for_each(m_mem, [this](auto& m) {
            m.m_recv_reqs.clear();
            for (auto& p0 : m.send_memory)
            {
                for (auto& p1 : p0.second)
                {
                    //p1.second.buffer.resize(0);
                    p1.second.size = 0;
                    p1.second.field_infos.resize(0);
                }
            }
            for (auto& p0 : m.recv_memory)
            {
                for (auto& p1 : p0.second)
                {
                    p1.second.size = 0;
                    p1.second.field_infos.resize(0);
                }
            }
        });
    }

    // allocation member functions

    /** @brief allocation member function (entry point).
      * This function calls in turn the allocation functions for receive and send,
      * respectively set_recv_size and allocate_send.
      * For the receive no allocation is actually needed, since field memory is used directly instead of buffers.
      * In this case the respective function (set_recv_size)
      * just sets the sizes and the other parameters needed for the receive operation.
      * Note that here memory (mem) is a pointer to a buffer_memory object and is passed by copy,
      * whereas in the inner functions it refers directly to the receive / send memory (respectively),
      * and is therefore passed by reference.
      * @tparam Arch the device on which the buffer / field memory is allocated
      * @tparam T buffer / field value type
      * @tparam Memory memory type (pointer to buffer_memory)
      * @tparam Field field (descriptor) type
      * @tparam O tag offset type (set to int in the exchange function)
      * @param mem buffer memory pointer
      * @param pattern pattern
      * @param field_ptr field pointer
      * @param dom_id domain id
      * @param device_id device id
      * @param tag_offset tag offset*/
    template<typename Arch, typename T, typename Memory, typename Field, typename O>
    void allocate(Memory mem, const pattern_type& pattern, Field* field_ptr, domain_id_type dom_id,
        typename arch_traits<Arch>::device_id_type device_id, O tag_offset)
    {
        set_recv_size<Arch, T>(
            mem->recv_memory[device_id], pattern.recv_halos(), dom_id, tag_offset, field_ptr);
        allocate_send<Arch, T>(
            mem->send_memory[device_id], pattern.send_halos(),
            [field_ptr](void* buffer, const index_container_type& c, void* arg) {
                field_ptr->pack(reinterpret_cast<T*>(buffer), c, arg);
            },
            dom_id, tag_offset, field_ptr);
    }

    template<typename Arch, typename ValueType, typename Memory, typename Halos,
        typename Field = void>
    void set_recv_size(Memory& memory, const Halos& halos, domain_id_type my_dom_id, int tag_offset,
        Field* field_ptr)
    {
        using byte = unsigned char;
        using buffer_type = recv_ipr_info;

        for (const auto& p_id_c : halos)
        {
            const auto num_elements = pattern_type::num_elements(p_id_c.second);
            if (num_elements < 1) continue;

            const auto     remote_rank = p_id_c.first.rank;
            const auto     remote_dom_id = p_id_c.first.id;
            const auto     tag = p_id_c.first.tag + tag_offset;
            domain_id_type left, right;
            left = my_dom_id;
            right = remote_dom_id;
            const auto d_p_t = domain_id_pair_and_tag{left, right, tag};
            auto       it = memory.find(d_p_t);

            if (it == memory.end())
            {
                it = memory
                         .insert(std::make_pair(d_p_t,
                             buffer_type{remote_rank, p_id_c.first.tag + tag_offset,
                                 ipr_message{reinterpret_cast<byte*>(&(field_ptr->operator()(
                                     p_id_c.second.front().local_indices().front(), 0)))},
                                 0, std::vector<typename buffer_type::field_info_type>(), {}}))
                         .first;
            }
            else if (it->second.size == 0)
            {
                it->second.rank = remote_rank;
                it->second.tag = p_id_c.first.tag + tag_offset;
                it->second.field_infos.resize(0);
            }

            const auto prev_size = it->second.size;
            const auto padding =
                ((prev_size + alignof(ValueType) - 1) / alignof(ValueType)) * alignof(ValueType) -
                prev_size;
            it->second.field_infos.push_back(typename buffer_type::field_info_type{
                &p_id_c.second, prev_size + padding, field_ptr});
            it->second.size += padding + static_cast<std::size_t>(num_elements) * sizeof(ValueType);
        }
    }

    template<typename Arch, typename ValueType, typename Memory, typename Halos, typename Function,
        typename Field = void>
    void allocate_send(Memory& memory, const Halos& halos, Function&& func,
        domain_id_type my_dom_id, int tag_offset, Field* field_ptr = nullptr)
    {
        using buffer_type = typename buffer_memory<Arch>::send_buffer_type;

        for (const auto& p_id_c : halos)
        {
            const auto num_elements = pattern_type::num_elements(p_id_c.second);
            if (num_elements < 1) continue;

            const auto     remote_rank = p_id_c.first.rank;
            const auto     remote_dom_id = p_id_c.first.id;
            const auto     tag = p_id_c.first.tag + tag_offset;
            domain_id_type left, right;
            left = remote_dom_id;
            right = my_dom_id;
            const auto d_p_t = domain_id_pair_and_tag{left, right, tag};
            auto       it = memory.find(d_p_t);

            if (it == memory.end())
            {
                it = memory
                         .insert(std::make_pair(
                             d_p_t, buffer_type{remote_rank, p_id_c.first.tag + tag_offset, {}, 0,
                                        std::vector<typename buffer_type::field_info_type>(), {}}))
                         .first;
            }
            else if (it->second.size == 0)
            {
                it->second.rank = remote_rank;
                it->second.tag = p_id_c.first.tag + tag_offset;
                it->second.field_infos.resize(0);
            }

            const auto prev_size = it->second.size;
            const auto padding =
                ((prev_size + alignof(ValueType) - 1) / alignof(ValueType)) * alignof(ValueType) -
                prev_size;
            it->second.field_infos.push_back(typename buffer_type::field_info_type{
                std::forward<Function>(func), &p_id_c.second, prev_size + padding, field_ptr});
            it->second.size += padding + static_cast<std::size_t>(num_elements) * sizeof(ValueType);
        }
    }
};

namespace detail
{
/** @brief creates a communication object (struct with implementation)*/
template<typename Index, typename DomainIdType>
struct make_communication_object_ipr_impl<unstructured::detail::grid<Index>, DomainIdType>
{
    using index_type = Index;
    using grid_type = unstructured::detail::grid<index_type>;
    using domain_id_type = DomainIdType;

    static auto apply(context& ctxt)
    {
        return communication_object_ipr<grid_type, domain_id_type>{ctxt};
    }
};

} // namespace detail

// } // namespace unstructured

} // namespace ghex
