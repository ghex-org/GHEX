/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ghex/config.hpp>
#include <ghex/context.hpp>
#include <ghex/util/for_each.hpp>
#include <ghex/util/test_eq.hpp>
#include <ghex/pattern_container.hpp>
#include <ghex/device/stream.hpp>
#include <ghex/device/guard.hpp>
#include <ghex/packer.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif
#include <map>
#include <stdio.h>
#include <functional>

namespace ghex
{
// forward declaration for optimization on regular grids
namespace structured
{
namespace regular
{
template<typename T, typename Arch, typename DomainDescriptor, typename Layout>
class field_descriptor;
} // namespace regular
} // namespace structured

// traits class for optimization on regular grids
namespace detail
{
template<typename T>
struct is_regular_gpu : public std::false_type
{
};
template<typename P, typename T, typename D, int... Order>
struct is_regular_gpu<buffer_info<P, gpu,
    structured::regular::field_descriptor<T, gpu, D, gridtools::layout_map<Order...>>>>
: public std::true_type
{
};
} // namespace detail

// forward declaration
struct generic_bulk_communication_object;
template<template<typename> class RangeGen, typename Pattern, typename... Fields>
class bulk_communication_object;

// forward declaration
template<typename GridType, typename DomainIdType>
class communication_object;

/** @brief handle type for waiting on asynchronous communication processes.
  * The wait function is stored in a member.
  * @tparam GridType grid tag type
  * @tparam DomainIdType domain id type*/
template<typename GridType, typename DomainIdType>
class communication_handle
{
  private: // friend class
    friend class communication_object<GridType, DomainIdType>;
    friend struct generic_bulk_communication_object;

  private: // member types
    using co_t = communication_object<GridType, DomainIdType>;

  private: // members
    co_t* m_co = nullptr;

  public: // public constructor
    /** @brief construct a ready handle
      * @param comm communicator */
    communication_handle() {}

  private: // private constructor
    /** @brief construct a handle */
    communication_handle(co_t* co)
    : m_co{co}
    {
    }

  public: // copy and move ctors
    communication_handle(communication_handle&&) = default;
    communication_handle(const communication_handle&) = delete;
    communication_handle& operator=(communication_handle&&) = default;
    communication_handle& operator=(const communication_handle&) = delete;

  public: // member functions
    /** @brief wait for communication to be finished */
    void wait();

#ifdef GHEX_CUDACC
    /**
     * \brief	Schedule a wait for the communication on `stream`.
     *
     * Add synchronization to `stream` such that all work that is scheduled _next_
     * on it will only start after _all_ communication has finished.
     * Thus it is important that when this function returns, the communication and
     * unpacking has not necessaraly concluded, but all work that is send to `stream`
     * will wait for it.
     *
     * However, the function will wait for all recive communication, not the unpacking,
     * has finished.
     */
    void schedule_wait(cudaStream_t stream);
#endif

    /** @brief check whether communication is finished */
    bool is_ready();
    /** @brief progress the communication */
    void progress();
};

/** @brief communication object responsible for exchanging halo data. Allocates storage depending on the 
         * device type and device id of involved fields.
         * @tparam GridType grid tag type
         * @tparam DomainIdType domain id type*/
template<typename GridType, typename DomainIdType>
class communication_object
{
    //TODO: Can we add the event pool as a member here? is this nice and okay from a MT point?

  public: // member types
    /** @brief handle type returned by exhange operation */
    using handle_type = communication_handle<GridType, DomainIdType>;
    using grid_type = GridType;
    using domain_id_type = DomainIdType;
    using pattern_type = pattern<GridType, DomainIdType>;
    using pattern_container_type = pattern_container<GridType, DomainIdType>;
    using this_type = communication_object<GridType, DomainIdType>;

    template<typename D, typename F>
    using buffer_info_type = buffer_info<pattern_type, D, F>;

  private: // friend class
    friend class communication_handle<GridType, DomainIdType>;
    template<template<typename> class RangeGen, typename Pattern, typename... Fields>
    friend class bulk_communication_object;

  private: // member types
    using communicator_type = oomph::communicator;
    using rank_type = oomph::rank_type;
    using index_container_type = typename pattern_type::index_container_type;
    using pack_function_type = std::function<void(void*, const index_container_type&, void*)>;
    using unpack_function_type =
        std::function<void(const void*, const index_container_type&, void*)>;
    using send_request_type = oomph::send_request;
    using recv_request_type = oomph::recv_request;

    /** @brief pair of domain ids with ordering */
    struct domain_id_pair
    {
        domain_id_type first_id;
        domain_id_type second_id;
        bool           operator<(const domain_id_pair& other) const noexcept
        {
            return (first_id < other.first_id
                        ? true
                        : (first_id > other.first_id ? false : (second_id < other.second_id)));
        }
    };

    /** @brief Holds a pointer to a set of iteration spaces and a callback function pointer 
      * which is used to store a field's pack or unpack member function. 
      * This class also stores the offset in the serialized buffer in bytes.
      * The type-erased field_ptr member is only used for the gpu-vector-interface.
      * @tparam Function Either pack or unpack function pointer type */
    template<typename Function>
    struct field_info
    {
        using index_container_type = typename pattern_type::map_type::mapped_type;
        Function                    call_back;
        const index_container_type* index_container;
        std::size_t                 offset;
        void*                       field_ptr;
    };

    /** @brief Holds serial buffer memory and meta information associated with it
      * @tparam Vector contiguous buffer memory type
      * @tparam Function Either pack or unpack function pointer type */
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

    /** @brief Holds maps of buffers for send and recieve operations indexed by a domain_id_pair and
      * a device id
      * @tparam Arch the device on which the buffer memory is allocated */
    template<typename Arch>
    struct buffer_memory
    {
        using arch_type = Arch;
        using device_id_type = typename arch_traits<Arch>::device_id_type;
        using send_buffer_type = buffer<pack_function_type>;
        using recv_buffer_type = buffer<unpack_function_type>;
        using send_memory_type =
            std::map<device_id_type, std::map<domain_id_pair, send_buffer_type>>;
        using recv_memory_type =
            std::map<device_id_type, std::map<domain_id_pair, recv_buffer_type>>;

        send_memory_type send_memory;
        recv_memory_type recv_memory;
    };

    /** tuple type of buffer_memory (one element for each device in arch_list) */
    using memory_type = transform<arch_list>::with<buffer_memory>;

    template<typename T, typename R>
    using disable_if_buffer_info = std::enable_if_t<!is_buffer_info<T>::value, R>;

  private: // members
    bool                           m_valid;
    communicator_type              m_comm;
    memory_type                    m_mem;
    std::vector<send_request_type> m_send_reqs;
    std::vector<recv_request_type> m_recv_reqs;

  public: // ctors
    communication_object(context& c)
    : m_valid(false)
    , m_comm(c.transport_context()->get_communicator())
    {
    }
    communication_object(const communication_object&) = delete;
    communication_object(communication_object&&) = default;

    communicator_type& communicator() { return m_comm; }

  public: // exchange arbitrary field-device-pattern combinations
    /** @brief non-blocking exchange of halo data
      * @tparam Archs list of device types
      * @tparam Fields list of field types
      * @param buffer_infos buffer_info objects created by binding a field descriptor to a pattern
      * @return handle to await communication */
    template<typename... Archs, typename... Fields>
    [[nodiscard]] handle_type exchange(buffer_info_type<Archs, Fields>... buffer_infos)
    {
        exchange_impl(buffer_infos...);
        post_recvs();
        pack_and_send();
        return {this};
    }

#if defined(GHEX_CUDACC) // TODO
    /**
     * \brief	Schedule an asynchronous exchange.
     *
     * In the asynchronous exchange the function does not block but schedules everything
     * on the device. The function will schedule all packing, i.e. putting the hallos
     * into continious memory, such that they wait on the passed stream. Thus no
     * packing will start before all work, that has been scheduled in `stream` has finished.
     *
     * The function will ensure that all exchanges, that have been started before have
     * concluded.
     *
     * The function will return when all send request have been completed.
     *
     * Note that this function must be matched by a call to `schedule_wait()` on the returned
     * handle.
     */
    template<typename... Archs, typename... Fields>
    [[nodiscard]] handle_type schedule_exchange(
        // TODO: Accept unmanaged (i.e. one that isn't freed) device::stream
        // and construct implicitly from cudaStream_t or hipStream_t?
        cudaStream_t stream, buffer_info_type<Archs, Fields>... buffer_infos)
    {
        std::cerr << "Using main schedule_exchange overload\n";
        std::cerr << "stream is " << stream << "\n";

        // make sure previous exchange finished
        // TODO: skip this? instead just keep adding to request vectors etc.
        // and require wait before destruction? allow explicitly calling
        // progress (currently private)?
        // Comment(phimuell): I do not think that keep appending is a good idea, because at one point
        // 	the thing becomes too big, so I would say we should clear it, but implement it as lightweight
        // 	as possible.
        wait();

        //Allocate memory, probably for the reciving buffers.
        exchange_impl(buffer_infos...);

        //Create the MPI handle for the reciving, are they using `IRecv`?
        post_recvs(stream);

        //TODO: the function will wait until the sends have been concluded, so it is not truely asynchronous.
        //	It is hard because this might lead to race conditions somewhere else.
        pack_and_send(stream);

        // Trigger unpacking, but don't wait for unpacking.
        // TODO: Not sure if this needed, because it makes it even less asynchrnous.
        // 	Furthermore, when `schedule_wait()` is called, this function is called again.
        // 	Thus I would remove it.
        m_comm.wait_all();

        return {this};

        // TODO: NCCL and MPI backends can be scheduled differently with
        // "async" functionality, but that exposes implementation details.
        // Should both be allowed? Can one be emulated in terms of the other to
        // support both modes? Caller has to know which mode to use...?
        // Concretely:
        // - MPI can be split into two (or three) phases: 1. post recv and trigger
        //   packing, 2. post sends, (3.) wait for recv, trigger unpacking
        // - NCCL can be scheduled all in one go, and should be scheduled all
        //   in one go as part of a single NCCL group (posting receives before
        //   sends can lead to deadlocks). But synchronizing unpacking should
        //   be done in a separate stage.
    }
#endif

    /** @brief  non-blocking exchange of halo data
                  * @tparam Iterator Iterator type to range of buffer_info objects
                  * @param first points to the begin of the range
                  * @param last points to the end of the range
                  * @return handle to await communication */
    template<typename Iterator>
    [[nodiscard]] disable_if_buffer_info<Iterator, handle_type> exchange(Iterator first,
        Iterator                                                                  last)
    {
        // call special function for a single range
        return exchange_u(first, last);
    }

    /** @brief  non-blocking exchange of halo data
                  * @tparam Iterator0 Iterator type to range of buffer_info objects
                  * @tparam Iterator1 Iterator type to range of buffer_info objects
                  * @tparam Iterators Iterator types to ranges of buffer_info objects
                  * @param first0 points to the begin of the range0
                  * @param last0 points to the end of the range0
                  * @param first1 points to the begin of the range1
                  * @param last1 points to the end of the range1
                  * @param iters first and last iterators for further ranges
                  * @return handle to await communication */
    template<typename Iterator0, typename Iterator1, typename... Iterators>
    [[nodiscard]] disable_if_buffer_info<Iterator0, handle_type> exchange(Iterator0 first0,
        Iterator0 last0, Iterator1 first1, Iterator1 last1, Iterators... iters)
    {
        static_assert(sizeof...(Iterators) % 2 == 0,
            "need even number of iteratiors: (begin,end) pairs");
        // call helper function to turn iterators into pairs of iterators
        return exchange_make_pairs(std::make_index_sequence<2 + sizeof...(iters) / 2>(), first0,
            last0, first1, last1, iters...);
    }

  private: // implementation
    // overload for pairs of iterators
    template<typename... Iterators>
    [[nodiscard]] handle_type exchange(std::pair<Iterators, Iterators>... iter_pairs)
    {
        exchange_impl(iter_pairs...);
        post_recvs();
        pack_and_send();
        return {this};
    }

    // helper function to turn iterators into pairs of iterators
    template<std::size_t... Is, typename... Iterators>
    [[nodiscard]] handle_type exchange_make_pairs(std::index_sequence<Is...>, Iterators... iters)
    {
        const std::tuple<Iterators...> iter_t{iters...};
        // call exchange with pairs of iterators
        return exchange(std::make_pair(std::get<2 * Is>(iter_t), std::get<2 * Is + 1>(iter_t))...);
    }

    // special function to handle one iterator pair (optimization for gpus below)
    template<typename Iterator>
#if defined(GHEX_COMM_OBJ_USE_U) && !defined(GHEX_GPU_MODE_EMULATE)
    [[nodiscard]] std::enable_if_t<
        !detail::is_regular_gpu<typename std::iterator_traits<Iterator>::value_type>::value,
        handle_type>
#else
    [[nodiscard]] handle_type
#endif
    exchange_u(Iterator first, Iterator last)
    {
        // call exchange with a pair of iterators
        return exchange(std::make_pair(first, last));
    }

#if defined(GHEX_COMM_OBJ_USE_U) && !defined(GHEX_GPU_MODE_EMULATE) && defined(GHEX_CUDACC)
    // optimized exchange for regular grids and a range of same-type fields
    template<typename Iterator>
    [[nodiscard]] std::enable_if_t<
        detail::is_regular_gpu<typename std::iterator_traits<Iterator>::value_type>::value,
        handle_type>
    exchange_u(Iterator first, Iterator last)
    {
        using gpu_mem_t = buffer_memory<gpu>;
        using field_type = std::remove_reference_t<decltype(first->get_field())>;
        using value_type = typename field_type::value_type;
        exchange_impl(std::make_pair(first, last));
        // post recvs
        auto& gpu_mem = std::get<gpu_mem_t>(m_mem);
        for (auto& p0 : gpu_mem.recv_memory)
        {
            const auto device_id = p0.first;
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u)
                {
                    if (!p1.second.buffer || p1.second.buffer.size() != p1.second.size ||
                        p1.second.buffer.device_id() != device_id)
                        p1.second.buffer =
                            arch_traits<gpu>::make_message(m_comm, p1.second.size, device_id);
                    // use callbacks for unpacking
                    auto ptr = &p1.second;
                    m_recv_reqs.push_back(
                        m_comm.recv(p1.second.buffer, p1.second.rank, p1.second.tag,
                            [ptr](context::message_type& m, context::rank_type, context::tag_type)
                            {
                                device::guard g(m);
                                packer<gpu>::unpack(*ptr, g.data());
                            }));
                }
            }
        }
        // pack
        packer<gpu>::template pack_u<value_type, field_type>(gpu_mem, m_send_reqs, m_comm);
        // return handle
        return {this};
    }
#endif

    // helper function to set up communicaton buffers (run-time case)
    template<typename... Iterators>
    void exchange_impl(std::pair<Iterators, Iterators>... iter_pairs)
    {
        const std::tuple<std::pair<Iterators, Iterators>...> iter_pairs_t{iter_pairs...};

        if (m_valid) throw std::runtime_error("earlier exchange operation was not finished");
        m_valid = true;

        // build a tag map
        using test_t = pattern_container<grid_type, domain_id_type>;
        std::map<const test_t*, int> pat_ptr_map;
        int                          max_tag = 0;
        for_each(iter_pairs_t,
            [&pat_ptr_map, &max_tag](std::size_t, auto iter_pair)
            {
                for (auto it = iter_pair.first; it != iter_pair.second; ++it)
                {
                    auto ptr = &(it->get_pattern_container());
                    auto p_it_bool = pat_ptr_map.insert(std::make_pair(ptr, max_tag));
                    if (p_it_bool.second == true) max_tag += ptr->max_tag() + 1;
                }
            });
        for_each(iter_pairs_t,
            [this, &pat_ptr_map](std::size_t, auto iter_pair)
            {
                using buffer_info_t =
                    typename std::remove_reference<decltype(*iter_pair.first)>::type;
                using arch_t = typename buffer_info_t::arch_type;
                using value_t = typename buffer_info_t::value_type;
                auto mem = &(std::get<buffer_memory<arch_t>>(m_mem));
                for (auto it = iter_pair.first; it != iter_pair.second; ++it)
                {
                    auto       field_ptr = &(it->get_field());
                    auto       tag_offset = pat_ptr_map[&(it->get_pattern_container())];
                    const auto my_dom_id = it->get_field().domain_id();
                    allocate<arch_t, value_t>(mem, it->get_pattern(), field_ptr, my_dom_id,
                        it->device_id(), tag_offset);
                }
            });
    }

    // helper function to set up communicaton buffers (compile-time case)
    template<typename... Archs, typename... Fields>
    void exchange_impl(buffer_info_type<Archs, Fields>... buffer_infos)
    {
        // check that arguments are compatible
        using test_t = pattern_container<grid_type, domain_id_type>;
        static_assert(
            test_eq_t<test_t,
                typename buffer_info_type<Archs, Fields>::pattern_container_type...>::value,
            "patterns are not compatible with this communication object");
        if (m_valid) throw std::runtime_error("earlier exchange operation was not finished");
        m_valid = true;

        // temporarily store address of pattern containers
        const test_t* ptrs[sizeof...(Fields)] = {&(buffer_infos.get_pattern_container())...};
        // build a tag map
        std::map<const test_t*, int> pat_ptr_map;
        int                          max_tag = 0;
        for (unsigned int k = 0; k < sizeof...(Fields); ++k)
        {
            auto p_it_bool = pat_ptr_map.insert(std::make_pair(ptrs[k], max_tag));
            if (p_it_bool.second == true) max_tag += ptrs[k]->max_tag() + 1;
        }
        // compute tag offset for each field
        int tag_offsets[sizeof...(Fields)] = {
            pat_ptr_map[&(buffer_infos.get_pattern_container())]...};
        // store arguments and corresponding memory in tuples
        using buffer_infos_ptr_t = std::tuple<std::remove_reference_t<decltype(buffer_infos)>*...>;
        using memory_t = std::tuple<buffer_memory<Archs>*...>;
        buffer_infos_ptr_t buffer_info_tuple{&buffer_infos...};
        memory_t           memory_tuple{&(std::get<buffer_memory<Archs>>(m_mem))...};
        // loop over buffer_infos/memory and compute required space
        for_each(memory_tuple, buffer_info_tuple,
            [this, &tag_offsets](std::size_t i, auto mem, auto bi)
            {
                using arch_type = typename std::remove_reference_t<decltype(*mem)>::arch_type;
                using value_type = typename std::remove_reference_t<decltype(*bi)>::value_type;
                auto                 field_ptr = &(bi->get_field());
                const domain_id_type my_dom_id = bi->get_field().domain_id();
                allocate<arch_type, value_type>(mem, bi->get_pattern(), field_ptr, my_dom_id,
                    bi->device_id(), tag_offsets[i]);
            });
    }

    /* Create the receve calls in blocking case. */

    void post_recvs() { post_recvs_impl<false>(); }

#ifdef GHEX_CUDACC
    /**
     * \brief	Create the receive calls for the asynchronous case.
     *
     * Packing will wait until all work on `stream` has finished.
     */
    void post_recvs(cudaStream_t stream)
    {
        //TODO: Maybe rename this function to `schedule_post_recvs()`?
        post_recvs_impl<true>(stream);
    }
#endif

    template<bool UseAsyncStream, typename... StreamType>
    void post_recvs_impl(StreamType&&... sync_streams)
    {
        static_assert(
            UseAsyncStream ? (sizeof...(sync_streams) > 0) : (sizeof...(sync_streams) == 0));

        for_each(m_mem,
            [this, sync_streams...](std::size_t, auto& m)
            {
                using arch_type = typename std::remove_reference_t<decltype(m)>::arch_type;
                for (auto& p0 : m.recv_memory)
                {
                    const auto device_id = p0.first;
                    for (auto& p1 : p0.second)
                    {
                        if (p1.second.size > 0u)
                        {
                            if (!p1.second.buffer || p1.second.buffer.size() != p1.second.size
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
                                || p1.second.buffer.device_id() != device_id
#endif
                            )
                                p1.second.buffer = arch_traits<arch_type>::make_message(m_comm,
                                    p1.second.size, device_id);
                            auto ptr = &p1.second;
                            // use callbacks for unpacking
                            // TODO: Reserve space in vector?
                            // TODO: Also think of where the vector is freed, depending on where we do wait.
                            m_recv_reqs.push_back(m_comm.recv(p1.second.buffer, p1.second.rank,
                                p1.second.tag,
                                [ptr, sync_streams...](context::message_type& m, context::rank_type,
                                    context::tag_type)
                                {
                                    device::guard g(m);
                                    packer<arch_type>::unpack(*ptr, g.data());

#ifdef GHEX_CUDACC
                                    if constexpr (UseAsyncStream && std::is_same_v<arch_type, gpu>)
                                    {
                                        // TODO: Cache/pool events. Relatively cheap to
                                        // create, but not free.
                                        // NOTE: Ideally we would write `StreamType` here, but this is not possible for some reason.
                                        // In that case we could drop the `ifdef`.
                                        auto record_streams =
                                            [ptr](cudaStream_t stream) -> std::uintptr_t
                                        {
                                            // NOTE: No race condition here, the event destruction, through the destructor of
                                            // 	`device::cuda_event`, will only be scheduled and performed by the runtime, when
                                            // 	the event happened.
                                            device::cuda_event event;
                                            GHEX_CHECK_CUDA_RESULT(
                                                cudaEventRecord(event.get(), ptr->m_stream));
                                            GHEX_CHECK_CUDA_RESULT(
                                                cudaStreamWaitEvent(stream, event.get()));
                                            return (std::uintptr_t)stream;
                                        };
                                        std::uintptr_t unused_variable_for_expansion[] = {
                                            record_streams(sync_streams)...};
                                        (void)unused_variable_for_expansion;
                                    }
#endif
                                }));
                        }
                    }
                }
            });
    }

    //Blocking version of `pack_and_send()`.
    void pack_and_send() { pack_and_send<false>(); }

#ifdef GHEX_CUDACC
    //Non-blocking version of `pack_and_send()` that will make sure that packing will wait until
    // everything submitted to stream `stream` has finished.
    void pack_and_send(cudaStream_t stream) { pack_and_send<true>(stream); };
#endif

    template<bool UseAsyncStream, typename... StreamType>
    void pack_and_send(StreamType&&... sync_streams)
    {
        static_assert(
            UseAsyncStream ? (sizeof...(sync_streams) > 0) : (sizeof...(sync_streams) == 0));
        for_each(m_mem,
            [this, sync_streams...](std::size_t, auto& m)
            {
                using arch_type = typename std::remove_reference_t<decltype(m)>::arch_type;
#ifdef GHEX_CUDACC
                if constexpr (UseAsyncStream && std::is_same_v<arch_type, gpu>)
                {
                    std::cerr << "creating cuda event\n";

                    //Put an event on the stream on which the packing is supposed to wait.
                    //NOTE: Currently only works for one stream because an event can only
                    //	be recorded to a single stream.
                    //NOTE: See not about `StreamType` in `post_recvs()`.
                    device::cuda_event event;
                    static_assert((not UseAsyncStream) || (sizeof...(sync_streams) == 1));
                    auto record_capturer = [&event](cudaStream_t stream) -> std::uintptr_t
                    {
                        std::cerr << "recording event on stream " << stream << "\n";
                        //TODO: Is a device guard needed here? What should be the memory?
                        GHEX_CHECK_CUDA_RESULT(cudaEventRecord(event.get(), stream));
                        return (std::uintptr_t)stream;
                    };
                    const std::uintptr_t unused_variable_for_expansion[] = {
                        record_capturer(sync_streams)...};
                    (void)unused_variable_for_expansion;

                    for (auto& p0 : m.send_memory)
                    {
                        for (auto& p1 : p0.second)
                        {
                            if (p1.second.size > 0u)
                            {
                                std::cerr << "adding wait on stream " << p1.second.m_stream.get()
                                          << "\n";
                                //Add the event to any stream that is used for packing, before starting the actuall
                                // packing. This ensures that packing will only start if any work has concluded.
                                //Is this device guard correct?
                                device::guard g(p1.second.buffer);
                                GHEX_CHECK_CUDA_RESULT(
                                    cudaStreamWaitEvent(p1.second.m_stream.get(), event.get()));
                            }
                        }
                    }
                }
#endif

                //NOTE: This function currently blocks until the send has been fully scheduled.
                std::cerr << "starting packing and creating the send request\n";
                packer<arch_type>::pack(m, m_send_reqs, m_comm);
            });
    }

  private: // wait functions
    void progress()
    {
        if (!m_valid) return;
        m_comm.progress();
    }

    bool is_ready()
    {
        if (!m_valid) return true;
        if (m_comm.is_ready())
        {
#ifdef GHEX_CUDACC
            sync_streams();
#endif
            clear();
            return true;
        }
        m_comm.progress();
        if (m_comm.is_ready())
        {
#ifdef GHEX_CUDACC
            sync_streams();
#endif
            clear();
            return true;
        }
        return false;
    }

    void wait()
    {
        if (!m_valid) return;
        // wait for data to arrive (unpack callback will be invoked)
        m_comm.wait_all();
#ifdef GHEX_CUDACC
        sync_streams();
#endif
        clear();
    }

#ifdef GHEX_CUDACC
    //See descripto of the handle.
    void schedule_wait(cudaStream_t stream)
    {
        if (!m_valid) return;
        // wait for data to arrive (unpack callback will be invoked)
        // This function calls `progress()` which is needed for MPI to make
        // progress and process the recieve operations.
        m_comm.wait_all();

        schedule_sync_streams(stream);
        // TODO: What is supposed to clear?
        // clear();
    }
#endif

#ifdef GHEX_CUDACC
  private: // synchronize (unpacking) streams
    //Ensures that all communication has finished.
    void sync_streams()
    {
        //NOTE: Depending on how `pack_and_send()` is modified here might be a race condition.
        //	This is because currently `pack_and_send()` waits until everything has been send,
        //	thus if we are here, we know that the send operations have concluded and we only
        //	have to check the recive buffer.
        using gpu_mem_t = buffer_memory<gpu>;
        auto& m = std::get<gpu_mem_t>(m_mem);
        for (auto& p0 : m.recv_memory)
        {
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u) { p1.second.m_stream.sync(); }
            }
        }
    }

    //Actuall implementation of the scheduled wait, for more information, see
    // the description of `communication_handle::schedule_wait()`.
    void schedule_sync_streams(cudaStream_t stream)
    {
        // TODO: Pool events.
        constexpr std::size_t                  num_events{128};
        static std::vector<device::cuda_event> events(num_events);
        static std::size_t                     event_index{0};

        //TODO: We only iterate over the recive buffers and not over the send streams.
        //	Currently this is not needed, because of how `pack_and_send()` is implemented,
        //	as it will wait until send has been completed, but depending on how the
        //	function is changed we have to modify this function.
        using gpu_mem_t = buffer_memory<gpu>;
        auto& m = std::get<gpu_mem_t>(m_mem);
        for (auto& p0 : m.recv_memory)
        {
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u)
                {
                    // Instead of doing a blocking wait, create events on each
                    // stream that the default stream waits for. This assumes
                    // that all kernels that need the unpacked data will use or
                    // synchronize with the default stream.
                    cudaEvent_t& e = events[event_index].get();
                    event_index = (event_index + 1) % num_events;
                    GHEX_CHECK_CUDA_RESULT(cudaEventRecord(e, p1.second.m_stream.get()));
                    GHEX_CHECK_CUDA_RESULT(cudaStreamWaitEvent(stream, e));
                }
            }
        }
    }
#endif

  private: // reset
    // clear the internal flags so that a new exchange can be started
    // important: does not deallocate
    void clear()
    {
        m_valid = false;
        m_send_reqs.clear();
        m_recv_reqs.clear();
        for_each(m_mem,
            [this](std::size_t, auto& m)
            {
                for (auto& p0 : m.send_memory)
                    for (auto& p1 : p0.second)
                    {
                        p1.second.size = 0;
                        p1.second.field_infos.resize(0);
                    }
                for (auto& p0 : m.recv_memory)
                    for (auto& p1 : p0.second)
                    {
                        p1.second.size = 0;
                        p1.second.field_infos.resize(0);
                    }
            });
    }

    //  private: // allocation member functions
    template<typename Arch, typename T, typename Memory, typename Field, typename O>
    void allocate(Memory& mem, const pattern_type& pattern, Field* field_ptr, domain_id_type dom_id,
        typename arch_traits<Arch>::device_id_type device_id, O tag_offset)
    {
        allocate<Arch, T, typename buffer_memory<Arch>::recv_buffer_type>(
            mem->recv_memory[device_id], pattern.recv_halos(),
            [field_ptr](const void* buffer, const index_container_type& c, void* arg)
            { field_ptr->unpack(reinterpret_cast<const T*>(buffer), c, arg); }, dom_id, tag_offset,
            true, field_ptr);
        allocate<Arch, T, typename buffer_memory<Arch>::send_buffer_type>(
            mem->send_memory[device_id], pattern.send_halos(),
            [field_ptr](void* buffer, const index_container_type& c, void* arg)
            { field_ptr->pack(reinterpret_cast<T*>(buffer), c, arg); }, dom_id, tag_offset, false,
            field_ptr);
    }

    // compute memory requirements to be allocated on the device
    template<typename Arch, typename ValueType, typename BufferType, typename Memory,
        typename Halos, typename Function, typename Field = void>
    void allocate(Memory& memory, const Halos& halos, Function&& func, domain_id_type my_dom_id,
        int tag_offset, bool receive, Field* field_ptr = nullptr)
    {
        for (const auto& p_id_c : halos)
        {
            const auto num_elements =
                pattern_type::num_elements(p_id_c.second) * field_ptr->num_components();
            if (num_elements < 1) continue;
            const auto     remote_rank = p_id_c.first.mpi_rank;
            const auto     remote_dom_id = p_id_c.first.id;
            domain_id_type left, right;
            if (receive)
            {
                left = my_dom_id;
                right = remote_dom_id;
            }
            else
            {
                left = remote_dom_id;
                right = my_dom_id;
            }
            const auto d_p = domain_id_pair{left, right};
            auto       it = memory.find(d_p);
            if (it == memory.end())
            {
                it = memory
                         .insert(std::make_pair(d_p,
                             BufferType{remote_rank, p_id_c.first.tag + tag_offset, {}, 0,
                                 std::vector<typename BufferType::field_info_type>(), {}}))
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
            it->second.field_infos.push_back(typename BufferType::field_info_type{
                std::forward<Function>(func), &p_id_c.second, prev_size + padding, field_ptr});
            it->second.size += padding + static_cast<std::size_t>(num_elements) * sizeof(ValueType);
        }
    }
};

template<typename GridType, typename DomainIdType>
void
communication_handle<GridType, DomainIdType>::wait()
{
    if (m_co) m_co->wait();
}

#ifdef GHEX_CUDACC
template<typename GridType, typename DomainIdType>
void
communication_handle<GridType, DomainIdType>::schedule_wait(cudaStream_t stream)
{
    if (m_co) m_co->schedule_wait(stream);
}
#endif

template<typename GridType, typename DomainIdType>
bool
communication_handle<GridType, DomainIdType>::is_ready()
{
    if (!m_co) return true;
    return m_co->is_ready();
}

template<typename GridType, typename DomainIdType>
void
communication_handle<GridType, DomainIdType>::progress()
{
    if (!m_co) return;
    m_co->progress();
}

/** @brief creates a communication object based on the pattern type
          * @tparam PatternContainer pattern type
          * @return communication object */
template<typename PatternContainer>
auto
make_communication_object(context& c)
{
    using grid_type = typename PatternContainer::value_type::grid_type;
    using domain_id_type = typename PatternContainer::value_type::domain_id_type;
    return communication_object<grid_type, domain_id_type>(c);
}

} // namespace ghex
