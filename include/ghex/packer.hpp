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
#include <ghex/arch_traits.hpp>
#include <ghex/device/guard.hpp>
#include <ghex/structured/field_utils.hpp>
#include <ghex/device/cuda/kernel_argument.hpp>
#include <ghex/device/cuda/future.hpp>
#include <gridtools/common/array.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

#ifdef GHEX_USE_NCCL
#include <nccl.h>

#define GHEX_CHECK_NCCL_RESULT(x) \
    if (x != ncclSuccess && x != ncclInProgress) \
        throw std::runtime_error(std::string("nccl call failed (") + std::to_string(x) + "):" + ncclGetErrorString(x));
#define GHEX_CHECK_NCCL_RESULT_NO_THROW(x) \
    if (x != ncclSuccess && x != ncclInProgress) { \
        std::cerr << "nccl call failed (" << std::to_string(x) << "): " << ncclGetErrorString(x) << '\n'; \
        std::terminate(); \
    }
#endif

#include <numeric>

namespace ghex
{
/** @brief generic implementation of pack and unpack */
template<typename Arch>
struct packer
{
    template<typename Map, typename Requests, typename Communicator>
    static void pack(Map& map, Requests& send_reqs, Communicator& comm)
    {
        for (auto& p0 : map.send_memory)
        {
            const auto device_id = p0.first;
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u)
                {
                    if (!p1.second.buffer || p1.second.buffer.size() != p1.second.size)
                        p1.second.buffer =
                            arch_traits<Arch>::make_message(comm, p1.second.size, device_id);
                    device::guard g(p1.second.buffer);
                    auto          data = g.data();
                    for (const auto& fb : p1.second.field_infos)
                        fb.call_back(data + fb.offset, *fb.index_container, nullptr);
                    send_reqs.push_back(comm.send(p1.second.buffer, p1.second.rank, p1.second.tag));
                }
            }
        }
    }

    template<typename Map, typename Requests, typename Communicator>
    static void pack2(Map& map, Requests& send_reqs, Communicator& comm)
    {
        pack(map, send_reqs, comm);
    }

    template<typename Map, typename Requests, typename Communicator>
    static void pack2_nccl(Map&, Requests&, Communicator&) {}

    template<typename Buffer>
    static void unpack(Buffer& buffer, unsigned char* data)
    {
        for (const auto& fb : buffer.field_infos)
            fb.call_back(data + fb.offset, *fb.index_container, nullptr);
    }
};

#ifdef GHEX_CUDACC

/** @brief wait for all futures in a range to finish and call 
  * a continuation with the future's value as argument. */
template<typename Future, typename Continuation>
inline void
await_futures(std::vector<Future>& range, Continuation&& cont)
{
    static thread_local std::vector<int> index_list;
    index_list.resize(range.size());
    std::iota(index_list.begin(), index_list.end(), 0);
    const auto begin = index_list.begin();
    auto       end = index_list.end();
    while (begin != end)
    {
        end =
            std::remove_if(begin, end, [&range, cont = std::forward<Continuation>(cont)](int idx) {
                if (range[idx].test())
                {
                    cont(range[idx].get());
                    return true;
                }
                else
                    return false;
            });
    }
}

template<typename PackIterationSpace, unsigned int N>
__global__ void
pack_kernel_u(device::kernel_argument<PackIterationSpace, N> args)
{
    using layout_t = typename PackIterationSpace::layout_map;
    using coordinate_type = typename PackIterationSpace::coordinate_t;
    static constexpr auto D = coordinate_type::size();
    const int             thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int             data_lu_index = blockIdx.y;

    auto&     arg = args[data_lu_index];
    const int size = arg.m_buffer_desc.m_size;
    if (thread_index < size)
    {
        // compute local coordinate
        coordinate_type local_coordinate;
        ghex::structured::detail::compute_coordinate<D>::template apply<layout_t>(
            arg.m_data_is.m_local_strides, local_coordinate, thread_index);
        // add offset
        const coordinate_type x = local_coordinate + arg.m_data_is.m_first;
        // assign
        arg.buffer(x) = arg.data(x);
    }
}

/** @brief specialization for gpus, including vector interface special functions */
template<>
struct packer<gpu>
{
    template<typename Map, typename Requests, typename Communicator>
    static void pack(Map& map, Requests& send_reqs, Communicator& comm)
    {
        using send_buffer_type = typename Map::send_buffer_type;
        using future_type = device::future<send_buffer_type*>;
        std::size_t num_streams = 0;

        for (auto& p0 : map.send_memory)
        {
            const auto device_id = p0.first;
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u)
                {
                    if (!p1.second.buffer || p1.second.buffer.size() != p1.second.size ||
                        p1.second.buffer.device_id() != device_id)
                        p1.second.buffer =
                            arch_traits<gpu>::make_message(comm, p1.second.size, device_id);
                    ++num_streams;
                }
            }
        }
        std::vector<future_type> stream_futures;
        stream_futures.reserve(num_streams);
        num_streams = 0;

        for (auto& p0 : map.send_memory)
        {
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u)
                {
                    for (const auto& fb : p1.second.field_infos)
                    {
                        device::guard g(p1.second.buffer);
                        fb.call_back(g.data() + fb.offset, *fb.index_container,
                            (void*)(&p1.second.m_stream.get()));
                    }
                    stream_futures.push_back(future_type{&(p1.second), p1.second.m_stream});
                    ++num_streams;
                }
            }
        }
        await_futures(stream_futures, [&comm, &send_reqs](send_buffer_type* b) {
            send_reqs.push_back(comm.send(b->buffer, b->rank, b->tag));
        });
    }

    template<typename Map, typename Requests, typename Communicator>
    static void pack2_nccl(Map& map, Requests&, Communicator& comm)
    {
#if 0
        constexpr std::size_t num_extra_streams{32};
        static std::vector<device::stream> streams(num_extra_streams);
        static std::size_t stream_index{0};
#endif

        constexpr std::size_t num_events{128};
        static std::vector<device::cuda_event> events(num_events);
        static std::size_t event_index{0};

        // Assume that send memory synchronizes with the default
        // stream so schedule pack kernels after an event on the
        // default stream.
        cudaEvent_t& e = events[event_index].get();
        event_index = (event_index + 1) % num_events;
        GHEX_CHECK_CUDA_RESULT(cudaEventRecord(e, 0));
        for (auto& p0 : map.send_memory)
        {
            const auto device_id = p0.first;
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u)
                {
                    if (!p1.second.buffer || p1.second.buffer.size() != p1.second.size ||
                        p1.second.buffer.device_id() != device_id)
                    {
                        // std::cerr << "pack2_nccl: making message\n";
                        p1.second.buffer =
                            arch_traits<gpu>::make_message(comm, p1.second.size, device_id);
                    }

                    device::guard g(p1.second.buffer);
#if 0
                    int count = 0;
#endif
		    // Make sure stream used for packing synchronizes with the
		    // default stream.
                    GHEX_CHECK_CUDA_RESULT(cudaStreamWaitEvent(p1.second.m_stream.get(), e));
                    for (const auto& fb : p1.second.field_infos)
                    {
                        // TODO:
                        // 1. launch pack kernels on separate streams for all data
                        // 1. (alternative) pack them all into the same kernel
                        // 2. trigger the send from a cuda host function
                        // 3. don't wait for futures here, but mixed with polling mpi for receives
#if 0
                        if (count == 0) {
#endif
                                // std::cerr << "pack2_nccl: calling pack call_back\n";
				fb.call_back(g.data() + fb.offset, *fb.index_container, (void*)(&p1.second.m_stream.get()));
#if 0
                        } else {
                                cudaStream_t& s = streams[stream_index].get();
                                stream_index = (stream_index + 1) % num_extra_streams;

				cudaEvent_t& e = events[event_index].get();
                                event_index = (event_index + 1) % num_events;

				fb.call_back(g.data() + fb.offset, *fb.index_container, (void*)(&s));

				// Use the main stream only to synchronize. Launch
				// the work on a separate stream and insert an event
				// to allow waiting for all work on the main stream.
				GHEX_CHECK_CUDA_RESULT(cudaEventRecord(e, s));
				GHEX_CHECK_CUDA_RESULT(cudaStreamWaitEvent(p1.second.m_stream.get(), e));
                        }
                        ++count;
#endif
                    }

                    // Warning: tag is not used. Messages have to be correctly ordered.
                    // This is just for debugging, don't do mpi and nccl send
                    // std::cerr << "pack2_nccl: triggering mpi_isend\n";
                    // comm.send(p1.second.buffer, p1.second.rank, p1.second.tag);
                    // std::cerr << "pack2_nccl: triggering ncclSend\n";
                    // std::cerr << "pack2_nccl: ptr is " << static_cast<void*>(p1.second.buffer.device_data()) << "\n";
                    // std::cerr << "pack2_nccl: g.data() is " << static_cast<void*>(g.data()) << "\n";
                    // std::cerr << "pack2_nccl: size is " << p1.second.buffer.size() << "\n";
                    // std::cerr << "pack2_nccl: ptr on device " << p1.second.buffer.on_device() << "\n";
                    // GHEX_CHECK_NCCL_RESULT(ncclSend(static_cast<const void*>(g.data()), p1.second.buffer.size() /* * sizeof(typename decltype(p1.second.buffer)::value_type) */, ncclChar, p1.second.rank, nccl_comm, p1.second.m_stream.get()));
                    // std::cerr << "pack2_nccl: triggered ncclSend\n";
                }
            }
        }
    }

    template<typename Buffer>
    static void unpack(Buffer& buffer, unsigned char* data)
    {
        auto& stream = buffer.m_stream;
        for (const auto& fb : buffer.field_infos)
            fb.call_back(data + fb.offset, *fb.index_container, (void*)(&stream.get()));
    }

    template<typename T, typename FieldType, typename Map, typename Requests, typename Communicator>
    static void pack_u(Map& map, Requests& send_reqs, Communicator& comm)
    {
        using send_buffer_type = typename Map::send_buffer_type;
        using arg_t = typename FieldType::pack_iteration_space;
        constexpr int block_size = 128;

        std::vector<arg_t> args;
        args.reserve(64);

        std::size_t num_streams = 0;
        for (auto& p0 : map.send_memory)
        {
            const auto device_id = p0.first;
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u)
                {
                    if (!p1.second.buffer || p1.second.buffer.size() != p1.second.size ||
                        p1.second.buffer.device_id() != device_id)
                        p1.second.buffer =
                            arch_traits<gpu>::make_message(comm, p1.second.size, device_id);
                    ++num_streams;
                }
            }
        }

        using future_type = device::future<send_buffer_type*>;
        std::vector<future_type> stream_futures;
        stream_futures.reserve(num_streams);

        num_streams = 0;
        for (auto& p0 : map.send_memory)
        {
            for (auto& p1 : p0.second)
            {
                if (p1.second.size > 0u)
                {
                    args.clear();
                    int num_blocks_y = 0;
                    int max_size = 0;
                    for (const auto& fb : p1.second.field_infos)
                    {
                        device::guard g(p1.second.buffer);
                        T*            buffer_address = reinterpret_cast<T*>(g.data() + fb.offset);
                        auto&         f = *reinterpret_cast<FieldType*>(fb.field_ptr);
                        for (const auto& it_space_pair : *fb.index_container)
                        {
                            ++num_blocks_y;
                            const int size = it_space_pair.size() * f.num_components();
                            max_size = std::max(size, max_size);
                            args.push_back(f.make_pack_is(it_space_pair, buffer_address, size));
                            buffer_address += size;
                        }
                    }
                    const int num_blocks_x = (max_size + block_size - 1) / block_size;
                    // unroll kernels: can fit at most 34 arguments as pack kernel argument
                    // invoke new kernels until all data is packed
                    unsigned int count = 0;
                    while (num_blocks_y)
                    {
                        if (num_blocks_y > 34)
                        {
                            dim3 dimBlock(block_size, 1);
                            dim3 dimGrid(num_blocks_x, 34);
                            pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_stream>>>(
                                device::make_kernel_arg<34>(args.data() + count, 34));
                            count += 34;
                            num_blocks_y -= 34;
                        }
                        else
                        {
                            dim3 dimBlock(block_size, 1);
                            dim3 dimGrid(num_blocks_x, num_blocks_y);
                            if (num_blocks_y < 7)
                            {
                                pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_stream>>>(
                                    device::make_kernel_arg<6>(args.data() + count, num_blocks_y));
                            }
                            else if (num_blocks_y < 13)
                            {
                                pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_stream>>>(
                                    device::make_kernel_arg<12>(args.data() + count, num_blocks_y));
                            }
                            else if (num_blocks_y < 25)
                            {
                                pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_stream>>>(
                                    device::make_kernel_arg<24>(args.data() + count, num_blocks_y));
                            }
                            else
                            {
                                pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_stream>>>(
                                    device::make_kernel_arg<34>(args.data() + count, num_blocks_y));
                            }
                            count += num_blocks_y;
                            num_blocks_y = 0;
                        }
                    }
                    stream_futures.push_back(future_type{&(p1.second), p1.second.m_stream});
                    ++num_streams;
                }
            }
        }
        await_futures(stream_futures, [&comm, &send_reqs](send_buffer_type* b) {
            send_reqs.push_back(comm.send(b->buffer, b->rank, b->tag));
        });
    }
};
#endif

} // namespace ghex
