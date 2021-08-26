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
#ifndef INCLUDED_GHEX_PACKER_HPP
#define INCLUDED_GHEX_PACKER_HPP

#include "./common/await_futures.hpp"
#include "./arch_list.hpp"
#include "./structured/field_utils.hpp"
#include "./cuda_utils/kernel_argument.hpp"
#include "./cuda_utils/future.hpp"
#include <gridtools/common/array.hpp>

#include "./common/defs.hpp"
#ifdef GHEX_CUDACC
#include "./common/cuda_runtime.hpp"
#endif

namespace gridtools {

    namespace ghex {

        /** @brief generic implementation of pack and unpack */
        template<typename Arch>
        struct packer
        {
            template<typename Map, typename Futures, typename Communicator>
            static void pack(Map& map, Futures& send_futures,Communicator& comm)
            {
                for (auto& p0 : map.send_memory)
                {
                    for (auto& p1: p0.second)
                    {
                        if (p1.second.size > 0u)
                        {
                            p1.second.buffer.resize(p1.second.size);
                            for (const auto& fb : p1.second.field_infos)
                                fb.call_back( p1.second.buffer.data() + fb.offset, *fb.index_container, nullptr);
                            send_futures.push_back(comm.send(p1.second.buffer, p1.second.address, p1.second.tag));
                        }
                    }
                }
            }
            
            template<typename Buffer>
            static void unpack(Buffer& buffer, unsigned char* data)
            {
                for (const auto& fb :  buffer.field_infos)
                    fb.call_back(data + fb.offset, *fb.index_container, nullptr);
            }

            template<typename BufferMem>
            static void unpack(BufferMem& m)
            {
                await_futures(
                    m.m_recv_futures,
                    [](typename BufferMem::hook_type hook)
                    {
                        for (const auto& fb :  hook->field_infos)
                            fb.call_back(hook->buffer.data() + fb.offset, *fb.index_container, nullptr);
                    });
            }
        };

        
#ifdef GHEX_CUDACC
        
        template<typename PackIterationSpace, unsigned int N>
        __global__ void pack_kernel_u(
            cuda::kernel_argument<PackIterationSpace, N> args)
        {
            using layout_t = typename PackIterationSpace::layout_map;
            using value_type = typename PackIterationSpace::value_t;
            using coordinate_type = typename PackIterationSpace::coordinate_t;
            static constexpr auto D = coordinate_type::size();
            const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
            const int data_lu_index = blockIdx.y;

            auto& arg = args[data_lu_index];
            const int size = arg.m_buffer_desc.m_size;
            if (thread_index < size)
            {
                // compute local coordinate
                coordinate_type local_coordinate;
                ::gridtools::ghex::structured::detail::compute_coordinate<D>::template apply<layout_t>(
                    arg.m_data_is.m_local_strides, local_coordinate, thread_index);
                // add offset
                const coordinate_type x = local_coordinate + arg.m_data_is.m_first;
                // assign
                arg.buffer(x) = arg.data(x);
            }
        }

        template<typename UnPackIterationSpace, unsigned int N>
        __global__ void unpack_kernel_u(
            cuda::kernel_argument<UnPackIterationSpace, N> args)
        {
            using layout_t = typename UnPackIterationSpace::layout_map;
            using value_type = typename UnPackIterationSpace::value_t;
            using coordinate_type = typename UnPackIterationSpace::coordinate_t;
            static constexpr auto D = coordinate_type::size();
            const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
            const int data_lu_index = blockIdx.y;

            auto& arg = args[data_lu_index];
            const int size = arg.m_buffer_desc.m_size;
            if (thread_index < size)
            {
                // compute local coordinate
                coordinate_type local_coordinate;
                ::gridtools::ghex::structured::detail::compute_coordinate<D>::template apply<layout_t>(
                    arg.m_data_is.m_local_strides, local_coordinate, thread_index);
                // add offset
                const coordinate_type x = local_coordinate + arg.m_data_is.m_first;
                // assign
                arg.data(x) = arg.buffer(x);
            }
        }

        /** @brief specialization for gpus, including vector interface special functions */
        template<>
        struct packer<gpu>
        {
            template<typename Map, typename Futures, typename Communicator>
            static void pack(Map& map, Futures& send_futures,Communicator& comm)
            {
                using send_buffer_type     = typename Map::send_buffer_type;
                using future_type = cuda::future<send_buffer_type*>;
                std::size_t num_streams = 0;
                for (auto& p0 : map.send_memory)
                {
                    for (auto& p1: p0.second)
                    {
                        if (p1.second.size > 0u)
                        {
                            p1.second.buffer.resize(p1.second.size);
                            ++num_streams;
                        }
                    }
                }
                std::vector<future_type> stream_futures;
                stream_futures.reserve(num_streams);
                num_streams = 0;
                for (auto& p0 : map.send_memory)
                {
                    for (auto& p1: p0.second)
                    {
                        if (p1.second.size > 0u)
                        {
                            for (const auto& fb : p1.second.field_infos)
                            {
                                fb.call_back( p1.second.buffer.data() + fb.offset, *fb.index_container, (void*)(&p1.second.m_cuda_stream.get()));
                            }
                            stream_futures.push_back( future_type{&(p1.second), p1.second.m_cuda_stream} );
                            ++num_streams;
                        }
                    }
                }
                await_futures(
                    stream_futures, 
                    [&comm,&send_futures](send_buffer_type* b)
                    {
                        send_futures.push_back(comm.send(b->buffer, b->address, b->tag));
                    });
            }
            
            template<typename Buffer>
            static void unpack(Buffer& buffer, unsigned char* data)
            {
                auto& stream = buffer.m_cuda_stream;
                for (const auto& fb :  buffer.field_infos)
                    fb.call_back(data + fb.offset, *fb.index_container, (void*)(&stream.get()));
            }

            template<typename BufferMem>
            static void unpack(BufferMem& m)
            {
                await_futures(
                    m.m_recv_futures,
                    [](typename BufferMem::hook_type hook)
                    {
                        for (const auto& fb : hook->field_infos)
                                fb.call_back(hook->buffer.data() + fb.offset, *fb.index_container, (void*)(&hook->m_cuda_stream.get()));

                    });
            }

            template<typename T, typename FieldType, typename Map, typename Futures, typename Communicator>
            static void pack_u(Map& map, Futures& send_futures, Communicator& comm)
            {
                using send_buffer_type     = typename Map::send_buffer_type;
                using field_info_type      = typename send_buffer_type::field_info_type;
                using index_container_type = typename field_info_type::index_container_type;
                using dimension            = typename index_container_type::value_type::dimension;
                using array_t              = array<int, dimension::value>;
                using arg_t                = typename FieldType::pack_iteration_space;
                constexpr int block_size   = 128;

                std::vector<arg_t> args;
                args.reserve(64);

                std::size_t num_streams = 0;
                for (auto& p0 : map.send_memory)
                {
                    for (auto& p1: p0.second)
                    {
                        if (p1.second.size > 0u)
                        {
                            p1.second.buffer.resize(p1.second.size);
                            ++num_streams;
                        }
                    }
                }

                using future_type = cuda::future<send_buffer_type*>;
                std::vector<future_type> stream_futures;
                stream_futures.reserve(num_streams);

                num_streams = 0;
                for (auto& p0 : map.send_memory)
                {
                    for (auto& p1: p0.second)
                    {
                        if (p1.second.size > 0u)
                        {
                            args.clear();
                            int num_blocks_y = 0;
                            int max_size = 0;
                            for (const auto& fb : p1.second.field_infos)
                            {
                                T* buffer_address = reinterpret_cast<T*>(p1.second.buffer.data()+fb.offset);
                                auto& f = *reinterpret_cast<FieldType*>(fb.field_ptr);
                                for (const auto& it_space_pair : *fb.index_container)
                                {
                                    ++num_blocks_y;
                                    const int size = it_space_pair.size() * f.num_components();
                                    max_size = std::max(size,max_size);
                                    args.push_back( f.make_pack_is(it_space_pair, buffer_address, size) );
                                    buffer_address += size;
                                }
                            }
                            const int num_blocks_x = (max_size+block_size-1)/block_size;
                            // unroll kernels: can fit at most 34 arguments as pack kernel argument
                            // invoke new kernels until all data is packed
                            unsigned int count = 0;
                            while (num_blocks_y)
                            {
                                if (num_blocks_y > 34)
                                {
                                    dim3 dimBlock(block_size, 1);
                                    dim3 dimGrid(num_blocks_x, 34);
                                    pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                        cuda::make_kernel_arg<34>(args.data()+count, 34)
                                    );
                                    count += 34;
                                    num_blocks_y -= 34;
                                }
                                else 
                                {
                                    dim3 dimBlock(block_size, 1);
                                    dim3 dimGrid(num_blocks_x, num_blocks_y);
                                    if (num_blocks_y < 7)
                                    {
                                        pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                            cuda::make_kernel_arg< 6>(args.data()+count, num_blocks_y)
                                        );
                                    }
                                    else if (num_blocks_y < 13)
                                    {
                                        pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                            cuda::make_kernel_arg<12>(args.data()+count, num_blocks_y)
                                        );
                                    }
                                    else if (num_blocks_y < 25)
                                    {
                                        pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                            cuda::make_kernel_arg<24>(args.data()+count, num_blocks_y)
                                        );
                                    }
                                    else
                                    {
                                        pack_kernel_u<<<dimGrid, dimBlock, 0, p1.second.m_cuda_stream>>>(
                                            cuda::make_kernel_arg<34>(args.data()+count, num_blocks_y)
                                        );
                                    }
                                    count += num_blocks_y;
                                    num_blocks_y = 0;
                                }
                            }
                            stream_futures.push_back( future_type{&(p1.second), p1.second.m_cuda_stream} );
                            ++num_streams;
                        }
                    }
                }
                await_futures(
                    stream_futures, 
                    [&comm,&send_futures](send_buffer_type* b)
                    {
                        send_futures.push_back(comm.send(b->buffer, b->address, b->tag));
                    });
            }
            
            template<typename T, typename FieldType, typename BufferMem>
            static void unpack_u(BufferMem& m)
            {
                using recv_buffer_type     = typename BufferMem::recv_buffer_type;
                using field_info_type      = typename recv_buffer_type::field_info_type;
                using index_container_type = typename field_info_type::index_container_type;
                using dimension            = typename index_container_type::value_type::dimension;
                using array_t              = ::gridtools::array<int, dimension::value>;
                using arg_t                = typename FieldType::unpack_iteration_space;
                constexpr int block_size   = 128;

                std::vector<arg_t> args;
                args.reserve(64);

                std::vector<cudaStream_t*> stream_ptrs;
                stream_ptrs.reserve(m.m_recv_futures.size());
                await_futures(
                    m.m_recv_futures,
                    [&block_size,&stream_ptrs,&args](typename BufferMem::hook_type hook)
                    {
                        auto stream_ptr = &hook->m_cuda_stream.get();
                        args.resize(0);
                        int num_blocks_y = 0;
                        int max_size = 0;
                        for (const auto& fb : hook->field_infos)
                        {
                            T* buffer_address = reinterpret_cast<T*>(hook->buffer.data()+fb.offset);
                            auto& f = *reinterpret_cast<FieldType*>(fb.field_ptr);
                            for (const auto& it_space_pair : *fb.index_container)
                            {
                                ++num_blocks_y;
                                const int size = it_space_pair.size() * f.num_components();
                                max_size = std::max(size,max_size);
                                args.push_back(f.make_unpack_is(it_space_pair, buffer_address, size));
                                buffer_address += size;
                            }
                        }
                        const int num_blocks_x = (max_size+block_size-1)/block_size;
                        // unroll kernels: can fit at most 34 arguments as unpack kernel argument
                        // invoke new kernels until all data is unpacked
                        unsigned int count = 0;
                        while (num_blocks_y)
                        {
                            if (num_blocks_y > 34)
                            {
                                dim3 dimBlock(block_size, 1);
                                dim3 dimGrid(num_blocks_x, 34);
                                unpack_kernel_u<<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                    cuda::make_kernel_arg<34>(args.data()+count, 34)
                                );
                                count += 34;
                                num_blocks_y -= 34;
                            }
                            else
                            {
                                dim3 dimBlock(block_size, 1);
                                dim3 dimGrid(num_blocks_x, num_blocks_y);
                                if (num_blocks_y < 7)
                                {
                                    unpack_kernel_u<<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                        cuda::make_kernel_arg<6>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 13)
                                {
                                    unpack_kernel_u<<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                        cuda::make_kernel_arg<12>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else if (num_blocks_y < 25)
                                {
                                    unpack_kernel_u<<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                        cuda::make_kernel_arg<24>(args.data()+count, num_blocks_y)
                                    );
                                }
                                else
                                {
                                    unpack_kernel_u<<<dimGrid, dimBlock, 0, *stream_ptr>>>(
                                        cuda::make_kernel_arg<34>(args.data()+count, num_blocks_y)
                                    );
                                }
                                count += num_blocks_y;
                                num_blocks_y = 0;
                            }
                        }
                        stream_ptrs.push_back(stream_ptr);
                    });
                for (auto x : stream_ptrs)
                {
                    cudaStreamSynchronize(*x);
                }
            }
        };
#endif

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_PACKER_HPP */

