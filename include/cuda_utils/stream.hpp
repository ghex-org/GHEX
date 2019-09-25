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
#ifndef INCLUDED_CUDA_STREAM_HPP
#define INCLUDED_CUDA_STREAM_HPP

#include <memory>

namespace gridtools {
    namespace ghex {

#ifdef __CUDACC__
        struct cuda_stream
        {
            cudaStream_t m_stream;
            bool m_valid = true;

            // default construct
            cuda_stream()
            {
                cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
            }

            cuda_stream(bool create)
            {
                if (create)
                    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
                m_valid = create;
            }

            // non-copyable
            cuda_stream(const cuda_stream&) noexcept = delete;
            cuda_stream& operator=(const cuda_stream&)= delete;

            // movable
            cuda_stream(cuda_stream&& other)
            : m_stream{std::move(other.m_stream)}
            {
                other.m_valid = false;
            }

            cuda_stream& operator=(cuda_stream&& other)
            {
                if (m_valid)
                    cudaStreamDestroy(m_stream);
                m_stream.~cudaStream_t();
                new((void*)(&m_stream)) cudaStream_t{std::move(other.m_stream)};
                m_valid = other.m_valid;
                other.m_valid = false;
                return *this;
            }

            ~cuda_stream()
            {
                if (m_valid)
                    cudaStreamDestroy(m_stream);
            }

            void sync()
            {
                cudaStreamSynchronize(m_stream);
            }

            void activate()
            {
                if (!m_valid)
                    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
                m_valid = true;    
            }

            bool active() const noexcept { return m_valid; }

            // implicit conversion
            operator       cudaStream_t&() noexcept { return m_stream; }
            operator const cudaStream_t&() const noexcept { return m_stream; }
        };
#else
        struct cuda_stream
        {
            // default construct
            cuda_stream() {}
            cuda_stream(bool) {}

            // non-copyable
            cuda_stream(const cuda_stream&) noexcept = delete;
            cuda_stream& operator=(const cuda_stream&)= delete;

            // movable
            cuda_stream(cuda_stream&& other) noexcept =  default;
            cuda_stream& operator=(cuda_stream&&) noexcept = default;
        };
#endif

    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_CUDA_STREAM_HPP */

