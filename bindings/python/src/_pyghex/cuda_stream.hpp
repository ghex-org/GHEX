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

#if defined(GHEX_CUDACC)

#include <cstdint>
#include <sstream>
#include <type_traits>

#include <ghex/device/cuda/runtime.hpp>

#include <nanobind/nanobind.h>

namespace pyghex
{

// Convert a Python stream object into a CUDA/HIP stream handle. The following
// Python objects are understood:
// - `None` is interpreted as the default stream, i.e. `nullptr`.
// - an object implementing Nvidia's `__cuda_stream__` protocol, see
//   https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
// - a CuPy stream, which exposes the stream handle via its `.ptr` attribute, see
//   https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.Stream.html#cupy-cuda-stream
inline cudaStream_t
extract_cuda_stream(nanobind::object python_stream)
{
    static_assert(std::is_pointer<cudaStream_t>::value);
    if (python_stream.is_none())
    {
        // NOTE: This is very C++ like, maybe remove and consider as an error?
        return static_cast<cudaStream_t>(nullptr);
    }
    else
    {
        if (nanobind::hasattr(python_stream, "__cuda_stream__"))
        {
            nanobind::tuple cuda_stream_protocol =
                nanobind::cast<nanobind::tuple>(python_stream.attr("__cuda_stream__")());
            if (cuda_stream_protocol.size() != 2)
            {
                std::stringstream error;
                error << "Expected a tuple of length 2, but got one with length "
                      << cuda_stream_protocol.size();
                throw nanobind::type_error(error.str().c_str());
            }

            const auto protocol_version = nanobind::cast<std::size_t>(cuda_stream_protocol[0]);
            if (protocol_version != 0)
            {
                std::stringstream error;
                error << "Expected `__cuda_stream__` protocol version 0, but got "
                      << protocol_version;
                throw nanobind::type_error(error.str().c_str());
            }

            const auto stream_address = nanobind::cast<std::uintptr_t>(cuda_stream_protocol[1]);
            return reinterpret_cast<cudaStream_t>(stream_address);
        }
        else if (nanobind::hasattr(python_stream, "ptr"))
        {
            std::uintptr_t stream_address =
                nanobind::cast<std::uintptr_t>(python_stream.attr("ptr"));
            return reinterpret_cast<cudaStream_t>(stream_address);
        }
        // TODO: Find out of how to extract the typename, i.e. `type(python_stream).__name__`.
        std::stringstream error;
        error << "Failed to convert the stream object into a CUDA stream.";
        throw nanobind::type_error(error.str().c_str());
    }
}

} // namespace pyghex

#endif
