#pragma once

#include <ghex/context.hpp>

#include <mpi_comm_shim.hpp>

namespace pyghex
{

struct context_shim
{
    bool          m_thread_safe;
    ghex::context m;
    mpi_comm_shim m_mpi_comm;

    context_shim(mpi_comm_shim mpi_comm_, bool thread_safe);
};

namespace util
{
template<>
std::string to_string(const context_shim& c);
} // namespace util

} // namespace pyghex
