// 
// GridTools
// 
// Copyright (c) 2014-2019, ETH Zurich
// All rights reserved.
// 
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
// 
#ifndef INCLUDED_MPI_HPP
#define INCLUDED_MPI_HPP

#include "communicator_base.hpp"
#include <boost/mpi/communicator.hpp>

namespace gridtools {

namespace protocol {

struct mpi {};

template<>
class communicator<mpi>
{
public:

    using protocol_type = mpi;
    using handle_type = boost::mpi::request;
    using address_type = int;
    template<typename T>
    using future = future_base<handle_type,T>;

public:
    communicator(const MPI_Comm& comm)
    :   m_comm(comm, boost::mpi::comm_attach) {}

    communicator(const communicator& other) 
    : communicator(other.m_comm) {} 

    address_type address() const { return m_comm.rank(); }
    
    address_type rank() const { return m_comm.rank(); }

    template<typename T>
    future<void> isend(address_type dest, int tag, const T* buffer, int n) const
    {
        return {std::move(m_comm.isend(dest, tag, reinterpret_cast<const char*>(buffer), sizeof(T)*n))};
    }

    template<typename T>
    future<void> irecv(address_type source, int tag, T* buffer, int n) const
    {
        return {std::move(m_comm.irecv(source, tag, reinterpret_cast<char*>(buffer), sizeof(T)*n))};
    }

    template<typename T, template<typename, typename> typename Vector, typename Allocator> 
    future<void> isend(address_type dest, int tag, const Vector<T,Allocator>& vec) const
    {
        return {std::move(m_comm.isend(dest, tag, reinterpret_cast<const char*>(vec.data()), sizeof(T)*vec.size()))};
    }

    template<typename T, template<typename, typename> typename Vector = std::vector, typename Allocator = std::allocator<T>> 
    [[nodiscard]] future<Vector<T,Allocator>> irecv(address_type source, int tag, int n, const Allocator& a = Allocator()) const
    {
        using vector_type = Vector<T,Allocator>;
        using size_type   = typename vector_type::size_type;
        vector_type vec( size_type(n), a );
        return { std::move(vec), std::move(m_comm.irecv(source, tag, reinterpret_cast<char*>(vec.data()), sizeof(T)*n)) };
    }

private:
    boost::mpi::communicator m_comm;
};

} // namespace protocol

} // namespace gridtools

#endif /* INCLUDED_MPI_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

