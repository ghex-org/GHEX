#ifndef INCLUDED_MPI_HPP
#define INCLUDED_MPI_HPP

#include "communicator_base.hpp"
#include <boost/mpi/communicator.hpp>
#include <vector>

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
    using future = future_base<protocol_type,T>;

public:
    communicator(const MPI_Comm& comm)
    :   m_comm(comm, boost::mpi::comm_attach) {}

    communicator(const communicator& other) 
    : communicator(other.m_comm) {} 

    address_type address() const { return m_comm.rank(); }
    address_type rank() const { return m_comm.rank(); }
    void barrier() { m_comm.barrier(); }

    template<typename T>
    void send(int dest, int tag, const T & value)
    {
        m_comm.send(dest, tag, reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template<typename T>
    boost::mpi::status recv(int source, int tag, T & value)
    {
        return m_comm.recv(source, tag, reinterpret_cast<char*>(&value), sizeof(T));
    }

    template<typename T>
    void send(int dest, int tag, const T* values, int n)
    {
        m_comm.send(dest, tag, reinterpret_cast<const char*>(values), sizeof(T)*n);
    }

    template<typename T>
    boost::mpi::status recv(int source, int tag, T* values, int n)
    {
        return m_comm.recv(source, tag, reinterpret_cast<char*>(values), sizeof(T)*n);
    }

    template<typename T> 
    void broadcast(T& value, int root)
    {
        BOOST_MPI_CHECK_RESULT(
            MPI_Bcast,
            (&value, sizeof(T), MPI_BYTE, root, m_comm));
    }

    template<typename T> 
    void broadcast(T * values, int n, int root)
    {
        BOOST_MPI_CHECK_RESULT(
            MPI_Bcast,
            (values, sizeof(T)*n, MPI_BYTE, root, m_comm));
    }

    template<typename T>
    future< std::vector<std::vector<T>> > all_gather(const std::vector<T>& payload, const std::vector<int>& sizes)
    {
        handle_type h;
        std::vector<int> displs(m_comm.size());
        std::vector<int> recvcounts(m_comm.size());
        std::vector<std::vector<T>> res(m_comm.size());
        for (int i=0; i<m_comm.size(); ++i)
        {
            res[i].resize(sizes[i]);
            recvcounts[i] = sizes[i]*sizeof(T);
            displs[i] = reinterpret_cast<char*>(&res[i][0]) - reinterpret_cast<char*>(&res[0][0]);
        }
        BOOST_MPI_CHECK_RESULT(
            MPI_Iallgatherv,
            (&payload[0], payload.size()*sizeof(T), MPI_BYTE,
            &res[0][0], &recvcounts[0], &displs[0], MPI_BYTE,
            m_comm, 
            &h.m_requests[0]));
        return {std::move(res), std::move(h)};
    }

    template<typename T>
    future< std::vector<T> > all_gather(const T& payload)
    {
        std::vector<T> res(m_comm.size());
        handle_type h;
        BOOST_MPI_CHECK_RESULT(
            MPI_Iallgather,
            (&payload, sizeof(T), MPI_BYTE,
            &res[0], sizeof(T), MPI_BYTE,
            m_comm,
            &h.m_requests[0]));
        return {std::move(res), std::move(h)};
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

