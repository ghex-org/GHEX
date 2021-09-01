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

#include <ghex/context.hpp>
#include <ghex/mpi/error.hpp>
#include <ghex/mpi/status.hpp>
#include <ghex/mpi/future.hpp>
#include <vector>
#include <cassert>
#include <algorithm>

namespace ghex
{
namespace mpi
{
/** @brief special mpi communicator used for setup phase */
class communicator
{
  public:
    using rank_type = int;
    using size_type = int;

  private:
    MPI_Comm  m_comm;
    rank_type m_rank;
    size_type m_size;

  public:
    communicator(context const& c)
    : m_comm{c.transport_context()->mpi_comm()}
    {
        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(m_comm, &m_rank));
        GHEX_CHECK_MPI_RESULT(MPI_Comm_size(m_comm, &m_size));
    }
    communicator(const communicator&) = default;
    communicator& operator=(const communicator&) = default;
    communicator(communicator&&) noexcept = default;
    communicator& operator=(communicator&&) noexcept = default;

    operator MPI_Comm() const noexcept { return m_comm; }

    inline rank_type rank() const noexcept { return m_rank; }

    inline size_type size() const noexcept { return m_size; }

    template<typename T>
    void send(int dest, int tag, const T& value) const
    {
        GHEX_CHECK_MPI_RESULT(
            MPI_Send(reinterpret_cast<const void*>(&value), sizeof(T), MPI_BYTE, dest, tag, *this));
    }

    template<typename T>
    status recv(int source, int tag, T& value) const
    {
        MPI_Status s;
        GHEX_CHECK_MPI_RESULT(
            MPI_Recv(reinterpret_cast<void*>(&value), sizeof(T), MPI_BYTE, source, tag, *this, &s));
        return {s};
    }

    template<typename T>
    void send(int dest, int tag, const T* values, int n) const
    {
        GHEX_CHECK_MPI_RESULT(MPI_Send(
            reinterpret_cast<const void*>(values), sizeof(T) * n, MPI_BYTE, dest, tag, *this));
    }

    template<typename T>
    status recv(int source, int tag, T* values, int n) const
    {
        MPI_Status s;
        GHEX_CHECK_MPI_RESULT(MPI_Recv(
            reinterpret_cast<void*>(values), sizeof(T) * n, MPI_BYTE, source, tag, *this, &s));
        return {s};
    }

    template<typename T>
    void broadcast(T& value, int root) const
    {
        GHEX_CHECK_MPI_RESULT(MPI_Bcast(&value, sizeof(T), MPI_BYTE, root, *this));
    }

    template<typename T>
    void broadcast(T* values, int n, int root) const
    {
        GHEX_CHECK_MPI_RESULT(MPI_Bcast(values, sizeof(T) * n, MPI_BYTE, root, *this));
    }

    template<typename T>
    future<std::vector<std::vector<T>>> all_gather(
        const std::vector<T>& payload, const std::vector<int>& sizes) const
    {
        std::vector<std::vector<T>> res(size());
        for (int neigh = 0; neigh < size(); ++neigh) { res[neigh].resize(sizes[neigh]); }
        std::vector<request> m_reqs;
        m_reqs.reserve((size()) * 2);
        for (int neigh = 0; neigh < size(); ++neigh)
        {
            m_reqs.push_back(request{});
            GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(res[neigh].data()),
                sizeof(T) * sizes[neigh], MPI_BYTE, neigh, 99, *this, &m_reqs.back().get()));
        }
        for (int neigh = 0; neigh < size(); ++neigh)
        {
            m_reqs.push_back(request{});
            GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(payload.data()),
                sizeof(T) * payload.size(), MPI_BYTE, neigh, 99, *this, &m_reqs.back().get()));
        }
        for (auto& r : m_reqs) r.wait();

        return {std::move(res), std::move(m_reqs.front())};

        /*request h;
                    std::vector<int> displs(size());
                    std::vector<int> recvcounts(size());
                    std::vector<std::vector<T>> res(size());
                    for (int i=0; i<size(); ++i)
                    {
                        res[i].resize(sizes[i]);
                        recvcounts[i] = sizes[i]*sizeof(T);
                        displs[i] = reinterpret_cast<char*>(&res[i][0]) - reinterpret_cast<char*>(&res[0][0]);
                    }
                    GHEX_CHECK_MPI_RESULT( MPI_Iallgatherv(
                        &payload[0], payload.size()*sizeof(T), MPI_BYTE,
                        &res[0][0], &recvcounts[0], &displs[0], MPI_BYTE,
                        *this, 
                        &h.m_req));
                    return {std::move(res), std::move(h)};*/
    }

    template<typename T>
    future<std::vector<T>> all_gather(const T& payload) const
    {
        std::vector<T> res(size());
        request        h;
        GHEX_CHECK_MPI_RESULT(MPI_Iallgather(
            &payload, sizeof(T), MPI_BYTE, &res[0], sizeof(T), MPI_BYTE, *this, &h.get()));
        return {std::move(res), std::move(h)};
    }

    /** @brief computes the max element of a vector<T> among all ranks */
    template<typename T>
    T max_element(const std::vector<T>& elems) const
    {
        T    local_max{*(std::max_element(elems.begin(), elems.end()))};
        auto all_max = all_gather(local_max).get();
        return *(std::max_element(all_max.begin(), all_max.end()));
    }

    /** @brief just a helper function using custom types to be used when send/recv counts can be deduced*/
    template<typename T>
    void all_to_all(const std::vector<T>& send_buf, std::vector<T>& recv_buf) const
    {
        int comm_size = this->size();
        assert(send_buf.size() % comm_size == 0);
        assert(recv_buf.size() % comm_size == 0);
        int send_count = send_buf.size() / comm_size * sizeof(T);
        int recv_count = recv_buf.size() / comm_size * sizeof(T);
        GHEX_CHECK_MPI_RESULT(
            MPI_Alltoall(reinterpret_cast<const void*>(send_buf.data()), send_count, MPI_BYTE,
                reinterpret_cast<void*>(recv_buf.data()), recv_count, MPI_BYTE, *this));
    }

    /** @brief just a wrapper using custom types*/
    template<typename T>
    void all_to_allv(const std::vector<T>& send_buf, const std::vector<int>& send_counts,
        const std::vector<int>& send_displs, std::vector<T>& recv_buf,
        const std::vector<int>& recv_counts, const std::vector<int>& recv_displs) const
    {
        int              comm_size = this->size();
        std::vector<int> send_counts_b(comm_size), send_displs_b(comm_size),
            recv_counts_b(comm_size), recv_displs_b(comm_size);
        for (auto i = 0; i < comm_size; ++i) send_counts_b[i] = send_counts[i] * sizeof(T);
        for (auto i = 0; i < comm_size; ++i) send_displs_b[i] = send_displs[i] * sizeof(T);
        for (auto i = 0; i < comm_size; ++i) recv_counts_b[i] = recv_counts[i] * sizeof(T);
        for (auto i = 0; i < comm_size; ++i) recv_displs_b[i] = recv_displs[i] * sizeof(T);
        GHEX_CHECK_MPI_RESULT(
            MPI_Alltoallv(reinterpret_cast<const void*>(send_buf.data()), &send_counts_b[0],
                &send_displs_b[0], MPI_BYTE, reinterpret_cast<void*>(recv_buf.data()),
                &recv_counts_b[0], &recv_displs_b[0], MPI_BYTE, *this));
    }

    void barrier() { MPI_Barrier(m_comm); }
};

} // namespace mpi
} // namespace ghex
