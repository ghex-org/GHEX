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
#ifndef INCLUDED_GHEX_TL_UCX_ENDPOINT_DB_PMI_HPP
#define INCLUDED_GHEX_TL_UCX_ENDPOINT_DB_PMI_HPP

#ifdef USE_PMIX

#define GHEX_USE_PMI
#include "../util/pmi/pmix/pmi.hpp"
using PmiType = gridtools::ghex::tl::pmi::pmi<gridtools::ghex::tl::pmi::pmix_tag>;

#include <map>
#include <vector>
#include <mutex>
#include <iostream>
#include "./error.hpp"
#include "./endpoint.hpp"
#include "./address.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace ucx {

                static std::mutex db_lock;

                struct address_db_pmi
                {
                    // PMI interface to obtain peer addresses
                    // per-communicator instance used to store/query connections
                    PmiType pmi_impl;

                    // global instance used to init/finalize the library
                    // it has to be here because of PMIx finalization bug
                    // https://github.com/openpmix/openpmix/issues/1558
                    static PmiType pmi_impl_static;

                    using key_t     = endpoint_t::rank_type;
                    using value_t   = address_t;

                    MPI_Comm m_mpi_comm;

                    // these should be PMIx ranks. might need remaping to MPI ranks
                    key_t m_rank;
                    key_t m_size;

                    address_db_pmi(MPI_Comm comm)
                        : m_mpi_comm{comm}
                    {
                        m_rank = pmi_impl_static.rank();
                        m_size = pmi_impl_static.size();

                        int mpi_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(comm) };
                        int mpi_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(comm) };

                        // TODO: m_mpi_comm should be use to obtain MPI rank to PMIx rank map
                        if(m_rank != mpi_rank || m_size != mpi_size)
                            throw std::runtime_error("PMIx and MPI ranks are different. Mapping not implemented yet.");
                    }

                    address_db_pmi(const address_db_pmi&) = delete;
                    address_db_pmi(address_db_pmi&&) = default;

                    key_t rank() const noexcept { return m_rank; }
                    key_t size() const noexcept { return m_size; }
                    int est_size() const noexcept { return m_size; }

                    value_t find(key_t k)
                    {
                        try {
                            return pmi_impl.get(k, "ghex-rank-address");
                        } catch(std::string err) {
                            throw std::runtime_error("PMIx could not find peer address: " + err);
                        }
                    }

                    void init(const value_t& addr)
                    {
                        std::vector<unsigned char> data(addr.data(), addr.data()+addr.size());
                        pmi_impl_static.set("ghex-rank-address", data);

                        // TODO: we have to call an explicit PMIx Fence due to
                        // https://github.com/open-mpi/ompi/issues/6982
                        pmi_impl_static.exchange();
                    }
                };

                PmiType address_db_pmi::pmi_impl_static;

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* USE_PMIX */

#endif /* INCLUDED_GHEX_TL_UCX_ENDPOINT_DB_MPI_HPP */
