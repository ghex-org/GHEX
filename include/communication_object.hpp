/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <mpi.h>
#include <vector>
#include <tuple>
#include "ghex_arch.hpp"
#include "ghex_protocol.hpp"
#include "utils.hpp"


namespace ghex {

    template <typename DomainId, typename Pattern, typename Protocol, typename Arch>
    class communication_object {};

    template <typename DomainId, typename Pattern>
    class communication_object<DomainId, Pattern, ghex::ghex_mpi, ghex::ghex_cpu> {

        using Byte = unsigned char;
        using IterationSpace = typename Pattern::iteration_space;

        const Pattern& m_pattern;
        std::vector<std::pair<DomainId, std::vector<IterationSpace>>>& m_receive_halos;
        std::vector<std::pair<DomainId, std::vector<IterationSpace>>>& m_send_halos;

        template <typename... DataDescriptor>
        std::size_t receive_buffer_size(const std::vector<IterationSpace>& iteration_spaces,
                                        const std::tuple<DataDescriptor...>& data_descriptors) {

            std::size_t size{0};

            for (const auto& is : iteration_spaces) {
                ghex::for_each(data_descriptors, [&is, &size](const auto& dd) {
                    size += is.size() * dd.data_type_size();
                });
            }

            return size;

        }

        template <typename... DataDescriptor>
        void pack(std::vector<std::vector<Byte>>& send_buffers,
                  const std::tuple<DataDescriptor...>& data_descriptors) {

            std::size_t halo_index{0};
            for (const auto& halo : m_send_halos) {

                auto iteration_spaces = halo.second;

                send_buffers[halo_index].resize(0);
                std::size_t buffer_index{0};

                /* The two loops are performed with this order
                 * in order to have as many data of the same type as possible in contiguos memory */
                ghex::for_each(data_descriptors, [&iteration_spaces, &send_buffers, &halo_index, &buffer_index](const auto& dd) {
                    for (const auto& is : iteration_spaces) {
                        dd.load(is, send_buffers[halo_index][buffer_index]);
                        buffer_index += is.size();
                    }
                });

                ++halo_index;

            }

        }

    public:

        template <typename... DataDescriptor>
        class handle {

            const std::vector<std::pair<DomainId, std::vector<IterationSpace>>>& m_receive_halos;
            std::vector<MPI_Request> m_receive_requests;
            std::vector<std::vector<Byte>> m_receive_buffers;
            std::tuple<DataDescriptor...> m_data_descriptors;

            void unpack() {

                std::size_t halo_index{0};
                for (const auto& halo : m_receive_halos) {

                    auto iteration_spaces = halo.second;

                    std::size_t buffer_index{0};

                    /* The two loops are performed with this order
                     * in order to have as many data of the same type as possible in contiguos memory */
                    ghex::for_each(m_data_descriptors, [this, &iteration_spaces, &halo_index, &buffer_index](const auto& dd) {
                        for (const auto& is : iteration_spaces) {
                            dd.store(is, m_receive_buffers[halo_index][buffer_index]);
                            buffer_index += is.size();
                        }
                    });

                    ++halo_index;

                }

            }

            handle(const std::vector<std::pair<DomainId, std::vector<IterationSpace>>>& receive_halos,
                   std::vector<MPI_Request>&& receive_requests,
                   std::vector<std::vector<Byte>>&& receive_buffers,
                   std::tuple<DataDescriptor...>&& data_descriptors) :
                m_receive_halos{receive_halos},
                m_receive_requests{std::move(receive_requests)},
                m_receive_buffers{std::move(receive_buffers)},
                m_data_descriptors{std::move(data_descriptors)} {}

        public:

            void wait() {

                auto n_halos = m_receive_halos.size();
                std::vector<MPI_Status> receive_statuses(n_halos);

                MPI_Waitall(n_halos, &m_receive_requests[0], &receive_statuses[0]);

                unpack();

            }

        };

        communication_object(const Pattern& p) :
            m_pattern{p},
            m_receive_halos{m_pattern.receive_halos()},
            m_send_halos {m_pattern.send_halos()} {}

        template <typename... DataDescriptor>
        handle<DataDescriptor...> exchange(DataDescriptor& ...dds) {

            std::size_t n_halos{m_receive_halos.size()};

            std::vector<std::vector<Byte>> send_buffers(n_halos);
            std::vector<MPI_Request> send_requests(n_halos);
            std::vector<MPI_Status> send_statuses(n_halos);

            std::vector<std::vector<Byte>> receive_buffers(n_halos);
            std::vector<MPI_Request> receive_requests(n_halos);

            auto data_descriptors = std::make_tuple(dds...);

            std::size_t halo_index;

            /* RECEIVE */

            halo_index = 0;
            for (const auto& halo : m_receive_halos) {

                int mpi_rank = halo.first.mpi_rank();
                int tag = halo.first.tag();
                auto& iteration_spaces = halo.second;

                receive_buffers[halo_index].resize(receive_buffer_size(iteration_spaces, data_descriptors));

                MPI_Irecv(&receive_buffers[halo_index][0],
                        static_cast<int>(receive_buffers[halo_index].size()),
                        MPI_CHAR,
                        mpi_rank,
                        tag,
                        MPI_COMM_WORLD, // Temporary
                        &receive_requests[halo_index]);

                ++halo_index;

            }

            /* SEND */

            pack(send_buffers, data_descriptors);

            halo_index = 0;
            for (const auto& halo : m_send_halos) {

                int mpi_rank = halo.first.mpi_rank();
                int tag = halo.first.tag();
                auto& iteration_spaces = halo.second;

                MPI_Isend(&send_buffers[halo_index][0],
                        static_cast<int>(send_buffers[halo_index].size()),
                        MPI_CHAR,
                        mpi_rank,
                        tag,
                        MPI_COMM_WORLD, // Temporary
                        &send_requests[halo_index]);

                ++halo_index;

            }

            MPI_Waitall(n_halos, &send_requests[0], &send_statuses[0]);

            return {m_receive_halos, std::move(receive_requests), std::move(receive_buffers), std::move(data_descriptors)};

        }

    };

}
