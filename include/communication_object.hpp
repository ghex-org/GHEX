/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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

        friend class handle;

        Pattern& m_pattern;
        std::vector<std::pair<DomainId, std::vector<IterationSpace>>>& m_receive_halos;
        std::vector<std::pair<DomainId, std::vector<IterationSpace>>>& m_send_halos;
        std::vector<std::vector<Byte>> m_send_buffers;
        std::vector<std::vector<Byte>> m_receive_buffers;

        template <typename... DataDescriptor>
        std::size_t receive_buffer_size(const std::tuple<DataDescriptor...>& data_descriptors) {

            std::size_t size{0};

            for (const auto& halo : m_receive_halos) {

                auto iteration_spaces = halo.second;

                for (const auto& is : iteration_spaces) {
                    ghex::for_each(data_descriptors, [&is, &size](const auto& dd) {
                        size += is.size() * dd.data_type_size();
                    });
                }

            }

            return size;

        }

        template <typename... DataDescriptor>
        void pack(const std::tuple<DataDescriptor...>& data_descriptors) {

            std::size_t halo_index{0};
            for (const auto& halo : m_send_halos) {

                auto iteration_spaces = halo.second;

                m_send_buffers[halo_index].resize(0);
                std::size_t buffer_index{0};

                /* The two loops are performed with this order
                 * in order to have as many data of the same type as possible in contiguos memory */
                ghex::for_each(data_descriptors, [this, &iteration_spaces, &halo_index, &buffer_index](const auto& dd) {
                    for (const auto& is : iteration_spaces) {
                        dd.load(is, m_send_buffers[halo_index][buffer_index]);
                        buffer_index += is.size();
                    }
                });

                ++halo_index;

            }

        }

    public:

        template <typename... DataDescriptor>
        class handle {

            std::vector<std::pair<DomainId, std::vector<IterationSpace>>>& m_receive_halos;
            std::vector<std::vector<Byte>>& m_receive_buffers;
            std::tuple<DataDescriptor...>& m_data_descriptors;

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

            handle(std::vector<std::pair<DomainId, std::vector<IterationSpace>>>& receive_halos,
                   std::vector<std::vector<Byte>>& receive_buffers,
                   std::tuple<DataDescriptor...>& data_descriptors) :
                m_receive_halos{receive_halos},
                m_receive_buffers{receive_buffers},
                m_data_descriptors {data_descriptors} {}

        public:

            void wait() {

                // MPI_Waitall
                unpack();

            }

        };

        communication_object(const Pattern& p) :
            m_pattern{p},
            m_receive_halos{m_pattern.get_receive_halos()},
            m_send_halos {m_pattern.get_send_halos()} {}

        template <typename... DataDescriptor>
        handle<DataDescriptor...> exchange(DataDescriptor& ...dds) {

            auto data_descriptors = std::make_tuple(dds...);

            /* RECEIVE */
            m_receive_buffers.resize(receive_buffer_size(data_descriptors));
            // ...

            /* SEND */
            pack(data_descriptors);
            // ...

            return {m_receive_halos, m_receive_buffers, data_descriptors};

        }

    };

}
