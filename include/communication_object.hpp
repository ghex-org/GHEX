/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef INCLUDED_COMMUNICATION_OBJECT_HPP
#define INCLUDED_COMMUNICATION_OBJECT_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include "gridtools_arch.hpp"
#include "utils.hpp"


namespace gridtools {

    template <typename Pattern, typename Arch>
    class communication_object {};

    template <typename Pattern>
    class communication_object<Pattern, gridtools::cpu> {

        using Byte = unsigned char;
        using IterationSpace = typename Pattern::iteration_space2;
        using MapType = typename Pattern::map_type;
        using Communicator = typename Pattern::communicator_type;
        using Future = typename Communicator::template future<void>;

        const Pattern& m_pattern;
        const MapType& m_send_halos;
        const MapType& m_receive_halos;
        const Communicator& m_communicator;

        template <typename... DataDescriptor>
        std::size_t buffer_size(const std::vector<IterationSpace>& iteration_spaces,
                                const std::tuple<DataDescriptor...>& data_descriptors) {

            std::size_t size{0};

            for (const auto& is : iteration_spaces) {
                gridtools::detail::for_each(data_descriptors, [&is, &size](const auto& dd) {
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

                const auto& iteration_spaces = halo.second;

                send_buffers[halo_index].resize(buffer_size(iteration_spaces, data_descriptors));
                std::size_t buffer_index{0};

                /* The two loops are performed with this order
                 * in order to have as many data of the same type as possible in contiguos memory */
                gridtools::detail::for_each(data_descriptors, [&iteration_spaces, &send_buffers, &halo_index, &buffer_index](const auto& dd) {
                    for (const auto& is : iteration_spaces) {
                        dd.get(is, &send_buffers[halo_index][buffer_index]);
                        buffer_index += is.size() * dd.data_type_size();
                    }
                });

                ++halo_index;

            }

        }

    public:

        template <typename... DataDescriptor>
        class handle {

            const MapType& m_receive_halos;
            std::vector<Future> m_receive_requests;
            std::vector<std::vector<Byte>> m_receive_buffers;
            std::tuple<DataDescriptor...> m_data_descriptors;

            void unpack() {

                std::size_t halo_index{0};
                for (const auto& halo : m_receive_halos) {

                    auto iteration_spaces = halo.second;

                    std::size_t buffer_index{0};

                    /* The two loops are performed with this order
                     * in order to have as many data of the same type as possible in contiguos memory */
                    gridtools::detail::for_each(m_data_descriptors, [this, &iteration_spaces, &halo_index, &buffer_index](auto& dd) {
                        for (const auto& is : iteration_spaces) {
                            dd.set(is, &m_receive_buffers[halo_index][buffer_index]);
                            buffer_index += is.size() * dd.data_type_size();
                        }
                    });

                    ++halo_index;

                }

            }

        public:

            handle(const MapType& receive_halos,
                   std::vector<Future>&& receive_requests,
                   std::vector<std::vector<Byte>>&& receive_buffers,
                   std::tuple<DataDescriptor...>&& data_descriptors) :
                m_receive_halos{receive_halos},
                m_receive_requests{std::move(receive_requests)},
                m_receive_buffers{std::move(receive_buffers)},
                m_data_descriptors{std::move(data_descriptors)} {}

            void wait() {

                int i{0};
                for (auto& r : m_receive_requests) {
                    std::cout << "DEBUG: waititng for receive request n. " << i++ << " \n";
                    std::cout.flush();
                    r.wait();
                }

                unpack();

            }

        };

        communication_object(const Pattern& p) :
            m_pattern{p},
            m_send_halos{m_pattern.send_halos()},
            m_receive_halos{m_pattern.recv_halos()},
            m_communicator{m_pattern.communicator()} {}

        template <typename... DataDescriptor>
        handle<DataDescriptor...> exchange(DataDescriptor& ...dds) {

            std::size_t n_send_halos{m_send_halos.size()};
            std::size_t n_receive_halos{m_receive_halos.size()};

            std::vector<std::vector<Byte>> send_buffers(n_send_halos);
            std::vector<std::vector<Byte>> receive_buffers(n_receive_halos);

            std::vector<Future> send_requests;
            send_requests.reserve(n_send_halos); // no default constructor

            std::vector<Future> receive_requests;
            receive_requests.reserve(n_receive_halos); // no default constructor

            auto data_descriptors = std::make_tuple(dds...);

            std::size_t halo_index;

            /* RECEIVE */

            halo_index = 0;
            for (const auto& halo : m_receive_halos) {

                auto source = halo.first.address;
                auto tag = halo.first.tag;
                const auto& iteration_spaces = halo.second;

                receive_buffers[halo_index].resize(buffer_size(iteration_spaces, data_descriptors));

                std::cout << "DEBUG: Starting receive request for halo_index = " << halo_index << " \n";
                std::cout.flush();

                receive_requests.push_back(m_communicator.irecv(source,
                        tag,
                        &receive_buffers[halo_index][0],
                        static_cast<int>(receive_buffers[halo_index].size())));

                ++halo_index;

            }

            /* SEND */

            pack(send_buffers, data_descriptors);

            halo_index = 0;
            for (const auto& halo : m_send_halos) {

                auto dest = halo.first.address;
                auto tag = halo.first.tag;

                std::cout << "DEBUG: Starting send request for halo_index = " << halo_index << " \n";
                std::cout.flush();

                send_requests.push_back(m_communicator.isend(dest,
                        tag,
                        &send_buffers[halo_index][0],
                        static_cast<int>(send_buffers[halo_index].size())));

                ++halo_index;

            }

            /* SEND WAIT */

            int i{0};
            for (auto& r : send_requests) {
                std::cout << "DEBUG: waititng for send request n. " << i++ << " \n";
                std::cout.flush();
                r.wait();
            }

            return {m_receive_halos, std::move(receive_requests), std::move(receive_buffers), std::move(data_descriptors)};

        }

    };

}

#endif /* INCLUDED_COMMUNICATION_OBJECT_HPP */
