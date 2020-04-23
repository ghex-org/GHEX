/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_GHEX_COMMUNICATION_OBJECT_HPP
#define INCLUDED_GHEX_COMMUNICATION_OBJECT_HPP

#include <vector>
#include <map>
#include <tuple>
#include "./arch_list.hpp"
#include "./common/utils.hpp"
#include "./transport_layer/message_buffer.hpp"

namespace gridtools {

    namespace ghex {

        /** @brief generic communication object for single pattern exchange
         * @tparam Pattern pattern type to be used for the communication
         * @tparam Arch architecture: cpu, gpu or others*/
        template <typename Pattern, typename Arch>
        class communication_object;

        /** @brief cpu template specialization of the communication object
         * @tparam Pattern pattern type to be used for the communication*/
        template <typename Pattern>
        class communication_object<Pattern, gridtools::ghex::cpu> {

            /** @brief buffer element type*/
            using byte_t = unsigned char;
            /** @brief extended domain id type, deduced form Pattern*/
            using extended_domain_id_t = typename Pattern::extended_domain_id_type;
            /** @brief iteration space type, deduced form Pattern*/
            using iteration_space_t = typename Pattern::iteration_space_pair;
            /** @brief map type for send and receive halos, deduced form Pattern*/
            using map_t = typename Pattern::map_type;
            /** @brief communication protocol, deduced form Pattern*/
            using communicator_t = typename Pattern::communicator_type;
            /** @brief future type, deduced form communicator, of type void*/
            using future_t = typename communicator_t::template future<void>;
            /** @brief send buffer type, for now set to vector of bytes*/
            using s_buffer_t = tl::message_buffer<>;
            /** @brief receive buffer type, for now set to vector of bytes*/
            using r_buffer_t = tl::message_buffer<>;
            /** @brief send request type, simply a future*/
            using s_request_t = future_t;
            /** @brief receive request type, 1:1 mapping between receive halo index, domain and receive request*/
            using r_request_t = std::tuple<std::size_t, extended_domain_id_t, future_t>;

            /** @brief domain id with size information, used for buffer ordering */
            struct ordered_domain_id_t {

                extended_domain_id_t m_domain_id;
                std::size_t m_size;

                bool operator < (const ordered_domain_id_t& other) const noexcept {
                    return (m_size < other.m_size ? true : (m_domain_id < other.m_domain_id));
                }

            };

            /** @brief map type for send and receive halos with ordered domain id */
            using ordered_map_t = std::map<ordered_domain_id_t, typename map_t::mapped_type>;
            /** @brief receive request type with ordered domain id */
            using ordered_r_request_t = std::tuple<std::size_t, ordered_domain_id_t, future_t>;

            const Pattern& m_pattern;
            const map_t& m_send_halos;
            const map_t& m_receive_halos;
            std::size_t m_n_send_halos;
            std::size_t m_n_receive_halos;
            std::vector<s_buffer_t> m_send_buffers;
            std::vector<r_buffer_t> m_receive_buffers;
            communicator_t m_communicator;
            ordered_map_t m_ordered_send_halos;
            ordered_map_t m_ordered_receive_halos;

            /** @brief sets the number of elements of a single buffer (single neighbor) given the halo sizes
             * @param iteration_spaces list of iteration spaces (halos), can be more than one per neighbor
             * @return std::size_t number of elements*/
            std::size_t iteration_spaces_size(const std::vector<iteration_space_t>& iteration_spaces) {

                std::size_t size{0};
                for (const auto& is : iteration_spaces) {
                    size += is.size();
                }

                return size;

            }

            /** @brief sets the size of a single buffer (single neighbor) given the halo sizes and the data type sizes
             * @tparam DataDescriptor list of data descriptors types
             * @param iteration_spaces list of iteration spaces (halos), can be more than one per neighbor
             * @param data_descriptors list of data descriptors
             * @return std::size_t buffer size*/
            template <typename... DataDescriptor>
            std::size_t buffer_size(const std::vector<iteration_space_t>& iteration_spaces,
                                    const std::tuple<DataDescriptor...>& data_descriptors) {

                std::size_t size{0};

                for (const auto& is : iteration_spaces) {
                    gridtools::ghex::detail::for_each(data_descriptors, [&is, &size](const auto& dd) {
                        size += is.size() * dd.data_type_size();
                    });
                }

                return size;

            }

            /** @brief packs the data for one neighbor into one single buffer per neighbor
             * @tparam DataDescriptor list of data descriptors types
             * @param halo_index neighbor index
             * @param domain neighbor extended domain id
             * @param data_descriptors tuple of data descriptors*/
            template <typename... DataDescriptor>
            void pack(const std::size_t halo_index, const extended_domain_id_t& domain, const std::tuple<DataDescriptor...>& data_descriptors) {

                const auto& iteration_spaces = m_send_halos.at(domain);

                m_send_buffers[halo_index].resize(buffer_size(iteration_spaces, data_descriptors));
                std::size_t buffer_index{0};

                /* The two loops are performed with this order
                 * in order to have as many data of the same type as possible in contiguos memory */
                gridtools::ghex::detail::for_each(data_descriptors, [this, &iteration_spaces, &halo_index, &buffer_index](const auto& dd) {
                    for (const auto& is : iteration_spaces) {
                        dd.get(is, m_send_buffers[halo_index].data()+buffer_index);
                        buffer_index += is.size() * dd.data_type_size();
                    }
                });

            }

        public:

            /** @brief handle to wait for and unpack the receive messages
             * @tparam DataDescriptor list of data descriptors types*/
            template <typename... DataDescriptor>
            class handle {

                const map_t& m_receive_halos;
                const std::vector<r_buffer_t>& m_receive_buffers;
                std::vector<ordered_r_request_t> m_receive_requests;
                std::tuple<DataDescriptor...> m_data_descriptors;

                /** @brief unpacks the buffer for one neighbor into all the data descriptors
                 * @param halo_index neighbor index, included in the receive request
                 * @param domain neighbor extended domain id, included in the receive request*/
                void unpack(const std::size_t halo_index, const extended_domain_id_t& domain) {

                    const auto& iteration_spaces = m_receive_halos.at(domain);

                    std::size_t buffer_index{0};

                    /* The two loops are performed with this order
                     * in order to have as many data of the same type as possible in contiguos memory */
                    gridtools::ghex::detail::for_each(m_data_descriptors, [this, &halo_index, &iteration_spaces, &buffer_index](auto& dd) {
                        for (const auto& is : iteration_spaces) {
                            dd.set(is, m_receive_buffers[halo_index].data()+buffer_index);
                            buffer_index += is.size() * dd.data_type_size();
                        }
                    });

                }

            public:

                /** @brief handle constructor
                 * @param receive_halos const reference to communication object's receive halos
                 * @param receive_buffers const reference to communication object's receive buffers
                 * @param receive_requests receive requests, moved from communication object' exchange()
                 * @param data_descriptors data descriptors, moved from communication object' exchange()*/
                handle(const map_t& receive_halos,
                       const std::vector<r_buffer_t>& receive_buffers,
                       std::vector<ordered_r_request_t>&& receive_requests,
                       std::tuple<DataDescriptor...>&& data_descriptors) :
                    m_receive_halos{receive_halos},
                    m_receive_buffers{receive_buffers},
                    m_receive_requests{std::move(receive_requests)},
                    m_data_descriptors{std::move(data_descriptors)} {}

                /** @brief waits for every receive request, and unpacks the corresponding data before moving to the next*/
                void wait() {

                    for (auto& r : m_receive_requests) {
                        std::get<2>(r).wait();
                        unpack(std::get<0>(r), std::get<1>(r).m_domain_id);
                    }

                }

            };

            /** @brief communication object constructor
             * @param p pattern*/
            communication_object(const Pattern& p, const communicator_t& comm) :
                m_pattern{p},
                m_send_halos{m_pattern.send_halos()},
                m_receive_halos{m_pattern.recv_halos()},
                m_n_send_halos{m_send_halos.size()},
                m_n_receive_halos(m_receive_halos.size()),
                m_send_buffers(m_n_send_halos),
                m_receive_buffers(m_n_receive_halos),
                m_communicator{comm} {

                for (const auto& halo : m_send_halos) {
                    const auto& domain_id = halo.first;
                    const auto& iteration_spaces = halo.second; // maybe not using a reference?
                    ordered_domain_id_t ordered_domain_id{domain_id, iteration_spaces_size(iteration_spaces)};
                    m_ordered_send_halos.insert(std::make_pair(ordered_domain_id, iteration_spaces));
                }

                for (const auto& halo : m_receive_halos) {
                    const auto& domain_id = halo.first;
                    const auto& iteration_spaces = halo.second;  // maybe not using a reference?
                    ordered_domain_id_t ordered_domain_id{domain_id, iteration_spaces_size(iteration_spaces)};
                    m_ordered_receive_halos.insert(std::make_pair(ordered_domain_id, iteration_spaces));
                }

            }

            /** @brief exchanges (receives, sends and waits for the send requests) halos for multiple data fields that shares the same pattern
             * @tparam DataDescriptor list of data descriptors types
             * @param dds list of data descriptors*/
            template <typename... DataDescriptor>
            handle<DataDescriptor...> exchange(DataDescriptor& ...dds) {

                std::vector<s_request_t> send_requests;
                send_requests.reserve(m_n_send_halos);

                std::vector<r_request_t> receive_requests;
                receive_requests.reserve(m_n_receive_halos);

                std::vector<ordered_r_request_t> ordered_receive_requests;
                ordered_receive_requests.reserve(m_n_receive_halos);

                auto data_descriptors = std::make_tuple(dds...);

                std::size_t halo_index;

                /* RECEIVE */

                halo_index = 0;
                for (const auto& halo : m_ordered_receive_halos) {

                    const auto& ordered_domain_id = halo.first;
                    const auto& domain_id = ordered_domain_id.m_domain_id;
                    auto source = domain_id.address;
                    auto tag = domain_id.tag;
                    const auto& iteration_spaces = halo.second;

                    m_receive_buffers[halo_index].resize(buffer_size(iteration_spaces, data_descriptors));

                    ordered_receive_requests.push_back(std::make_tuple(halo_index, ordered_domain_id, 
                                                                       m_communicator.recv(m_receive_buffers[halo_index],source,tag)));

                    ++halo_index;

                }

                /* SEND */

                halo_index = 0;
                for (const auto& halo : m_ordered_send_halos) {

                    const auto& ordered_domain_id = halo.first;
                    const auto& domain_id = ordered_domain_id.m_domain_id;
                    auto dest = domain_id.address;
                    auto tag = domain_id.tag;

                    pack(halo_index, domain_id, data_descriptors);

                    send_requests.push_back(m_communicator.send(m_send_buffers[halo_index], dest, tag));

                    ++halo_index;

                }

                /* SEND WAIT */

                for (auto& r : send_requests) {
                    r.wait();
                }

                return {m_receive_halos, m_receive_buffers, std::move(ordered_receive_requests), std::move(data_descriptors)};

            }

        };

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_COMMUNICATION_OBJECT_HPP */

