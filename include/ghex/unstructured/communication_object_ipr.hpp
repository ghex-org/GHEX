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
#include <ghex/context.hpp>
#include <ghex/arch_traits.hpp>
#include <ghex/buffer_info.hpp>
#include <ghex/device/guard.hpp>
#include <ghex/device/stream.hpp>
#include <ghex/unstructured/grid.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/unstructured/pattern.hpp>

namespace ghex
{
namespace unstructured
{
template<typename Field>
class communication_object_ipr;

template<typename Arch, typename DomainIdType, typename IndexType, typename T>
class communication_object_ipr<data_descriptor<Arch, DomainIdType, IndexType, T>>
{
  public:
    using communicator_type = context::communicator_type;
    using field_type = data_descriptor<Arch, DomainIdType, IndexType, T>;
    using arch_type = Arch;
    using domain_id_type = DomainIdType;
    using global_index_type = IndexType;
    using value_type = T;
    using domain_descriptor_type = domain_descriptor<domain_id_type, global_index_type>;
    using grid_type = typename grid::template type<domain_descriptor_type>;
    using pattern_container_type = pattern_container<grid_type, domain_id_type>;
    using pattern_type = pattern<grid_type, domain_id_type>;
    using buffer_info_type = buffer_info<pattern_type, arch_type, field_type>;
    using message_type = typename arch_traits<arch_type>::message_type;
    using index_container_type = typename pattern_type::index_container_type;
    using index_vector_type = typename pattern_type::iteration_space::local_indices_type;
    using device_id_type = typename arch_traits<arch_type>::device_id_type;

  private:
    struct halo
    {
        int                         remote_rank;
        int                         tag;
        index_container_type const& index_container;
        message_type                message;
    };

    struct status
    {
        std::size_t        num_completed;
        std::size_t        num_total;
        communicator_type& comm;
    };

  public:
    class handle
    {
      public:
        handle() noexcept {}
        handle(status* s) noexcept
        : m_status{s}
        {
        }
        handle(handle&&) noexcept = default;
        handle& operator=(handle&&) noexcept = default;

      private:
        status* m_status = nullptr;

      public: // member functions
        /** @brief wait for communication to be finished */
        void wait()
        {
            if (m_status)
                while (m_status->num_completed < m_status->num_total) { m_status->comm.progress(); }
        }
        /** @brief check whether communication is finished */
        bool is_ready()
        {
            if (!m_status) return true;
            return (m_status->num_completed == m_status->num_total);
        }
        /** @brief progress the communication */
        void progress()
        {
            if (m_status)
                while (m_status->num_completed < m_status->num_total) { m_status->comm.progress(); }
        }
    };

  private:
    context&                m_context;
    communicator_type       m_comm;
    device_id_type          m_device_id;
    field_type              m_field;
    std::vector<halo>       m_send_halos;
    std::vector<halo>       m_recv_halos;
    std::unique_ptr<status> m_status;
    device::stream          m_stream;

  public:
    communication_object_ipr(context& c, buffer_info_type bi)
    : m_context{c}
    , m_comm{m_context.get_communicator()}
    , m_device_id{bi.device_id()}
    , m_field{bi.get_field()}
    {
        //domain_id_type const domain_id = m_field.domain_id();
        //int const            max_tag = bi.get_pattern_container().max_tag();
        pattern_type const& pattern = bi.get_pattern();

        for (auto const& send_halo : pattern.send_halos())
        {
            auto const&                 extended_id = send_halo.first;
            index_container_type const& index_container = send_halo.second;
            auto const&                 iteration_space = index_container[0];
            std::size_t const           size =
                iteration_space.size() * m_field.num_components() * sizeof(value_type);

            m_send_halos.push_back(halo{extended_id.mpi_rank, extended_id.tag, index_container,
                arch_traits<arch_type>::make_message(m_comm, size, m_device_id)});
        }
        for (auto const& recv_halo : pattern.recv_halos())
        {
            auto const&                 extended_id = recv_halo.first;
            index_container_type const& index_container = recv_halo.second;
            auto const&                 iteration_space = index_container[0];
            std::size_t const           size =
                iteration_space.size() * m_field.num_components() * sizeof(value_type);
            index_vector_type const& index_vector = iteration_space.local_indices();

            m_recv_halos.push_back(halo{extended_id.mpi_rank, extended_id.tag, index_container,
                arch_traits<arch_type>::make_message(m_comm,
                    m_field.get_address_at(index_vector.front(), 0), size, m_device_id)});
        }

        m_status = std::unique_ptr<status>(
            new status{0ul, m_recv_halos.size() + m_send_halos.size(), m_comm});
    }

    handle exchange()
    {
        post_recvs();
        pack();
        //while (m_status->num_completed < m_status->num_total) { m_status->comm.progress(); }
        return {m_status.get()};
    }

  private:
    void post_recvs()
    {
        for (halo& h : m_recv_halos)
        {
            m_comm.recv(h.message, h.remote_rank, h.tag,
                [s = m_status.get()](message_type&, int, int) { ++s->num_completed; });
        }
    }

    void pack()
    {
        for (halo& h : m_send_halos)
        {
            device::guard g(h.message);
            auto          data = g.data();
            m_field.pack(reinterpret_cast<T*>(data), h.index_container,
                std::is_same<arch_type, cpu>::value ? nullptr : &m_stream);
            m_stream.sync();
            m_comm.send(h.message, h.remote_rank, h.tag,
                [s = m_status.get()](message_type&, int, int) { ++s->num_completed; });
        }
    }
};

template<typename Arch, typename DomainIdType, typename IndexType, typename T>
communication_object_ipr<data_descriptor<Arch, DomainIdType, IndexType, T>>
make_communication_object_ipr(context& c,
    buffer_info<pattern<typename grid::template type<domain_descriptor<DomainIdType, IndexType>>,
                    DomainIdType>,
        Arch, data_descriptor<Arch, DomainIdType, IndexType, T>>
        bi)
{
    return {c, bi};
}

} // namespace unstructured
} // namespace ghex
