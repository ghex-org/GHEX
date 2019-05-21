#ifndef INCLUDED_COMMUNICATOR_BASE_HPP
#define INCLUDED_COMMUNICATOR_BASE_HPP

#include <utility>

namespace gridtools {

namespace protocol {

template<typename P>
class communicator 
{
    // using protocol_type = P;
    // using handle_type   = ...;
    // using address_type  = ...;
    // template<typename T>
    // using future = future_base<protocol_type,T>;
};

template<typename P, typename T>
struct future_base
{
    using handle_type = typename communicator<P>::handle_type;
    using value_type  = T;

    future_base(value_type&& data, handle_type&& h) 
    :   m_data(std::move(data)),
        m_handle(std::move(h))
    {}
    future_base(const future_base&) = delete;
    future_base(future_base&&) = default;
    future_base& operator=(const future_base&) = delete;
    future_base& operator=(future_base&&) = default;

    void wait()
    {
        m_handle.wait();
    }

    value_type get() noexcept 
    {
        wait(); 
        return std::move(m_data); 
    }

    value_type m_data;
    handle_type m_handle;
};

} // namespace protocol

} // namespace gridtools

#endif /* INCLUDED_COMMUNICATOR_BASE_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

