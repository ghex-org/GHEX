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
#ifndef INCLUDED_GHEX_TL_CALLBACK_UTILS_HPP
#define INCLUDED_GHEX_TL_CALLBACK_UTILS_HPP

#include <boost/callable_traits.hpp>

/** @brief checks the arguments of callback function object */
#define GHEX_CHECK_CALLBACK_F(MESSAGE_TYPE, RANK_TYPE, TAG_TYPE)                              \
    using args_t = boost::callable_traits::args_t<CallBack>;                                  \
    using arg0_t = std::tuple_element_t<0, args_t>;                                           \
    using arg1_t = std::tuple_element_t<1, args_t>;                                           \
    using arg2_t = std::tuple_element_t<2, args_t>;                                           \
    static_assert(std::tuple_size<args_t>::value==3,                                          \
        "callback must have 3 arguments");                                                    \
    static_assert(std::is_convertible<arg1_t,RANK_TYPE>::value,                               \
        "rank_type is not convertible to second callback argument type");                     \
    static_assert(std::is_convertible<arg2_t,TAG_TYPE>::value,                                \
        "tag_type is not convertible to third callback argument type");                       \
    static_assert(std::is_convertible<arg0_t,MESSAGE_TYPE>::value,                            \
        "first callback argument type is not a message_type");

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace cb {

                // shared request state
                struct request_state
                {
                    // volatile is needed to prevent the compiler
                    // from optimizing away the check of this member
                    volatile bool m_ready = false;
                    request_state() = default;
                    request_state(bool r) noexcept : m_ready{r} {}
                    bool is_ready() const noexcept { return m_ready; }
                };

                // simple request class which is returned from send and recv calls
                struct request
                {
                    std::shared_ptr<request_state> m_request_state;
                    bool is_ready() const noexcept { return m_request_state->is_ready(); }
                };

                // simple wrapper around an l-value reference message (stores pointer and size)
                template<typename T>
                struct ref_message
                {
                    using value_type = T;//unsigned char;
                    T* m_data;
                    std::size_t m_size;
                    T* data() noexcept { return m_data; }
                    const T* data() const noexcept { return m_data; }
                    std::size_t size() const noexcept { return m_size; }
                };

                // type-erased message
                struct any_message
                {
                    using value_type = unsigned char;

                    struct iface
                    {
                        virtual unsigned char* data() noexcept = 0;
                        virtual const unsigned char* data() const noexcept = 0;
                        virtual std::size_t size() const noexcept = 0;
                        virtual ~iface() {}
                    };

                    template<class Message>
                    struct holder final : public iface
                    {
                        using value_type = typename Message::value_type;
                        Message m_message;
                        holder(Message&& m): m_message{std::move(m)} {}

                        unsigned char* data() noexcept override { return reinterpret_cast<unsigned char*>(m_message.data()); }
                        const unsigned char* data() const noexcept override { return reinterpret_cast<const unsigned char*>(m_message.data()); }
                        std::size_t size() const noexcept override { return sizeof(value_type)*m_message.size(); }
                    };

                    unsigned char* __restrict m_data = nullptr;
                    std::size_t m_size;
                    std::unique_ptr<iface> m_ptr;
                    std::shared_ptr<std::size_t> m_ptr2;

                    template<class Message>
                    any_message(Message&& m)
                    : m_data{reinterpret_cast<unsigned char*>(m.data())}
                    , m_size{m.size()*sizeof(typename Message::value_type)}
                    , m_ptr{std::make_unique<holder<Message>>(std::move(m))}
                    {}

                    template<typename T>
                    any_message(ref_message<T>&& m)
                    : m_data{reinterpret_cast<unsigned char*>(m.data())}
                    , m_size{m.size()*sizeof(T)}
                    {}

                    template<typename Message>
                    any_message(std::shared_ptr<Message>& sm)
                    : m_data{reinterpret_cast<unsigned char*>(sm->data())}
                    , m_size{sm->size()*sizeof(typename Message::value_type)}
                    , m_ptr2(sm,&m_size)
                    {}
                    

                    any_message(any_message&&) = default;
                    any_message& operator=(any_message&&) = default;

                    unsigned char* data() noexcept { return m_data; /*m_ptr->data();*/ }
                    const unsigned char* data() const noexcept { return m_data; /*m_ptr->data();*/ }
                    std::size_t size() const noexcept { return m_size; /*m_ptr->size();*/ }

                    std::size_t use_count()
                    {
                        if(nullptr != m_ptr2.get()) return m_ptr2.use_count();
                        return 1;
                    }
                };


                // simple shared message which is internally used for send_multi
                template<typename Message>
                struct shared_message
                {
                    using value_type = typename Message::value_type;
                    std::shared_ptr<Message> m_message;

                    shared_message(Message&& m) : m_message{std::make_shared<Message>(std::move(m))} {}
                    shared_message(const shared_message&) = default;
                    shared_message(shared_message&&) = default;

                    value_type* data() noexcept { return m_message->data(); }
                    const value_type* data() const noexcept { return m_message->data(); }
                    std::size_t size() const noexcept { return m_message->size(); }
                };

                // type-erased future
                struct any_future
                {
                    struct iface
                    {
                        virtual bool ready() = 0;
                        virtual ~iface() {}
                    };

                    template<class Future>
                    struct holder final : public iface
                    {
                        Future m_future;
                        holder() = default;
                        holder(Future&& fut): m_future{std::move(fut)} {}
                        bool ready() override { return m_future.ready(); }
                    };

                    std::unique_ptr<iface> m_ptr;

                    template<class Future>
                    any_future(Future&& fut) : m_ptr{std::make_unique<holder<Future>>(std::move(fut))} {}

                    bool ready() { return m_ptr->ready(); }
                };


            } // cb
        } // tl
    } // ghex
} // gridtools

#endif /* INCLUDED_GHEX_TL_CALLBACK_UTILS_HPP */


