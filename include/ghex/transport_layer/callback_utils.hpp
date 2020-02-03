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
    static_assert(std::is_same<arg1_t,RANK_TYPE>::value,                                      \
        "rank_type is not convertible to second callback argument type");                     \
    static_assert(std::is_same<arg2_t,TAG_TYPE>::value,                                       \
        "tag_type is not convertible to third callback argument type");                       \
    static_assert(std::is_same<arg0_t,MESSAGE_TYPE>::value,                                   \
        "first callback argument type is not a message_type");

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace cb {

                /** @brief shared request state for completion handlers returned by callback based sends/recvs. */
                struct request_state {
                    // volatile is needed to prevent the compiler
                    // from optimizing away the check of this member
                    volatile bool m_ready = false;
                    unsigned int m_index = 0;
                    request_state() = default;
                    request_state(bool r) noexcept : m_ready{r} {}
                    request_state(bool r, unsigned int i) noexcept : m_ready{r}, m_index{i} {}
                    bool is_ready() const noexcept { return m_ready; }
                    int queue_index() const noexcept { return m_index; }
                };

                /** @brief simple request with shared state as member returned by callback based send/recvs. */
                struct request
                {
                    std::shared_ptr<request_state> m_request_state;
                    bool is_ready() const noexcept { return m_request_state->is_ready(); }
                    void reset() { m_request_state.reset(); }
                    int queue_index() const noexcept { return m_request_state->queue_index(); }
                };

                /** @brief simple wrapper around an l-value reference message (stores pointer and size)
                  * @tparam T value type of the messages content */
                template<typename T>
                struct ref_message
                {
                    using value_type = T;
                    T* m_data;
                    std::size_t m_size;
                    T* data() noexcept { return m_data; }
                    const T* data() const noexcept { return m_data; }
                    std::size_t size() const noexcept { return m_size; }
                };

                /** @brief type erased message capable of holding any message. Uses optimized initialization for  
                  * ref_messages and std::shared_ptr pointing to messages. */
                struct any_message
                {
                    using value_type = unsigned char;

                    // common interface to a message
                    struct iface
                    {
                        virtual unsigned char* data() noexcept = 0;
                        virtual const unsigned char* data() const noexcept = 0;
                        virtual std::size_t size() const noexcept = 0;
                        virtual ~iface() {}
                    };

                    // struct which holds the actual message and provides access through the interface iface
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

                    unsigned char* __restrict m_data;
                    std::size_t m_size;
                    std::unique_ptr<iface> m_ptr;
                    std::shared_ptr<char> m_ptr2;

                    /** @brief Construct from an r-value: moves the message inside the type-erased structure.
                      * Requires the message not to reallocate during the move. Note, that this operation will allocate
                      * storage on the heap for the holder structure of the message.
                      * @tparam Message a message type
                      * @param m a message */
                    template<class Message>
                    any_message(Message&& m)
                    : m_data{reinterpret_cast<unsigned char*>(m.data())}
                    , m_size{m.size()*sizeof(typename Message::value_type)}
                    , m_ptr{std::make_unique<holder<Message>>(std::move(m))}
                    {}

                    /** @brief Construct from a reference: copies the pointer to the data and size of the data.
                      * Note, that this operation will not allocate storage on the heap.
                      * @tparam T a message type
                      * @param m a ref_message to a message. */
                    template<typename T>
                    any_message(ref_message<T>&& m)
                    : m_data{reinterpret_cast<unsigned char*>(m.data())}
                    , m_size{m.size()*sizeof(T)}
                    {}

                    /** @brief Construct from a shared pointer: will share ownership with the shared pointer (aliasing)
                      * and keeps the message wrapped by the shared pointer alive. Note, that this operation may
                      * allocate on the heap, but does not allocate storage for the holder structure.
                      * @tparam Message a message type
                      * @param sm a shared pointer to a message*/
                    template<typename Message>
                    any_message(std::shared_ptr<Message>& sm)
                    : m_data{reinterpret_cast<unsigned char*>(sm->data())}
                    , m_size{sm->size()*sizeof(typename Message::value_type)}
                    , m_ptr2(sm,reinterpret_cast<char*>(sm.get()))
                    {}

                    any_message(any_message&&) = default;
                    any_message& operator=(any_message&&) = default;

                    unsigned char* data() noexcept { return m_data;}
                    const unsigned char* data() const noexcept { return m_data; }
                    std::size_t size() const noexcept { return m_size; }
                };

                /** @brief A container for storing callbacks and progressing them.
                  * @tparam FutureType a future type
                  * @tparam RankType the rank type (integer)
                  * @tparam TagType the tag type (integer) */
                template<class FutureType, typename RankType = int, typename TagType = int>
                class callback_queue{
                  public: // member types
                    using message_type = any_message;
                    using future_type = FutureType;
                    using rank_type = RankType;
                    using tag_type = TagType;
                    using cb_type = std::function<void(message_type, rank_type, tag_type)>;

                    // internal element which is stored in the queue
                    struct element_type {
                        message_type m_msg;
                        rank_type m_rank;
                        tag_type m_tag;
                        cb_type m_cb;
                        future_type m_future;
                        request m_request;
                    };

                    using queue_type = std::vector<element_type>;

                  private: // members
                    queue_type m_queue;

                  public:
                    int m_progressed_cancels = 0;

                  public: // ctors
                    callback_queue() { m_queue.reserve(256); }

                  public: // member functions
                    /** @brief Add a callback to the queue and receive a completion handle (request).
                      * @tparam Callback callback type
                      * @param msg the message (data)
                      * @param rank the destination/source rank
                      * @param tag the message tag
                      * @param fut the send/recv operation's return value
                      * @param cb the callback
                      * @return returns a completion handle */
                    template<typename Callback>
                    request enqueue(message_type&& msg, rank_type rank, tag_type tag, future_type&& fut, Callback&& cb) {
                        request m_req{std::make_shared<request_state>(false,m_queue.size())};
                        m_queue.push_back(element_type{std::move(msg), rank, tag, std::forward<Callback>(cb), std::move(fut),
                                             m_req});
                        return m_req;
                    }

                    /** @brief progress the queue and call the callbacks if the futures are ready. Note, that the order
                      * of progression is not defined.
                      * @return number of progressed elements */
                    int progress() {
                        int completed = 0;
                        for (unsigned int i = 0; i < m_queue.size(); ++i) {
                            auto& element = m_queue[i];
                            if (element.m_future.ready()) {
                                element.m_cb(std::move(element.m_msg), element.m_rank, element.m_tag);
                                ++completed;
                                element.m_request.m_request_state->m_ready = true;
                                if (i + 1 < m_queue.size()) {
                                    element = std::move(m_queue.back());
                                    element.m_request.m_request_state->m_index = i;
                                    --i;
                                }
                                m_queue.pop_back();
                            }
                        }
                        return completed;
                    }

                    /** @brief Cancel a callback
                      * @param index the queue index - access to this index is given through the request returned when
                      * enqueing.
                      * @return true if cancelling was successful */
                    bool cancel(unsigned int index)
                    {
                        auto& element = m_queue[index];
                        auto res = element.m_future.cancel();
                        if (!res) return false;
                        if (m_queue.size() > index+1)
                        {
                            element = std::move(m_queue.back());
                            element.m_request.m_request_state->m_index = index;
                        }
                        m_queue.pop_back();
                        ++m_progressed_cancels;
                        return true;
                    }
                };

                /** @brief a class to return the number of progressed callbacks */
                struct progress_status {
                    int m_num_sends = 0;
                    int m_num_recvs = 0;
                    int m_num_cancels = 0;
 
                    int num() const noexcept { return m_num_sends+m_num_recvs+m_num_cancels; }
                    int num_sends() const noexcept { return m_num_sends; }
                    int num_recvs() const noexcept { return m_num_recvs; }
                    int num_cancels() const noexcept { return m_num_cancels; }

                    progress_status& operator+=(const progress_status& other) noexcept {
                        m_num_sends += other.m_num_sends;
                        m_num_recvs += other.m_num_recvs;
                        m_num_cancels += other.m_num_cancels;
                        return *this;
                    }
                };

                progress_status operator+(progress_status a, progress_status b) { return a+=b; }

            } // cb
        } // tl
    } // ghex
} // gridtools

#endif /* INCLUDED_GHEX_TL_CALLBACK_UTILS_HPP */


