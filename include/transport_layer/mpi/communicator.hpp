#include <iostream>
#include <mpi.h>
#include <future>
#include <functional>
#include <unordered_map>
#include <tuple>
#include <cassert>
#include "message.hpp"

namespace mpi {

    /** Ironic name (ha! ha!) for the future returned by the send and receive
     * operations of a communicator object to check or wait on their status.
     */
    struct my_dull_future {
        MPI_Request m_req;

        my_dull_future(MPI_Request req) : m_req{req} {}

        /** Function to wait until the operation completed */
        void wait() {
            MPI_Status status;
            MPI_Wait(&m_req, &status);
        }

        /** Function to test if the operation completed
         *
         * @return True if the operation is completed
        */
        bool ready() {
            MPI_Status status;
            int res;
            MPI_Test(&m_req, &res, &status);
            return !res;
        }
    };


    #ifdef NDEBUG
    #define CHECK_MPI_ERROR(x) x;
    #else
    #define CHECK_MPI_ERROR(x) if (x != MPI_SUCCESS) throw std::runtime_error("MPI Call failed " + std::string(#x) + " in " + std::string(__FILE__) + ":"  +  std::to_string(__LINE__));
    #endif

    /** Class that provides the functions to send and receive messages. A message
     * is an object with .data() that returns a pointer to `unsigned char`
     * and .size(), with the same behavior of std::vector<unsigned char>.
     * Each message will be sent and received with a tag, bot of type int
     */
    struct communicator {

        using tag_type = int;
        using rank_type = int;

        std::unordered_map<MPI_Request, std::tuple<std::function<void(rank_type, tag_type)>, rank_type, tag_type>> m_call_backs;

        MPI_Comm m_mpi_comm = MPI_COMM_WORLD;

        ~communicator() {
            if (m_call_backs.size() != 0) {
                std::cout << "There are " << m_call_backs.size() << " pending requests that have not been serviced\n";
                std::terminate();
            }
        }

        /** Send a message to a destination with the given tag.
         * It returns a future that can be used to check when the message is available
         * again for the user.
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         *
         * @param msg Const reference to a message to send
         * @param dst Destination of the message
         * @param tag Tag associated with the message
         *
         * @return A future that will be ready when the message can be reused (e.g., filled with new data to send)
         */
        template <typename MsgType>
        my_dull_future send(MsgType const& msg, rank_type dst, tag_type tag) {
            MPI_Request req;
            MPI_Status status;
            CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
            return req;
        }

        /** Send a message to a destination with the given tag. When the message is sent, and
         * the message ready to be reused, the given call-back is invoked with the destination
         *  and tag of the message sent.
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         * @tparam CallBack Funciton to call when the message has been sent and the message ready for reuse
         *
         * @param msg Const reference to a message to send
         * @param dst Destination of the message
         * @param tag Tag associated with the message
         * @param cb  Call-back function with signature void(int, int)
         */
        template <typename MsgType, typename CallBack>
        void send(MsgType const& msg, rank_type dst, tag_type tag, CallBack&& cb) {
            MPI_Request req;
            MPI_Status status;
            CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
            m_call_backs.emplace(std::make_pair(req, std::make_tuple(cb, dst, tag) ));
        }

        /** Send a message to a destination with the given tag. This function blocks until the message has been sent and
         * the message ready to be reused
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         *
         * @param msg Const reference to a message to send
         * @param dst Destination of the message
         * @param tag Tag associated with the message
         */
        template <typename MsgType>
        void send_safe(MsgType const& msg, rank_type dst, tag_type tag) {
            MPI_Request req;
            MPI_Status status;
            CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
            CHECK_MPI_ERROR(MPI_Wait(&req, &status));
        }

        /** Receive a message from a destination with the given tag.
         * It returns a future that can be used to check when the message is available
         * to be read.
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         *
         * @param msg Const reference to a message that will contain the data
         * @param src Source of the message
         * @param tag Tag associated with the message
         *
         * @return A future that will be ready when the message can be read
         */
        template <typename MsgType>
        my_dull_future recv(MsgType& msg, rank_type src, tag_type tag) {
            MPI_Request request;
            CHECK_MPI_ERROR(MPI_Irecv(msg.data(), msg.size(), MPI_BYTE, src, tag, m_mpi_comm, &request));
            return request;
        }

        /** Receive a message from a source with the given tag. When the message arrives, and
         * the message ready to be read, the given call-back is invoked with the destination
         *  and tag of the message sent.
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         * @tparam CallBack Funciton to call when the message has been sent and the message ready for reuse
         *
         * @param msg Const reference to a message that will contain the data
         * @param src Source of the message
         * @param tag Tag associated with the message
         * @param cb  Call-back function with signature void(int, int)
         */
        template <typename MsgType, typename CallBack>
        void recv(MsgType& msg, rank_type src, tag_type tag, CallBack&& cb) {
            MPI_Request request;
            CHECK_MPI_ERROR(MPI_Irecv(msg.data(), msg.size(), MPI_BYTE, src, tag, m_mpi_comm, &request));

            m_call_backs.emplace(std::make_pair(request, std::make_tuple(cb, src, tag) ));
        }

        /** Function to invoke to poll the transport layer and check for the completions
         * of the operations without a future associated to them (that is, they are associated
         * to a call-back). When an operation completes, the corresponfing call-back is invoked
         * with the rank and tag associated with that request.
         */
        void progress() {
            for (auto & i : m_call_backs) {
                int res;
                MPI_Status status;
                MPI_Request r = i.first;
                CHECK_MPI_ERROR(MPI_Test(&r, &res, &status));

                if (res) {
                    std::get<0>(i.second)(std::get<1>(i.second), std::get<2>(i.second));
                    m_call_backs.erase(i.first); // must use i.first andnot r, since r is modified
                }
            }
        }
    };

} //namespace mpi
