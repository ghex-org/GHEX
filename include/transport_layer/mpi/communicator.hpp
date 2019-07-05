#include <iostream>
#include <mpi.h>
#include <future>
#include <functional>
#include <unordered_map>
#include <tuple>
#include <cassert>
#include "message.hpp"

namespace mpi {

struct my_dull_future {
    MPI_Request m_req;

    my_dull_future(MPI_Request req) : m_req{req} {}

    void wait() {
        MPI_Status status;
        MPI_Wait(&m_req, &status);
    }

    bool ready() {
        MPI_Status status;
        int res;
        MPI_Test(&m_req, &res, &status);
        return !res;
    }
};


#ifdef NDEBUG
#define CHECK_MPI_ERROR(x) if x;
#else
#define CHECK_MPI_ERROR(x) if (x != MPI_SUCCESS) throw std::runtime_error("MPI Call failed " + std::string(#x) + " in " + std::string(__FILE__) + ":"  +  std::to_string(__LINE__));
#endif

template <typename RankType, typename TagType>
struct communicator {

    std::unordered_map<MPI_Request, std::tuple<std::function<void(RankType, TagType)>, RankType, TagType>> m_call_backs;

    MPI_Comm m_mpi_comm = MPI_COMM_WORLD;

    ~communicator() {
        if (m_call_backs.size() != 0) {
            std::cout << "There are " << m_call_backs.size() << " pending requests that have not been serviced\n";
        }
    }

    template <typename MsgType>
    my_dull_future send(MsgType const& msg, RankType dst, TagType tag) {
        MPI_Request req;
        MPI_Status status;
        CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
        return req;
    }

    template <typename MsgType, typename CallBack>
    void send(MsgType const& msg, RankType dst, TagType tag, CallBack&& cb) {
        MPI_Request req;
        MPI_Status status;
        CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
        m_call_backs.emplace(std::make_pair(req, std::make_tuple(cb, dst, tag) ));
    }

    template <typename MsgType>
    void send_safe(MsgType const& msg, RankType dst, TagType tag) {
        MPI_Request req;
        MPI_Status status;
        CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
        CHECK_MPI_ERROR(MPI_Wait(&req, &status));
    }

    template <typename MsgType>
    my_dull_future recv(MsgType& msg, RankType src, TagType tag) {
        MPI_Request request;
        CHECK_MPI_ERROR(MPI_Irecv(msg.data(), msg.size(), MPI_BYTE, src, tag, m_mpi_comm, &request));
        return request;
    }

    template <typename MsgType, typename CallBack>
    void recv(MsgType& msg, RankType src, TagType tag, CallBack&& cb) {
        MPI_Request request;
        CHECK_MPI_ERROR(MPI_Irecv(msg.data(), msg.size(), MPI_BYTE, src, tag, m_mpi_comm, &request));

        m_call_backs.emplace(std::make_pair(request, std::make_tuple(cb, src, tag) ));
    }

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

    template <typename Allc>
    message<Allc> acquire() const {
        return {};
    }

    template <typename Allc>
    void release(message<Allc> && msg) {
        // destruct msg
    }
};

} //namespace mpi
