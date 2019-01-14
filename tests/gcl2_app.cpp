#ifdef MT_MPI
#include "gcl2.cpp"
#elif OS_MPI
#include "gcl2_onesided.cpp"
#else
#include "gcl2_managed.cpp"
#endif

void work(int local_thread_id) {
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int global_thread_id = local_thread_id + threads_per_node * rank;

    gcl_object gcl1(local_thread_id,
                   (local_thread_id==(threads_per_node-1))?(rank+1)%size:rank,
                   (local_thread_id==(threads_per_node-1))?0:local_thread_id+1,
                   (local_thread_id==0)?((rank==0)?(size-1):rank-1):rank,
                   (local_thread_id==0)?threads_per_node-1:local_thread_id-1);

    gcl_object gcl2(local_thread_id,
                   (local_thread_id==(threads_per_node-1))?(rank+1)%size:rank,
                   (local_thread_id==(threads_per_node-1))?0:local_thread_id+1,
                   (local_thread_id==0)?((rank==0)?(size-1):rank-1):rank,
                    (local_thread_id==0)?threads_per_node-1:local_thread_id-1);

    //    gcl2_barrier();

    std::string payload1("Hi, I am " + std::to_string(local_thread_id) +
                        " From processor " + std::to_string(rank));
    std::string payload2("HI, I AM " + std::to_string(local_thread_id) +
                        " FROM PROCESSOR " + std::to_string(rank));
    std::string payload3("IH, I MA " + std::to_string(local_thread_id) +
                        " RFMO RPCOSEOSR " + std::to_string(rank));

    char resv1[100];
    char resv2[100];

    auto hdl1 = gcl1.exchange(payload1.c_str(), payload1.length()+1, resv1);
    auto hdl2 = gcl2.exchange(payload2.c_str(), payload2.length()+1, resv2);

    //std::this_thread::sleep_for(std::chrono::seconds{1+rank*threads_per_node+local_thread_id});

    hdl1.get();

    std::string res1("<" + std::to_string(rank) + ", " + std::to_string(local_thread_id) + ">: " + resv1 + "\n");
    safe_output(res1);

    auto hdl3 = gcl1.exchange(payload3.c_str(), payload3.length()+1, resv1);

    hdl3.get();
    hdl2.get();

    std::string res3("<" + std::to_string(rank) + ", " + std::to_string(local_thread_id) + ">: " + resv1 + " <SECOND>\n");
    std::string res2("<" + std::to_string(rank) + ", " + std::to_string(local_thread_id) + ">: " + resv2 + "\n");

    safe_output(res2);
    safe_output(res3);

    //gcl2_barrier();

}

int main(int argc, char** argv) {
    gcl2_init(argc, argv);

    auto handle0 = std::thread(work, 0);
    auto handle1 = std::thread(work, 1);
    auto handle2 = std::thread(work, 2);
    handle0.join();
    handle1.join();
    handle2.join();

    // auto handle0 = std::async(std::launch::async, // mandatory
    //                          work, 0);
    // auto handle1 = std::async(std::launch::async, // mandatory
    //                          work, 1);
    // auto handle2 = std::async(std::launch::async, // mandatory
    //                          work, 2);

    // handle0.get();
    // handle1.get();
    // handle2.get();

    MPI_Finalize();
}
