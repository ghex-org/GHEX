#include <iostream>
#ifdef _OPENMP
#include <transport_layer/primitives/openmp.hpp>
#else
#include <transport_layer/primitives/threads.hpp>
#endif
#include <vector>
#include <thread>
#include <fstream>

int main() {
    std::vector<std::thread> ts;

    threads controller(4);

#ifdef _OPENMP
#pragma omp parallel
    {
        int i = omp_get_thread_num();
        controller.critical([i] { std::cout << "Critical " << i << "\n"; std::cout.flush(); });

        auto token = controller.get_token();
        controller.barrier(token);
        controller.barrier(token);
        controller.barrier(token);
        controller.master(token, [] { std::cout << "Only ONE does this\n"; } );
        controller.barrier(token);
        controller.barrier(token);
    }
#else
    for (int i = 0; i < 4; ++i) {
        ts.push_back(std::thread{[&controller, i] {

                    controller.critical([i] { std::cout << "Critical " << i << "\n"; std::cout.flush(); });

                    auto token = controller.get_token();
                    controller.barrier(token);
                    controller.barrier(token);
                    controller.barrier(token);
                    controller.master(token, [] { std::cout << "Only ONE does this\n"; } );
                    controller.barrier(token);
                    controller.barrier(token);
                }});
    }

    for (auto& t : ts) {
        t.join();
    }
#endif
    std::cout << "Done\n";
}
