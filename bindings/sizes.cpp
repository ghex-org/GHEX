#include <string.h>
#include "context_bind.hpp"

int main(int argc, char *argv[])
{
    if(argc!=2){
        std::cerr << "wrong number of arguments\n";
        return 1;
    }
    if(!strcmp(argv[1], "GHEX_REQUEST_SIZE"))
        std::cout << sizeof(gridtools::ghex::fhex::communicator_type::request_cb_type);
    else if(!strcmp(argv[1], "GHEX_FUTURE_SIZE"))
        std::cout << sizeof(gridtools::ghex::fhex::communicator_type::future<void>);
    else if(!strcmp(argv[1], "GHEX_FUTURE_MULTI_SIZE"))
        std::cout << sizeof(std::vector<gridtools::ghex::fhex::communicator_type::future<void>>);
    else {
        std::cout << "unknown argument\n";
        return 1;
    }
    return 0;
}
