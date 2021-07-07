#include "context_bind.hpp"

int main()
{
    std::cout << "#ifndef GHEX_CONFIG_H_INCLUDED\n";
    std::cout << "#define GHEX_CONFIG_H_INCLUDED\n";    
    std::cout << "\n";    
    std::cout << "#define GHEX_REQUEST_SIZE " << sizeof(gridtools::ghex::fhex::communicator_type::request_cb) << "\n";
    std::cout << "#define GHEX_FUTURE_SIZE " << sizeof(gridtools::ghex::fhex::communicator_type::future<void>) << "\n";
    std::cout << "#define GHEX_REQUEST_MULTI_SIZE " << sizeof(std::vector<gridtools::ghex::fhex::communicator_type::future<void>>) << "\n";
    std::cout << "#define GHEX_FUTURE_MULTI_SIZE " << sizeof(std::vector<gridtools::ghex::fhex::communicator_type::request_cb>) << "\n";
    std::cout << "\n";
    std::cout << "#endif /* GHEX_CONFIG_H_INCLUDED */\n";
}
