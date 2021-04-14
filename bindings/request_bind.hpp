#ifndef GHEX_FORTRAN_REQUEST_BIND_INCLUDED_HPP
#define GHEX_FORTRAN_REQUEST_BIND_INCLUDED_HPP

#include <cstdint>

#define GHEX_REQUEST_SIZE 24

namespace gridtools {
    namespace ghex {
        namespace fhex {

            struct frequest_type {
                int8_t data[GHEX_REQUEST_SIZE];
            };
        }
    }
}

#endif /* GHEX_FORTRAN_REQUEST_BIND_INCLUDED_HPP */
