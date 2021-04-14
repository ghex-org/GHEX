#ifndef INCLUDED_GHEX_FORTRAN_REQUEST_BIND_HPP
#define INCLUDED_GHEX_FORTRAN_REQUEST_BIND_HPP

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

#endif /* INCLUDED_GHEX_FORTRAN_REQUEST_BIND_HPP */
