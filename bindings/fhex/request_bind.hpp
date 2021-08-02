#ifndef INCLUDED_GHEX_FORTRAN_REQUEST_BIND_HPP
#define INCLUDED_GHEX_FORTRAN_REQUEST_BIND_HPP

#include <cstdint>
#include "ghex_config.h"

namespace gridtools {
    namespace ghex {
        namespace fhex {

            struct frequest_type {
                int8_t data[GHEX_REQUEST_SIZE];
            };

            struct frequest_multi_type {
                int8_t data[GHEX_REQUEST_MULTI_SIZE];
            };
        }
    }
}

#endif /* INCLUDED_GHEX_FORTRAN_REQUEST_BIND_HPP */
