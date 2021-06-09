#ifndef INCLUDED_GHEX_FORTRAN_FUTURE_BIND_HPP
#define INCLUDED_GHEX_FORTRAN_FUTURE_BIND_HPP

#include <cstdint>
#include "ghex_config.h"

namespace gridtools {
    namespace ghex {
        namespace fhex {
            
            struct ffuture_type {
                int8_t data[GHEX_FUTURE_SIZE];
            };

            struct ffuture_multi_type {
                int8_t data[GHEX_FUTURE_MULTI_SIZE];
            };
        }
    }
}

#endif /* INCLUDED_GHEX_FORTRAN_FUTURE_BIND_HPP */
