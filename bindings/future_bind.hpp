#ifndef GHEX_FORTRAN_FUTURE_BIND_INCLUDED_HPP
#define GHEX_FORTRAN_FUTURE_BIND_INCLUDED_HPP

#include <cstdint>

#define GHEX_FUTURE_SIZE 8
#define GHEX_FUTURE_MULTI_SIZE 24

struct ffuture_type {
    int8_t data[GHEX_FUTURE_SIZE];
};

struct ffuture_multi_type {
    int8_t data[GHEX_FUTURE_MULTI_SIZE];
};

#endif /* GHEX_FORTRAN_FUTURE_BIND_INCLUDED_HPP */
