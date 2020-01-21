#ifndef GHEX_FORTRAN_FUTURE_BIND_INCLUDED_HPP
#define GHEX_FORTRAN_FUTURE_BIND_INCLUDED_HPP

#include <cstdint>

#define GHEX_FUTURE_SIZE 8

struct ffuture_type {
    int8_t data[GHEX_FUTURE_SIZE];
};

#endif /* GHEX_FORTRAN_FUTURE_BIND_INCLUDED_HPP */
