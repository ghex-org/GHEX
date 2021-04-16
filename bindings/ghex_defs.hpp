#ifndef INCLUDED_GHEX_FORTRAN_DEFS_HPP
#define INCLUDED_GHEX_FORTRAN_DEFS_HPP

namespace gridtools {
    namespace ghex {
        namespace fhex {
            using fp_type                   = GHEX_FORTRAN_FP;
        }
    }
}

#define GHEX_DIMS                 3

#define DeviceUnknown 0
#define DeviceCPU     1
#define DeviceGPU     2

#define LayoutFieldLast  1
#define LayoutFieldFirst 2

/* barrier types */
#define BarrierGlobal 1
#define BarrierThread 2
#define BarrierRank   3

#endif /* INCLUDED_GHEX_FORTRAN_DEFS_HPP */
