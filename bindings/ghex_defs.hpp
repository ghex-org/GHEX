#ifndef GHEX_FORTRAN_DEFS_INCLUDED_HPP
#define GHEX_FORTRAN_DEFS_INCLUDED_HPP

namespace gridtools {
    namespace ghex {
        namespace fhex {
            
#ifdef GHEX_FORTRAN_FP_DOUBLE
            using fp_type                   = double;
#else
            using fp_type                   = float;
#endif
            
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

#endif /* GHEX_FORTRAN_DEFS_INCLUDED_HPP */
