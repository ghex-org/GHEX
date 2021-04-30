#ifndef INCLUDED_GHEX_FORTRAN_DEFS_HPP
#define INCLUDED_GHEX_FORTRAN_DEFS_HPP

namespace gridtools {
    namespace ghex {
        namespace fhex {
            using fp_type                   = GHEX_FORTRAN_FP;
        }
    }
}

#define GHEX_DIMS         3

#define GhexDeviceUnknown 0
#define GhexDeviceCPU     1
#define GhexDeviceGPU     2

#define GhexLayoutFieldLast  1
#define GhexLayoutFieldFirst 2

/* barrier types */
#define GhexBarrierGlobal 1
#define GhexBarrierThread 2
#define GhexBarrierRank   3

#endif /* INCLUDED_GHEX_FORTRAN_DEFS_HPP */
