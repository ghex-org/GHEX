MODULE ghex_defs
  integer, public, parameter :: GHEX_ANY_SOURCE = -1

  integer, public, parameter :: ghex_fp_kind = 4

  integer, public, parameter :: GhexDeviceUnknown = 0
  integer, public, parameter :: GhexDeviceCPU = 1
  integer, public, parameter :: GhexDeviceGPU = 2

  integer, public, parameter :: GhexLayoutFieldLast  = 1
  integer, public, parameter :: GhexLayoutFieldFirst = 2

  integer, public, parameter :: GhexBarrierGlobal = 1
  integer, public, parameter :: GhexBarrierThread = 2
  integer, public, parameter :: GhexBarrierRank   = 3
  
END MODULE ghex_defs
