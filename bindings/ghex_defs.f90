MODULE ghex_defs
  integer, public, parameter :: GHEX_ANY_SOURCE = -1

  integer, public, parameter :: ghex_fp_kind = 4

  integer, public, parameter :: DeviceUnknown = 0
  integer, public, parameter :: DeviceCPU = 1
  integer, public, parameter :: DeviceGPU = 2

  integer, public, parameter :: LayoutFieldLast  = 1
  integer, public, parameter :: LayoutFieldFirst = 2

  integer, public, parameter :: CartOrderXYZ = 1
  integer, public, parameter :: CartOrderXZY = 2
  integer, public, parameter :: CartOrderZYX = 3
  integer, public, parameter :: CartOrderYZX = 4
  integer, public, parameter :: CartOrderZXY = 5
  integer, public, parameter :: CartOrderYXZ = 6
  integer, public, parameter :: CartOrderDefault = 1

  integer, public, parameter :: GhexBarrierGlobal = 1
  integer, public, parameter :: GhexBarrierThread = 2
  integer, public, parameter :: GhexBarrierRank   = 3
  
END MODULE ghex_defs
