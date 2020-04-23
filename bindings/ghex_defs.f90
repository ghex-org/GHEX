MODULE ghex_defs
  integer, public, parameter :: ghex_fp_kind = 4

  integer, public, parameter :: DeviceUnknown = 0
  integer, public, parameter :: DeviceCPU = 1
  integer, public, parameter :: DeviceGPU = 2

  integer, public, parameter :: LayoutFieldLast  = 1
  integer, public, parameter :: LayoutFieldFirst = 2
END MODULE ghex_defs
