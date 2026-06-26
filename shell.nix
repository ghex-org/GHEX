{ pkgs ? import <nixpkgs> { } }:
let
  python = pkgs.python3.withPackages (ps: [
    ps.nanobind
    ps.mpi4py
    ps.numpy
    ps.pytest
    ps.pytest-mpi
  ]);
in
pkgs.mkShell {
  name = "ghex-dev";

  nativeBuildInputs = [
    pkgs.cmake
    pkgs.ninja
    pkgs.gcc
    pkgs.git
    python
  ];

  buildInputs = [
    pkgs.openmpi
    pkgs.boost
    pkgs.numactl
  ];

  # cmake's Python test infrastructure creates a venv and pip-installs
  # mpi4py into it. That venv needs to find the Nix-provided MPI libs.
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.openmpi}/lib:${pkgs.numactl}/lib''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH

    echo "ghex dev shell"
    echo "  cmake -B build -S . -G Ninja \\"
    echo "    -DGHEX_USE_BUNDLED_LIBS=ON \\"
    echo "    -DGHEX_GIT_SUBMODULE=OFF \\"
    echo "    -DGHEX_BUILD_PYTHON_BINDINGS=ON \\"
    echo "    -DGHEX_WITH_TESTING=ON \\"
    echo "    -DGHEX_USE_GPU=OFF \\"
    echo "    -DGHEX_TRANSPORT_BACKEND=MPI"
  '';
}
