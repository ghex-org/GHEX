# -*- coding: utf-8 -*-
import multiprocessing
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess

# build dependencies
import mpi4py
import pybind11


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        out = subprocess.run(["cmake", "--version"], capture_output=True)
        if out.returncode != 0:
            raise RuntimeError(out.stderr)

        # retrieve useful directories
        this_dir = os.path.dirname(__file__)
        source_dir = os.path.abspath(os.path.join(this_dir, "../.."))
        build_dir = os.path.abspath(os.path.join(this_dir, "build"))
        build_lib_dir = os.path.join(build_dir, "lib")
        pybind11_dir = pybind11.get_cmake_dir()
        mpi4py_dir = os.path.abspath(os.path.join(mpi4py.get_include(), ".."))
        ext_name = self.extensions[0].name
        install_dir = self.get_ext_fullpath(ext_name).rsplit("/", maxsplit=1)[0]

        # cmake arguments: default
        cmake_args = [
            f"-B{build_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGHEX_BUILD_PYTHON_BINDINGS=ON",
            "-DHAVE_MPI4PY=True",
            f"-Dpybind11_DIR={pybind11_dir}",
            f"-DPY_MPI4PY={mpi4py_dir}",
            "-DGHEX_USE_BUNDLED_LIBS=ON",
            f"-DCMAKE_BUILD_RPATH={install_dir}"
        ]

        # cmake arguments: GPU
        ghex_use_gpu = os.environ.get("GHEX_USE_GPU", "False")
        cmake_args.append(f"-DGHEX_USE_GPU={ghex_use_gpu}")
        if bool(ghex_use_gpu):
            ghex_gpu_type = os.environ.get("GHEX_GPU_TYPE", "AUTO")
            cmake_args.append(f"-DGHEX_GPU_TYPE={ghex_gpu_type}")
            ghex_gpu_arch = os.environ.get("GHEX_GPU_ARCH", "")
            if ghex_gpu_type == "NVIDIA":
                cxx = os.environ.get("CXX", "g++")
                cmake_args.append(f"-DCMAKE_CUDA_HOST_COMPILER={cxx}")
                cmake_args.append(f"-DGHEX_GPU_ARCH={ghex_gpu_arch}")
            elif ghex_gpu_type == "AMD":
                cmake_args.append(f"-DAMDGPU_TARGETS={ghex_gpu_arch}")

        # cmake arguments: transport backend
        ghex_transport_backend = os.environ.get("GHEX_TRANSPORT_BACKEND", "MPI")
        cmake_args.append(f"-DGHEX_TRANSPORT_BACKEND={ghex_transport_backend}")

        # build
        subprocess.run(["cmake", source_dir, *cmake_args], capture_output=False)
        subprocess.run(
            ["cmake", "--build", build_dir, "--", f"--jobs={multiprocessing.cpu_count()}"],
            capture_output=False,
        )

        # install shared libraries
        libs = os.listdir(build_lib_dir)
        for lib in libs:
            src_path = os.path.join(build_lib_dir, lib)
            trg_path = os.path.join(install_dir, lib)
            self.copy_file(src_path, trg_path)

        # install version.txt
        src_path = os.path.join(build_dir, "version.txt")
        trg_path = os.path.join(install_dir, "ghex/version.txt")
        self.copy_file(src_path, trg_path)


if __name__ == "__main__":
    setup(ext_modules=[CMakeExtension("_pyghex")], cmdclass={"build_ext": CMakeBuild})
