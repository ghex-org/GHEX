# -*- coding: utf-8 -*-
from distutils.dir_util import copy_tree
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

        # default cmake arguments
        cmake_args = [
            f"-B{build_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGHEX_BUILD_PYTHON_BINDINGS=ON",
            "-DHAVE_MPI4PY=True",
            f"-Dpybind11_DIR={pybind11_dir}",
            f"-DPY_MPI4PY={mpi4py_dir}",
            "-DGHEX_USE_BUNDLED_LIBS=ON",
            "-DGHEX_USE_BUNDLED_GRIDTOOLS=ON",
            "-DGHEX_USE_BUNDLED_OOMPH=ON",
        ]

        # extra cmake arguments set via env variables
        # ghex_use_gpu = os.environ.get("GHEX_USE_GPU", "False")
        # cmake_args.append(f"-DUSE_GPU={ghex_use_gpu}")
        # if bool(ghex_use_gpu):
        #     ghex_gpu_type = os.environ.get("GHEX_GPU_TYPE", "AUTO")
        #     ghex_gpu_arch = os.environ.get("GHEX_GPU_ARCH", "")
        #     cmake_args += [
        #         f"-DGHEX_GPU_TYPE={ghex_gpu_type}",
        #         f"-DGHEX_GPU_ARCH={ghex_gpu_arch}",
        #     ]

        # build
        subprocess.run(["cmake", source_dir, *cmake_args], capture_output=False)
        subprocess.run(
            ["cmake", "--build", build_dir, "--", "--jobs=8"], capture_output=False
        )

        # install shared libraries
        copy_tree(build_lib_dir, install_dir)

        # install version.txt
        src_path = os.path.join(build_dir, "version.txt")
        trg_path = os.path.join(install_dir, "ghex/version.txt")
        self.copy_file(src_path, trg_path)


if __name__ == "__main__":
    setup(ext_modules=[CMakeExtension("_pyghex")], cmdclass={"build_ext": CMakeBuild})
