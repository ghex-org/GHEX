# -*- coding: utf-8 -*-
import multiprocessing
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import shutil
import site
import subprocess
import tempfile

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

        # set path to useful directories
        this_dir = os.path.dirname(__file__)
        source_dir = os.path.abspath(os.path.join(this_dir, "../.."))
        build_prefix = os.environ.get("GHEX_BUILD_PREFIX", tempfile.gettempdir())
        build_dir = tempfile.mkdtemp(dir=build_prefix)
        pybind11_dir = pybind11.get_cmake_dir()
        mpi4py_dir = os.path.abspath(os.path.join(mpi4py.get_include(), ".."))
        ext_name = self.extensions[0].name
        ext_filename = self.get_ext_filename(ext_name)
        ext_fullpath = self.get_ext_fullpath(ext_name)
        install_dir = ext_fullpath.rsplit("/", maxsplit=1)[0]
        install_prefix = os.path.join(site.getsitepackages()[0], "ghex")

        # cmake arguments: default
        cmake_args = [
            f"-B{build_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            "-DGHEX_BUILD_PYTHON_BINDINGS=ON",
            "-DGHEX_USE_BUNDLED_LIBS=ON",
            "-DHAVE_MPI4PY=True",
            f"-DPY_MPI4PY={mpi4py_dir}",
            f"-Dpybind11_DIR={pybind11_dir}",
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

        # build and install
        subprocess.run(["cmake", source_dir, *cmake_args], capture_output=False)
        subprocess.run(
            ["cmake", "--build", build_dir, "--", f"--jobs={multiprocessing.cpu_count()}"],
            capture_output=False,
        )
        subprocess.run(["cmake", "--install", build_dir], capture_output=False)

        # copy shared library _pyghex into top-level distribution folder
        src_path = os.path.join(install_prefix, ext_filename)
        self.copy_file(src_path, ext_fullpath)

        # install version.txt
        src_path = os.path.join(build_dir, "version.txt")
        trg_path = os.path.join(install_dir, "ghex/version.txt")
        self.copy_file(src_path, trg_path)

        # delete build directory
        shutil.rmtree(build_dir)


if __name__ == "__main__":
    setup(ext_modules=[CMakeExtension("_pyghex")], cmdclass={"build_ext": CMakeBuild})
