# Python bindings

This is a proof-of-concept implementation.

## Build instructions

```
mkdir build
pushd build
cmake .. -DGHEX_BUILD_TESTS=On -DGHEX_BUILD_BENCHMARKS=On -DCMAKE_BUILD_TYPE=Debug -DGHEX_BUILD_PYTHON_BINDINGS=On
make ghex_py_binding
popd
```

## Tests

```
export GHEX_PY_LIB_PATH="$(pwd)/build"
cd bindings/python/tests
mpiexec -n 2 pytest -rP test_mpi4py.py
```