.. include:: ../defs.hrst

.. _introduction:

============
Introduction
============

-----------------
What Is GHEX
-----------------

^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^

|GHEX| requires at least a header-only installation of Boost_. It depends on ``Boost Preprocessor Library``.

Additionally, |GHEX| requires a recent version of CMake_.

.. _Boost: https://www.boost.org/
.. _CMake: https://www.cmake.org/

|GHEX| requires a modern compiler. A list of supported compilers can be found on `github <https://github.com/GridTools/gridtools>`_.


.. _installation:

--------------------
Installation and Use
--------------------

^^^^^^^^^^^^^
Simple Script
^^^^^^^^^^^^^

We first provide a sample of the commands needed to enable the multicore and CUDA backends, install them in ``/usr/local``,
and run the gridtools tests.

.. code-block:: shell

 git clone http://github.com/GridTools/GHEX.git
 cd GHEX
 mkdir build && cd build
 cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
 make install -j4
 make test

|GHEX| uses CMake as building system. The installation can be configured using `ccmake`.

------------
Contributing
------------

Contributions to the |GHEX| set of libraries are welcome. However, our policy is that we will be the official maintainers and providers of the GridTools code. We believe that this will provide our users with a clear reference point for support and guarantees on timely interactions. For this reason, we require that external contributions to |GT| will be accepted after their authors provide to us a signed copy of a copyright release form to ETH Zurich.
