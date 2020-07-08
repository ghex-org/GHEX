.. include:: ../defs.hrst

.. _scope:

============================
Scope and objectives of GHEX
============================

|GHEX| is a C++ library to perform halo-update operations in mesh/grid applications
in modern HPC architecures. The objective of |GHEX| is to enable halo-updates operations

    - For traditional domain decomposed distributed memory applications (i.e., one domain per node) either on CPUs or GPUs

    - For applications applying oversubscriptioning on a node, either for latency hiding of exploiting multi-threading

    - For application exploiting hybrid systems (nodes with multiple address spaces and multiple computing devices)

    - For applications regardless of the specific representation of the grid/mesh (by using *adaptors*)

    - On architectures that gives access to transport mechanisms other than MPI (e.g., ``UCX`` and ``Libfabric``) whose performance may be higher

In order to accomplish all of the above, the interfaces to |GHEX| requires a non trivial amount of work on the user side.
The reward for this work is: portability to multiple architectures, with and without accelerators, with the possibility to exploit
native transport mechanisms. Depending on the complexity of the application, a user can easily adapt it to use different number of threads,
or different types of threads. |GHEX| can accommodate these requirements quite flexibly.

--------------------------
Type of interfaces
--------------------------

|GHEX| has a layered strctured. The user can enter at different leyers, depending on their needs. The highest level is the ``halo exchange`` level,
where the user instructs |GHEX| to take a mesh or grid representation and produce a ``communication pattern`` to then perform the halo update operations.

In order to enable all the previously mentioned features, like oversusubscription, alternate transport layers and hybrid computations, the steps to create
a pattern and use it to communicate can seem overly complicated. While we are working on shortening the number of steps to take for simple cases, more
complex cases seem to require these complications, and hence they cannot be avoided. As we expect applications to become more and more complex in the future,
we think the use cases in which the interfaces provided by |GHEX| will increase and overshadow the traditional distributed memory applications simply based
on MPI.

The main concepts needed by |GHEX| in order to provide all the above features are:

    - **Context** : A computing node, that can be identified with a process, or an MPI rank, for instance, needs some information about how it is connected to other processes and which are those processes. The context provide this information to the application. In addition, the context must be informed of how many threads will be used for communication (in |GHEX| threads can communicate halos to each other) it, since this information needs to be provided by the user when constructing the context.

    - **Token** : When the user schedule computations on different threads, contexts would need to know which threads are actually part of the communication bunch. For this reason, each thread must obtain a token from the context to identify itself. It's responsibility of the user to guarantee that th tokens are rightly associated to the threads. In many occasions these tokens are needed only to obtain the communicatot objects, but for re-obtaining them, or to perform barrier operations, the tokens are necessary. In this latter case, the tokens are needed to identify the threads that particiate in the berriers (the barriers are provided as debugging tools, and are not performance optimized).

    - **Communicator**:

    - **Coomunication Pattern**:

    - **Communication Object**:

