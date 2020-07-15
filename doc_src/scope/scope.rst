.. include:: ../defs.hrst

.. _scope:

============================
Scope and objectives of GHEX
============================

|GHEX| is a C++ library to perform halo-update operations in mesh/grid applications
in modern HPC architecures. Domain decomposition refers the technique that applications use to distribute the work across different processes when numerically solving differentail equations. For instance in the next figure aschematic depiction this idea.

.. figure:: figures/domain_decomp.png
    :width: 600px
    :align: center
    :alt: This should not be visible
    :figclass: align-center

    Example of Domain Decomposition. The physical domain, on the right, is split into 4 different sub-domains. To allow for the computation to process without having to remotely access avery single element that resides on remote processes, the application uses *ghost regions*, or *halos* (in pink) that need to be updated whenever the computation requires accesing that region again (and the values in that regions have been updated). The blue arrows show the communication pattern, also referred in the manual as *halo-update*, or *halo-exchange* operation.



The objective of |GHEX| is to enable halo-update operations

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

    - **Context** : A computing node, that can be identified with a process, or an MPI rank, for instance, needs some information about how it is connected to other processes and which are those processes. The context provide this information to the application.

    - **Communicator**: A communicator can be seen representing the end-point of a communication channel. These communication channels are obtained from the context.

    - **Coomunication Pattern**:

    - **Communication Object**:

The interfaces for context and communicator are slightly richer than what is required by the *pure* halo-update operation. We deisgned contexts and communicators in order to cope with more complex scenarious where the domain decomposition more dynamic. The interfaces for communication patterns and communication objects are advantageous only

------------------------
Context
------------------------

A computing node, that can be identified with a process, or an MPI rank, for instance, needs some information about how it is connected to other processes and which are those processes. The context provide this information to the application.

In addition, contexts maintain state information that the *communicators* need in order to function. In other words, communicators (presented below) cannot outlive the contexts from where they were obtained.

GHEX, assumes the platform where application runs, provide an implementation of the MPI library. We assume this is a safe assumption since the vast majority of the HPC application rely on MPI, directly or indirectly (for instance by using PGAS languages or runtime-systems such as HPX). This makes is possible to simplify creation of contexts and collect information about which processes participate to the computation. This is the reason why the creation of a context requires an MPI Communicator as runtime argument, in addition to other possible transport specific arcguments. The passed MPI Communicator will be cloned by the context constructor. For this reason the context should be destroyed before the call to ``MPI_Finalize``.

Contexts are constructed on a specific *transport layers*. The available tranport layers are: MPI, UCX, and Libfabric. The transport layer is a *tag* passed as template argument to the context type.

In order to guarantee an uniform initilization of the contexts, which are highly platform dependent, the user should not call the constructor of the context direcly, but instead the use should use a factory, which returns a ``std::unique_ptr`` to context object instantiated with the proper transport layer. The syntax looks like this:

.. code-block:: gridtools

    #ifdef USE_UCX
    using transport = gridtools::ghex::tl::ucx_tag;
    #else
    using transport = gridtools::ghex::tl::mpi_tag;
    #endif

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);

In the above example the use of ``#ifdef`` is an example of how code can be made portable by minimal use of macros, and a recompilation.

From the context, additional information can be obtained, such as the ``context<TransportTag>::rank()``, which is the unique ID of the process in the parallel execution, and ranges from 0 to ``context<TransportTag>::size() - 1``.

A context object is instantiated by the processes, but different processes in a computation should instantiate contexts in the same order. The contexts instatiated, one per process, form some kind of *distributed context*, since they are *connected* to one another. Suppose we have the following code, executed by a certain number `P` of processes:

.. code-block:: gridtools

    auto context_ptrA = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto context_ptrB = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);

The instances of ``context_ptrA`` in the `P` processes form a distributed context, those instances are connected to one another. The communicators (see below) obtained by it can communcate to one another directly. The same for the communicators obtianed from ``context_ptrB``. Communicators from ``context_ptrA`` cannot communicate directly to communicators from ``context_ptrB``.

------------------------
Communicator
------------------------

A context generates and keeps communicators. A communicator can be seen representing the end-point of a communi/cation channel. These communication channels are obtained from the context. Communicators coming from different contexts cannot communicate with one another in any way, creating isolation of communications, which is useful for program composition and creating abstractions.

To get a communicator, the user calls

.. code-block:: cpp

    auto comm = context.get_communicator();

In order to keep the API simple, ``get_communicator`` will return a *new* communicator every time the function is invoked. The function is thread safe, and each thread of the computation can call it and obtain a unique communicator object. Anyway, a thread should call ``get_communicator`` only to get the communicator objects it needs and should *never* invoke ``get_communicator`` to retrieve a previously generated communicator. The user is *responsible* for keeping the communicators alive for the duration needed by the computation. ``get_communicator`` must be called once for each communicator instance needed.

Once communicator is obtained, it can be used to send message to, and receive from, other communicators obtained from the same distributed context, as explained in the previous Section.

Communicators can communicate with one-another by sending messages with tags. There are two types of message exchanges: `future based` and `call-back based`. The destination of a message is a ``rank``, that identifies a context, and a ``tag`` is used to match a receive on that rank.

Communicators do not direcly communicae to one-another, they rather send a message to a *context* and by using tag-matching the messages are delivered to the proper communicator. Communicators are mostly needed to increase concurrency, since different channels can be multiplexed or demultiplexed, depending on the characteristics of the transport layer.

For instance, when using the MPI transport layer, multiple ``Isends`` and ``Irecvs`` are issued by different communicators, and the concurrency is managed by the MPI runtime system that should be initialized with ``MPI_THREAD_MULTI`` option. In UCX we can exploit concurrency differently: each communicator has its private end-point, while the receives are all channeled through a single end-point on the process. This choice was dictated by benchmarks that showed this solution was the most efficient, and by using the communicator obstraction, the code did not need to change. The same code runs well with MPI and UCX transport layers, despite a different way of handling concurrency and addressing latency hiding.

.. note::

    While this leave the tag management to the user, the design was decided in order to avoid shifting the problem to managing tags in MPI, which are a scarce resource in some implementations (that is, the tags only allow the use of few bits). Identifying the communicators directly would have required predermining the number of bits to assign for the local identifyers and the bits used for rank identification, which can lead to potentially hard to catch bugs. We may revise this decision in the future.

Let's take a look at the main interfaces to exchange messages using communicators:

.. code-block:: gridtools

    template<typename Message>
    [[nodiscard]] future<void> send(const Message& msg, rank_type dst, tag_type tag);
    template<typename Message>
    [[nodiscard]] future<void> recv(Message& msg, rank_type src, tag_type tag);
    template<typename CallBack>
    request_cb_type send(message_type&& msg, rank_type dst, tag_type tag, CallBack&& callback)
    template<typename CallBack>
    request_cb_type recv(message_type&& msg, rank_type src, tag_type tag, CallBack&& callback)

The first function sends a message to a given rank with a given tag. The message can be any class with ``.data()`` member function returning a pointer to a region of memory of size ``.size()`` bytes. The function returns a future-like value that can be checked using ``.wait()`` to check that the message has been sent (the future does not guarantee the message has been delivered).

Similarly, the second function reveives into a message, with the same requirements as before, and returns a future that, when ``.wait()`` returns guarantee the message has been received.

The last two functions accept call-backs that are called either after the message has been sent (not necessarily delivered) or received, respectively. The message type, in this case cannot be arbitrary. The call-backs are a function taking three arguments: the message, the destination or the source, and the tag. The message is passed always by move, and the ownership of it is handled to the call-back. The call-back can, for instance, reuse the message in a receive recurvively, re-passing the message to the comunicator.

.. note::

    It would not make much sense to pass a message that cannot be moved (a message can have arbirary allocators that makes the move semantics not viable). To use the call-backs then, ``communicator::message_type`` must be used.

The send/recv functions accepting call-backs, also return request values that can be used to check if an operation succeded (PLEASE EXPLAIN WHY THIS IS NOT A FUTURE)

These four different function covers use-cases that covers more that simple halo-update operations, and these use-cases are explained in more detail in a separate Section.

------------------------
Communication Pattern
------------------------

In order to perform halo-update operations, the user needs to provide to |GHEX| information about the domain, the domain decomposition, the sizes of the halos and the accessing of the data. One of the most important aspects of |GHEX| is the choice of not imposing domain decomposition strateegy, that would have resulted in sub-optimal solutions that few users would have agreed with. So the user is providing descriptions of the above mentioned concepts as adaptors to their implementation choices. After all, all domain decomposed applications have to refer to similar information, even though the encoding of this information differs in all sort of details in different applications. The user of |GHEX| needs to provide standard functions that |GHEX| will call to gather/access the necessary information, and these functions form a thin layer that interfaces the specific domain decomposition implementation and |GHEX|. |GHEX| developers are providing some components directly, in order to facilitate the interfacing in the most common cases.





------------------------
Communication Object
------------------------
