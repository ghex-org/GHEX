.. include:: ../defs.hrst

.. _scope:

============================
Scope and objectives of GHEX
============================

|GHEX| is a C++ library to perform halo-update operations in mesh/grid applications in modern HPC
architecures. Domain decomposition refers to the technique that applications use to distribute the
work across different processes when numerically solving differential equations. For instance, see
the next figure for a schematic depiction of this idea.

.. figure:: figures/domain_decomp.png
    :width: 600px
    :align: center
    :alt: This should not be visible
    :figclass: align-center

    Example of Domain Decomposition. The physical domain, on the left, is split into 4 different
    sub-domains. To allow for the computation to progress without having to remotely access every
    single element that resides on a remote process, the application uses **ghost regions**, or
    **halos** (in pink), that need to be updated whenever the computation requires accesss to that
    region.  The blue arrows show the communication pattern, also referred in the manual as
    **halo-update**, or **halo-exchange** operation.

.. figure:: figures/symmetric.png
    :width: 300px
    :align: center
    :alt: This should not be visible
    :figclass: align-center

    Traditional distributed memory data distribution: one process (MPI rank) is responsible for one
    sub-domain of the decomposed domain.

.. figure:: figures/oversubscription.png
    :width: 300px
    :align: center
    :alt: This should not be visible
    :figclass: align-center

    Each process in a node can manage more than one sub-domain, for instance to achieve load
    balancing (over-subscription).

.. figure:: figures/multi_threads.png
    :width: 300px
    :align: center
    :alt: This should not be visible
    :figclass: align-center

    The over-subscription can be a tool to improve the parallelization in each process by running
    computations in independent threads.

.. figure:: figures/hybrid.png
    :width: 300px
    :align: center
    :alt: This should not be visible
    :figclass: align-center

    Using accelerators is certainly one of the most attractive options for running HPC applications
    on current and near future platforms. The execution can be symmetric, or hybrid.

The objective of |GHEX| is to enable halo-update operations

    - for traditional domain decomposed distributed memory applications (i.e., one domain per node)
      either on CPUs or GPUs

    - for applications applying over-subscription on a node, either for latency hiding or for
      exploiting multi-threading

    - for application exploiting hybrid systems (nodes with multiple address spaces and multiple
      computing devices)

    - regardless of the specific representation of the grid/mesh (by using *adaptors*)

    - on architectures that provide access to transport mechanisms other than MPI (e.g., ``UCX`` and
      ``Libfabric``) whose performance may be higher

In order to accomplish all of the above, the interfaces to |GHEX| requires a non trivial amount of
work on the user side.  The reward for this work is: portability to multiple architectures, with and
without accelerators, with the possibility to exploit native transport mechanisms. Depending on the
complexity of the application, a user can easily adapt it to use different number of threads, or
different types of threads. |GHEX| can accommodate these requirements quite flexibly.

----------
Features
----------

|GHEX| employs a number of different communication strategies to improve performance of information
exchange. In particular, the following features are available:

    - off-node access: use of a buffered communication for remote neighbors and reduction of the
      amount of messages by coallescing data with the same destination into larger chunks
    
    - in-node access: taking advantage of direct memory access within a shared memory region when
      run with multiple threads (native) or when run with multiple processes (through *xpmem*)

    - latency hiding: computation - communication overlap is possible through an explicit
      future-like API

    - cache friendliness: structured grids are traversed in a cache-friendly way to avoid cache
      misses when serializing/deserializing data (for remote connections) and when directly
      accessing memory (for node-local connections)

    - avoid serialization: certain types of unstructured meshes can be configured such that
      serialization on one side of the communication can be avoided

    - heterogeneity of data: different types of data (with different neighbor regions) may be
      exchanged in one go

    - GPU accelerators: carefully tuned serialization kernels which exploit the bandwidth and
      asynchronicity of execution


--------------------------
Type of interfaces
--------------------------

|GHEX| has a layered strctured. The user can enter at different layers, depending on their needs.
The highest level is the ``halo exchange`` level, where the user instructs |GHEX| to take a mesh or
grid representation and produce a ``communication pattern`` to then perform the halo update
operations.

In order to enable all the previously mentioned features, like oversusubscription, alternate
transport layers and hybrid computations, the steps to create a pattern and use it to communicate
can seem overly complicated. While we are working on shortening the number of steps to take for
simple cases, more complex cases seem to require these complications, and hence they cannot be
avoided. As we expect applications to become more and more complex in the future, we think the use
cases in which the interfaces provided by |GHEX| will increase and overshadow the traditional
distributed memory applications simply based on MPI.

The main concepts needed by |GHEX| are:

    - **Context** : Provides information to the application about network connectivity. A computing
      node, that can be identified with a process (or an MPI rank, for instance) needs some
      information about how it is connected to other processes.

    - **Communicator**: Represents an end-point of a communication, and is obtained from the
      context.

    - **Communication Pattern**: Provides application-specific neighbor conntectivity information
      (encodes exposed halos for sending and receiving, for instance)

    - **Communication Object**: Is responsible for executing communications by tying together
      *Communication Pattern*, *Communicator*, and user-data.

.. The interfaces for *context* and *communicator* are slightly richer than what is required by
   *pure* halo-update operations. We designed contexts and communicators in order to cope with more
   complex scenarious, where the domain decomposition more dynamic. The interfaces for communication
   patterns and communication objects are advantageous only

------------------------
Context
------------------------

The context manages the underlying transport layer. Among its tasks are initialization, connectivity
setup, commpunication end-point management, network topology exploration. It is the first entity
that is created and the last that is being destroyed.  Contexts maintain state information that the
*communicators* need in order to function. In other words, communicators (presented below) cannot
outlive the contexts from where they were obtained.
A context can be compared to a richer version of an MPI-Communicator, such as *MPI_COMM_WORLD*, and
is encapsulated in a template class of type **communicator**.

|GHEX| assumes the availabilty of an MPI library implementation on the platform where it is run.
While this is not strictly needed conceptually, the vast majority of HPC applications rely on MPI
(for instance by using PGAS languages or runtime-systems such as HPX) and, thus, we can take
advantage of the infrastructre provided through MPI for our implementation. This simplifies creation
of contexts and collection of information about which processes participate in the computation.
Therefore, the context requires an MPI Communicator as runtime argument, in addition to other
possible transport specific arcguments.  The passed MPI Communicator will be cloned by the context.
For this reason the context should be destroyed before the call to ``MPI_Finalize``.

Contexts are constructed for a specific *transport layer*. The available tranport layers are: MPI,
UCX, and Libfabric. The transport layer is represented by a *tag* passed as template argument to the
context type.

In order to guarantee uniform initilization of the contexts, which are highly platform dependent,
the user should not call the constructor of the context direcly, but instead the use should use a
factory, which returns a ``std::unique_ptr`` to context object instantiated with the proper
transport layer. The syntax looks like this:

.. code-block:: gridtools

    #ifdef USE_UCX
    using transport = gridtools::ghex::tl::ucx_tag;
    #else
    using transport = gridtools::ghex::tl::mpi_tag;
    #endif

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);

In the above example the use of ``#ifdef`` is an example of how code can be made portable by minimal
use of macros, and a recompilation.

From the context, additional information can be obtained, such as the
``context<TransportTag>::rank()``, which is the unique ID of the process in the parallel execution,
and ranges from 0 to ``context<TransportTag>::size() - 1``.

A context object is instantiated by the processes, but different processes in a computation should
instantiate contexts in the same order. The contexts instatiated, one per process, form a kind of
*distributed context*, since they are *connected* to one another. Suppose we have the following
code, executed by a certain number `P` of processes:

.. code-block:: gridtools

    auto context_ptrA = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto context_ptrB = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);

The instances of ``context_ptrA`` in the `P` processes form a distributed context, those instances
are connected to one another. The communicators (see below) obtained from it can communcate to one
another directly. The same for the communicators obtianed from ``context_ptrB``. Communicators from
``context_ptrA`` cannot communicate directly to communicators from ``context_ptrB``.

.. note::

   **Thread safety:** A context shall be created in a serial part of the code. Usage of a context is
   thread-safe.

------------------------
Communicator
------------------------

A context generates and keeps communicators. A communicator represents the end-point of a
communication and is obtained from a context.  Communicators coming from different contexts cannot
communicate with one another in any way, creating isolation of communications, which is useful for
program composition and creating abstractions.

To get a communicator, the user calls

.. code-block:: cpp

    auto comm = context_ptr->get_communicator();

In order to keep the API simple, ``get_communicator`` will return a *new* communicator every time
the function is invoked. The function is thread safe, and each thread of the computation can call it
and obtain a unique communicator object. A thread should call ``get_communicator`` only to get the
communicator objects it needs and should *never* invoke ``get_communicator`` to retrieve a
previously generated communicator. The user is *responsible* for keeping the communicators alive for
the duration needed by the computation. ``get_communicator`` must be called once for each
communicator instance needed.

.. note::

   If an application is only concerned with avaialble high-level halo-exchange facilities
   (implemented in the *patterns* and *communication objects*, see below), a communicator is never
   directly used by the application.  The API of the communicator explained below may however be
   used to implement more complex scenarios where the applcation does not follow the typical bulk
   halo exchange strategy.

Once a communicator is obtained, it can be used to send messages to, and receive from, other
communicators obtained from the same distributed context, as explained in the previous Section.

Communicators can communicate with one-another by sending messages with tags. There are two types of
message exchanges: `future based` and `call-back based`. The destination of a message is a ``rank``,
that identifies a process/context, and a ``tag`` is used to match a receive on that rank.

Communicators do not direcly communicate to one-another, they rather send a message to a *context*
and by using tag-matching the messages are delivered to the proper communicator. Communicators are
mostly needed to increase concurrency, since different channels can be multiplexed or demultiplexed,
depending on the characteristics of the transport layer.

For instance, when using the MPI transport layer, multiple ``Isends`` and ``Irecvs`` are issued by
different communicators, and the concurrency is managed by the MPI runtime system that should be
initialized with ``MPI_THREAD_MULTI`` option. In UCX we can exploit concurrency differently: each
communicator has its private end-point, while the receives are all channeled through a single
end-point on the process. This choice was dictated by benchmarks that showed this solution was the
most efficient. The same code runs well with MPI and UCX transport layers, despite a different way
of handling concurrency and addressing latency hiding.

.. note::

    While this leaves the tag management to the user, it avoids unnecessary restrictions for the
    avaialbe tags, which are a scarce resource in some implementations (that is, the tags only allow
    the use of few bits). Identifying the communicators directly would have required predetermining
    the number of bits to assign for the local identyfiers and the bits used for rank
    identification, which can lead to potentially hard to catch bugs. We may revise this decision in
    the future.

Let's take a look at the main interfaces to exchange messages using communicators:

.. code-block:: gridtools

    template<typename Message>
    future<void> send(const Message& msg, rank_type dst, tag_type tag);
    
    template<typename Message>
    future<void> recv(Message& msg, rank_type src, tag_type tag);
    
    template <typename Message, typename Neighs>
    std::vector<future<void>> send_multi(Message& msg, const Neighs& neighs, tag_type tag);

The first function sends a message to a given rank with a given tag. The message can be any class
with ``.data()`` member function returning a pointer to a region of *contiguous* memory of
``.size()`` elements. Additionally, the message type has to expose a typedef ``value_type``
identifiying the type of element that is sent. The function returns a future-like value that can be
checked using ``.wait()`` to check that the message has been sent (the future does not guarantee the
message has been delivered). This variant **does not take ownership of the message** (but refers to
the address of the memory only) an the user is responsible to keep the message alive until the
communication has finished.
                
Similarly, the second function reveives into a message, with the same requirements as before, and
returns a future that, when ``.wait()`` returns guarantee the message has been received.

The third function conveniently allows to send the same message to multiple neighbors, and returns a
vector of futures to wait on.


The communicator also has a secondary API where a user-defined callback may be registered, which is
called upon completion of the communication:

.. code-block:: gridtools

    template<typename Message, typename CallBack>
    request_cb send(Message&& msg, rank_type dst, tag_type tag, CallBack&& callback);
                
    template<typename Message, typename CallBack>
    request_cb recv(Message&& msg, rank_type src, tag_type tag, CallBack&& callback);

    template <typename Message, typename Neighs, typename CallBack>
    std::vector<request_cb> send_multi(Message&& msg, Neighs const &neighs, tag_type tag, const CallBack& callback);

Here, the functions accept callbacks that are called either after the message has been sent (not
necessarily delivered) or received, respectively. The requirements on the message type stay the same
as before, however, the type of the message will be type erased in the process and the callback must
expose the following signature:

.. code-block:: gridtools

   void operator()(message_type msg, rank_type rank, tag_type tag);

where ``message_type`` is a class defined by |GHEX| fulfilling the above message contract with
``value_type`` being ``unsigned char``. Thus, the ``msg`` object provides access to the same data as the
message passed to |GHEX| originally, reinterpreted as raw memory.

|GHEX| makes a distinction based on the type of the message which is passed to the functions above:

    - if the message is moved in (has r-value reference type), the communicator **takes ownership of
      the message** and keeps the message alive internally. The user is free to delete or re-use the
      (moved) message directly after calling the function.

    - if the message is passed by l-value reference, the same requirements apply as above: the
      communicator **does not take ownership of the message** and the user is responsible to keep
      the message alive until the communication has finished.

.. note::

   When the callback is invoked, the message ownership is again passed to the user, i.e. the user is
   free to delete or re-use the message inside the callback body. However, when the user re-submits
   a further callback based communication through the above API from with the callback (recursive
   call) , the message must passed by r-value reference (through ``std::move``).

.. note::

   Since GHEX relies on move semantics of the message internally, the message type must not
   re-allocate memory during move construction and the memory address of the data must remain
   constant during the communication.
                
The send/recv functions accepting call-backs, also return request values that can be used to check
if an operation succeded. However, these requests may not be used like futures: they cannot be
waited upon. Instead, to progress the communication and ensure the completion, the communicator has
to be progressed explicitly:

.. code-block:: gridtools

    progress_status progress();

This function progresses the transport layer and returns a status object. The status object can be
queried for the number of progressed send and receive callbacks.

A third API is provided for messages wrapped in a ``shared_ptr``:

.. code-block:: gridtools

    template<typename Message, typename CallBack>
    request_cb send(std::shared_ptr<Message> shared_msg_ptr, rank_type dst, tag_type tag, CallBack&& callback);

    template<typename Message, typename CallBack>
    request_cb recv(std::shared_ptr<Message> shared_msg_ptr, rank_type src, tag_type tag, CallBack&& callback);

Here, the owenership is obviously shared between the user and |GHEX|.


.. note::

   **Thread safety:** A communicator is thread-compatible, i.e. it is created per thread. One must
   not use the same communicator from more than one thread.

------------------------
Communication Pattern
------------------------

In order to perform halo-update operations, the user needs to provide to |GHEX| information about the domain, the domain decomposition, the sizes of the halos and the accessing of the data. One of the most important aspects of |GHEX| is the choice of not imposing domain decomposition strateegy, that would have resulted in sub-optimal solutions that few users would have agreed with. So the user is providing descriptions of the above mentioned concepts as adaptors to their implementation choices. After all, all domain decomposed applications have to refer to similar information, even though the encoding of this information differs in all sort of details in different applications. The user of |GHEX| needs to provide standard functions that |GHEX| will call to gather/access the necessary information, and these functions form a thin layer that interfaces the specific domain decomposition implementation and |GHEX|. |GHEX| developers are providing some components directly, in order to facilitate the interfacing in the most common cases.





------------------------
Communication Object
------------------------
