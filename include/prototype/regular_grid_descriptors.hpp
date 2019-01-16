/** @file We describe here the way of representing the regular grids
    (multi-dimensional arrays) could be decsribed in a general way, so
    that a back-end can take this information to pack-unpack or do
    whatever with this data.

    Concepts and Notation:
    ======================

    - A processing-grid (for regural grids) is a $D$-dimensional mesh. A
      node of the mesh is identified by the usual tuple of indices
      $(p_0, ..., p_{D-1})$. The mesh may or not be toroidal in some
      dimension.

    - A gird (in this context) is a $d$-dimensional array of values,
      where $d \ge D$.

    (Lower dimensionalities can be simulated by setting the sizes of
    the extra dimensions to 1)

    - $\eta \in \{-1, 0, 1\}^D$ identifies a neighbor of a mesh-node
      in the computing grid. Given ane identifier of a node $x = (x_0,
      x_1, \ldots, x_{D-1}$, the neighbor identifier is simply
      $x+\eta$.

    - The number od neighors in a D-dimensional processing-grid is
      $3^D-1$ (easy to proove considering the number of possible
      neighbors).

    - The number of principal neighbors (the ones along main
      directions) is $2D$ (same way of proofing as before).

    - The grid has a layout map (we consider only affine layouts so
      far). If $S \in N^d$ is the tuple of strides, then the offset of
      an element is at $\sum_{i=0}^{d-1} S_ix_i$, where $x_i$ are the
      coordinate indices of the elements and the strides are listed in
      the same order as the indices of the coordinates. It is always
      possible to re-order the coordinates and strides to be so that
      the first $D$ indices correspond to the dimensions that are
      partitioned along the processing-grid. We use this convention in
      what follows, but the code will make sure the permutations are
      accounted in the proper way.

    - A halo descriptor is a tuple of 4 integer values indicating, in
      a specific dimension, the halo in the minus direction, the halo
      in the plus direction, the begin and end of the core region.

    - A grid is then partitioned into $3^D$ regions corresponding to
      the halo descriptors in each dimension that is
      partitioned. There are $D$ dimensions that are partitioned. The
      regions can be identified using a tuple $\eta \in \{-1, 0,
      1\}^D$, exactly as the neighbors. In fact they identify the
      elements to be shipped to the neighbors. They correspond to the/
      coordinate indices of the neighbors, if we take the central part
      to be $\eta = (0, \ldots, 0) \in Z^D$. The dimensions being
      partitioned are the first $D$ dimensions of the $d$ dimensional
      data.

    - Let us introduce the range concept. A range concept is simply an
      integer interval (we can assume it is semi-open like the typical
      C++ pair of iterators). A range $R = [a, b)$, where $b>=a$. If
      $b=a$ then the range is empty.

    - To indicate a region of a grid we can use ranges, in the same
      way it is done in Matlab or Python. If $A$ is the data of a
      grid, then $A(R_0, R_1, \ldtos, R_{D-1})$. A single index $i$
      can be interpreted ad a range as $[i, i+1)$. A range that takes
      all the elements in one dimension is indicated as $:$.

    - A halo descriptor $H$ can provide information about the ranges
      used for data to be sent and data to be received. We will focus
      on the data to be sent, which is the interior, but it will
      probably not be noticed in the discussion in this comment.

    - We can index the range of a halo descriptor using the same
      indices used in the $\eta$ tuples. So that given a halo
      descriptor $H$, $H^{-1}$, $H^0$, and $H^1$ should have an
      obvious meaning (this is abuse of notation since a halo
      descriptor is told to be a tuple, but the indices there actually
      are not accessing the elements of the tuple but rather the
      ranges associated with them - in this case for data to be sent).

    - A iteration space is the Cartesian product of ranges.

    - Given the identifier of a neighbor $\eta$, the iteration space
      containing the elements to be sent to that neighbor is specified
      as:

      $$(H_0^{\eta_0}, H_1^{\eta_1}, \ldtos, H_{D-1}^{\eta_{D-1}}, :, \ldots)$$

      Where $H_i$ is the halo descriptor for dimension $i$. The $:$
      will take all the other dimensions in full until we cover all
      the $d$ dimensions of the grid elements.
 */

#include <common/layout_map.hpp>
#include "./halo_descriptor.hpp"
