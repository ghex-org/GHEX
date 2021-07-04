# -*- coding: utf-8 -*-
#
# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union, Any, Dict, Sequence, Tuple, Literal
import copy

from .index_space import IndexSpace, UnitRange, union, ProductSet

# todo: subgrid to be used for stencils in libraries

class PeriodicImageId:
    """Identifier of a periodic image."""
    id_: Any  # index space id
    dir: Tuple[int, int, int]  # direction, e.g. (-1, -1, 0) is north-west

    def __init__(self, id_, dir):
        self.id_ = id_
        self.dir = dir


class DimSymbol:
    """Symbol representing a dimension, e.g. I or I - 1/2."""
    name: str
    offset: float

    def __init__(self, name, offset=0):
        self.name = name
        self.offset = offset

    def __eq__(self, other):
        return self.name == other.name and self.offset == other.offset

    def __hash__(self):
        return hash((self.name, self.offset))

    def __add__(self, offset):
        assert self.offset == 0 or self.offset+offset == 0
        return DimSymbol(self.name, self.offset + offset)

    def __sub__(self, other):
        return self+(-other)

    def __repr__(self):
        if self.offset == 0:
            return self.name
        elif self.offset > 0:
            return f"{self.name} + {self.offset}"
        elif self.offset < 0:
            return f"{self.name} - {abs(self.offset)}"
        raise RuntimeError()


class Grid3D:
    index_spaces: Dict[Any, IndexSpace]

    index_conventions: Dict[Any, Dict[Any, Sequence[int]]]

    periodic_images: Dict[PeriodicImageId, IndexSpace]

    class Dims:
        # just use symbols to identify the axis/dimensions
        I = DimSymbol("I")
        J = DimSymbol("J")
        K = DimSymbol("K")


    def __init__(self, Nx, Ny, Nz, *, bcx=1, bcy=1, bcz=1):
        # bc == 0 -> prescribed, bc == 1 -> periodic
        assert(Nx >= 3 and Ny >= 3 and Nz >= 3)

        I, J, K = type(self).Dims.I, type(self).Dims.J, type(self).Dims.K

        # todo: add periodic layers to the left
        ijk = self._init_index_space((Nx, Ny, Nz), (bcx, bcy, bcz), (1, 1, 1), (False, False, False))
        ijk_sx = self._init_index_space((Nx + 1, Ny, Nz), (bcx, bcy, bcz), (1, 0, 0), (True, False, False))
        ijk_sy = self._init_index_space((Nx, Ny + 1, Nz), (bcx, bcy, bcz), (0, 1, 0), (False, True, False))
        ijk_sz = self._init_index_space((Nx, Ny, Nz + 1), (bcx, bcy, bcz), (0, 0, 1), (False, False, True))

        ij = self._init_index_space((Nx, Ny, 1), (bcx, bcy, bcz), (1, 1, 0), (True, True, False))

        self.index_spaces = {
            (I, J, K): ijk, # unstaggered
            (I - 1 / 2, J, K): ijk_sx, # staggered-x
            (I, J - 1 / 2, K): ijk_sy, # staggered-y
            (I, J, K - 1 / 2): ijk_sz, # staggered-z

            # pseudo two-dimensional index space for stencils running only in the horizontal
            (I, J): ij
        }

        self.index_conventions = {
            (I, J, K): {
                (I, J, K): (0, 0, 0),
                (I - 1 / 2, J, K): (-1, 0, 0),
                (I, J - 1 / 2, K): (0, -1, 0),
                (I, J, K - 1 / 2): (0, 0, -1)
            },
            (I - 1 / 2, J, K): {(I - 1 / 2, J, K): (0, 0, 0), (I, J, K): (0, 0, 0)},
            (I, J - 1 / 2, K): {(I, J - 1 / 2, K): (0, 0, 0), (I, J, K): (0, 0, 0)},
            (I, J, K - 1 / 2): {(I, J, K - 1 / 2): (0, 0, 0), (I, J, K): (0, 0, 0)},

            (I, J): {(I, J): (0, 0, 0)}
        }

        self._init_periodic_images()

    def _init_index_space(self, shape: Tuple[int, int, int], bcs: Tuple[int, int, int],
                          periodic_layers: Tuple[int, int, int], has_boundary: Tuple[bool, bool, bool]):
        def label_from_dir(dir_):
            labels = [
                ["south", "north"],
                ["west", "east"],
                ["bottom", "top"]]
            permuted_dir = (dir_[1], dir_[0], dir_[2])  # meridional direction comes first
            return "-".join(labels[l][0 if dir_l == -1 else 1] for l, dir_l in enumerate(permuted_dir) if dir_l != 0)

        def slices_from_dir(dir_):
            def slices_from_dir_l(dir_l, size):
                dir_l_to_slice = (slice(0, 1), slice(1, -1), slice(-1, None))
                if size == 1:
                    assert dir_l == 0
                    return slice(0, None)
                return dir_l_to_slice[dir_l + 1]

            return tuple(slices_from_dir_l(dir_l, size) for dir_l, size in zip(dir_, shape))

        def is_interior(dir_):
            return all(not has_boundary[dim] or dir_l in ([-1, 0] if bc == 1 else [0]) for dim, (dir_l, bc) in enumerate(zip(dir_, bcs)))

        def is_prescribed(dir_):
            return any(has_boundary[dim] and dir_l in [-1, 1] and bc == 0 for dim, (dir_l, bc) in enumerate(zip(dir_, bcs)))

        def is_periodic(dir_):
            return any(has_boundary[dim] and dir_l == 1 and bc == 1 for dim, (dir_l, bc) in enumerate(zip(dir_, bcs)))

        def is_corner(dir_):
            return not any(dir_l == 0 for dir_l in dir_)

        def is_part_of(dir_, candidate_dir):
            return all(candidate_dir_l == dir_l or candidate_dir_l == 0 for candidate_dir_l, dir_l in zip(candidate_dir, dir_))

        index_space = IndexSpace.from_sizes(*shape)

        empty_set = UnitRange(0, 0) * UnitRange(0, 0) * UnitRange(0, 0)

        interior = empty_set
        prescribed_boundary = empty_set
        periodic_boundary = empty_set

        prescribed_boundary_parts = {dir_: empty_set for dir_ in UnitRange(-1, 2) * UnitRange(-1, 2) * UnitRange(-1, 2) if is_prescribed(dir_) and not is_corner(dir_)}

        for dir_ in UnitRange(-1, 2) * UnitRange(-1, 2) * UnitRange(-1, 2):
            # skip if any non-center direction has a size of 1, e.g. top and bottom for pseudo horizontal
            if any(dir_l != 0 and size==1 for dir_l, size in zip(dir_, shape)):
                assert not any(dir_l != 0 and size == 1 and hb for dir_l, size, hb in zip(dir_, shape, has_boundary))
                continue

            part = index_space.subset["definition"][slices_from_dir(dir_)]

            if is_interior(dir_):
                interior = union(interior, part, simplify=False)

            if is_prescribed(dir_):
                prescribed_boundary = union(prescribed_boundary, part, simplify=False)
                if is_corner(dir_):
                    index_space.add_subset(("prescribed_boundary", label_from_dir(dir_)), part)
                for candidate_dir in prescribed_boundary_parts.keys():
                    if is_part_of(dir_, candidate_dir):
                        prescribed_boundary_parts[candidate_dir] = union(prescribed_boundary_parts[candidate_dir],
                                                                         part)

            if is_periodic(dir_):
                periodic_boundary = union(periodic_boundary, part, simplify=False)

        index_space.add_subset("interior", interior)
        for dir_, part in prescribed_boundary_parts.items():
            index_space.add_subset(("prescribed_boundary", label_from_dir(dir_)), part)
        index_space.add_subset("prescribed_boundary", prescribed_boundary)
        index_space.add_subset("periodic_boundary", periodic_boundary)
        index_space.add_subset("physical", index_space.subset["definition"].without(index_space.subset["periodic_boundary"]))
        index_space.add_subset("periodic_layers", index_space.subset["physical"].extend(
            *(el if hb and bc == 1 else 0 for hb, bc, el in zip(has_boundary, bcs, periodic_layers))).without(index_space.subset["physical"]))

        # used for point-wise stencils, might change in the future
        index_space.add_subset("covering", index_space.covering)

        # used for stencils expecting a west+east, north+south, bottom+top neighbor, e.g. centered diff
        # todo: more meaningful name
        index_space.add_subset(("interior", (1, 0, 0)), index_space.covering.simplify().extend(-1, 0, 0))
        index_space.add_subset(("interior", (0, 1, 0)), index_space.covering.simplify().extend(0, -1, 0))
        if shape[2] != 1: # not meaningful for pseudo horizontal
            index_space.add_subset(("interior", (0, 0, 1)), index_space.covering.simplify().extend(0, 0, -1))

        assert union(interior, prescribed_boundary, periodic_boundary) == index_space.subset["definition"]
        assert union(index_space.subset["physical"], periodic_boundary) == index_space.subset["definition"]

        return index_space

    def _init_periodic_images(self):
        self.periodic_images = {}

        original_index_spaces = copy.deepcopy(self.index_spaces)
        for id_, index_space in original_index_spaces.items():
            self.periodic_images[id_] = {}

            for dir in UnitRange(-1, 2) * UnitRange(-1, 2) * UnitRange(-1, 2):
                if dir == (0, 0, 0):
                    continue

                offsets = tuple(dir_i * l for dir_i, l in zip(dir, index_space.subset["physical"].shape))

                image_id = PeriodicImageId(id_, dir)
                image_indices = index_space.bounds.translate(*offsets)
                image_index_space = IndexSpace(image_indices)
                image_index_space.add_subset("periodic_layers",
                                             index_space.subset["physical"].translate(*offsets).intersect(
                                                 index_space.subset["periodic_layers"]))

                if image_index_space.subset["periodic_layers"].empty:
                    continue

                self.index_spaces[image_id] = image_index_space
                self.periodic_images[id_][image_id] = image_index_space
                self.index_conventions[image_id] = {
                    image_id: (0, 0, 0),
                    id_: (0, 0, 0)
                }

            # ensure periodic images cover entire boundary
            assert (len(self.periodic_images[id_]) == 0 and index_space.subset["periodic_layers"].empty) or union(
                *(image_index_space.subset["periodic_layers"] for image_index_space in
                  self.periodic_images[id_].values())) == index_space.subset["periodic_layers"]