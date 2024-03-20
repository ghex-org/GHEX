#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
from __future__ import annotations
from copy import copy
import functools
import itertools
import math
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, Literal, Sequence, Tuple, Union, TypeAlias

    integer: TypeAlias = Union[int, Literal[math.inf], Literal[-math.inf]]


def is_integer_like(val):
    return isinstance(val, int) or val == math.inf or val == -math.inf


class Set:
    pass


class IntegerSet(Set):
    """A set containing integers."""

    def empty_set(self):
        return UnitRange(0, 0)

    def universe(self):
        return UnitRange(-math.inf, math.inf)

    def shrink(self, arg: Union[int, Tuple[int, int]]):
        if isinstance(arg, int):
            arg = (arg, arg)

        return self.extend(tuple(-v for v in arg))

    @staticmethod
    def primitive_type():
        return UnitRange

    @staticmethod
    def union_type():
        return UnionRange


class UnitRange(IntegerSet):
    """Range from `start` to `stop` with step size one."""

    start: integer
    stop: integer

    def __init__(self, start: integer, stop: integer):
        assert stop >= start

        self.start = start
        self.stop = stop

        # canonicalize
        if self.empty:
            self.start = 0
            self.stop = 0

    @property
    def size(self) -> int:
        """Return the number of elements."""
        assert self.start <= self.stop
        return self.stop - self.start

    @property
    def empty(self) -> bool:
        """Return if the range is empty"""
        return self.start >= self.stop

    @property
    def bounds(self) -> UnitRange:
        """Smallest range containing all elements. In this case itelf."""
        return self

    def __eq__(self, other: Any) -> bool:
        """Return if `self` and `other` contain the same elements."""
        if isinstance(other, Set):
            return self.issubset(other) and other.issubset(self)

        return False

    def __contains__(self, arg: integer) -> bool:
        """Is (the integer-like) `arg` element of this range"""
        assert is_integer_like(arg)

        return self.start <= arg < self.stop

    def issubset(self, arg: Set) -> bool:
        """Return if `self` is a subset of `arg`."""
        return arg.complement(simplify=False).intersect(self).empty

    def __getitem__(self, arg: Union[int, slice]):
        """Return element(s) at relative index (slice)"""
        if isinstance(arg, slice):
            assert arg.step in [1, None]

            if arg.start is None:
                start = self.start
            elif arg.start < 0:
                start = self.stop + arg.start
            elif arg.start >= 0:
                start = self.start + arg.start

            if arg.stop is None:
                stop = self.stop
            elif arg.stop < 0:
                stop = self.stop + arg.stop
            elif arg.stop >= 0:
                stop = self.start + arg.stop

            return UnitRange(start, stop)
        elif isinstance(arg, int):
            result = (self.start if arg >= 0 else self.stop) + arg

            if result not in self:
                raise IndexError()

            return result

        raise ValueError(f"Invalid argument `{arg}`")

    def __str__(self):
        return f"UnitRange({self.start}, {self.stop})"

    def __mul__(self, other: UnitRange):
        """Cartesian product of `self` with `other`"""
        if isinstance(other, ProductSet):
            return ProductSet(self, *other.args)
        elif isinstance(other, UnitRange):
            return ProductSet(self, other)
        elif isinstance(other, UnionRange) or isinstance(other, UnionCartesian):
            return union(
                *(self * arg for arg in other.args),
                disjoint=other.disjoint,
                simplify=False,
            )

        raise NotImplementedError()

    def __iter__(self):
        """Return an iterator over all elements of the set"""
        return range(self.start, self.stop).__iter__()

    def __hash__(self):
        return hash((self.start, self.stop))

    def intersect(self, other: Set):
        """Return intersection of `self` with `other`"""
        if isinstance(other, UnitRange):
            start = max(self.start, other.start)
            stop = max(start, min(self.stop, other.stop))
            return UnitRange(start, stop)
        elif isinstance(other, UnionRange):
            return other.intersect(self)

        raise NotImplementedError()

    def without(
        self,
        other: Union[UnitRange, UnionRange],
        *tail: Union[UnitRange, UnionRange],
        simplify=True,
    ):
        """Return range containing all elements in self, but not in other, i.e. the complement of `other` with `self`"""
        result = other.complement(self, simplify=simplify)

        return result if len(tail) == 0 else result.without(*tail, simplify=simplify)

    def complement(self, other: Union[None, Set] = None, simplify=True):
        """Return the complement of self in other."""
        result = union(
            UnitRange(-math.inf, self.start),
            UnitRange(self.stop, math.inf),
            disjoint=True,
            simplify=simplify,
        )

        return result.intersect(other) if other else result

    def union(self, *others: Set):
        """Return the union of `self` with `other`"""
        return union(self, *others)

    def extend(self, arg: Union[int, Tuple[int, int]]):
        if self.empty:
            raise ValueError("The empty set can not be extended.")

        if isinstance(arg, int):
            arg = (arg, arg)

        return UnitRange(self.start - arg[0], self.stop + arg[1])

    def translate(self, arg: int):
        """Return a range shifted by arg."""
        if self.empty:
            raise ValueError("The empty set can not be translated.")

        return UnitRange(self.start + arg, self.stop + arg)

    def as_tuple(self):
        return (self.start, self.stop)

    def __repr__(self):
        return f"UnitRange({self.start}, {self.stop})"


def union(*args: Set, simplify=True, disjoint=False):
    assert len(args) > 0
    empty_set = args[0].empty_set()
    union_type = args[0].union_type()

    # remove empty sets
    args = [arg for arg in args if not arg.empty]

    # flatten
    args = functools.reduce(
        operator.add,
        [list(arg.args) if isinstance(arg, UnionMixin) else [arg] for arg in args],
        [],
    )

    if len(args) == 0:
        return empty_set
    if len(args) == 1:
        return args[0]

    result = union_type(*args, disjoint=disjoint)

    return result.simplify() if simplify else result


from functools import reduce


def intersect(a, *args: Set):
    if len(args) == 0:
        return a
    return reduce(lambda a, b: a.intersect(b), args, a)


class UnionMixin:
    # todo: abstract bounds property
    args: Sequence[UnitRange]

    disjoint: bool

    def __init__(self, *args, disjoint=False):
        assert len(args) > 1

        if not all(isinstance(arg, self.primitive_type()) for arg in args):
            raise ValueError(
                "Union can only be constructed from primitive sets. Use `union` instead."
            )

        if any(arg.empty for arg in args):
            raise ValueError("Empty set given as argument. Use `union` instead.")

        self.args = args
        self.disjoint = disjoint

    @property
    def size(self) -> int:
        overlap = 0
        if not self.disjoint:
            for i, arg1 in enumerate(self.args):
                for j, arg2 in enumerate(self.args[i + 1 :], start=i + 1):
                    overlap += arg1.intersect(arg2).size

        return functools.reduce(operator.add, (arg.size for arg in self.args)) - overlap

    @property
    def empty(self) -> bool:
        return all(arg.empty for arg in self.args)

    def union(self, *args: Sequence[Union[UnitRange, UnionRange]]):
        return union(*self.args, *args)

    def without(self, *others: Sequence[Set], simplify=True):
        return union(
            *(s.without(*others, simplify=simplify) for s in self.args),
            disjoint=self.disjoint,
            simplify=simplify,
        )

    def complement(self, other: Union[None, Set] = None, simplify=True):
        if not other:
            other = self.universe()

        return other.without(*self.args, simplify=simplify)

    def intersect(self, other: Set):
        return union(
            *(s.intersect(other) for s in self.args),
            disjoint=self.disjoint,
            simplify=False,
        )

    def translate(self, *args):
        return union(*(s.translate(*args) for s in self.args), disjoint=self.disjoint)

    def simplify(self):
        if not self.disjoint:
            return self.make_disjoint()

        return self

    def make_disjoint(self):
        if self.disjoint:
            return self

        args = list(self.args)
        for i, arg1 in enumerate(args):
            for j, arg2 in enumerate(args[i + 1 :], start=i + 1):
                args[j] = arg2.without(arg1, simplify=False)

        return union(*args, disjoint=True, simplify=False)

    def __iter__(self):
        for arg in self.args:
            for p in arg:
                yield p

    def __eq__(self, other):
        if isinstance(other, Set):
            return self.issubset(other) and other.issubset(self)

        return False

    def issubset(self, other):
        """Return if `self` is a subset of `other`."""
        return other.complement(simplify=False).intersect(self).empty

    def __contains__(self, arg: integer):
        return any(arg in comp for comp in self.args)

    def __repr__(self):
        return "union(" + ", ".join(str(arg) for arg in self.args) + ")"


class UnionRange(IntegerSet, UnionMixin):
    """Union of a set of integer sets"""

    def __init__(self, *args, **kwargs):
        UnionMixin.__init__(self, *args, **kwargs)

    @property
    def bounds(self) -> UnitRange:
        """Smallest UnitRange containing all elements"""
        return UnitRange(
            functools.reduce(min, (arg.start for arg in self.args)),
            functools.reduce(max, (arg.stop for arg in self.args)),
        )

    def simplify(self) -> IntegerSet:
        if not self.disjoint:
            # note: just return as UnionMixin.simplify indirectly calls UnionRange.simplify after it made its components
            #  disjoint
            return UnionMixin.simplify(self)

        # do some basic fusing
        assert all(isinstance(arg, UnitRange) for arg in self.args)
        args = sorted(self.args, key=lambda arg: (arg.start, arg.stop))
        fused_args = [args[0]]
        for arg in args[1:]:
            if fused_args[-1].stop == arg.start:
                fused_args = [
                    *fused_args[0:-1],
                    UnitRange(fused_args[-1].start, arg.stop),
                ]
            else:
                fused_args.append(arg)

        return union(*fused_args, simplify=False, disjoint=self.disjoint)

    def __mul__(self, other):
        # todo: may user facing interface should simplify
        return union(*(arg * other for arg in self.args), disjoint=self.disjoint, simplify=False)

    def __hash__(self):
        return hash(tuple(hash(arg) for arg in self.args))


_empty_cartesian_cache = {}


class CartesianSet(Set):
    """A set of (cartesian indices, i.e. tuples of integers)"""

    # todo: implement abstract methods
    def empty_set(self):
        if self.dim not in _empty_cartesian_cache:
            _empty_cartesian_cache[self.dim] = functools.reduce(
                operator.mul, itertools.repeat(UnitRange(0, 0), self.dim)
            )
        return _empty_cartesian_cache[self.dim]

    def universe(self):
        return functools.reduce(
            operator.mul, itertools.repeat(UnitRange(-math.inf, math.inf), self.dim)
        )

    def shrink(self, *args: Sequence[Union[int, Tuple[int, int]]]):
        args = tuple((-arg, -arg) if isinstance(arg, int) else (-arg[0], -arg[1]) for arg in args)
        return self.extend(*args)

    @staticmethod
    def union_type():
        return UnionCartesian

    @staticmethod
    def primitive_type():
        return ProductSet

    def simplify(self):
        return self


class ProductSet(CartesianSet):
    """Cartesian product of a set of `UnitRange`s"""

    args: Sequence[UnitRange]

    def __init__(self, *args: Sequence[UnitRange]):
        assert all(isinstance(arg, UnitRange) for arg in args)
        assert len(args) > 1

        self.args = args

    @classmethod
    def from_coords(cls, p1: Tuple, p2: Tuple):
        assert len(p1) == len(p2)
        return functools.reduce(
            operator.mul, (UnitRange(first, last + 1) for (first, last) in zip(p1, p2))
        )

    @property
    def size(self):
        return functools.reduce(operator.mul, self.shape)

    @property
    def bounds(self):
        return self

    @property
    def shape(self):
        return tuple(arg.size for arg in self.args)

    @property
    def empty(self):
        return any(arg.empty for arg in self.args)

    @property
    def dim(self):
        return len(self.args)

    def without(self, other: ProductSet, *tail: ProductSet, simplify=True):
        if isinstance(other, ProductSet):
            # if there is no overlap in any dimension nothing is to be removed
            if any(r1.intersect(r2).empty for r1, r2 in zip(self.args, other.args)):
                result = self
            else:
                if len(self.args) == 2:  # break recursion
                    result = union(
                        self.args[0].without(other.args[0], simplify=simplify) * self.args[1],
                        self.args[0].intersect(other.args[0]) * self.args[1].without(other.args[1]),
                        simplify=simplify,
                    )
                else:
                    result = union(
                        self.args[0].without(other.args[0], simplify=simplify)
                        * ProductSet(*self.args[1:]),
                        self.args[0].intersect(other.args[0])
                        * ProductSet(*self.args[1:]).without(
                            ProductSet(*other.args[1:]), simplify=simplify
                        ),
                        simplify=simplify,
                    )

            return result if len(tail) == 0 else result.without(*tail, simplify=simplify)
        elif isinstance(other, UnionCartesian):
            return self.without(other.args[0], *other.args[1:], *tail)

        raise NotImplementedError()

    def complement(self, arg: Union[None, ProductSet] = None, simplify=True):
        if not arg:
            arg = self.universe()

        return arg.without(self, simplify=simplify)

    def intersect(self, other: Union[ProductSet, UnionCartesian]):
        if isinstance(other, ProductSet):
            return functools.reduce(
                operator.mul,
                (arg1.intersect(arg2) for (arg1, arg2) in zip(self.args, other.args)),
            )
        elif isinstance(other, UnionCartesian):
            return other.intersect(self)

        raise ValueError(f"Invalid argument `{other}`")

    def extend(self, *args: Sequence[Union[int, Tuple[int, int]]]):
        if self.empty:
            raise ValueError("Empty set can not be extended")
        assert len(self.args) == len(args)
        return functools.reduce(operator.mul, (r.extend(arg) for r, arg in zip(self.args, args)))

    def translate(self, *args: Sequence[int]):
        if self.empty:
            raise ValueError("Empty set can not be translated")
        assert len(self.args) == len(args)
        return functools.reduce(operator.mul, (r.translate(arg) for r, arg in zip(self.args, args)))

    def as_tuple(self):
        return tuple(arg.as_tuple() for arg in self.args)

    def __iter__(self):
        # memory-lightweight itertools.product like iterator
        for i in self.args[0]:
            if len(self.args[1:]) > 1:
                for tail in functools.reduce(operator.mul, self.args[1:]):
                    yield i, *tail
            else:  # break recursion
                for j in self.args[1]:
                    yield i, j

    def __eq__(self, other):
        if isinstance(other, Set):
            return self.issubset(other) and other.issubset(self)

        return False

    def __contains__(self, arg: Sequence[integer]):
        assert all(is_integer_like(i) for i in arg)
        assert len(arg) == len(self.args)

        return all(i in r for r, i in zip(self.args, arg))

    def issubset(self, arg):
        # if isinstance(arg, ProductSet):
        #    assert len(arg.args) == len(self.args)
        #    return all(subr.issubset(r) for subr, r in zip(self.args, arg.args))
        if isinstance(arg, Set):
            return arg.complement(simplify=False).intersect(self).empty

        raise ValueError(f"Invalid argument `{arg}`")

    def __getitem__(self, args):
        if all(isinstance(arg, int) for arg in args):
            return tuple(r[i] for r, i in zip(self.args, args))
        elif all(isinstance(arg, slice) for arg in args):
            return ProductSet(*(r[s] for r, s in zip(self.args, args)))

        raise ValueError(f"Invalid argument `{args}`")

    def __mul__(self, other: UnitRange):
        if not isinstance(other, UnitRange):
            raise ValueError(f"Invalid argument `{other}`")

        return ProductSet(
            *(self.args if not other.empty else self.empty_set().args),
            other if not self.empty else other.empty_set(),
        )

    def __hash__(self):
        return hash(tuple(hash(arg) for arg in self.args))

    def __repr__(self):
        return " * ".join(repr(arg) for arg in self.args)


class UnionCartesian(CartesianSet, UnionMixin):
    """(set)union of a set of cartesian sets"""

    def __init__(self, *args, **kwargs):
        UnionMixin.__init__(self, *args, **kwargs)

    @property
    def bounds(self) -> Set:
        return functools.reduce(
            operator.mul,
            (
                union(*comp_ranges, simplify=False).bounds
                for comp_ranges in zip(*(ps.args for ps in self.args))
            ),
        )

    @property
    def dim(self):
        assert all(arg.dim == self.args[0].dim for arg in self.args)

        return self.args[0].dim

    def simplify(self):
        a = UnionMixin.simplify(self)

        if isinstance(a, ProductSet):
            return a

        converged = False
        while not converged:
            converged = True
            for curr in a.args:
                touching_sets = [
                    other
                    for other in a.args
                    if not curr.extend(*(1 for _ in range(0, self.dim))).intersect(other).empty
                    and other != curr
                ]
                for touching_set in touching_sets:
                    covering = union(touching_set, curr, simplify=False).bounds
                    if covering.issubset(a):
                        rest = [
                            set_.without(curr)
                            for set_ in a.args
                            if not set_.issubset(covering) and set_ != curr and set_ != touching_set
                        ]
                        merged = union(
                            union(*rest, disjoint=a.disjoint, simplify=False)
                            if len(rest) > 0
                            else a.empty_set(),
                            covering,
                            disjoint=False,
                            simplify=False,
                        )
                        if isinstance(merged, ProductSet):
                            return merged
                        elif isinstance(merged, UnionCartesian):
                            merged = UnionMixin.simplify(merged)
                            if len(merged.args) < len(a.args):
                                # we found something that has lower complexity
                                converged = False
                                a = merged
                                break
                        else:
                            raise RuntimeError()
                # reset simplification to merged a
                if converged == False:
                    break
        return a.make_disjoint()

    def __hash__(self):
        return hash(tuple(hash(arg) for arg in self.args))


class IndexSpace:
    """
    An index_space is a collection of cartesian sets associated with a label
    """

    subset: Dict[Any, CartesianSet]

    def __init__(self, definition: CartesianSet = None):
        self.subset = {}
        if definition:
            self.subset["definition"] = definition

    @classmethod
    def from_sizes(cls, n_i: int, n_j: int, n_k: int):
        return cls(UnitRange(0, n_i) * UnitRange(0, n_j) * UnitRange(0, n_k))

    def __getitem__(self, arg):
        return self.subset["definition"][arg]

    @property
    def bounds(self):
        return self.covering.bounds

    @property
    def covering(self):
        # todo: simplification is expensive, so cache the value
        return union(*(subset for subset in self.subset.values()), simplify=False)

    @property
    def default_origin(self):
        """A tuple of the lowest indices in each dimension"""
        return tuple(bound.start for bound in self.subset["definition"].bounds.args)

    @property
    def shape(self):
        """The maximum size of each dimensions"""
        return tuple(bound.size for bound in self.bounds.args)

    def translate(self, *args):
        """Translate each subset"""
        assert len(args) == 3
        new_space = copy(self)
        new_space.subset = {k: s.translate(*args) for k, s in self.subset.items()}
        return new_space

    def add_subset(self, name, subset):
        self.subset[name] = subset.simplify()

    def decompose(self, parts_per_dim: Tuple[int, int, int]):
        def dim_splitters(n, length):
            "Divide `length` long dimension into `n` parts"
            interval_length = math.floor(length / n)
            return [i * interval_length for i in range(n)] + [length]

        splitters = [
            dim_splitters(num_parts, self.covering.shape[dim])
            for dim, num_parts in enumerate(parts_per_dim)
        ]

        coords = list(itertools.product(*[range(0, parts) for parts in parts_per_dim]))
        index_spaces = {coord: None for coord in coords}

        for coord in coords:
            coord_idx_space = IndexSpace()
            for name, subset in self.subset.items():
                subset_part = subset[
                    tuple(
                        slice(splitters[dim][coord_l], splitters[dim][coord_l + 1])
                        for dim, coord_l in enumerate(coord)
                    )
                ]
                coord_idx_space.add_subset(name, subset_part)
            index_spaces[coord] = coord_idx_space

        return index_spaces

    def __str__(self):
        result = f"{self.__repr__()}\n  subsets:\n"
        for label, value in self.subset.items():
            result += f"    {label}: {str(value)}\n"
        return result


class index_convention:
    index_spaces: Dict[Any, IndexSpace]
    origins: Dict[Any, Sequence[int]]
