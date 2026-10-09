"""A Mesh's full uniform k-grid, indexed in a basis where the first zone is compact (#302).

In a sheared reciprocal basis the first zone spans many periods along some axis, so a pad of
the stored grid misses part of the zone and marching cubes runs on long thin cells. The k-points
are still one lattice of samples. ``periodic_grid`` reads that lattice once and picks a reduced
basis of it, and ``PeriodicGrid.drawn_mesh`` copies the stored samples onto the box around the
zone in that basis. No value is interpolated.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyprocar.core.brillouin_zone import BrillouinZone, reduced_basis_steps
from pyprocar.core.kpoints import KGridInfo
from pyprocar.utils.log_utils import warn_user

if TYPE_CHECKING:
    from pyprocar.core.ebs import ElectronicBandStructureMesh

GRID_TOLERANCE = 1e-2
"""Grid spacings within which a k-point is taken as its uniform-grid point.

k-points printed with 5 decimals stay inside it up to N = 2000 points per axis, with 4
decimals up to N = 200. A grid stretched as k^1.05 on 16 points is 0.29 off, and falls back.
Accepted k-points are treated as their exact grid points, by the drawn surface and the cut.
"""

REDUCED_TOLERANCE = 1e-4
"""Relative slack on sorted row lengths within which a basis counts as already reduced.

POSCAR rounding makes equal hexagonal rows differ: BiSb's |b1 - b2| is 3.2e-6 below |b1|.
"""

_ON_GRID = 1e-6
"""Grid spacings within which a zone corner is taken as the grid point it sits on."""


@dataclass(frozen=True, eq=False)
class PeriodicGrid:
    """One full uniform k-grid, indexed in a drawn basis where it is diagonal.

    Drawn grid point m (any integers) is fractional (m + shift) / n of ``lattice`` and holds
    source row ``_rows[(m @ _to_source + _offset) mod N]``. ``lattice`` is the period of the
    drawn data: the given basis B on the identity map, T @ B for a grid with equal N, and a
    sublattice of the reciprocal lattice when no reciprocal basis makes the grid diagonal.
    ``reciprocal_lattice`` is always B, the basis the user's integers refer to.
    """

    reciprocal_lattice: np.ndarray
    lattice: np.ndarray
    n: tuple[int, int, int]
    shift: np.ndarray
    """In drawn grid units, in [0, 1)."""
    _source: ElectronicBandStructureMesh
    _to_source: np.ndarray
    _offset: np.ndarray
    _start: np.ndarray
    """Drawn index of the stored grid's first point."""
    _rows: np.ndarray

    @property
    def is_identity(self) -> bool:
        """The given basis is reduced or an obtuse superbase, so the grid is drawn as stored."""
        return bool((self._to_source == np.eye(3, dtype=int)).all())

    @property
    def tile_multiple(self) -> int:
        """Drawn tile points per stored grid point: 1 whenever a reciprocal basis makes the grid
        diagonal."""
        return math.prod(self.n) // self._rows.size

    def tile(self, values: np.ndarray) -> np.ndarray:
        """One period of per-k-point ``values``, shaped (n1, n2, n3, ...)."""
        return values[self._rows_at(box_indices(np.zeros(3, dtype=int), np.asarray(self.n)))]

    def drawn_mesh(self, padding: int, *, keep_pad: bool) -> ElectronicBandStructureMesh:
        """The source Mesh on the zone's bounding box plus one point on each side, joined with
        the pad box when ``keep_pad``: one drawn period from the stored grid's first point and
        ``padding`` points beyond it on each side, ``pad(padding)``'s box on the identity map.

        On the identity map whose pad already holds the zone's box this is ``pad(padding)``,
        today's mesh bit for bit.
        """
        n = np.asarray(self.n)
        live = n > 1
        zone_lo, zone_hi = self._zone_box()
        pad_lo, pad_hi = self._start - padding, self._start + n + padding
        if self.is_identity and ((pad_lo <= zone_lo) & (pad_hi >= zone_hi))[live].all():
            return self._source.pad(padding=padding, inplace=False)
        if keep_pad:
            zone_lo, zone_hi = np.minimum(zone_lo, pad_lo), np.maximum(zone_hi, pad_hi)
        lo = np.where(live, zone_lo, self._start)
        hi = np.where(live, zone_hi, self._start + 1)
        m = box_indices(lo, hi)
        flat = m.reshape(-1, 3, order="F")
        kgrid_info = KGridInfo(
            kgrid=self.n,
            kgrid_mode=self._source.kgrid_info.kgrid_mode,
            kshift=(float(self.shift[0]), float(self.shift[1]), float(self.shift[2])),
        )
        return self._source._take(
            rows=self._rows_at(flat),
            points=(flat + self.shift) / n,
            kgrid=_size3(hi - lo),
            kgrid_info=kgrid_info,
            reciprocal_lattice=self.lattice,
        )

    def _zone_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Half-open drawn index box over the zone corners, widened by one point each side,
        so central differences hold at every cell that touches the zone. A corner within
        rounding of a grid point counts as on it."""
        corners = np.asarray(BrillouinZone(self.reciprocal_lattice).points)
        g = corners @ np.linalg.inv(self.lattice) * np.asarray(self.n) - self.shift
        lo = np.floor(g.min(axis=0) + _ON_GRID).astype(int)
        hi = np.ceil(g.max(axis=0) - _ON_GRID).astype(int)
        return lo - 1, hi + 2

    def _rows_at(self, m: np.ndarray) -> np.ndarray:
        index = (m @ self._to_source + self._offset) % np.asarray(self._rows.shape)
        return self._rows[index[..., 0], index[..., 1], index[..., 2]]


def drawn_mesh(
    ebs: ElectronicBandStructureMesh, padding: int, drawing: str, *, keep_pad: bool
) -> tuple[PeriodicGrid | None, ElectronicBandStructureMesh]:
    """The grid of ``ebs`` and the mesh to draw ``drawing`` on: ``PeriodicGrid.drawn_mesh``, or
    today's pad in the given basis when the k-points are not one full uniform grid. A sheared
    given basis then warns that part of the zone is missing. A drawing that may be shown
    unclipped sets ``keep_pad``; one always clipped to the zone needs only the zone's box."""
    grid = periodic_grid(ebs)
    if grid is not None:
        return grid, grid.drawn_mesh(padding, keep_pad=keep_pad)
    lattice = ebs.reciprocal_lattice
    live = np.asarray(ebs.kgrid) > 1
    if lattice is not None and not is_reduced_basis(np.asarray(lattice), live):
        warn_user(
            f"The k-points are not one uniform grid, so the {drawing} is drawn in the given "
            + "reciprocal basis, which is not reduced; parts of the first Brillouin zone beyond "
            + f"{padding} padded k-points are missing. Give the full uniform k-grid to draw it all."
        )
    return None, ebs.pad(padding=padding, inplace=False)


def periodic_grid(ebs: ElectronicBandStructureMesh) -> PeriodicGrid | None:
    """The k-points of ``ebs`` as one uniform grid in a drawn basis.

    None when the k-points do not fill the grid ``ebs.kgrid`` once within ``GRID_TOLERANCE``,
    when more than one axis has a single point, or without a reciprocal lattice.
    """
    if ebs.reciprocal_lattice is None:
        return None
    big_n = np.asarray(ebs.kgrid, dtype=int)
    if (big_n < 1).any() or (big_n == 1).sum() > 1:
        return None
    scaled = np.asarray(ebs.kpoints, dtype=np.float64) * big_n
    shift = _snap_shift(scaled[0] - np.floor(scaled[0]))
    nearest = np.rint(scaled - shift)
    if np.abs(scaled - shift - nearest).max() > GRID_TOLERANCE:
        return None
    index = nearest.astype(int) % big_n
    rows = np.full(tuple(big_n), -1)
    rows[index[:, 0], index[:, 1], index[:, 2]] = np.arange(len(scaled))
    if len(scaled) != rows.size or (rows < 0).any():
        return None

    basis = np.asarray(ebs.reciprocal_lattice, dtype=np.float64)
    live = big_n > 1
    if is_reduced_basis(basis, live) or _obtuse_superbase(basis[live]):
        to_source = np.eye(3, dtype=int)
    else:
        # Reduce the k-point lattice, not the reciprocal lattice: for unequal N only it is diagonal.
        k_lattice = basis / big_n[:, None]
        to_source = _nearest_identity(_reducing_steps(k_lattice, live), k_lattice, live)
    from_source = np.rint(np.linalg.inv(to_source)).astype(int)
    drawn_shift = _snap_shift((shift @ from_source) % 1.0)
    offset = np.rint(drawn_shift @ to_source - shift).astype(int)
    # n_i is the order of drawn step i modulo the reciprocal lattice.
    n = np.lcm.reduce(big_n // np.gcd(to_source, big_n), axis=1)
    steps = n[:, None] * to_source // big_n[None, :]
    start = np.rint((nearest.min(axis=0) + shift) @ from_source - drawn_shift).astype(int)
    return PeriodicGrid(
        basis,
        steps @ basis,
        _size3(n),
        drawn_shift,
        ebs,
        to_source,
        offset,
        start,
        rows,
    )


def is_reduced_basis(basis: np.ndarray, live: np.ndarray) -> bool:
    """The rows on ``live`` axes are as short as a reduced basis of the lattice they span."""
    if live.sum() < 2:
        return True
    given = np.sort(np.linalg.norm(basis[live], axis=1))
    reduced = np.sort(np.linalg.norm((_reducing_steps(basis, live) @ basis)[live], axis=1))
    return bool((given <= reduced * (1 + REDUCED_TOLERANCE)).all())


def _reducing_steps(lattice: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Integer unimodular W with W @ lattice reduced; a single-point axis keeps its row, and only
    the two in-plane rows of such a grid are reduced."""
    if live.all():
        return reduced_basis_steps(lattice)
    steps = np.eye(3, dtype=int)
    steps[np.ix_(live, live)] = _gauss_steps(lattice[live])
    return steps


def _gauss_steps(rows: np.ndarray) -> np.ndarray:
    """Integer 2x2 W with W @ rows Lagrange-Gauss reduced: the two shortest lattice vectors."""
    steps = np.eye(2, dtype=int)
    u, v = rows[0].astype(np.float64), rows[1].astype(np.float64)
    while True:
        if u @ u > v @ v:
            u, v = v, u
            steps = steps[[1, 0]]
        mu = int(np.rint((u @ v) / (u @ u)))
        if mu == 0:
            return steps
        v = v - mu * u
        steps[1] = steps[1] - mu * steps[0]


def _nearest_identity(steps: np.ndarray, lattice: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Of the bases with the sorted row lengths of the reduced basis ``steps @ lattice``, the one
    closest to the identity in index space, so a shear of a reduced basis returns to it exactly.
    Ties go to the fewest changed entries, which undoes a single shear b2 + 3 b1 of an fcc basis
    rather than reaching (b1, b1 + b2 + b3, b3), then to the first in the order of the entries:
    integers only, so rounding noise in the lattice cannot change the choice.

    Candidate rows are the -1, 0, 1 combinations of the reduced rows. A hexagonal plane has
    reduced bases that are no signed permutation of each other, (b1, b2) and (b1, b2 - b1).
    """
    axes = np.flatnonzero(live)
    reduced_lengths = np.sort(np.linalg.norm((steps @ lattice)[axes], axis=1))
    combos = np.array([c for c in itertools.product((-1, 0, 1), repeat=len(axes)) if any(c)])
    rows = combos @ steps[axes]
    picks = np.array(list(itertools.permutations(range(len(rows)), len(axes))))
    bases = np.tile(np.eye(3, dtype=int), (len(picks), 1, 1))
    bases[:, axes, :] = rows[picks]
    drawn = bases @ lattice
    lengths = np.sort(np.linalg.norm(drawn[:, axes, :], axis=2), axis=1)
    usable = (np.abs(np.rint(np.linalg.det(bases))) == 1) & (
        lengths <= reduced_lengths * (1 + REDUCED_TOLERANCE)
    ).all(axis=1)
    change = bases - np.eye(3, dtype=int)
    distance = np.abs(change).sum(axis=(1, 2))
    changed = np.count_nonzero(change, axis=(1, 2))
    ranked = np.lexsort((*bases.reshape(-1, 9).T[::-1], changed, distance))
    return bases[ranked[usable[ranked]][0]]


def _obtuse_superbase(rows: np.ndarray) -> bool:
    """The rows and minus their sum meet pairwise at angles whose cosine is below
    -REDUCED_TOLERANCE: a Selling-reduced basis, of a lattice whose reduction is unique.

    Such a basis is drawn as given, as before #302, though it may not be the shortest. The
    rhombohedral primitive cell of Bi2Se3 is one; drawn in its reduced bases, (b1 + b2 + b3)
    and two of the b_i, marching cubes on 16^3 cuts the orbit along hexagonal [1 0 3] as 0.357
    or, 0.01 above, as two orbits; the analytic band has one of 0.283, the given basis 0.281.
    """
    superbase = np.vstack([rows, -rows.sum(axis=0)])
    unit = superbase / np.linalg.norm(superbase, axis=1, keepdims=True)
    cosines = unit @ unit.T
    return bool((cosines[~np.eye(len(superbase), dtype=bool)] < -REDUCED_TOLERANCE).all())


def _snap_shift(shift: np.ndarray) -> np.ndarray:
    """A shift within GRID_TOLERANCE of 0 or 1 is 0: (-29/60) * 60 rounds to -29.000000000000004."""
    return np.where((shift < GRID_TOLERANCE) | (shift > 1 - GRID_TOLERANCE), 0.0, shift)


def box_indices(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Integer points lo <= m < hi, shaped (hi - lo) + (3,)."""
    axes = [np.arange(a, b) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _size3(n: np.ndarray) -> tuple[int, int, int]:
    return (int(n[0]), int(n[1]), int(n[2]))
