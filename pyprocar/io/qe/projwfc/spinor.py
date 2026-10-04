"""Turn projwfc.x spinor projections into total, Sx, Sy and Sz per real (l, m) orbital.

In a non-collinear run projwfc.x projects each band onto spinor atomic states:
``|l m s_z>`` without spin-orbit coupling and ``|l j m_j>`` with it. The
coefficients come from QE 7.2: ``spinor`` and ``sph_ind`` (upflib) build each
``|l j m_j>`` from complex spherical harmonics, and ``rot_ylm`` (upflib
init_us_1.f90) writes those in the real harmonics that ``ylmr2`` defines, whose
order m = 1..2l+1 projwfc.x prints as z, x, y for p and z2, xz, yz, x2-y2, xy for d.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Self

import numpy as np
import numpy.typing as npt

N_ORBITALS = 16


@dataclass(frozen=True, slots=True)
class SpinorState:
    """One projwfc.x atomic state: |l j m_j> with spin-orbit, |l m s_z> without."""

    atom: int
    wfc: int
    ang: int
    j: float | None
    m_j: float | None
    m: int | None
    s_z: float | None

    @classmethod
    def from_record(cls, record: Mapping[str, int | float | None]) -> Self:
        def optional(key: str) -> float | None:
            value = record[key]
            return None if value is None else float(value)

        m = record["m"]
        return cls(
            atom=int(record["atom_num"] or 0),
            wfc=int(record["wfc_num"] or 0),
            ang=int(record["l"] or 0),
            j=optional("j"),
            m_j=optional("m_j"),
            m=None if m is None else int(m),
            s_z=optional("s_z"),
        )

    @property
    def is_upper(self) -> bool:
        return self.j is not None and abs(self.j - self.ang - 0.5) < 1e-8

    def in_real_harmonics(self) -> npt.NDArray[np.complex128]:
        """The state's components on (real m, spin up/down), shape (2l+1, 2)."""
        ang = self.ang
        components = np.zeros((2 * ang + 1, 2), dtype=np.complex128)
        if self.s_z is not None and self.m is not None:
            components[self.m - 1, 0 if self.s_z > 0 else 1] = 1.0
            return components
        assert self.j is not None and self.m_j is not None
        m = round(self.m_j - 0.5 if self.is_upper else self.m_j + 0.5)
        for spin in (1, 2):
            k = _sph_ind(self.is_upper, m, spin)
            if abs(k) <= ang:
                components[:, spin - 1] = _spinor(ang, self.is_upper, m, spin) * _complex_to_real(
                    ang, k
                )
        return components


def _spinor(ang: int, upper: bool, m: int, spin: int) -> float:
    if upper:
        return np.sqrt((ang + m + 1 if spin == 1 else ang - m) / (2 * ang + 1))
    if m < -ang + 1:
        return 0.0
    return (
        np.sqrt((ang - m + 1) / (2 * ang + 1)) if spin == 1 else -np.sqrt((ang + m) / (2 * ang + 1))
    )


def _sph_ind(upper: bool, m: int, spin: int) -> int:
    if upper:
        return m if spin == 1 else m + 1
    return m - 1 if spin == 1 else m


def _complex_to_real(ang: int, k: int) -> npt.NDArray[np.complex128]:
    """Coefficients of the complex harmonic m = k on the 2l+1 real harmonics (rot_ylm)."""
    row = np.zeros(2 * ang + 1, dtype=np.complex128)
    if k == 0:
        row[0] = 1.0
        return row
    mm = abs(k)
    sign = (-1.0) ** mm if k < 0 else 1.0
    row[2 * mm - 1] = sign / np.sqrt(2.0)
    row[2 * mm] = (-1j if k < 0 else 1j) * sign / np.sqrt(2.0)
    return row


def _shells(states: Sequence[SpinorState]) -> list[list[int]]:
    """Group state indices into shells that span all real m and both spins of one l.

    projwfc.x lists the j = l - 1/2 and j = l + 1/2 states of a fully relativistic
    pseudopotential as separate wavefunctions; the n-th of each pair up per atom and l.
    """
    by_wfc: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        by_wfc[(state.atom, state.wfc)].append(index)
    halves: dict[tuple[int, int, bool], list[list[int]]] = defaultdict(list)
    shells: list[list[int]] = []
    for (atom, _), indices in by_wfc.items():
        first = states[indices[0]]
        if first.j is None or first.ang == 0 or len({states[i].j for i in indices}) == 2:
            shells.append(indices)
        else:
            halves[(atom, first.ang, first.is_upper)].append(indices)
    for atom, ang in dict.fromkeys((atom, ang) for atom, ang, _ in halves):
        for lower, upper in zip_longest(
            halves[(atom, ang, False)], halves[(atom, ang, True)], fillvalue=[]
        ):
            shells.append(lower + upper)
    return shells


def spinor_projections(
    amplitudes: npt.NDArray[np.complexfloating],
    records: Sequence[Mapping[str, int | float | None]],
    n_atoms: int,
) -> npt.NDArray[np.float64]:
    """Total, Sx, Sy and Sz of each band on each atom's real (l, m) orbitals.

    ``amplitudes`` holds <state|band> with shape (n_kpoints, n_bands, n_states), and
    ``records`` describes each state as ProjwfcOut.atm_wfcs does. Returns shape
    (n_kpoints, n_bands, 4, n_atoms, 16), orbitals in projwfc.x's real-harmonic
    order. Shells with different radial functions add incoherently.
    """
    states = [SpinorState.from_record(record) for record in records]
    n_kpoints, n_bands, _ = amplitudes.shape
    projected = np.zeros((n_kpoints, n_bands, 4, n_atoms, N_ORBITALS))
    for shell in _shells(states):
        ang, atom = states[shell[0]].ang, states[shell[0]].atom - 1
        basis = np.array([states[i].in_real_harmonics() for i in shell])
        components = np.einsum("kbs,smt->kbmt", amplitudes[..., shell], basis)
        up, down = components[..., 0], components[..., 1]
        up_down = np.conj(up) * down
        density = np.abs(up) ** 2 + np.abs(down) ** 2
        sz = np.abs(up) ** 2 - np.abs(down) ** 2
        spin = np.stack([density, 2 * up_down.real, 2 * up_down.imag, sz], axis=2)
        projected[:, :, :, atom, ang * ang : (ang + 1) ** 2] += spin
    return projected
