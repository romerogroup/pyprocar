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
from itertools import zip_longest

import numpy as np
import numpy.typing as npt

N_ORBITALS = 16

State = Mapping[str, int | float | None]


def _spinor(l: int, j: float, m: int, spin: int) -> float:
    if abs(j - l - 0.5) < 1e-8:
        return np.sqrt((l + m + 1 if spin == 1 else l - m) / (2 * l + 1))
    if m < -l + 1:
        return 0.0
    return np.sqrt((l - m + 1) / (2 * l + 1)) if spin == 1 else -np.sqrt((l + m) / (2 * l + 1))


def _sph_ind(l: int, j: float, m: int, spin: int) -> int:
    if abs(j - l - 0.5) < 1e-8:
        return m if spin == 1 else m + 1
    return m - 1 if spin == 1 else m


def _complex_to_real(l: int, k: int) -> npt.NDArray[np.complex128]:
    """Coefficients of the complex harmonic m = k on the 2l+1 real harmonics (rot_ylm)."""
    row = np.zeros(2 * l + 1, dtype=np.complex128)
    if k == 0:
        row[0] = 1.0
        return row
    mm = abs(k)
    sign = (-1.0) ** mm if k < 0 else 1.0
    row[2 * mm - 1] = sign / np.sqrt(2.0)
    row[2 * mm] = (-1j if k < 0 else 1j) * sign / np.sqrt(2.0)
    return row


def _state_in_real_harmonics(state: State) -> npt.NDArray[np.complex128]:
    """The state's components on (real m, spin up/down), shape (2l+1, 2)."""
    l = int(state["l"] or 0)
    components = np.zeros((2 * l + 1, 2), dtype=np.complex128)
    if state["s_z"] is not None:
        components[int(state["m"] or 0) - 1, 0 if float(state["s_z"]) > 0 else 1] = 1.0
        return components
    j, m_j = float(state["j"] or 0.0), float(state["m_j"] or 0.0)
    m = round(m_j - 0.5 if abs(j - l - 0.5) < 1e-8 else m_j + 0.5)
    for spin in (1, 2):
        k = _sph_ind(l, j, m, spin)
        if abs(k) <= l:
            components[:, spin - 1] = _spinor(l, j, m, spin) * _complex_to_real(l, k)
    return components


def _shells(states: Sequence[State]) -> list[list[int]]:
    """Group state indices into shells that span all real m and both spins of one l.

    projwfc.x lists the j = l - 1/2 and j = l + 1/2 states of a fully relativistic
    pseudopotential as separate wavefunctions; the n-th of each pair up per atom and l.
    """
    by_wfc: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        by_wfc[(int(state["atom_num"] or 0), int(state["wfc_num"] or 0))].append(index)
    lowers: dict[tuple[int, int], list[list[int]]] = defaultdict(list)
    uppers: dict[tuple[int, int], list[list[int]]] = defaultdict(list)
    shells: list[list[int]] = []
    for (atom, _), indices in by_wfc.items():
        l = int(states[indices[0]]["l"] or 0)
        js = {states[i]["j"] for i in indices}
        if None in js or len(js) == 2 or l == 0:
            shells.append(indices)
        elif abs(float(js.pop() or 0.0) - l - 0.5) < 1e-8:
            uppers[(atom, l)].append(indices)
        else:
            lowers[(atom, l)].append(indices)
    for key in dict.fromkeys([*lowers, *uppers]):
        for lower, upper in zip_longest(lowers[key], uppers[key], fillvalue=[]):
            shells.append(lower + upper)
    return shells


def spinor_projections(
    amplitudes: npt.NDArray[np.complexfloating], states: Sequence[State], n_atoms: int
) -> npt.NDArray[np.float64]:
    """Total, Sx, Sy and Sz of each band on each atom's real (l, m) orbitals.

    ``amplitudes`` holds <state|band> with shape (n_kpoints, n_bands, n_states).
    Returns shape (n_kpoints, n_bands, 4, n_atoms, 16), orbitals in projwfc.x's
    real-harmonic order. Shells with different radial functions add incoherently.
    """
    n_kpoints, n_bands, _ = amplitudes.shape
    projected = np.zeros((n_kpoints, n_bands, 4, n_atoms, N_ORBITALS))
    for shell in _shells(states):
        l = int(states[shell[0]]["l"] or 0)
        atom = int(states[shell[0]]["atom_num"] or 0) - 1
        basis = np.array([_state_in_real_harmonics(states[i]) for i in shell])
        components = np.einsum("kbs,smt->kbmt", amplitudes[..., shell], basis)
        up, down = components[..., 0], components[..., 1]
        up_down = np.conj(up) * down
        spin = np.stack(
            [
                np.abs(up) ** 2 + np.abs(down) ** 2,
                2 * up_down.real,
                2 * up_down.imag,
                np.abs(up) ** 2 - np.abs(down) ** 2,
            ],
            axis=2,
        )
        projected[:, :, :, atom, l * l : (l + 1) ** 2] += spin
    return projected
