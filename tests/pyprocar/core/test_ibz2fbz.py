import re
import warnings
from typing import cast

import numpy as np
import pytest

from pyprocar.core import Structure, kpoints
from pyprocar.core.ebs import ElectronicBandStructure, ElectronicBandStructureMesh, ibz2fbz
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

C4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
C4_GROUP = np.array([np.linalg.matrix_power(C4, n) for n in range(4)])
HEXAGONAL = np.array([[3.0, 0.0, 0.0], [-1.5, 1.5 * np.sqrt(3), 0.0], [0.0, 0.0, 5.0]])
VASP_ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"]


def key(k):
    wrapped = np.round(np.mod(np.round(np.asarray(k, dtype=float), 6), 1.0), 6) % 1.0
    return tuple(wrapped + 0.0)


def gamma_grid(n):
    return kpoints.get_kpoints_from_kgrid(kgrid=n, kshift=(0, 0, 0), mode=KGRID_MODE.GAMMA)


def gamma_info(n):
    return KGridInfo(kgrid=n, kgrid_mode=KGRID_MODE.GAMMA, kshift=(0, 0, 0))


def wedge(grid, k_actions, first=()):
    """One k-point per orbit of the matrices acting on fractional k, the points in first leading."""
    kept, seen = [], set()
    for k in [*np.asarray(first, dtype=float).reshape(-1, 3), *grid]:
        if key(k) not in seen:
            kept.append(k)
            seen.update(key(m @ k) for m in k_actions)
    return np.array(kept)


def asymmetric_field(k):
    x, y, z = 2 * np.pi * np.atleast_2d(k).T
    odd = np.sin(x + 2 * y + 3 * z) + 0.3 * np.sin(z) + 0.2 * np.sin(y)
    even = 0.5 * np.cos(2 * x - z) + 0.4 * np.cos(x + 3 * y - z) + 0.25 * np.cos(3 * x + y + 2 * z)
    return odd + even


def invariant(k, k_actions):
    """sum over g of asymmetric_field(g^-1 k): invariant under that group and no other."""
    k = np.atleast_2d(k)
    return np.sum([asymmetric_field(k @ np.linalg.inv(m).T) for m in k_actions], axis=0)


def band_values(ebs):
    assert ebs.bands is not None
    return ebs.bands.to_array()[:, 0, 0]


def single_site(lattice=None, rotations=None, **kwargs):
    return Structure(
        atoms=["X"],
        fractional_coordinates=np.zeros((1, 3)),
        lattice=np.eye(3) if lattice is None else lattice,
        rotations=rotations,
        **kwargs,
    )


def nonmagnetic_c4_wedge():
    grid = gamma_grid((3, 3, 3))
    k_actions = np.concatenate([C4_GROUP, -C4_GROUP])
    ibz = wedge(grid, k_actions)
    return ibz, k_actions


def test_ibz2fbz_unfolds_with_the_rotations_it_is_given():
    ibz, k_actions = nonmagnetic_c4_wedge()
    ebs = ElectronicBandStructure(
        kpoints=ibz,
        bands=invariant(ibz, k_actions).reshape(-1, 1, 1),
        reciprocal_lattice=cast(kpoints.RECIPROCAL_LATTICE_DTYPE, np.eye(3)),
        structure=single_site(),
    )

    ibz2fbz(ebs, rotations=C4_GROUP, kgrid_info=gamma_info((3, 3, 3)))

    assert (len(ibz), ebs.n_kpoints) == (6, 27)
    exact = np.round(ebs.kpoints * 3) / 3
    np.testing.assert_allclose(band_values(ebs), invariant(exact, k_actions), atol=1e-12)


def test_ibz2fbz_without_a_kgrid_keeps_every_distinct_image():
    ibz, k_actions = nonmagnetic_c4_wedge()
    ebs = ElectronicBandStructure(
        kpoints=ibz,
        bands=invariant(ibz, k_actions).reshape(-1, 1, 1),
        reciprocal_lattice=cast(kpoints.RECIPROCAL_LATTICE_DTYPE, np.eye(3)),
        structure=single_site(rotations=C4_GROUP),
    )

    ibz2fbz(ebs, kgrid_info=None)

    assert ebs.n_kpoints == 27
    assert {key(k) for k in ebs.kpoints} == {key(k) for k in gamma_grid((3, 3, 3))}
    np.testing.assert_allclose(band_values(ebs), invariant(ebs.kpoints, k_actions), atol=1e-12)


def test_a_magnetic_group_without_inversion_puts_each_value_at_its_own_image():
    grid = gamma_grid((4, 4, 4))
    ibz = wedge(grid, C4_GROUP)
    bands = invariant(ibz, C4_GROUP)
    assert np.abs(invariant(-ibz, C4_GROUP) - bands).max() > 0.5

    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=bands.reshape(-1, 1, 1),
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=single_site(rotations=C4_GROUP),
        kgrid_info=gamma_info((4, 4, 4)),
    )

    assert (len(ibz), ebs.n_kpoints) == (24, 64)
    exact = np.round(ebs.kpoints * 4) / 4
    np.testing.assert_allclose(band_values(ebs), invariant(exact, C4_GROUP), atol=1e-12)


def test_an_operation_combined_with_time_reversal_sends_k_to_minus_rk_and_flips_the_spin():
    # Magnetic group 2'2'2 with the moment along z: C2z, and C2x and C2y combined with time
    # reversal. On k they act as C2z, -C2x and -C2y, a group without inversion.
    c2z, c2x, c2y = np.diag([-1.0, -1, 1]), np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1])
    rotations = np.array([np.eye(3), c2z, c2x, c2y])
    time_reversals = np.array([False, False, True, True])
    signs = np.where(time_reversals, -1.0, 1.0)
    k_actions = signs[:, None, None] * rotations

    def spin(k):
        """S(g k) = s det(R) R S(k) for g = (R, s): a group sum of a field without symmetry."""
        k = np.atleast_2d(k)
        total = np.zeros((len(k), 3))
        for r, s, m in zip(rotations, signs, k_actions, strict=True):
            source = k @ np.linalg.inv(m).T
            field = np.stack(
                [
                    asymmetric_field(source),
                    asymmetric_field(source + 0.1),
                    asymmetric_field(source + 0.2),
                ],
                axis=-1,
            )
            total += field @ (s * np.linalg.det(r) * r).T
        return total

    grid = gamma_grid((4, 4, 4))
    ibz = wedge(grid, k_actions)
    projected = np.concatenate([np.ones((len(ibz), 1)), spin(ibz)], axis=1)
    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=invariant(ibz, k_actions).reshape(-1, 1, 1),
        projected=projected.reshape(-1, 1, 4, 1, 1),
        orbital_names=["s"],
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=single_site(rotations=rotations, time_reversals=time_reversals),
        kgrid_info=gamma_info((4, 4, 4)),
    )

    assert ebs.n_kpoints == 64 and ebs.projected is not None
    exact = np.round(ebs.kpoints * 4) / 4
    np.testing.assert_allclose(band_values(ebs), invariant(exact, k_actions), atol=1e-12)
    np.testing.assert_allclose(ebs.projected.to_array()[:, 0, 1:, 0, 0], spin(exact), atol=1e-12)


def test_is_grid_is_false_for_an_ibz_whose_coordinates_form_a_smaller_grid():
    # The 4x1x1 grid reduced by k ~ -k is {0, 1/4, 1/2}: three evenly spaced values, not four
    ibz = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])

    assert ElectronicBandStructure(kpoints=ibz).is_grid is False
    assert ElectronicBandStructure(kpoints=gamma_grid((4, 1, 1))).is_grid is True

    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.array([1.0, 2.0, 3.0]).reshape(3, 1, 1),
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=single_site(rotations=np.array([np.eye(3)])),
        kgrid_info=gamma_info((4, 1, 1)),
    )

    band_at = {key(k): band_values(ebs)[i] for i, k in enumerate(ebs.kpoints)}
    assert band_at == {
        key([0, 0, 0]): 1.0,
        key([0.25, 0, 0]): 2.0,
        key([0.5, 0, 0]): 3.0,
        key([-0.25, 0, 0]): 2.0,
    }


def test_symmetry_images_swap_the_atoms_the_operation_swaps():
    # C2z maps the site (0.2, 0.1, 0) onto (0.8, 0.9, 0) and back
    structure = Structure(
        atoms=["X", "X"],
        fractional_coordinates=[[0.2, 0.1, 0], [0.8, 0.9, 0]],
        lattice=np.eye(3),
        rotations=np.array([np.eye(3), np.diag([-1.0, -1, 1])]),
    )
    k_actions = np.array([np.eye(3), np.diag([-1.0, -1, 1]), -np.eye(3), np.diag([1.0, 1, -1])])
    swaps = np.array([False, True, False, True])

    def weights(k):
        """P(g k, g a) = P(k, a): site a sums the field at g^-1 k over the g taking site 0 to a."""
        k = np.atleast_2d(k)
        result = np.zeros((len(k), 2))
        for m, swap in zip(k_actions, swaps, strict=True):
            result[:, int(swap)] += asymmetric_field(k @ np.linalg.inv(m).T) + 2
        return result

    grid = gamma_grid((4, 4, 4))
    ibz = wedge(grid, k_actions)
    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 1, 1)),
        projected=weights(ibz).reshape(-1, 1, 1, 2, 1),
        orbital_names=["s"],
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=structure,
        kgrid_info=gamma_info((4, 4, 4)),
    )

    assert ebs.n_kpoints == 64 and ebs.projected is not None
    exact = np.round(ebs.kpoints * 4) / 4
    np.testing.assert_allclose(ebs.projected.to_array()[:, 0, 0, :, 0], weights(exact), atol=1e-12)


def test_symmetry_images_cycle_the_three_oxygens_the_way_the_third_turn_does():
    # C3 about [111] sends (x, y, z) to (z, x, y): the O at (1/2, 1/2, 0) lands on the one at
    # (0, 1/2, 1/2), that one on (1/2, 0, 1/2), and that one back on (1/2, 1/2, 0).
    c3 = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    structure = Structure(
        atoms=["O", "O", "O"],
        fractional_coordinates=[[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        lattice=np.eye(3),
        rotations=np.array([np.eye(3), c3, c3 @ c3]),
    )
    turns = [np.eye(3), c3, c3 @ c3]
    k_actions = np.array([sign * turn for sign in (1, -1) for turn in turns])
    lands_on = [0, 2, 1, 0, 2, 1]

    def weights(k):
        """P(g k, g a) = P(k, a): site a sums the field at g^-1 k over the g taking site 0 to a."""
        k = np.atleast_2d(k)
        result = np.zeros((len(k), 3))
        for m, site in zip(k_actions, lands_on, strict=True):
            result[:, site] += asymmetric_field(k @ np.linalg.inv(m).T) + 2
        return result

    grid = gamma_grid((4, 4, 4))
    ibz = wedge(grid, k_actions)
    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 1, 1)),
        projected=weights(ibz).reshape(-1, 1, 1, 3, 1),
        orbital_names=["s"],
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=structure,
        kgrid_info=gamma_info((4, 4, 4)),
    )

    assert ebs.n_kpoints == 64 and ebs.projected is not None
    exact = np.round(ebs.kpoints * 4) / 4
    np.testing.assert_allclose(ebs.projected.to_array()[:, 0, 0, :, 0], weights(exact), atol=1e-12)


def pure_orbital_projections(n_kpoints, orbitals):
    """Band i is the pure real orbital orbitals[i] at every k-point."""
    projected = np.zeros((n_kpoints, len(orbitals), 1, 1, len(VASP_ORBITALS)))
    for band, orbital in enumerate(orbitals):
        projected[:, band, 0, 0, VASP_ORBITALS.index(orbital)] = 1.0
    return projected


def weights_by_name(row):
    return {
        name: round(float(w), 12)
        for name, w in zip(VASP_ORBITALS, row, strict=True)
        if abs(w) > 1e-12
    }


def test_a_quarter_turn_carries_px_to_py_and_dxz_to_dyz():
    grid = gamma_grid((4, 4, 1))
    ibz = wedge(grid, np.concatenate([C4_GROUP, -C4_GROUP]), first=[0.25, 0, 0])
    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 3, 1)),
        projected=pure_orbital_projections(len(ibz), ["px", "dxz", "x2-y2"]),
        orbital_names=VASP_ORBITALS,
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=single_site(rotations=C4_GROUP),
        kgrid_info=gamma_info((4, 4, 1)),
    )

    assert ebs.projected is not None
    at = {key(k): i for i, k in enumerate(ebs.kpoints)}
    # (0, 1/4, 0) is the quarter turn of (1/4, 0, 0): x -> y, so px -> py and xz -> yz
    image = ebs.projected.to_array()[at[key([0, 0.25, 0])], :, 0, 0]
    assert [weights_by_name(row) for row in image] == [{"py": 1.0}, {"dyz": 1.0}, {"x2-y2": 1.0}]


def hexagonal_c3():
    b_t = np.linalg.inv(HEXAGONAL)
    angle = 2 * np.pi / 3
    turn = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    fractional = np.array(
        [np.round(np.linalg.solve(b_t, np.linalg.matrix_power(turn, n) @ b_t)) for n in range(3)]
    )
    return turn, fractional, b_t.T


def test_a_third_turn_splits_a_pure_px_state_a_quarter_px_and_three_quarters_py():
    turn, rotations, reciprocal_lattice = hexagonal_c3()
    grid = gamma_grid((6, 6, 1))
    source = np.array([1 / 6, 0, 0])
    ibz = wedge(grid, np.concatenate([rotations, -rotations]), first=source)
    spin = np.array([0.1, 0.2, 0.3])
    projected = np.zeros((len(ibz), 2, 4, 1, len(VASP_ORBITALS)))
    projected[:, 0, 0, 0, VASP_ORBITALS.index("px")] = 1.0
    projected[:, 0, 1:, 0, VASP_ORBITALS.index("px")] = spin
    projected[:, 1, 0, 0, VASP_ORBITALS.index("dxy")] = 1.0
    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 2, 1)),
        projected=projected,
        orbital_names=VASP_ORBITALS,
        fermi=0.0,
        reciprocal_lattice=reciprocal_lattice,
        structure=single_site(lattice=HEXAGONAL, rotations=rotations),
        kgrid_info=gamma_info((6, 6, 1)),
    )

    assert ebs.projected is not None
    image_cartesian = turn @ (source @ reciprocal_lattice)
    image = np.linalg.solve(reciprocal_lattice.T, image_cartesian)
    unfolded = ebs.projected.to_array()[{key(k): i for i, k in enumerate(ebs.kpoints)}[key(image)]]
    # cos^2(120) = 1/4 and sin^2(120) = 3/4; d_xy turns by twice the angle, cos^2(240) = 1/4
    assert weights_by_name(unfolded[0, 0, 0]) == {"py": 0.75, "px": 0.25}
    assert weights_by_name(unfolded[1, 0, 0]) == {"dxy": 0.25, "x2-y2": 0.75}
    turned = turn @ spin
    np.testing.assert_allclose(
        unfolded[0, 1:, 0, VASP_ORBITALS.index("px")], 0.25 * turned, atol=1e-12
    )
    np.testing.assert_allclose(
        unfolded[0, 1:, 0, VASP_ORBITALS.index("py")], 0.75 * turned, atol=1e-12
    )


def test_time_reversal_conjugates_the_projection_phase():
    t = 1 / 3
    ibz = np.array([[0, 0, 0], [t, 0, 0], [0, t, 0], [t, t, 0], [t, -t, 0]])
    phase = (1.0 + 1j * np.arange(5)).reshape(5, 1, 1, 1, 1)
    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.zeros((5, 1, 1)),
        projected=np.abs(phase) ** 2,
        projected_phase=phase,
        orbital_names=["s"],
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=single_site(rotations=np.array([np.eye(3)])),
        kgrid_info=gamma_info((3, 3, 1)),
    )

    assert ebs.projected_phase is not None
    phase_at = {
        key(k): complex(ebs.projected_phase.to_array()[i, 0, 0, 0, 0])
        for i, k in enumerate(ebs.kpoints)
    }
    assert phase_at[key([t, 0, 0])] == pytest.approx(1 + 1j)
    assert phase_at[key([-t, 0, 0])] == pytest.approx(1 - 1j)
    assert phase_at[key([t, -t, 0])] == pytest.approx(1 + 4j)
    assert phase_at[key([-t, t, 0])] == pytest.approx(1 - 4j)
    assert phase_at[key([0, 0, 0])] == pytest.approx(1)


def test_a_quarter_turn_carries_the_px_phase_to_py():
    grid = gamma_grid((4, 4, 1))
    ibz = wedge(grid, np.concatenate([C4_GROUP, -C4_GROUP]), first=[0.25, 0, 0])
    phase = np.zeros((len(ibz), 1, 1, 1, len(VASP_ORBITALS)), dtype=complex)
    phase[..., VASP_ORBITALS.index("px")] = 0.6 + 0.8j
    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 1, 1)),
        projected=np.abs(phase) ** 2,
        projected_phase=phase,
        orbital_names=VASP_ORBITALS,
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=single_site(rotations=C4_GROUP),
        kgrid_info=gamma_info((4, 4, 1)),
    )

    assert ebs.projected_phase is not None
    at = {key(k): i for i, k in enumerate(ebs.kpoints)}
    image = ebs.projected_phase.to_array()[at[key([0, 0.25, 0])], 0, 0, 0]
    expected = np.zeros(len(VASP_ORBITALS), dtype=complex)
    expected[VASP_ORBITALS.index("py")] = 0.6 + 0.8j
    np.testing.assert_allclose(image, expected, atol=1e-12)


def test_a_phase_moved_by_a_reciprocal_lattice_vector_turns_by_the_atom_position():
    # VASP's phases use Bloch sums with exp(i k.(R + tau)), so c(k + G) = exp(-2 pi i G.tau) c(k).
    # The third turns send (1/6, 1/2, 0) to (-2/3, 1/6, 0) and (1/2, -2/3, 0), stored as
    # (1/3, 1/6, 0) and (1/2, 1/3, 0), so G = (1, 0, 0) and (0, 1, 0). The atom sits at the
    # threefold site (1/3, 2/3, 0): G.tau is 1/3 and 2/3, which tell exp(-2 pi i) from exp(+2 pi i).
    _, rotations, reciprocal_lattice = hexagonal_c3()
    grid = gamma_grid((6, 6, 1))
    source = [1 / 6, 1 / 2, 0]
    ibz = wedge(grid, np.concatenate([rotations, -rotations]), first=source)
    phase = np.full((len(ibz), 1, 1, 1, 1), 0.6 + 0.8j)
    structure = Structure(
        atoms=["X"],
        fractional_coordinates=[[1 / 3, 2 / 3, 0]],
        lattice=HEXAGONAL,
        rotations=rotations,
    )
    ebs = ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 1, 1)),
        projected=np.abs(phase) ** 2,
        projected_phase=phase,
        orbital_names=["s"],
        fermi=0.0,
        reciprocal_lattice=reciprocal_lattice,
        structure=structure,
        kgrid_info=gamma_info((6, 6, 1)),
    )

    assert ebs.projected_phase is not None
    phase_at = {
        key(k): complex(ebs.projected_phase.to_array()[i, 0, 0, 0, 0])
        for i, k in enumerate(ebs.kpoints)
    }
    assert phase_at[key(source)] == pytest.approx(0.6 + 0.8j)
    assert phase_at[key([1 / 3, 1 / 6, 0])] == pytest.approx(np.exp(-2j * np.pi / 3) * (0.6 + 0.8j))
    assert phase_at[key([1 / 2, 1 / 3, 0])] == pytest.approx(np.exp(-4j * np.pi / 3) * (0.6 + 0.8j))


def test_phases_decide_how_a_third_turn_splits_equal_px_and_py_weights():
    # (px + py)/sqrt(2) and (px + i py)/sqrt(2) have the same weights, 1/2 and 1/2, but a 120
    # degree turn sends the first to (1 + sqrt(3)/2)/2 px and the second to 1/2 px.
    turn, rotations, reciprocal_lattice = hexagonal_c3()
    grid = gamma_grid((6, 6, 1))
    source = np.array([1 / 6, 0, 0])
    ibz = wedge(grid, np.concatenate([rotations, -rotations]), first=source)
    phase = np.zeros((len(ibz), 2, 1, 1, len(VASP_ORBITALS)), dtype=complex)
    phase[:, :, 0, 0, VASP_ORBITALS.index("px")] = 1 / np.sqrt(2)
    phase[:, 0, 0, 0, VASP_ORBITALS.index("py")] = 1 / np.sqrt(2)
    phase[:, 1, 0, 0, VASP_ORBITALS.index("py")] = 1j / np.sqrt(2)

    def unfold(projected_phase):
        return ElectronicBandStructureMesh(
            kpoints=ibz,
            bands=np.zeros((len(ibz), 2, 1)),
            projected=np.abs(phase) ** 2,
            projected_phase=projected_phase,
            orbital_names=VASP_ORBITALS,
            fermi=0.0,
            reciprocal_lattice=reciprocal_lattice,
            structure=single_site(lattice=HEXAGONAL, rotations=rotations),
            kgrid_info=gamma_info((6, 6, 1)),
        )

    image = np.linalg.solve(reciprocal_lattice.T, turn @ (source @ reciprocal_lattice))
    px, py = VASP_ORBITALS.index("px"), VASP_ORBITALS.index("py")
    for projected_phase, expected in [
        (phase, [[(2 + np.sqrt(3)) / 4, (2 - np.sqrt(3)) / 4], [0.5, 0.5]]),
        (None, [[0.5, 0.5], [0.5, 0.5]]),
    ]:
        ebs = unfold(projected_phase)
        assert ebs.projected is not None
        at = {key(k): i for i, k in enumerate(ebs.kpoints)}
        weights = ebs.projected.to_array()[at[key(image)], :, 0, 0]
        np.testing.assert_allclose(weights[:, [px, py]], expected, atol=1e-12)


def quarter_turn_mesh(n_orbitals, phase=None):
    grid = gamma_grid((4, 4, 1))
    ibz = wedge(grid, np.concatenate([C4_GROUP, -C4_GROUP]))
    return ElectronicBandStructureMesh(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 1, 1)),
        projected=np.full((len(ibz), 1, 1, 1, n_orbitals), 0.5),
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=single_site(rotations=C4_GROUP),
        projected_phase=None if phase is None else np.full((len(ibz), 1, 1, 1, n_orbitals), phase),
        kgrid_info=gamma_info((4, 4, 1)),
    )


def test_phases_over_orbitals_without_real_harmonic_names_are_dropped_with_a_warning():
    # QE spin-orbit phases come over (l, j, m_j) states with no orbital names. Time reversal
    # also sends m_j to -m_j there, so conjugating each coefficient in place would be wrong.
    with pytest.warns(UserWarning, match="projected_phase is dropped"):
        ebs = quarter_turn_mesh(2, phase=0.6 + 0.8j)

    assert ebs.n_kpoints == 16
    assert ebs.projected_phase is None
    assert ebs.projected is not None
    np.testing.assert_allclose(ebs.projected.to_array(), 0.5)


def test_the_unrotated_orbital_warning_lists_the_orbitals_in_numeric_order():
    names = ", ".join(f"'orbital {i}'" for i in range(11))

    with pytest.warns(UserWarning, match=re.escape(f"[{names}]")):
        quarter_turn_mesh(11)


def time_reversed_wedge():
    # The calculation reduced the 4x4x4 grid by C2z and plain time reversal, k ~ -k
    c2z = np.diag([-1.0, -1, 1])
    k_actions = np.array([np.eye(3), c2z, -np.eye(3), -c2z])
    return wedge(gamma_grid((4, 4, 4)), k_actions), c2z


def test_time_reversal_filling_points_of_a_magnetic_group_warns():
    # Listed as the magnetic group {E, C2z with time reversal}, k -> -C2z k reaches only
    # half of the missing points; plain time reversal, not in that group, fills the rest.
    ibz, c2z = time_reversed_wedge()
    structure = single_site(
        rotations=np.array([np.eye(3), c2z]), time_reversals=np.array([False, True])
    )
    ebs = ElectronicBandStructure(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 1, 1)),
        reciprocal_lattice=cast(kpoints.RECIPROCAL_LATTICE_DTYPE, np.eye(3)),
        structure=structure,
    )

    with pytest.warns(UserWarning, match="time reversal"):
        ibz2fbz(ebs, kgrid_info=gamma_info((4, 4, 4)))

    assert ebs.n_kpoints == 64


@pytest.mark.guards_existing_behaviour(
    reason="time reversal stays a silent fill-in for groups without time-reversal flags"
)
def test_time_reversal_filling_points_of_a_group_without_flags_does_not_warn():
    ibz, c2z = time_reversed_wedge()
    ebs = ElectronicBandStructure(
        kpoints=ibz,
        bands=np.zeros((len(ibz), 1, 1)),
        reciprocal_lattice=cast(kpoints.RECIPROCAL_LATTICE_DTYPE, np.eye(3)),
        structure=single_site(rotations=np.array([np.eye(3), c2z])),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ibz2fbz(ebs, kgrid_info=gamma_info((4, 4, 4)))

    assert ebs.n_kpoints == 64


@pytest.mark.data
def test_reduced_bisb_orbital_weights_and_spin_follow_each_rotation():
    spglib = pytest.importorskip("spglib")
    from pyprocar.core.ebs import get_ebs_from_code, get_ebs_from_data
    from tests.utils import DATA_DIR

    # The full 60x60x1 non-collinear mesh (ISYM=-1) is the truth. Reduce it under the point
    # group and k ~ -k, unfold, and compare each image with the full mesh, bands 0-59.
    full = get_ebs_from_code("vasp", str(DATA_DIR / "examples" / "fermi2d" / "bisb_monolayer"))
    structure = full.structure
    assert isinstance(full, ElectronicBandStructureMesh) and structure is not None
    assert full.bands is not None and full.projected is not None
    full_projected = full.projected.to_array()
    cell = (structure.lattice, structure.fractional_coordinates, structure.atomic_numbers)
    rotations = np.array([w.T for w in spglib.get_symmetry(cell, symprec=1e-3)["rotations"]])
    structure._rotations = rotations.astype(float)
    full_keys = [key(np.round(k, 4)) for k in full.kpoints]
    kept, seen = [], set()
    for i, k in enumerate(full.kpoints):
        if full_keys[i] not in seen:
            kept.append(i)
            seen.update(key(np.round(s * (r @ k), 4)) for r in rotations for s in (1, -1))
    reduced = get_ebs_from_data(
        kpoints=full.kpoints[kept],
        bands=full.bands.to_array()[kept],
        projected=full_projected[kept],
        fermi=full.fermi,
        reciprocal_lattice=full.reciprocal_lattice,
        orbital_names=full.orbital_names,
        structure=structure,
        kgrid_info=full.kgrid_info,
    )
    assert reduced.projected is not None and reduced.n_kpoints == 3600
    index = {k: i for i, k in enumerate(full_keys)}
    target = [index[key(np.round(k, 4))] for k in reduced.kpoints]
    unfolded, true = reduced.projected.to_array()[:, :60], full_projected[target, :60]

    # Rotations by 120 degrees mix px with py and dxy with dx2-y2. Weights alone fix an image
    # only up to the interference between the mixed orbitals, which LORBIT=11 does not keep.
    # Unmixed, the mean errors are 0.0037 (weights) and 0.0049 (spin); mixed, 0.0013 and 0.0024.
    weight_error = np.abs(unfolded[:, :, 0] - true[:, :, 0]).mean()
    spin_error = np.abs(unfolded[:, :, 1:] - true[:, :, 1:]).mean()
    assert weight_error < 0.0025
    assert spin_error < 0.0035
    for columns in ([1, 2, 3], [4, 5, 6, 7, 8]):
        shell_error = np.abs(
            unfolded[:, :, 0][..., columns].sum(-1) - true[:, :, 0][..., columns].sum(-1)
        )
        assert shell_error.mean() < 4e-4
