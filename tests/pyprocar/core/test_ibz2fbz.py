from typing import cast

import numpy as np

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


def v(k):
    """A periodic function of fractional k with no symmetry at all."""
    x, y, z = 2 * np.pi * np.atleast_2d(k).T
    odd = np.sin(x + 2 * y + 3 * z) + 0.3 * np.sin(z) + 0.2 * np.sin(y)
    even = 0.5 * np.cos(2 * x - z) + 0.4 * np.cos(x + 3 * y - z) + 0.25 * np.cos(3 * x + y + 2 * z)
    return odd + even


def invariant(k, k_actions):
    """sum over g of v(g^-1 k): invariant under the group the matrices form, and nothing else."""
    k = np.atleast_2d(k)
    return np.sum([v(k @ np.linalg.inv(m).T) for m in k_actions], axis=0)


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
    # C4 alone: no inversion, and time reversal is broken, so E(-k) != E(k)
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
            field = np.stack([v(source), v(source + 0.1), v(source + 0.2)], axis=-1)
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
    # C2z maps the site (1/4, 0, 0) onto (3/4, 0, 0) and back
    structure = Structure(
        atoms=["X", "X"],
        fractional_coordinates=[[0.25, 0, 0], [0.75, 0, 0]],
        lattice=np.eye(3),
        rotations=np.array([np.eye(3), np.diag([-1.0, -1, 1])]),
    )
    k_actions = np.array([np.eye(3), np.diag([-1.0, -1, 1]), -np.eye(3), np.diag([1.0, 1, -1])])
    swaps = np.array([False, True, False, True])

    def weights(k):
        """P(g k, g a) = P(k, a): site a sums v(g^-1 k) over the g that bring site 0 to a."""
        k = np.atleast_2d(k)
        result = np.zeros((len(k), 2))
        for m, swap in zip(k_actions, swaps, strict=True):
            result[:, int(swap)] += v(k @ np.linalg.inv(m).T) + 2
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
    # The 3x3x1 grid reduced by k ~ -k alone
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
    assert phase_at[key([t, 0, 0])] == 1 + 1j and phase_at[key([-t, 0, 0])] == 1 - 1j
    assert phase_at[key([t, -t, 0])] == 1 + 4j and phase_at[key([-t, t, 0])] == 1 - 4j
    assert phase_at[key([0, 0, 0])] == 1


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
    assert {VASP_ORBITALS[i]: complex(c) for i, c in enumerate(image) if abs(c) > 1e-12} == {
        "py": 0.6 + 0.8j
    }
