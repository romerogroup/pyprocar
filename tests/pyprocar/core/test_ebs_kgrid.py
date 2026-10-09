"""The Mesh carries its grid size instead of re-inferring it from histogram peaks (#302, item I)."""

import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo, get_kpoints_from_kgrid


def grid_offset(mode: KGRID_MODE, n: int) -> float:
    """k * n - offset is an integer on an n-point axis: Gamma j/n, Monkhorst-Pack (2j+1-n)/2n."""
    return 0.0 if mode is KGRID_MODE.GAMMA else (1 - n) / 2


def cell_code(kpoints: np.ndarray, kgrid: tuple[int, int, int], mode: KGRID_MODE) -> np.ndarray:
    """An integer per k-point that names its grid point modulo a reciprocal lattice vector."""
    n = np.array(kgrid)
    offsets = np.array([grid_offset(mode, int(size)) for size in n])
    index = np.rint(kpoints * n - offsets).astype(int) % n
    return index[:, 0] + 1000 * index[:, 1] + 1000_000 * index[:, 2]


def grid_mesh(
    kgrid: tuple[int, int, int], mode: KGRID_MODE = KGRID_MODE.GAMMA
) -> ElectronicBandStructureMesh:
    """A full grid whose one band is the periodic ``cell_code`` of each k-point."""
    kpoints = get_kpoints_from_kgrid(kgrid, (0.0, 0.0, 0.0), mode)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=kgrid, kgrid_mode=mode, kshift=(0.0, 0.0, 0.0)),
        kpoints=kpoints,
        bands=cell_code(kpoints, kgrid, mode).astype(float)[:, np.newaxis, np.newaxis],
        reciprocal_lattice=np.eye(3),
    )


def test_kgrid_is_the_known_grid_past_the_histogram_limit():
    """I1: the histogram reads a 492-point axis as 491, and pad then fails to reshape."""
    ebs = grid_mesh((492, 2, 2))

    assert tuple(ebs.kgrid) == (492, 2, 2)
    assert tuple(ebs.pad(10, inplace=False).kgrid) == (512, 22, 22)


def test_is_fbz_compares_with_the_known_grid():
    """I2: a padded grid holds more points than the calculation's grid, so it is not that grid."""
    ebs = grid_mesh((16, 16, 16))
    assert ebs.is_fbz
    assert not ebs.is_ibz

    padded = ebs.pad(10, inplace=False)

    assert not padded.is_fbz
    assert not padded.is_ibz


@pytest.mark.guards_existing_behaviour(
    reason="carrying the size leaves pad's arrays as they were where the histogram reads the grid"
)
@pytest.mark.parametrize(
    ("kgrid", "mode"),
    [((16, 16, 16), KGRID_MODE.GAMMA), ((12, 8, 6), KGRID_MODE.MONKHORST)],
)
def test_pad_arrays_are_unchanged(kgrid: tuple[int, int, int], mode: KGRID_MODE):
    """I3: pad(10) extends every axis by 10 grid steps and copies values periodically."""
    ebs = grid_mesh(kgrid, mode)
    lows = [
        int(np.rint(ebs.kpoints[:, axis].min() * n - grid_offset(mode, n)))
        for axis, n in enumerate(kgrid)
    ]
    axes = [
        (np.arange(low - 10, low + n + 10) + grid_offset(mode, n)) / n
        for low, n in zip(lows, kgrid, strict=True)
    ]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    expected_kpoints = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    padded = ebs.pad(10, inplace=False)

    np.testing.assert_allclose(padded.kpoints, expected_kpoints, rtol=0, atol=1e-12)
    bands = padded.get_property("bands")
    assert bands is not None
    assert np.array_equal(bands.value[:, 0, 0], cell_code(expected_kpoints, kgrid, mode))


def test_interpolate_carries_the_size():
    ebs = grid_mesh((250, 2, 2))

    assert tuple(ebs.interpolate(2, inplace=False).kgrid) == (500, 4, 4)


@pytest.mark.parametrize(
    ("kgrid", "new_kgrid"),
    [((4, 6, 2), (8, 12, 4)), ((5, 3, 4), (10, 6, 8)), ((6, 4, 1), (12, 8, 1))],
)
def test_interpolate_carries_the_size_of_its_kpoints(
    kgrid: tuple[int, int, int], new_kgrid: tuple[int, int, int]
):
    """Every axis doubles except a single-point one, and the carried size is the k-points' grid."""
    mesh = grid_mesh(kgrid).interpolate(2, inplace=False)

    assert tuple(mesh.kgrid) == new_kgrid
    assert mesh.n_kpoints == np.prod(new_kgrid)
    for axis, n in enumerate(new_kgrid):
        assert len(np.unique(mesh.kpoints[:, axis].round(9))) == n
    # Reshaped kx fastest, mesh axis i varies only k-component i.
    kpoints_mesh = mesh.kpoints.reshape(*new_kgrid, 3, order="F")
    for axis in range(3):
        assert np.allclose(np.diff(kpoints_mesh[..., axis], axis=(axis + 1) % 3), 0)
        assert np.allclose(np.diff(kpoints_mesh[..., axis], axis=(axis + 2) % 3), 0)


def test_expand_single_dimension_carries_the_size():
    ebs = grid_mesh((500, 2, 1))

    assert tuple(ebs.expand_single_dimension(inplace=False).kgrid) == (500, 2, 3)


@pytest.mark.guards_existing_behaviour(
    reason="an ebs.pkl cache written before the Mesh carried its size still reads its grid"
)
def test_cache_without_a_carried_size_falls_back_to_the_histogram():
    ebs = grid_mesh((6, 4, 2))
    # A Mesh pickled before this change has no _kgrid in its __dict__.
    vars(ebs).pop("_kgrid", None)

    assert tuple(ebs.kgrid) == (6, 4, 2)
