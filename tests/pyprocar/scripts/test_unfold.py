"""pyprocar.unfold on the MgB2 2x2x2 supercell, checked against the primitive cell calculation."""

import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection, PolyCollection

import pyprocar
from pyprocar.core import ElectronicBandStructurePath
from tests.utils import DATA_DIR

UNFOLDING = DATA_DIR / "examples" / "bands" / "unfolding"
SUPERCELL = np.diag([2, 2, 2])
FERMI = 5.2182
MG_ATOMS = list(range(8))
N_KPOINTS, N_BANDS = 150, 80

pytestmark = pytest.mark.data


def _copy(tmp_path, name):
    src = UNFOLDING / name
    if not src.exists():
        pytest.skip(f"fixture bands/unfolding/{name} not downloaded")
    dst = tmp_path / name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("CHG*", "*.pkl"))
    return dst


@pytest.fixture(scope="module")
def supercell(tmp_path_factory):
    return _copy(tmp_path_factory.mktemp("unfold"), "supercell")


@pytest.fixture(scope="module")
def unfolded(supercell):
    ebs = ElectronicBandStructurePath.from_code("vasp", supercell)
    ebs.unfold(transformation_matrix=SUPERCELL, structure=ebs.structure)
    return ebs


@pytest.fixture(scope="module")
def primitive_bands(tmp_path_factory):
    calc = _copy(tmp_path_factory.mktemp("primitive"), "primitive")
    bands = ElectronicBandStructurePath.from_code("vasp", calc).bands
    assert bands is not None
    return np.asarray(bands.value)[..., 0]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _midpoints(values):
    return (values[:-1] + values[1:]) / 2


def _weights(ebs):
    return np.asarray(ebs.weights.value)[..., 0]


class TestUnfoldedWeights:
    def test_mg_2p_semicore_unfolds_to_the_three_primitive_bands(self, unfolded, primitive_bands):
        bands = np.asarray(unfolded.bands.value)[..., 0]
        semicore = bands < -20

        assert set((primitive_bands < -20).sum(axis=1).tolist()) == {3}
        assert set(semicore.sum(axis=1).tolist()) == {24}
        np.testing.assert_allclose((_weights(unfolded) * semicore).sum(axis=1), 3.0, atol=5e-3)

    def test_unfolded_bands_are_the_primitive_bands(self, unfolded, primitive_bands):
        bands = np.asarray(unfolded.bands.value)[..., 0]
        weights = _weights(unfolded)
        unmatched_supercell, unmatched_primitive = [], []
        for ik in range(N_KPOINTS):
            visible = bands[ik][(weights[ik] > 0.25) & (bands[ik] > -10) & (bands[ik] < 12)]
            reference = primitive_bands[ik][(primitive_bands[ik] > -9.5) & (primitive_bands[ik] < 11.5)]
            unmatched_supercell += [e for e in visible if np.abs(primitive_bands[ik] - e).min() > 0.5]
            unmatched_primitive += [e for e in reference if np.abs(visible - e).min() > 0.5]

        assert unmatched_supercell == []
        assert unmatched_primitive == []

    def test_weights_are_fractions(self, unfolded):
        weights = _weights(unfolded)

        assert weights.shape == (N_KPOINTS, N_BANDS)
        assert weights.min() > -1e-9
        assert weights.max() < 1 + 1e-9


class TestUnfoldPlot:
    def test_color_mode_colors_each_band_by_its_weight(self, supercell, unfolded, tmp_path):
        out = tmp_path / "unfold.png"

        fig, ax = pyprocar.unfold(
            code="vasp",
            dirname=supercell,
            mode="plain",
            unfold_mode="color",
            transformation_matrix=SUPERCELL,
            fermi=FERMI,
            elimit=[-5, 5],
            title="MgB2 unfolded",
            show=False,
            savefig=out,
        )

        collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(collections) == N_BANDS
        for iband in (0, 30, 79):
            np.testing.assert_allclose(
                collections[iband].get_array(), _midpoints(_weights(unfolded)[:, iband])
            )
        assert len(fig.axes) == 2
        assert out.stat().st_size > 0
        assert ax.get_title() == "MgB2 unfolded"
        assert ax.get_ylim() == (-5.0, 5.0)

    def test_plain_mode_draws_the_supercell_bands_shifted_by_fermi(self, supercell, unfolded):
        _, ax = pyprocar.unfold(
            code="vasp",
            dirname=supercell,
            unfold_mode="color",
            transformation_matrix=SUPERCELL,
            fermi=FERMI,
            show=False,
        )

        lines = [line for line in ax.lines if len(line.get_ydata()) == N_KPOINTS]
        bands = np.asarray(unfolded.bands.value)[..., 0]
        assert len(lines) == N_BANDS
        np.testing.assert_allclose(lines[40].get_ydata(), bands[:, 40] - FERMI)

    def test_thickness_mode_sets_each_segment_width_by_the_weight(self, supercell, unfolded):
        _, ax = pyprocar.unfold(
            code="vasp",
            dirname=supercell,
            unfold_mode="thickness",
            transformation_matrix=SUPERCELL,
            fermi=FERMI,
            show=False,
        )

        collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(collections) == N_BANDS
        widths = np.asarray(collections[30].get_linewidth())
        expected = _midpoints(_weights(unfolded)[:, 30])
        np.testing.assert_allclose(widths / widths.max(), expected / expected.max())
        assert collections[30].get_array() is None

    def test_parametric_colors_by_projection_and_widths_by_weight(self, supercell, unfolded):
        projection = np.asarray(unfolded.compute_projected_sum(atoms=MG_ATOMS).value)[..., 0]

        _, ax = pyprocar.unfold(
            code="vasp",
            dirname=supercell,
            mode="parametric",
            unfold_mode="thickness",
            atoms=MG_ATOMS,
            transformation_matrix=SUPERCELL,
            fermi=FERMI,
            show=False,
        )

        collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        np.testing.assert_allclose(collections[30].get_array(), _midpoints(projection[:, 30]))
        widths = np.asarray(collections[30].get_linewidth())
        expected = _midpoints(_weights(unfolded)[:, 30])
        np.testing.assert_allclose(widths / widths.max(), expected / expected.max())

    def test_scatter_sizes_points_by_weight(self, supercell, unfolded):
        _, ax = pyprocar.unfold(
            code="vasp",
            dirname=supercell,
            mode="scatter",
            unfold_mode="both",
            transformation_matrix=SUPERCELL,
            fermi=FERMI,
            show=False,
        )

        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == N_BANDS
        sizes = scatters[30].get_sizes()
        expected = _weights(unfolded)[:, 30]
        np.testing.assert_allclose(sizes / sizes.max(), expected / expected.max())

    def test_overlay_species_fill_is_the_species_weight_times_the_unfolding_weight(
        self, supercell, unfolded
    ):
        mg = np.asarray(unfolded.compute_projected_sum(species=["Mg"]).value)[..., 0]
        bands = np.asarray(unfolded.bands.value)[..., 0] - FERMI

        _, ax = pyprocar.unfold(
            code="vasp",
            dirname=supercell,
            mode="overlay_species",
            unfold_mode="thickness",
            transformation_matrix=SUPERCELL,
            fermi=FERMI,
            show=False,
        )

        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["B", "Mg"]
        fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(fills) == 2 * N_BANDS
        lower_edge = fills[N_BANDS + 30].get_paths()[0].vertices[1 : N_KPOINTS + 1, 1]
        expected = bands[:, 30] - mg[:, 30] * _weights(unfolded)[:, 30] / 2
        np.testing.assert_allclose(lower_edge, expected, atol=1e-9)

    def test_color_mode_needs_plain_mode(self, supercell):
        with pytest.raises(ValueError, match="unfold_mode='color'"):
            pyprocar.unfold(
                code="vasp",
                dirname=supercell,
                mode="parametric",
                unfold_mode="color",
                atoms=MG_ATOMS,
                transformation_matrix=SUPERCELL,
                show=False,
            )

    def test_calculation_without_phases_is_rejected(self, tmp_path):
        calc = _copy(tmp_path, "primitive")

        with pytest.raises(ValueError, match="LORBIT = 12"):
            pyprocar.unfold(
                code="vasp", dirname=calc, transformation_matrix=SUPERCELL, show=False
            )
