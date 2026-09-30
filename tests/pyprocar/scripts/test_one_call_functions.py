"""Smoke tests for the one-call plotting functions, called the way the example notebooks call them."""

import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import LineCollection, PathCollection

import pyprocar
from tests.utils import DATA_DIR

FERMI = 5.3017
TICK_NAMES = [r"$\Gamma$", "M", r"$\Gamma$", "R", "X"]


def _calc(tmp_path, relpath):
    src = DATA_DIR / "examples" / relpath
    if not src.exists():
        pytest.skip(f"fixture {relpath} not downloaded")
    dst = tmp_path / "calc"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.pkl", "CHG*", "WAVECAR", "*.pdf"))
    return dst


def _rendered(png):
    """A screenshot with more than one distinct color drew something besides the background."""
    image = plt.imread(png)
    return bool((image != image[0, 0]).any())


def _lines_of_length(ax, n):
    return [line for line in ax.lines if len(line.get_xdata()) == n]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestBandsplot:
    def test_plain_draws_one_line_per_band_with_kpath_ticks(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")
        out = tmp_path / "bands.png"

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="plain",
            fermi=FERMI,
            elimit=[-5, 5],
            title="Basic Band Structure",
            show=False,
            savefig=out,
        )

        assert out.stat().st_size > 0
        assert len(_lines_of_length(ax, 200)) == 20
        assert [t.get_text() for t in ax.get_xticklabels()] == TICK_NAMES
        assert ax.get_ylim() == (-5.0, 5.0)
        assert ax.get_title() == "Basic Band Structure"

    def test_parametric_spin_polarized_colors_both_channels(self, tmp_path):
        calc = _calc(tmp_path, "bands/spin-polarized")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            atoms=[1],
            orbitals=[4, 5, 6, 7, 8],
            elimit=[-5, 5],
            show=False,
        )

        collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(collections) == 60
        assert len(fig.axes) == 2

    def test_parametric_single_spin_channel(self, tmp_path):
        calc = _calc(tmp_path, "bands/spin-polarized")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            atoms=[1],
            orbitals=[4, 5, 6, 7, 8],
            spins=[1],
            elimit=[-5, 5],
            show=False,
        )

        collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(collections) == 30

    def test_scatter_draws_one_collection_per_band(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="scatter",
            fermi=FERMI,
            atoms=[1],
            orbitals=[4, 5, 6, 7, 8],
            elimit=[-5, 5],
            show=False,
        )

        assert len([c for c in ax.collections if isinstance(c, PathCollection)]) == 20

    def test_overlay_species_labels_each_species(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="overlay_species",
            fermi=FERMI,
            orbitals=[1, 2, 3],
            elimit=[-5, 5],
            show=False,
        )

        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["O", "Sr", "V"]

    def test_overlay_orbitals_labels_each_orbital_group(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="overlay_orbitals",
            fermi=FERMI,
            atoms=[1],
            elimit=[-5, 5],
            show=False,
        )

        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["s", "p", "d"]

    def test_overlay_items_labels_each_species(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="overlay",
            fermi=FERMI,
            items={"O": [1, 2, 3], "V": [4, 5, 6, 7, 8]},
            elimit=[-5, 5],
            show=False,
        )

        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["O", "V"]

    def test_non_collinear_parametric_colors_one_channel_by_spin_component(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-colinear")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            spins=[3],
            elimit=[-5, 5],
            show=False,
        )

        assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 50

    def test_ipr_colors_both_spin_channels(self, tmp_path):
        calc = _calc(tmp_path, "bands/spin-polarized")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="ipr",
            fermi=FERMI,
            spins=[0, 1],
            clim=[0, 0.1],
            elimit=[-5, 5],
            show=False,
        )

        collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(collections) == 60
        assert collections[0].get_clim() == (0.0, 0.1)
        assert fig.axes[1].get_ylabel() == "Inverse Participation Ratio"

    def test_atomic_levels_draw_one_segment_per_level(self, tmp_path):
        calc = _calc(tmp_path, "bands/atomic_levels/hBN-CNCN")

        fig, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="atomic",
            elimit=[-6, 6],
            clim=[-0.4, 0.4],
            atoms=[96, 97],
            fermi=-4.6628,
            show=False,
        )

        assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 2 * 246
        assert ax.get_ylim() == (-6.0, 6.0)

    def test_draws_onto_a_given_axes(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")
        fig, ax = plt.subplots()

        returned_fig, returned_ax = pyprocar.bandsplot(
            code="vasp", dirname=calc, mode="plain", fermi=FERMI, ax=ax, show=False
        )

        assert returned_ax is ax
        assert returned_fig is fig
        assert len(_lines_of_length(ax, 200)) == 20


class TestDosplot:
    def test_plain_draws_total_dos_and_keeps_user_files(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-spin-polarized")
        sentinel = calc / "dos.pkl"
        sentinel.write_bytes(b"user file")
        out = tmp_path / "dos.png"

        fig, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="plain",
            fermi=FERMI,
            elimit=[-6, 4],
            show=False,
            savefig=out,
        )

        assert out.stat().st_size > 0
        assert sentinel.read_bytes() == b"user file"
        assert ax.get_xlim() == (-6.0, 4.0)
        assert [line.get_label() for line in ax.lines if not line.get_label().startswith("_")] == [
            "Total"
        ]

    def test_parametric_fills_total_with_projection_colors(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-spin-polarized")

        fig, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            elimit=[-6, 4],
            dos_limit=[0, 30],
            atoms=[2, 3, 4],
            orbitals=[1, 2, 3],
            show=False,
        )

        assert len(ax.images) == 1
        assert len(fig.axes) == 2
        assert ax.get_ylim() == (0.0, 30.0)

    def test_stack_species_fills_one_band_per_species(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-spin-polarized")

        fig, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="stack_species",
            fermi=FERMI,
            elimit=[-6, 4],
            orbitals=[1, 2, 3],
            show=False,
        )

        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "Total",
            r"$\mathrm{O}_{2-4}-(p)[\uparrow]$",
            r"$\mathrm{Sr}_{0}-(p)[\uparrow]$",
            r"$\mathrm{V}_{1}-(p)[\uparrow]$",
        ]
        assert len(ax.collections) == 3

    def test_vertical_overlay_orbitals(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-spin-polarized")

        fig, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="overlay_orbitals",
            fermi=FERMI,
            elimit=[-4, 4],
            atoms=[1],
            orientation="vertical",
            show=False,
        )

        assert ax.get_ylim() == (-4.0, 4.0)
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "Total",
            r"$\mathrm{V}_{1}-(s)[\uparrow]$",
            r"$\mathrm{V}_{1}-(p)[\uparrow]$",
            r"$\mathrm{V}_{1}-(d)[\uparrow]$",
        ]

    def test_spin_channels_can_share_one_axes(self, tmp_path):
        calc = _calc(tmp_path, "dos/spin-polarized")

        fig, ax = pyprocar.dosplot(
            code="vasp", dirname=calc, mode="plain", fermi=FERMI, spins=[0], show=False
        )
        pyprocar.dosplot(
            code="vasp", dirname=calc, mode="plain", fermi=FERMI, spins=[1], ax=ax, show=False
        )

        labels = [line.get_label() for line in ax.lines if not line.get_label().startswith("_")]
        assert labels == [r"$Total - \uparrow$", r"$Total - \downarrow$"]


class TestFermi2D:
    def test_plain_draws_contours(self, tmp_path):
        calc = _calc(tmp_path, "fermi2d/non-spin-polarized")
        out = tmp_path / "fermi2d.png"

        fig, ax = pyprocar.fermi2D(
            code="vasp",
            dirname=calc,
            mode="plain",
            fermi=FERMI,
            energy=0.0,
            k_z_plane=0.0,
            show=False,
            savefig=out,
        )

        assert out.stat().st_size > 0
        segments = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(segments) == 1
        assert len(segments[0].get_segments()) > 0

    def test_parametric_adds_colorbar(self, tmp_path):
        calc = _calc(tmp_path, "fermi2d/non-spin-polarized")

        fig, ax = pyprocar.fermi2D(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            energy=0.0,
            k_z_plane=0.0,
            atoms=[1],
            orbitals=[4, 5, 6, 7, 8],
            show=False,
        )

        assert len(fig.axes) == 2
        segments = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert segments[0].get_array() is not None


class TestFermiHandler:
    @pytest.fixture
    def handler(self, tmp_path):
        calc = _calc(tmp_path, "fermi3d/non-spin-polarized")
        handler = pyprocar.FermiHandler(code="vasp", dirname=calc, fermi=FERMI, verbose=0)
        shutil.rmtree(calc)
        return handler

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "plain"},
            {"mode": "parametric", "atoms": [1], "orbitals": [4, 5, 6, 7, 8], "spins": [0]},
            {"mode": "fermi_speed"},
        ],
        ids=["plain", "parametric", "fermi_speed"],
    )
    def test_plot_fermi_surface_saves_a_screenshot(self, handler, tmp_path, kwargs):
        out = tmp_path / "fs.png"

        handler.plot_fermi_surface(**kwargs, show=False, save_2d=out, off_screen=True)

        assert _rendered(out)


class TestBandsdosplot:
    def test_bands_and_dos_share_the_energy_axis(self, tmp_path):
        bands_calc = _calc(tmp_path / "b", "bands/non-spin-polarized")
        dos_calc = _calc(tmp_path / "d", "dos/non-spin-polarized")
        out = tmp_path / "bandsdos.png"

        fig, ax_bands, ax_dos = pyprocar.bandsdosplot(
            bands_settings={"mode": "plain", "dirname": bands_calc, "fermi": FERMI},
            dos_settings={"mode": "plain", "dirname": dos_calc, "fermi": FERMI},
            elimit=[-4, 4],
            show=False,
            savefig=out,
        )

        assert out.stat().st_size > 0
        assert ax_bands.get_ylim() == (-4.0, 4.0)
        assert ax_dos.get_ylim() == (-4.0, 4.0)
        assert len(_lines_of_length(ax_bands, 200)) == 20

    def test_default_settings_are_not_mutated(self, tmp_path):
        import inspect

        from pyprocar.scripts.scriptBandsDosplot import bandsdosplot

        defaults_before = {
            name: param.default
            for name, param in inspect.signature(bandsdosplot).parameters.items()
        }
        bands_calc = _calc(tmp_path / "b", "bands/non-spin-polarized")
        dos_calc = _calc(tmp_path / "d", "dos/non-spin-polarized")

        bandsdosplot(
            bands_settings={"mode": "plain", "dirname": bands_calc, "fermi": FERMI},
            dos_settings={"mode": "plain", "dirname": dos_calc, "fermi": FERMI},
            show=False,
        )

        defaults_after = {
            name: param.default
            for name, param in inspect.signature(bandsdosplot).parameters.items()
        }
        assert defaults_after == defaults_before
        assert defaults_after["bands_settings"] is None
        assert defaults_after["dos_settings"] is None


class TestBandStructure2DHandler:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "plain"},
            {"mode": "parametric", "atoms": [0, 1], "orbitals": [1, 2, 3], "spins": [0]},
        ],
        ids=["plain", "parametric"],
    )
    def test_plot_band_structure_saves_a_screenshot(self, tmp_path, kwargs):
        calc = _calc(tmp_path, "bands/2d-bands/graphene")
        handler = pyprocar.BandStructure2DHandler(code="vasp", dirname=calc, fermi=-0.795606)
        shutil.rmtree(calc)
        out = tmp_path / "bs2d.png"

        handler.plot_band_structure(
            **kwargs,
            add_fermi_plane=True,
            fermi_plane_size=4,
            energy_lim=[-2.5, 2.0],
            show=False,
            render_offscreen=True,
            save_2d=out,
        )

        assert _rendered(out)
