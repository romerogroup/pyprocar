"""Smoke tests for the one-call plotting functions, called the way the notebooks call them."""

import ast
import inspect
import json
import re
import shutil
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection, PolyCollection

import pyprocar
from pyprocar.core import (
    DensityOfStates,
    ElectronicBandStructureMesh,
    ElectronicBandStructurePath,
    FermiSurface,
)
from pyprocar.core.property_store import Property
from tests.utils import DATA_DIR, ROOT_DIR

FERMI = 5.3017
V_ATOM = [1]
D_ORBITALS = [4, 5, 6, 7, 8]
TICK_NAMES = [r"$\Gamma$", "X", "M", r"$\Gamma$", "R", "X"]


def _calc(tmp_path, relpath):
    src = DATA_DIR / "examples" / relpath
    if not src.exists():
        pytest.skip(f"fixture {relpath} not downloaded")
    dst = tmp_path / "calc"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.pkl", "CHG*", "WAVECAR", "*.pdf"))
    return dst


def _legend_texts(ax):
    legend = ax.get_legend()
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


def _projection(source, **selection) -> Property:
    return cast(Property, source.compute_projected_sum(**selection))


def _line_collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def _midpoints(values):
    return (values[:-1] + values[1:]) / 2


def _fill_extents(ax):
    """(max y, min y) of each fill_between polygon, in drawing order."""
    fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
    ys = [np.concatenate([p.vertices[:, 1] for p in fill.get_paths()]) for fill in fills]
    return [(y.max(), y.min()) for y in ys]


def _lines_of_length(ax, n):
    return [line for line in ax.lines if len(line.get_xdata()) == n]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.data
class TestBandsplot:
    def test_plain_draws_one_line_per_band_with_kpath_ticks(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")
        out = tmp_path / "bands.png"

        _, ax = pyprocar.bandsplot(
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

        _, ax = pyprocar.bandsplot(
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

        _, ax = pyprocar.bandsplot(
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

        _, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="overlay_species",
            fermi=FERMI,
            orbitals=[1, 2, 3],
            elimit=[-5, 5],
            show=False,
        )

        assert _legend_texts(ax) == ["O", "Sr", "V"]

    def test_overlay_orbitals_labels_each_orbital_group(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")

        _, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="overlay_orbitals",
            fermi=FERMI,
            atoms=[1],
            elimit=[-5, 5],
            show=False,
        )

        assert _legend_texts(ax) == ["s", "p", "d"]

    def test_overlay_items_labels_each_species(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")

        _, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="overlay",
            fermi=FERMI,
            items={"O": [1, 2, 3], "V": [4, 5, 6, 7, 8]},
            elimit=[-5, 5],
            show=False,
        )

        assert _legend_texts(ax) == ["O", "V"]

    def test_non_collinear_parametric_colors_one_channel_by_spin_component(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-colinear")

        _, ax = pyprocar.bandsplot(
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

        _, ax = pyprocar.bandsplot(
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

    def test_plain_shifts_bands_by_fermi_once(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")
        ebs = ElectronicBandStructurePath.from_code("vasp", calc)
        bands = cast(Property, ebs.bands).value

        _, ax = pyprocar.bandsplot(code="vasp", dirname=calc, mode="plain", fermi=FERMI, show=False)

        plotted = np.concatenate([line.get_ydata() for line in _lines_of_length(ax, 200)])
        assert plotted.min() == pytest.approx(bands.min() - FERMI)
        assert plotted.max() == pytest.approx(bands.max() - FERMI)

    def test_parametric_colors_equal_the_projected_sum(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")
        ebs = ElectronicBandStructurePath.from_code("vasp", calc)
        expected = ebs.compute_projected_sum(atoms=[1], orbitals=[4, 5, 6, 7, 8]).value

        _, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            atoms=[1],
            orbitals=[4, 5, 6, 7, 8],
            show=False,
        )

        collections = _line_collections(ax)
        for iband in (0, 10, 19):
            np.testing.assert_allclose(
                collections[iband].get_array(), _midpoints(expected[:, iband, 0])
            )

    def test_non_collinear_defaults_to_the_total_component(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-colinear")
        total = ElectronicBandStructurePath.from_code("vasp", calc).compute_projected_sum(spins=[0])

        _, ax = pyprocar.bandsplot(
            code="vasp", dirname=calc, mode="parametric", fermi=FERMI, show=False
        )

        collections = _line_collections(ax)
        assert len(collections) == 50
        np.testing.assert_allclose(collections[0].get_array(), _midpoints(total.value[:, 0, 0]))

    def test_signed_spin_component_is_not_clipped(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-colinear")

        _, ax = pyprocar.bandsplot(
            code="vasp", dirname=calc, mode="parametric", fermi=FERMI, spins=[1], show=False
        )

        collections = _line_collections(ax)
        data_min = min(c.get_array().min() for c in collections)
        assert data_min < 0
        assert collections[0].get_clim()[0] <= data_min

    def test_non_collinear_rejects_several_spin_components(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-colinear")

        with pytest.raises(ValueError, match="one spin component"):
            pyprocar.bandsplot(
                code="vasp", dirname=calc, mode="parametric", fermi=FERMI, spins=[1, 2], show=False
            )

    def test_non_collinear_ipr_uses_the_total_component(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-colinear")
        ipr = ElectronicBandStructurePath.from_code("vasp", calc).compute_ebs_ipr().value

        _, ax = pyprocar.bandsplot(code="vasp", dirname=calc, mode="ipr", fermi=FERMI, show=False)

        collections = _line_collections(ax)
        assert len(collections) == 50
        np.testing.assert_allclose(collections[0].get_array(), _midpoints(ipr[:, 0, 0]))

    def test_joined_spin_channels_reject_projection_modes(self, tmp_path):
        calc = _calc(tmp_path, "bands/spin-polarized")

        with pytest.raises(ValueError, match="plain mode"):
            pyprocar.bandsplot(
                code="vasp",
                dirname=calc,
                mode="parametric",
                fermi=FERMI,
                spins=[-1, 1],
                show=False,
            )

    def test_overlay_items_accepts_a_list_of_mappings(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")

        _, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=calc,
            mode="overlay",
            fermi=FERMI,
            items=[{"V": ["d"]}, {"O": ["p"]}],
            show=False,
        )

        assert _legend_texts(ax) == ["V", "O"]

    def test_draws_onto_a_given_axes(self, tmp_path):
        calc = _calc(tmp_path, "bands/non-spin-polarized")
        fig, ax = plt.subplots()

        returned_fig, returned_ax = pyprocar.bandsplot(
            code="vasp", dirname=calc, mode="plain", fermi=FERMI, ax=ax, show=False
        )

        assert returned_ax is ax
        assert returned_fig is fig
        assert len(_lines_of_length(ax, 200)) == 20


@pytest.mark.data
class TestDosplot:
    def test_plain_draws_total_dos_and_keeps_user_files(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-spin-polarized")
        sentinel = calc / "dos.pkl"
        sentinel.write_bytes(b"user file")
        out = tmp_path / "dos.png"

        _, ax = pyprocar.dosplot(
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

        _, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="stack_species",
            fermi=FERMI,
            elimit=[-6, 4],
            orbitals=[1, 2, 3],
            show=False,
        )

        assert _legend_texts(ax) == [
            "Total",
            r"$\mathrm{O}_{2-4}-(p)[\uparrow]$",
            r"$\mathrm{Sr}_{0}-(p)[\uparrow]$",
            r"$\mathrm{V}_{1}-(p)[\uparrow]$",
        ]
        assert len(ax.collections) == 3

    def test_stack_fills_accumulate_the_projected_sums(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-spin-polarized")
        dos = DensityOfStates.from_code("vasp", calc)
        o_p, sr_p = (
            _projection(dos, species=[s], orbitals=[1, 2, 3]).value[:, 0] for s in ("O", "Sr")
        )

        _, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="stack_species",
            orbitals=[1, 2, 3],
            fermi=FERMI,
            show=False,
        )

        fills = _fill_extents(ax)
        assert fills[0] == (pytest.approx(o_p.max()), 0.0)
        assert fills[1][0] == pytest.approx((o_p + sr_p).max())

    def test_stack_draws_spin_down_below_zero(self, tmp_path):
        calc = _calc(tmp_path, "dos/spin-polarized")
        o_p = _projection(
            DensityOfStates.from_code("vasp", calc), species=["O"], orbitals=[1, 2, 3]
        )

        _, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="stack_species",
            orbitals=[1, 2, 3],
            fermi=FERMI,
            show=False,
        )

        up, down = _fill_extents(ax)[:2]
        assert up == (pytest.approx(o_p.value[:, 0].max()), 0.0)
        assert down == (0.0, pytest.approx(-o_p.value[:, 1].max()))

    def test_parametric_spin_down_colors_equal_its_projection_fraction(self, tmp_path):
        calc = _calc(tmp_path, "dos/spin-polarized")
        both = _projection(
            DensityOfStates.from_code("vasp", calc),
            atoms=[1],
            orbitals=[4, 5, 6, 7, 8],
            spins=[0, 1],
            norm_mode="total_projection",
        )

        _, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            atoms=[1],
            orbitals=[4, 5, 6, 7, 8],
            spins=[1],
            show=False,
        )

        colors = np.asarray(ax.images[0].get_array()).ravel()
        np.testing.assert_allclose(colors, both.value[:, 1])
        assert colors.max() <= 1.0

    def test_non_collinear_stack_sums_the_total_component(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-colinear")

        _, ax = pyprocar.dosplot(
            code="vasp", dirname=calc, mode="stack_species", fermi=FERMI, show=False
        )

        fills = _fill_extents(ax)
        assert len(fills) == 3
        assert min(bottom for _, bottom in fills) == 0.0

    def test_joined_spin_channels_plot_the_summed_total(self, tmp_path):
        calc = _calc(tmp_path, "dos/spin-polarized")
        dos = DensityOfStates.from_code("vasp", calc)
        summed = dos.total.to_array().sum(axis=1)

        _, ax = pyprocar.dosplot(
            code="vasp", dirname=calc, mode="plain", fermi=FERMI, spins=[-1, 1], show=False
        )

        labelled = [line for line in ax.lines if not line.get_label().startswith("_")]
        assert [line.get_label() for line in labelled] == ["Total"]
        np.testing.assert_allclose(labelled[0].get_ydata(), summed)
        assert summed.max() == pytest.approx(58.38, abs=0.01)

    def test_joined_spin_channels_reject_projection_modes(self, tmp_path):
        calc = _calc(tmp_path, "dos/spin-polarized")

        with pytest.raises(ValueError, match="plain mode"):
            pyprocar.dosplot(
                code="vasp",
                dirname=calc,
                mode="parametric",
                fermi=FERMI,
                atoms=[1],
                spins=[-1, 1],
                show=False,
            )

    def test_open_ended_limits_keep_the_data_bound(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-spin-polarized")
        top = DensityOfStates.from_code("vasp", calc).energies.max() - FERMI

        _, ax = pyprocar.dosplot(
            code="vasp",
            dirname=calc,
            mode="plain",
            fermi=FERMI,
            elimit=[-2, None],
            dos_limit=[0, None],
            show=False,
        )

        assert ax.get_xlim() == (-2.0, pytest.approx(top))
        assert ax.get_ylim()[0] == 0.0
        assert ax.get_ylim()[1] > 0.0

    def test_non_collinear_rejects_several_spin_components(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-colinear")

        with pytest.raises(ValueError, match="one spin component"):
            pyprocar.dosplot(
                code="vasp", dirname=calc, mode="parametric", fermi=FERMI, spins=[1, 2], show=False
            )

    def test_vertical_overlay_orbitals(self, tmp_path):
        calc = _calc(tmp_path, "dos/non-spin-polarized")

        _, ax = pyprocar.dosplot(
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
        assert _legend_texts(ax) == [
            "Total",
            r"$\mathrm{V}_{1}-(s)[\uparrow]$",
            r"$\mathrm{V}_{1}-(p)[\uparrow]$",
            r"$\mathrm{V}_{1}-(d)[\uparrow]$",
        ]

    def test_spin_channels_can_share_one_axes(self, tmp_path):
        calc = _calc(tmp_path, "dos/spin-polarized")

        _, ax = pyprocar.dosplot(
            code="vasp", dirname=calc, mode="plain", fermi=FERMI, spins=[0], show=False
        )
        pyprocar.dosplot(
            code="vasp", dirname=calc, mode="plain", fermi=FERMI, spins=[1], ax=ax, show=False
        )

        labels = [line.get_label() for line in ax.lines if not line.get_label().startswith("_")]
        assert labels == [r"$Total - \uparrow$", r"$Total - \downarrow$"]


@pytest.mark.data
class TestFermi2D:
    def test_plain_draws_contours(self, tmp_path):
        calc = _calc(tmp_path, "fermi2d/non-spin-polarized")
        out = tmp_path / "fermi2d.png"

        _, ax = pyprocar.fermi2D(
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

    def test_plot_line_kwargs_style_every_contour(self, tmp_path):
        calc = _calc(tmp_path, "fermi2d/spin-polarized")

        _, ax = pyprocar.fermi2D(
            code="vasp",
            dirname=calc,
            mode="plain",
            fermi=FERMI,
            energy=0.0,
            plot_line_kwargs={"colors": "purple", "linewidths": 2.0, "linestyles": "dashed"},
            show=False,
        )

        (lines,) = _line_collections(ax)
        assert np.asarray(lines.get_edgecolor()).tolist() == [
            [0.5019607843137255, 0.0, 0.5019607843137255, 1.0]
        ]
        assert np.asarray(lines.get_linewidth()).tolist() == [2.0]
        assert lines.get_linestyle() == [(0.0, [7.4, 3.2])]

    def test_parametric_spin_channel_uses_cmap(self, tmp_path):
        calc = _calc(tmp_path, "fermi2d/spin-polarized")

        fig, ax = pyprocar.fermi2D(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            energy=0.0,
            k_z_plane=0.0,
            atoms=V_ATOM,
            orbitals=D_ORBITALS,
            spins=[0],
            show_colorbar=True,
            cmap="viridis",
            use_cache=False,
            verbose=2,
            show=False,
        )

        (lines,) = _line_collections(ax)
        assert lines.get_cmap().name == "viridis"
        assert len(fig.axes) == 2

    def test_show_colorbar_false_draws_no_colorbar(self, tmp_path):
        calc = _calc(tmp_path, "fermi2d/non-spin-polarized")

        fig, _ = pyprocar.fermi2D(
            code="vasp",
            dirname=calc,
            mode="parametric",
            fermi=FERMI,
            atoms=V_ATOM,
            orbitals=D_ORBITALS,
            show_colorbar=False,
            show=False,
        )

        assert len(fig.axes) == 1


def _documented_fermi2d_calls():
    """Yield (location, keyword names) for every pyprocar.fermi2D call in the notebooks and docs."""
    sources = []
    for notebook in sorted((ROOT_DIR / "examples").rglob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        sources += [
            (f"{notebook.name}#cell{i}", "".join(cell["source"]))
            for i, cell in enumerate(cells)
            if cell["cell_type"] == "code"
        ]
    for page in sorted((ROOT_DIR / "docs").rglob("*.rst")):
        text = page.read_text(encoding="utf-8")
        sources += [
            (f"{page.name}#{i}", match.group(0))
            for i, match in enumerate(re.finditer(r"pyprocar\.fermi2D\(.*?\)", text, re.DOTALL))
        ]
    for location, source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "pyprocar.fermi2D":
                yield location, [kw.arg for kw in node.keywords if kw.arg]


class TestFermi2DDocumentedCalls:
    def test_every_documented_keyword_is_accepted(self):
        accepted = set(inspect.signature(pyprocar.fermi2D).parameters)
        calls = list(_documented_fermi2d_calls())

        rejected = {
            location: sorted(set(keywords) - accepted)
            for location, keywords in calls
            if set(keywords) - accepted
        }

        assert rejected == {}
        assert len(calls) == 26


class TestFermiHandlerSignature:
    def test_options_after_fermi_are_keyword_only(self, tmp_path):
        with pytest.raises(TypeError):
            positional: list[Any] = ["vasp", tmp_path, FERMI, True]
            pyprocar.FermiHandler(*positional)


@pytest.mark.data
class TestFermiHandler:
    def test_parametric_colors_each_surface_by_its_own_band(self, tmp_path):
        calc = _calc(tmp_path, "fermi3d/non-spin-polarized")
        ebs = cast(ElectronicBandStructureMesh, ElectronicBandStructureMesh.from_code("vasp", calc))
        fs = FermiSurface.from_ebs(ebs, isovalue=FERMI)
        projection = fs.get_property(
            "projected_sum", atoms=V_ATOM, orbitals=D_ORBITALS, spins=[0]
        ).value
        handler = pyprocar.FermiHandler(code="vasp", dirname=calc, fermi=FERMI, verbose=0)
        shutil.rmtree(calc)

        plotter = handler.plot_fermi_surface(
            mode="parametric",
            atoms=V_ATOM,
            orbitals=D_ORBITALS,
            spins=[0],
            show=False,
            off_screen=True,
        )

        assert plotter is not None
        assert len(fs.band_spin_mask) == 3
        for (iband, ispin), mask in fs.band_spin_mask.items():
            colors = plotter.values_dict[f"band_{iband}_spin_{ispin}_scalars"]
            np.testing.assert_allclose(colors, projection[mask, iband, ispin])
            assert colors.min() > 0.79
            assert colors.max() < 0.85

    def test_fermi_speed_colors_each_surface(self, tmp_path):
        calc = _calc(tmp_path, "fermi3d/non-spin-polarized")
        handler = pyprocar.FermiHandler(code="vasp", dirname=calc, fermi=FERMI, verbose=0)

        plotter = handler.plot_fermi_surface(mode="fermi_speed", show=False, off_screen=True)
        assert plotter is not None

        scalars = [v for k, v in plotter.values_dict.items() if k.endswith("_scalars")]
        assert len(scalars) == 3
        assert all(s.ndim == 1 and s.min() > 0 for s in scalars)

    def test_plain_draws_only_the_requested_spin_one_color_per_band(self, tmp_path):
        calc = _calc(tmp_path, "fermi3d/spin-polarized")
        handler = pyprocar.FermiHandler(code="vasp", dirname=calc, fermi=FERMI, verbose=0)

        spin_up = handler.plot_fermi_surface(mode="plain", spins=[0], show=False, off_screen=True)
        spin_down = handler.plot_fermi_surface(mode="plain", spins=[1], show=False, off_screen=True)
        assert spin_up is not None and spin_down is not None

        up = {k: v for k, v in spin_up.values_dict.items() if k.endswith("_scalars")}
        down = [k for k in spin_down.values_dict if k.endswith("_scalars")]
        assert sorted(up) == [f"band_{b}_spin_0_scalars" for b in (16, 17, 18)]
        assert down == []
        colors = [np.unique(v) for v in up.values()]
        assert [c.tolist() for c in colors] == [[0], [1], [2]]

    def test_save_2d_writes_a_screenshot(self, tmp_path):
        calc = _calc(tmp_path, "fermi3d/non-spin-polarized")
        handler = pyprocar.FermiHandler(code="vasp", dirname=calc, fermi=FERMI, verbose=0)
        out = tmp_path / "fs.png"

        handler.plot_fermi_surface(mode="plain", show=False, save_2d=out, off_screen=True)

        assert out.stat().st_size > 0


@pytest.mark.data
class TestBandsdosplot:
    def test_bands_and_dos_share_the_energy_axis(self, tmp_path):
        bands_calc = _calc(tmp_path / "b", "bands/non-spin-polarized")
        dos_calc = _calc(tmp_path / "d", "dos/non-spin-polarized")
        out = tmp_path / "bandsdos.png"

        _, ax_bands, ax_dos = pyprocar.bandsdosplot(
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

    def test_caller_settings_are_not_mutated(self, tmp_path):
        bands_calc = _calc(tmp_path / "b", "bands/non-spin-polarized")
        dos_calc = _calc(tmp_path / "d", "dos/non-spin-polarized")
        bands_settings = {"mode": "plain", "dirname": bands_calc, "fermi": FERMI}
        dos_settings = {"mode": "plain", "dirname": dos_calc, "fermi": FERMI}

        pyprocar.bandsdosplot(bands_settings=bands_settings, dos_settings=dos_settings, show=False)

        assert bands_settings == {"mode": "plain", "dirname": bands_calc, "fermi": FERMI}
        assert dos_settings == {"mode": "plain", "dirname": dos_calc, "fermi": FERMI}


@pytest.mark.data
class TestBandStructure2DHandler:
    @pytest.fixture
    def handler(self, tmp_path):
        calc = _calc(tmp_path, "bands/2d-bands/graphene")
        handler = pyprocar.BandStructure2DHandler(code="vasp", dirname=calc, fermi=-0.795606)
        shutil.rmtree(calc)
        return handler

    @staticmethod
    def _points(plotter):
        return np.concatenate([v for k, v in plotter.values_dict.items() if k.endswith("_points")])

    def test_parametric_colors_each_surface_with_a_projection_fraction(self, handler, tmp_path):
        out = tmp_path / "bs2d.png"

        plotter = handler.plot_band_structure(
            mode="parametric",
            atoms=[0, 1],
            orbitals=[1, 2, 3],
            spins=[0],
            show=False,
            render_offscreen=True,
            save_2d=out,
        )

        scalars = [v for k, v in plotter.values_dict.items() if k.endswith("_scalars")]
        assert out.stat().st_size > 0
        assert len(scalars) == 2
        for values in scalars:
            finite = values[np.isfinite(values)]
            assert values.ndim == 1
            assert finite.min() >= 0.0
            assert finite.max() <= 1.0

    def test_surfaces_are_clipped_to_the_brillouin_zone(self, handler):
        clipped = handler.plot_band_structure(mode="plain", show=False, render_offscreen=True)
        unclipped = handler.plot_band_structure(
            mode="plain", show=False, render_offscreen=True, clip_brillouin_zone=False
        )

        inside, outside = self._points(clipped), self._points(unclipped)
        assert len(inside) < len(outside)
        assert np.nanmax(np.abs(inside[:, 0])) < np.nanmax(np.abs(outside[:, 0]))
