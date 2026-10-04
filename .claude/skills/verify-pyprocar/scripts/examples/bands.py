# Bands along a k-path: object API (plain/parametric/scatter/quiver/overlay/flip)
# and legacy bandsplot/bandsdosplot.
# Fixture: data/examples/bands/non-spin-polarized
import shutil

import matplotlib.pyplot as plt
import numpy as np
from verify_steps import CALC, EV, REPO, finish, png, step

import pyprocar
from pyprocar.core.ebs import ElectronicBandStructurePath
from pyprocar.plotter.bs_plot import BandStructurePlotter as P

loaded = ElectronicBandStructurePath.from_code(code="vasp", dirpath=str(CALC))
assert isinstance(loaded, ElectronicBandStructurePath)
ebs: ElectronicBandStructurePath = loaded
w = ebs.compute_projected_sum(atoms=[1], orbitals=[4, 5, 6, 7, 8])


def facts():
    ax = plt.gcf().axes[0]
    return {
        "n_lines": len(ax.lines),
        "n_coll": len(ax.collections),
        "n_axes": len(plt.gcf().axes),
        "ylim": [round(v, 2) for v in ax.get_ylim()],
        "xticks": [t.get_text() for t in ax.get_xticklabels()],
    }


@step("obj_plain")
def _():
    P().plot(ebs.bands, scalars_mode="none")
    return {**facts(), "png": png("obj_plain")}


@step("obj_parametric")
def _():
    P().plot(ebs.bands, scalars_data=w, scalars_mode="parametric")
    return {**facts(), "png": png("obj_parametric")}


@step("obj_scatter")
def _():
    P().plot(ebs.bands, scalars_data=w, scalars_mode="scatter")
    return {**facts(), "png": png("obj_scatter")}


@step("obj_quiver_plot_quiver")
def _():
    v = ebs.get_property("bands_velocity")
    assert ebs.bands is not None and v is not None
    P().plot_quiver(ebs.kpath, ebs.bands.to_array(), np.asarray(v.to_array()))
    return {**facts(), "png": png("obj_quiver")}


@step("obj_quiver_vectors_mode")  # the call examples/general/new_bands_examples.py makes
def _():
    P().plot(ebs.bands, vectors_data=ebs.get_property("bands_velocity"), vectors_mode="quiver")
    return {"png": png("obj_quiver_vectors_mode")}


@step("obj_overlay_species")
def _():
    props = ebs.build_overlay_species_weights(orbitals=[4, 5, 6, 7, 8])
    P().plot_overlay(
        ebs.kpath,
        ebs.bands,  # pyright: ignore[reportArgumentType]  # the call new_bands_examples.py makes
        [p.to_array() for p in props],
        labels=[p.label or "" for p in props],
    )
    return {**facts(), "png": png("obj_overlay_species")}


@step("obj_channel_flip")
def _():
    P().plot(ebs.bands, scalars_mode="none", channel_mode="flip")
    return {**facts(), "png": png("obj_channel_flip")}


for mode, kw in {
    "plain": {},
    "parametric": dict(atoms=[1], orbitals=[4, 5, 6, 7, 8]),
    "scatter": dict(atoms=[1], orbitals=[4, 5, 6, 7, 8]),
    "overlay_species": dict(orbitals=[1, 2, 3]),
}.items():

    @step(f"legacy_bandsplot_{mode}")
    def _(mode=mode, kw=kw):
        _, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=str(CALC),
            mode=mode,
            fermi=5.3017,
            elimit=[-5, 5],
            show=False,
            savefig=str(EV / f"legacy_{mode}.png"),
            **kw,
        )
        return {"n_lines": len(ax.lines), "n_coll": len(ax.collections)}


@step("legacy_bandsdosplot")
def _():
    dos_dir = CALC.parent / "dos_calc"
    shutil.copytree(REPO / "data/examples/dos/non-spin-polarized", dos_dir, dirs_exist_ok=True)
    pyprocar.bandsdosplot(
        bands_settings=dict(mode="plain", dirname=str(CALC), fermi=5.3017),
        dos_settings=dict(mode="plain", dirname=str(dos_dir), fermi=5.3017),
        code="vasp",
        show=False,
        savefig=str(EV / "legacy_bandsdosplot.png"),
    )


finish()
