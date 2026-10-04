# 2D Fermi slice: object API (FermiSlicePlotter slices internally)
# and legacy fermi2D (str and enum modes).
# Fixture: data/examples/fermi2d/non-spin-polarized
import numpy as np
from verify_steps import CALC, EV, finish, png, step

import pyprocar
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.plotter.fs_slice_plot import FermiSlicePlotter
from pyprocar.scripts.scriptFermi2D import Fermi2DMode

fs = FermiSurface.from_code(code="vasp", dirpath=CALC, fermi=5.3017)


@step("obj_slice_no_scalars")
def _():
    p = FermiSlicePlotter(fs, normal=np.array([0, 0, 1]), origin=np.zeros(3))
    arts = p.plot()
    return {
        "artist_keys": list(arts),
        "drew_anything": bool(p.ax.collections or p.ax.lines),
        "png": png("obj_slice_plain", p.fig),
    }


@step("obj_slice_projected")
def _():
    fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[0])
    p = FermiSlicePlotter(fs, normal=np.array([0, 0, 1]), origin=np.zeros(3))
    lc = p.plot(scalars_name="projected_sum")["scalars"]
    return {
        "n_segments": len(lc.get_segments()),
        "colored": lc.get_array() is not None,
        "png": png("obj_slice_projected", p.fig),
    }


for label, mode, kw in [
    ("str_plain", "plain", {}),
    ("enum_plain", Fermi2DMode.plain, {}),
    ("str_parametric", "parametric", dict(atoms=[1], orbitals=[4, 5, 6, 7, 8])),
]:

    @step(f"legacy_fermi2D_{label}")
    def _(label=label, mode=mode, kw=kw):
        _, ax = pyprocar.fermi2D(
            code="vasp",
            dirname=str(CALC),
            mode=mode,
            fermi=5.3017,
            energy=0.0,
            k_z_plane=0.0,
            show=False,
            savefig=str(EV / f"legacy_{label}.png"),
            **kw,
        )
        return {"n_coll": len(ax.collections)}


finish()
