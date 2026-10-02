# 3D Fermi surface: object API (FermiSurface + FermiPlotter off screen) and legacy FermiHandler.
# Fixture: data/examples/fermi3d/non-spin-polarized
from verify_steps import CALC, EV, finish, distinct_colors, step

import pyprocar
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.plotter.fs_plot import FermiPlotter

fs = FermiSurface.from_code(code="vasp", dirpath=CALC, fermi=5.3017)


@step("obj_surface_with_bz")
def _():
    p = FermiPlotter(off_screen=True)
    meshes = p.plot(fs, show_brillouin_zone=True)
    out = EV / "obj_plain.png"
    p.screenshot(str(out))
    p.close()
    return {"n_points": fs.n_points, "n_cells": fs.n_cells, "band_indices": [list(map(int, k)) for k in fs.band_indices],
            "n_meshes": len(meshes), "distinct_colors": distinct_colors(out)}


@step("obj_scalars_projected_sum")
def _():
    fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[0])
    p = FermiPlotter(off_screen=True)
    p.plot(fs, scalars_data="projected_sum")
    out = EV / "obj_projected.png"
    p.screenshot(str(out))
    p.close()
    return {"distinct_colors": distinct_colors(out)}


for mode, kw in {"plain": {}, "parametric": dict(atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[0]), "fermi_speed": {}}.items():

    @step(f"legacy_handler_{mode}")
    def _(mode=mode, kw=kw):
        h = pyprocar.FermiHandler(code="vasp", dirname=str(CALC), fermi=5.3017, use_cache=False, verbose=0)
        out = EV / f"legacy_{mode}.png"
        h.plot_fermi_surface(mode=mode, show=False, save_2d=str(out), off_screen=True, **kw)
        return {"distinct_colors": distinct_colors(out)}


finish()
