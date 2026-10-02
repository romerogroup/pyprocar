# Density of states: object API (total, line-colored projection, vertical + legend) and legacy dosplot.
# Fixture: data/examples/dos/non-spin-polarized
import numpy as np
from verify_steps import CALC, EV, SUMMARY, finish, png, step

import pyprocar
from pyprocar.core.dos import DensityOfStates
from pyprocar.plotter.dos_plot import DOSPlotter

before = {p.name for p in CALC.iterdir()}
dos = DensityOfStates.from_code(code="vasp", dirpath=CALC)


@step("obj_total")
def _():
    p = DOSPlotter(orientation="horizontal")
    p.plot(dos.total)
    e, t = np.asarray(dos.energies), np.asarray(dos.total.to_array())
    return {"energy_range": [float(e.min()), float(e.max())], "total_shape": list(t.shape),
            "total_max": float(t.max()), "n_lines": len(p.ax.lines), "png": png("obj_total", p.fig)}


@step("obj_total_colored_by_projection")
def _():
    proj = dos.compute_projected_sum(atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[0])
    p = DOSPlotter(orientation="horizontal")
    p.plot(dos.total, scalars_data=proj, scalars_mode="line")
    pa, ta = np.asarray(proj.to_array()), np.asarray(dos.total.to_array())
    return {"proj_le_total": bool((pa[..., 0] <= ta[..., 0] + 1e-6).all()), "n_axes": len(p.fig.axes),
            "n_coll": len(p.ax.collections), "png": png("obj_projected", p.fig)}


@step("obj_vertical_projected_legend")
def _():
    proj = dos.compute_projected_sum(atoms=[2, 3, 4], orbitals=[1, 2, 3], norm_mode="integral")
    p = DOSPlotter(orientation="vertical")
    p.plot(dos.total)
    p.plot(proj)
    p.legend()
    return {"n_lines": len(p.ax.lines), "legend": [t.get_text() for t in p.ax.get_legend().get_texts()],
            "png": png("obj_vertical", p.fig)}


SUMMARY["obj_new_files_in_calc"] = sorted({p.name for p in CALC.iterdir()} - before)

for mode, kw in {"plain": {}, "parametric": dict(atoms=[1], orbitals=[4, 5, 6, 7, 8])}.items():

    @step(f"legacy_dosplot_{mode}")
    def _(mode=mode, kw=kw):
        fig, ax = pyprocar.dosplot(code="vasp", dirname=CALC, mode=mode, fermi=5.3017, elimit=[-6, 4],
                                   show=False, savefig=EV / f"legacy_{mode}.png", **kw)
        return {"xlim": list(ax.get_xlim()), "n_lines": len(ax.lines)}


finish()
