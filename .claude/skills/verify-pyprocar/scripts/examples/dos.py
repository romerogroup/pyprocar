# Density of states: object API (total, line-colored projection, vertical + legend)
# and legacy dosplot.
# Fixture: data/examples/dos/non-spin-polarized
from typing import Any

import numpy as np
from verify_steps import CALC, EV, SUMMARY, finish, png, step

import pyprocar
from pyprocar.core.dos import DensityOfStates
from pyprocar.core.property_store import Property
from pyprocar.plotter.dos_plot import DOSPlotter

before = {p.name for p in CALC.iterdir()}
dos = DensityOfStates.from_code(code="vasp", dirpath=str(CALC))


@step("obj_total")
def _():
    p = DOSPlotter(orientation="horizontal")
    assert p.ax is not None
    p.plot(dos.total)
    e, t = np.asarray(dos.energies), np.asarray(dos.total.to_array())
    return {
        "energy_range": [float(e.min()), float(e.max())],
        "total_shape": list(t.shape),
        "total_max": float(t.max()),
        "n_lines": len(p.ax.lines),
        "png": png("obj_total", p.fig),
    }


@step("obj_total_colored_by_projection")
def _():
    proj = dos.compute_projected_sum(atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[0])
    assert isinstance(proj, Property)
    p = DOSPlotter(orientation="horizontal")
    assert p.ax is not None
    p.plot(dos.total, scalars_data=proj, scalars_mode="line")
    pa, ta = np.asarray(proj.to_array()), np.asarray(dos.total.to_array())
    return {
        "proj_le_total": bool((pa[..., 0] <= ta[..., 0] + 1e-6).all()),
        "n_axes": len(p.fig.axes),
        "n_coll": len(p.ax.collections),
        "png": png("obj_projected", p.fig),
    }


@step("obj_vertical_projected_legend")
def _():
    proj = dos.compute_projected_sum(atoms=[2, 3, 4], orbitals=[1, 2, 3], norm_mode="integral")
    assert isinstance(proj, Property)
    p = DOSPlotter(orientation="vertical")
    p.plot(dos.total)
    p.plot(proj)
    p.legend()
    assert p.ax is not None
    legend = p.ax.get_legend()
    assert legend is not None
    return {
        "n_lines": len(p.ax.lines),
        "legend": [t.get_text() for t in legend.get_texts()],
        "png": png("obj_vertical", p.fig),
    }


SUMMARY["obj_new_files_in_calc"] = sorted({p.name for p in CALC.iterdir()} - before)

V_D: dict[str, Any] = dict(atoms=[1], orbitals=[4, 5, 6, 7, 8])
MODES: list[tuple[str, str, dict[str, Any]]] = [
    ("plain", "plain", {}),
    ("parametric", "parametric", V_D),
    ("parametric_line", "parametric_line", V_D),
    ("stack_species", "stack_species", {}),
    ("stack_orbitals", "stack_orbitals", dict(atoms=[1])),
    ("stack_no_items", "stack", {}),  # the docstring says it falls back to stack_species
    ("overlay_items", "overlay", dict(items={"O": [1, 2, 3], "V": [4, 5, 6, 7, 8]})),
    ("overlay_species", "overlay_species", {}),
    ("overlay_orbitals", "overlay_orbitals", dict(atoms=[1])),
]
for label, mode, kw in MODES:

    @step(f"legacy_dosplot_{label}")
    def _(label=label, mode=mode, kw=kw):
        _, ax = pyprocar.dosplot(
            code="vasp",
            dirname=str(CALC),
            mode=mode,
            fermi=5.3017,
            elimit=[-6, 4],
            show=False,
            savefig=str(EV / f"legacy_{label}.png"),
            **kw,
        )
        legend = ax.get_legend()
        return {
            "xlim": list(ax.get_xlim()),
            "n_lines": len(ax.lines),
            "n_coll": len(ax.collections),
            "legend": [t.get_text() for t in legend.get_texts()] if legend else [],
        }


finish()
