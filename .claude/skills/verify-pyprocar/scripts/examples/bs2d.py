# 2D band structure surface: object API (BandStructure2D + BS2DPlotter)
# and legacy BandStructure2DHandler.
# Fixture: data/examples/bands/2d-bands (uses its graphene/ subdir)
from verify_steps import CALC, EV, distinct_colors, finish, step

import pyprocar
from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.plotter.bs_2d_plot import BS2DPlotter

G = CALC / "graphene"


bs = BandStructure2D.from_code(code="vasp", dirpath=str(G), grid_interpolation=(20, 20))


@step("obj_surface")
def _():
    p = BS2DPlotter(bs, off_screen=True)
    p.plot(scalars_data="bands", show_brillouin_zone=False)
    out = EV / "obj_bs2d.png"
    p.screenshot(str(out))
    p.close()
    return {"n_points": int(bs.n_points), "distinct_colors": distinct_colors(out)}


@step("obj_surface_with_bz")
def _():
    p = BS2DPlotter(bs, off_screen=True)
    p.plot(scalars_data="bands", show_brillouin_zone=True)
    out = EV / "obj_bs2d_bz.png"
    p.screenshot(str(out))
    actors = sorted(p.actors)
    p.close()
    return {"actors": actors, "distinct_colors": distinct_colors(out)}


for label, kw in {
    "plain": {},
    "plain_notebook_kwargs": dict(add_fermi_plane=True, fermi_plane_size=4, energy_lim=[-2.5, 2.0]),
}.items():

    @step(f"legacy_handler_{label}")
    def _(kw=kw, label=label):
        h = pyprocar.BandStructure2DHandler(code="vasp", dirname=str(G), fermi=-0.795606, verbose=0)
        out = EV / f"legacy_{label}.png"
        h.plot_band_structure(
            mode="plain",
            grid_interpolation=(20, 20),
            show=False,
            render_offscreen=True,
            save_2d=str(out),
            **kw,
        )
        return {"distinct_colors": distinct_colors(out)}


finish()
