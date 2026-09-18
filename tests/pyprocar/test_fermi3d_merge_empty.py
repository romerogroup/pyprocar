"""Regression test: merging Fermi surfaces when no band crosses the isovalue."""

import pytest
import pyvista as pv

from pyprocar.plotter.fermi3d_plot import FermiDataHandler


def test_merge_fermi_surfaces_all_empty_raises_value_error():
    handler = FermiDataHandler.__new__(FermiDataHandler)
    empty_surfaces = [pv.PolyData(), pv.PolyData()]

    with pytest.raises(ValueError, match="No Fermi surface found"):
        handler._merge_fermi_surfaces(empty_surfaces)
