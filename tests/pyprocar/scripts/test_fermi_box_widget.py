import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import pyprocar
from pyprocar.core.ebs import ElectronicBandStructureMesh
from tests.pyprocar.core.test_fermisurface_noncollinear import sphere_mesh


def _inked_pixels(path) -> int:
    image = plt.imread(path)[..., :3]
    return int(np.count_nonzero(image.min(axis=-1) < 0.5))


def test_box_widget_saves_the_3d_view_and_the_2d_slice(monkeypatch, tmp_path):
    ebs = sphere_mesh(1, np.full((2, 1, 2, 1), 0.5))

    def from_code(_cls: type[ElectronicBandStructureMesh], *_args: object, **_kwargs: object):
        return ebs

    monkeypatch.setattr(ElectronicBandStructureMesh, "from_code", classmethod(from_code))
    view, cut = tmp_path / "view.png", tmp_path / "slice.png"

    handler = pyprocar.FermiHandler(code="vasp", dirname="calc", fermi=0.1)
    handler.plot_fermi_cross_section_box_widget(
        mode="plain", slice_normal=(0, 0, 1), save_2d=view, save_2d_slice=cut, show=False
    )

    assert _inked_pixels(view) > 1000
    assert _inked_pixels(cut) > 100
    plt.close("all")
