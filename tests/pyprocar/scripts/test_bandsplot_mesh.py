import matplotlib

matplotlib.use("Agg")

import pytest

import pyprocar
from tests.pyprocar.scripts.test_fermi2d_kz_mesh import cylinder_mesh


@pytest.mark.parametrize("mode", ["plain", "parametric", "overlay_species"])
def test_bandsplot_of_a_k_mesh_says_it_needs_a_k_path(tmp_path, mode):
    cylinder_mesh(1).save(tmp_path / "ebs.pkl")

    with pytest.raises(ValueError, match="is a k-point mesh, not a band structure along a k-path"):
        pyprocar.bandsplot(
            code="vasp", dirname=str(tmp_path), mode=mode, use_cache=True, show=False
        )
