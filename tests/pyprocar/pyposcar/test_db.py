import numpy as np
import pytest

from pyprocar.pyposcar.db import DB
from pyprocar.pyposcar.defects import FindDefect
from pyprocar.pyposcar.poscar import Poscar


def test_vanadium_bond_uses_cordero_radius():
    # Cordero et al. single-bond radii: V 153 pm, O 66 pm
    assert DB().estimateBond("V", "O") == pytest.approx(2.19)


def test_find_defect_on_srvo3_finds_none():
    poscar = Poscar()
    poscar.load_from_data(
        direct_positions=np.array(
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        ),
        lattice=3.84 * np.eye(3),
        elements=["Sr", "V", "O", "O", "O"],
    )

    assert list(FindDefect(poscar).all_defects) == []
