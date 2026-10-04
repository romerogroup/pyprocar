import numpy as np
import pytest

from pyprocar.io.lobster.doscar_lobster import LOBSTER_ORBITALS
from pyprocar.io.lobster.parser import LobsterParser
from tests.pyprocar.io.lobster.test_doscar_lobster import DOSCAR_CONTENT
from tests.pyprocar.io.lobster.test_fatband import FATBAND_CONTENT
from tests.pyprocar.io.lobster.test_lobsterout import LOBSTEROUT_CONTENT


def test_fatband_weights_land_on_their_spin_ion_and_orbital(tmp_path):
    for name, content in {
        "lobsterout": LOBSTEROUT_CONTENT,
        "DOSCAR.lobster": DOSCAR_CONTENT,
        "FATBAND_Fe_s.lobster": FATBAND_CONTENT,
    }.items():
        (tmp_path / name).write_text(content)

    ebs = LobsterParser(tmp_path).ebs

    assert ebs is not None and ebs.projected is not None
    assert ebs.orbital_names == LOBSTER_ORBITALS
    projected = ebs.projected.to_array()
    assert projected.shape[2:] == (1, 2, 9)
    fe, s = 0, 0
    expected = np.array([[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45]])
    assert projected[:, :, 0, fe, s] == pytest.approx(expected)
    projected[:, :, 0, fe, s] = 0.0
    assert np.all(projected == 0.0)
