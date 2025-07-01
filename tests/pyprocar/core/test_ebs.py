import logging

import numpy as np
import pytest

from pyprocar.core import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from tests.utils import DATA_DIR, BaseTest

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)
user_logger = logging.getLogger("user")


@pytest.fixture
def mesh_calc_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi3d" / "non-spin-polarized"

@pytest.fixture
def ebs(mesh_calc_dir):
    return ElectronicBandStructureMesh.from_code(code="vasp", dirpath=mesh_calc_dir)


class TestElectronicBandStructureMesh:
    """Test class for ElectronicBandStructureMesh functionality."""
    
    def test_pad(self, ebs):
        """Test the pad method of ElectronicBandStructureMesh."""
        
        padding = 11
        padded_ebs = ebs.pad(padding=padding, inplace=False)
        
        logger.info(f"Padded_ebs kpoints mesh: {padded_ebs.kgrid}")
        
        assert np.allclose(ebs.kpoints_mesh, 
                           padded_ebs.kpoints_mesh[padding:-padding,padding:-padding,padding:-padding,:], 
                           rtol=1e-3)
  
        assert np.allclose(bands_mesh, 
                           padded_bands_mesh[padding:-padding,padding:-padding,padding:-padding,:], 
                           rtol=1e-3)


