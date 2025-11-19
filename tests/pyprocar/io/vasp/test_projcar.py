import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io import vasp
from pyprocar.utils.log_utils import set_verbose_level
from tests.utils.base_test import BaseTest

logger = logging.getLogger(__name__)

VERBOSE = 2
set_verbose_level(VERBOSE)

PROJCAR_STRING = """
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :  n= 1  za=  1.0000

 k-point:     1  spin:   1

   band         py             pz             px            dxy            dyz            dz2            dxz           dx2-y2  
      1   -0.033 -0.009   0.003  0.001   0.007 -0.041  -0.000  0.000   0.000  0.000   0.000 -0.000  -0.000  0.000  -0.000 -0.000
      2    0.033 -0.025   0.008  0.002  -0.022 -0.025   0.000  0.000   0.000 -0.000  -0.000 -0.000   0.000  0.000  -0.000  0.000

 k-point:     2  spin:   1

   band         py             pz             px            dxy            dyz            dz2            dxz           dx2-y2  
      1   -0.000 -0.000   0.000 -0.000   0.000  0.005   0.000 -0.000  -0.000 -0.000  -0.041  0.004   0.000 -0.000   0.068 -0.007
      2   -0.030 -0.023  -0.001  0.007  -0.000  0.000   0.036 -0.046   0.000  0.000   0.000  0.000  -0.011 -0.001  -0.000 -0.000

   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :  n= 1  za=  1.0000

 k-point:     1  spin:   1

   band         py             pz             px            dxy            dyz            dz2            dxz           dx2-y2  
      1    0.532  0.138  -0.046 -0.011  -0.120  0.659   0.000 -0.000  -0.000 -0.000   0.000  0.000   0.000 -0.000  -0.000 -0.000
      2   -0.533  0.399  -0.138 -0.032   0.360  0.402  -0.000 -0.000   0.000  0.000  -0.000 -0.000  -0.000 -0.000  -0.000  0.000

 k-point:     2  spin:   1

   band         py             pz             px            dxy            dyz            dz2            dxz           dx2-y2  
      1   -0.000  0.000  -0.000  0.000  -0.470 -0.381   0.000  0.000   0.000 -0.000   0.010 -0.012   0.000  0.000  -0.017  0.021
      2    0.859 -0.102  -0.101 -0.123  -0.000 -0.000   0.001  0.007   0.000 -0.000   0.000 -0.000   0.001 -0.001  -0.000  0.000
"""


@pytest.fixture
def projcar_file(tmp_path: Path) -> Path:
    """Create a temporary PROJCAR file for testing."""
    projcar_file = tmp_path / "PROJCAR"
    projcar_file.write_text(PROJCAR_STRING)
    return projcar_file

class TestProjcar(BaseTest):
    """Test Projcar parser with lazy loading."""
    
    def test_projcar_lazy_loading(self, projcar_file: Path) -> None:
        """Test that Projcar doesn't parse on initialization."""
        projcar = vasp.Projcar(projcar_file)
        # File shouldn't be read yet
        assert projcar._file_str == "" # pyright: ignore[reportPrivateUsage]
    
    def test_projcar_dimensions(self, projcar_file: Path) -> None:
        """Test that Projcar correctly parses dimensions."""
        projcar = vasp.Projcar(projcar_file)
        assert projcar.n_k == 2
        assert projcar.n_bands == 2
        assert projcar.n_spins == 1
        assert projcar.n_atoms == 2
        assert projcar.n_orbitals == 8
    
    def test_projcar_frac_coords(self, projcar_file: Path) -> None:
        """Test fractional coordinates parsing."""
        projcar = vasp.Projcar(projcar_file)
        frac_coords = projcar.frac_coords
        
        assert isinstance(frac_coords, np.ndarray)
        assert frac_coords.shape == (2, 3)
        assert frac_coords.dtype == np.float64
        
        # Check specific values
        np.testing.assert_array_almost_equal(
            frac_coords[0], [0.0, 0.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            frac_coords[1], [-0.5, -0.5, -0.5]
        )
    
    def test_projcar_radial_specs(self, projcar_file: Path) -> None:
        """Test radial specifications parsing."""
        projcar = vasp.Projcar(projcar_file)
        radial_specs = projcar.radial_specs
        
        assert isinstance(radial_specs, list)
        assert len(radial_specs) == 2
        
        for spec in radial_specs:
            assert "type" in spec
            assert "params" in spec
            assert spec["type"] == "Hydrogen-like"
            
            # Type assertion: params should be a dict
            params = spec["params"]
            assert isinstance(params, dict)
            assert params["N"] == 1
            assert params["za"] == 1.0
    
    def test_projcar_angular_types(self, projcar_file: Path) -> None:
        """Test angular types parsing."""
        projcar = vasp.Projcar(projcar_file)
        angular_types = projcar.angular_types
        
        expected = ["py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]
        assert angular_types == expected
    
    def test_projcar_projections_shape(self, projcar_file: Path) -> None:
        """Test projections array shape."""
        projcar = vasp.Projcar(projcar_file)
        projections = projcar.projections
        
        assert isinstance(projections, np.ndarray)
        assert projections.shape == (2, 2, 1, 2, 8)
        assert np.iscomplexobj(projections)
        assert projections.dtype in (np.complex128, np.complex64)
    
    def test_projcar_projections_values(self, projcar_file: Path) -> None:
        """Test specific projection values."""
        projcar = vasp.Projcar(projcar_file)
        projections = projcar.projections
        
        # Check a specific value: k=0, band=0, spin=0, atom=0, orbital=0 (py)
        # Expected: -0.033 -0.009
        val = projections[0, 0, 0, 0, 0]
        assert np.isclose(val.real, -0.033, atol=1e-3)
        assert np.isclose(val.imag, -0.009, atol=1e-3)
        
        # Check another value: k=0, band=0, spin=0, atom=1, orbital=0 (py)
        # Expected: 0.532  0.138
        val = projections[0, 0, 0, 1, 0]
        assert np.isclose(val.real, 0.532, atol=1e-3)
        assert np.isclose(val.imag, 0.138, atol=1e-3)
    
    def test_projcar_to_dict(self, projcar_file: Path) -> None:
        """Test to_dict method."""
        projcar = vasp.Projcar(projcar_file)
        data = projcar.to_dict()
        
        assert "frac_coords" in data
        assert "radial_specs" in data
        assert "projections" in data
        
        assert isinstance(data["frac_coords"], np.ndarray)
        assert isinstance(data["radial_specs"], list)
        assert isinstance(data["projections"], np.ndarray)
    
    def test_projcar_from_str(self):
        """Test creating Projcar from string."""
        projcar = vasp.Projcar.from_str(PROJCAR_STRING)
        
        assert projcar.n_k == 2
        assert projcar.n_bands == 2
        assert projcar.n_spins == 1
        assert projcar.n_atoms == 2
        assert projcar.n_orbitals == 8
    
    def test_projcar_mapping_interface(self, projcar_file: Path) -> None:
        """Test that Projcar implements Mapping interface."""
        projcar = vasp.Projcar(projcar_file)
        

        
        # Test __getitem__
        assert projcar["frac_coords"] is projcar.frac_coords
        assert projcar["radial_specs"] is projcar.radial_specs
        assert projcar["projections"] is projcar.projections
        
        # Test __len__
        assert len(projcar) > 0
