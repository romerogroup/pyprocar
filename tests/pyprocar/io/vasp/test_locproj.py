import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io import vasp
from pyprocar.utils.log_utils import set_verbose_level
from tests.utils import DATA_DIR

logger = logging.getLogger(__name__)

VERBOSE = 2
set_verbose_level(VERBOSE)


LOCPROJ_STRING = """
     1    2    2    16  # of spin, # of k-points, # of bands, # of proj
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :     py   
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :     pz   
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :     px   
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :    dxy   
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :    dyz   
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :    dz2   
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :    dxz   
   ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :   dx2-y2 
   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :     py   
   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :     pz   
   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :     px   
   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :    dxy   
   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :    dyz   
   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :    dz2   
   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :    dxz   
   ISITE:     2    R=     -0.5000000    -0.5000000    -0.5000000  Hydrogen-like    :   dx2-y2 
 
orbital     1     1     1      -34.3671700505        1.0000000000
     1       -0.0328454260       -0.0085019871
     2        0.0028357944        0.0006686975
     3        0.0074214687       -0.0407260402
     4       -0.0000000000        0.0000000000
     5        0.0000000000        0.0000000000
     6        0.0000000000       -0.0000000000
     7       -0.0000000000        0.0000000000
     8       -0.0000000000       -0.0000000000
     9        0.5318719912        0.1376742327
    10       -0.0459205371       -0.0108283402
    11       -0.1201772002        0.6594842183
    12        0.0000000000       -0.0000000000
    13       -0.0000000000       -0.0000000000
    14        0.0000000000        0.0000000000
    15        0.0000000000       -0.0000000000
    16       -0.0000000000       -0.0000000000
 
orbital     1     1     2      -34.3671700505        1.0000000000
     1        0.0328945347       -0.0246276625
     2        0.0084925158        0.0019750730
     3       -0.0222060342       -0.0248108719
     4        0.0000000000        0.0000000000
     5        0.0000000000       -0.0000000000
     6       -0.0000000000       -0.0000000000
     7        0.0000000000        0.0000000000
     8       -0.0000000000        0.0000000000
     9       -0.5326672165        0.3988002429
    10       -0.1375208620       -0.0319827185
    11        0.3595863740        0.4017669875
    12       -0.0000000000       -0.0000000001
    13        0.0000000000        0.0000000000
    14       -0.0000000000       -0.0000000000
    15       -0.0000000000       -0.0000000000
    16       -0.0000000000        0.0000000000
    
    
orbital     1     2     1      -34.3847119961        1.0000000000
     1       -0.0000000000       -0.0000000000
     2        0.0000000000       -0.0000000000
     3        0.0004806987        0.0045843902
     4        0.0000000000       -0.0000000000
     5       -0.0000000000       -0.0000000000
     6       -0.0406924646        0.0042668300
     7        0.0000000000       -0.0000000000
     8        0.0676840511       -0.0070970471
     9       -0.0000000000        0.0000000000
    10       -0.0000000000        0.0000000000
    11       -0.4697714040       -0.3806047604
    12        0.0000000000        0.0000000000
    13        0.0000000000       -0.0000000000
    14        0.0098611334       -0.0121713624
    15        0.0000000000        0.0000000000
    16       -0.0167489711        0.0206728567
 
orbital     1     2     2      -34.3663568191        1.0000000000
     1       -0.0296454034       -0.0233658533
     2       -0.0006864077        0.0069222297
     3       -0.0000000000        0.0000000000
     4        0.0364789275       -0.0462826034
     5        0.0000000000        0.0000000000
     6        0.0000000000        0.0000000000
     7       -0.0108070317       -0.0010716244
     8       -0.0000000000       -0.0000000000
     9        0.8587680169       -0.1017270126
    10       -0.1010186296       -0.1232578683
    11       -0.0000000000       -0.0000000000
    12        0.0008482941        0.0071612037
    13        0.0000000000       -0.0000000000
    14        0.0000000000       -0.0000000000
    15        0.0010278384       -0.0008423870
    16       -0.0000000000        0.0000000000
"""



@pytest.fixture
def locproj_filepath(tmp_path: Path) -> Path:
    """Create a temporary LOCPROJ file for testing."""
    locproj_file = tmp_path / "LOCPROJ"
    locproj_file.write_text(LOCPROJ_STRING)
    return locproj_file


class TestLocproj:
    def test_locproj_dimensions(self, locproj_filepath: Path) -> None:
        """Test that parsed dimensions match expected values from test data."""
        locproj = vasp.Locproj(locproj_filepath)

        # From LOCPROJ_STRING: "1    2    2    16"
        assert locproj.n_spins == 1
        assert locproj.n_k == 2
        assert locproj.n_bands == 2
        assert locproj.n_proj == 16

    def test_locproj_frac_coords(self, locproj_filepath: Path) -> None:
        """Test that fractional coordinates are correctly parsed."""
        locproj = vasp.Locproj(locproj_filepath)
        
        assert locproj.frac_coords.shape == (16, 3)
        assert locproj.frac_coords.dtype == np.float64
        
        # Check first coordinate (all zeros)
        assert np.allclose(locproj.frac_coords[0], [0.0, 0.0, 0.0])
        
        # Check 9th coordinate (second atom)
        assert np.allclose(locproj.frac_coords[8], [-0.5, -0.5, -0.5])

    def test_locproj_angular_types(self, locproj_filepath: Path) -> None:
        """Test that angular types are correctly extracted."""
        locproj = vasp.Locproj(locproj_filepath)

        # From LOCPROJ_STRING, we have 8 orbitals per atom × 2 atoms
        # py, pz, px, dxy, dyz, dz2, dxz, dx2-y2 (repeated twice)
        expected_orbitals = ["py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]
        
        assert len(locproj.angular_types) == 16
        assert locproj.angular_types[:8] == expected_orbitals
        assert locproj.angular_types[8:16] == expected_orbitals

    def test_locproj_radial_specs(self, locproj_filepath: Path) -> None:
        """Test that radial specifications are correctly parsed."""
        locproj = vasp.Locproj(locproj_filepath)
        
        assert len(locproj.radial_specs) == 16
        
        # All should be Hydrogen-like in this test case
        for spec in locproj.radial_specs:
            assert spec["type"] == "Hydrogen-like"
            assert isinstance(spec["params"], dict)

    def test_locproj_projections_shape(self, locproj_filepath: Path) -> None:
        """Test that projections array has correct shape and dtype."""
        locproj = vasp.Locproj(locproj_filepath)
        
        assert locproj.projections.shape == (2, 2, 1, 16)
        assert locproj.projections.dtype in [np.complex128, np.complex64]
        assert np.iscomplexobj(locproj.projections)

    def test_locproj_projections_values(self, locproj_filepath: Path) -> None:
        """Test that projections contain expected values."""
        locproj = vasp.Locproj(locproj_filepath)
        
        # Check first projection value for k=1, band=1, spin=1, proj=1
        # From LOCPROJ_STRING: "1       -0.0328454260       -0.0085019871"
        expected_val = complex(-0.0328454260, -0.0085019871)
        actual_val = locproj.projections[0, 0, 0, 0]
        assert np.isclose(actual_val, expected_val)

    def test_locproj_from_str(self) -> None:
        """Test parsing from string."""
        locproj = vasp.Locproj.from_str(LOCPROJ_STRING)
        
        assert locproj.n_spins == 1
        assert locproj.n_k == 2
        assert locproj.n_bands == 2
        assert locproj.n_proj == 16
        assert locproj.frac_coords.shape == (16, 3)
        assert len(locproj.angular_types) == 16
        assert len(locproj.radial_specs) == 16
        assert locproj.projections.shape == (2, 2, 1, 16)

    def test_locproj_to_dict(self, locproj_filepath: Path):
        """Test to_dict method."""
        locproj = vasp.Locproj(locproj_filepath)
        data = locproj.to_dict()

        assert "frac_coords" in data
        assert "angular_types" in data
        assert "radial_specs" in data
        assert "projections" in data
        
        assert isinstance(data["frac_coords"], np.ndarray)
        assert isinstance(data["angular_types"], list)
        assert isinstance(data["radial_specs"], list)
        assert isinstance(data["projections"], np.ndarray)

    def test_locproj_real_file(self):
        """Test parsing a real LOCPROJ file from test data."""
        locproj_path = DATA_DIR / "examples" / "other" / "FULL3d" / "LOCPROJ"
        
        if not locproj_path.exists():
            pytest.skip(f"Real LOCPROJ test file not found at {locproj_path}")
        
        locproj = vasp.Locproj(locproj_path)
        
        # Verify that data was parsed
        assert locproj.n_proj > 0
        assert locproj.n_k > 0
        assert locproj.n_bands > 0
        assert locproj.n_spins > 0
        
        # Verify array shapes
        assert locproj.frac_coords.shape[0] == locproj.n_proj
        assert len(locproj.angular_types) == locproj.n_proj
        assert len(locproj.radial_specs) == locproj.n_proj
        assert locproj.projections.shape == (
            locproj.n_k, 
            locproj.n_bands, 
            locproj.n_spins, 
            locproj.n_proj
        )
        
        # Verify data types
        assert locproj.projections.dtype in [np.complex128, np.complex64]
        assert np.iscomplexobj(locproj.projections)
