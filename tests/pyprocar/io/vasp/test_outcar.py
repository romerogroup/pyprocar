import json
import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io import vasp
from pyprocar.utils.log_utils import set_verbose_level
from tests.utils import DATA_DIR, BaseTest

logger = logging.getLogger(__name__)

VERBOSE = 2
set_verbose_level(VERBOSE)

OUTCAR_DATA_DIR = DATA_DIR / "io" / "vasp" / "outcar"


def get_test_id(outcar_filepath: Path) -> str:
    """Creates a nice, readable ID for each test run."""
    return f"{outcar_filepath.stem}"


outcar_files = []
for filepath in OUTCAR_DATA_DIR.glob("OUTCAR_*"):
    print(filepath)
    suffix = filepath.suffix
    if suffix in [".json", ".py"]:
        continue
    outcar_files.append(filepath)


@pytest.fixture(
    params=outcar_files,
    ids=get_test_id,
)
def outcar_filepath(request):
    """Fixture that provides OUTCAR file paths for testing."""
    return request.param



OUTCAR_v544 = """ vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Jun 02 2023 20:29:41) complex          
  
 executed on             LinuxIFC date 2023.12.27  15:16:11
 running on    4 total cores
 distrk:  each k-point on    4 cores,    1 groups
 distr:  one band on NCORES_PER_BAND=   1 cores,    4 groups
 
 E-fermi :   5.6251    

   volume of cell :      105.52
      direct lattice vectors                 reciprocal lattice vectors
     2.137553973 -1.234117362  0.000000000     0.233912222 -0.405147853  0.000000000
     2.137553973  1.234117362  0.000000000     0.233912222  0.405147853  0.000000000
     0.000000000  0.000000000 20.000000000     0.000000000  0.000000000  0.050000000
 
  Subroutine PRICEL returns:
 Original cell was already a primitive cell.
 

 Routine SETGRP: Setting up the symmetry group for a 
 hexagonal supercell.


 Subroutine GETGRP returns: Found 24 space group operations
 (whereof 12 operations were pure point group operations)
 out of a pool of 24 trial point group operations.


The dynamic configuration has the point symmetry D_3h.
 The point group associated with its full space group is D_6h.


 Subroutine INISYM returns: Found 24 space group operations
 (whereof 12 operations are pure point group operations),
 and found     1 'primitive' translations

 
 
 KPOINTS: k-points along high symmetry lines      
  interpolating k-points between supplied coordinates
  k-points in reciprocal lattice
Space group operators:
 irot       det(A)        alpha          n_x          n_y          n_z        tau_x        tau_y        tau_z
    1     1.000000     0.000000     1.000000     0.000000     0.000000     0.000000     0.000000     0.000000
    2    -1.000000    60.000000     0.000000     0.000000     1.000000     0.000000     0.000000     0.000000
    3     1.000000   120.000000     0.000000     0.000000     1.000000     0.000000     0.000000     0.000000
    4    -1.000000   180.000000     0.000000     0.000000     1.000000     0.000000     0.000000     0.000000
    5     1.000000   120.000000     0.000000     0.000000    -1.000000     0.000000     0.000000     0.000000
    6    -1.000000    60.000000     0.000000     0.000000    -1.000000     0.000000     0.000000     0.000000
    7    -1.000000   180.000000     0.000000     1.000000     0.000000     0.000000     0.000000     0.000000
    8     1.000000   180.000000     0.500000    -0.866025     0.000000     0.000000     0.000000     0.000000
    9    -1.000000   180.000000     0.866025    -0.500000     0.000000     0.000000     0.000000     0.000000
   10     1.000000   180.000000     1.000000     0.000000     0.000000     0.000000     0.000000     0.000000
   11    -1.000000   180.000000     0.866025     0.500000     0.000000     0.000000     0.000000     0.000000
   12     1.000000   180.000000    -0.500000    -0.866025     0.000000     0.000000     0.000000     0.000000
   13    -1.000000     0.000000     1.000000     0.000000     0.000000     0.333333    -0.666667     0.000000
   14     1.000000    60.000000     0.000000     0.000000     1.000000     0.333333    -0.666667     0.000000
   15    -1.000000   120.000000     0.000000     0.000000     1.000000     0.333333    -0.666667     0.000000
   16     1.000000   180.000000     0.000000     0.000000     1.000000     0.333333    -0.666667     0.000000
   17    -1.000000   120.000000     0.000000     0.000000    -1.000000     0.333333    -0.666667     0.000000
   18     1.000000    60.000000     0.000000     0.000000    -1.000000     0.333333    -0.666667     0.000000
   19     1.000000   180.000000     0.000000     1.000000     0.000000     0.333333    -0.666667     0.000000
   20    -1.000000   180.000000     0.500000    -0.866025     0.000000     0.333333    -0.666667     0.000000
   21     1.000000   180.000000     0.866025    -0.500000     0.000000     0.333333    -0.666667     0.000000
   22    -1.000000   180.000000     1.000000     0.000000     0.000000     0.333333    -0.666667     0.000000
   23     1.000000   180.000000     0.866025     0.500000     0.000000     0.333333    -0.666667     0.000000
   24    -1.000000   180.000000    -0.500000    -0.866025     0.000000     0.333333    -0.666667     0.000000


--------------------------------------------------------------------------------------------------------
"""


OUTCAR_v544_ibzkpt = """ vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Jun 11 2019 10:12:38) gamma-only       
  
 executed on             LinuxIFC date 2019.11.21  15:39:40
 running on   20 total cores
 distrk:  each k-point on   20 cores,    1 groups
 distr:  one band on NCORES_PER_BAND=  20 cores,    1 groups

 E-fermi :   5.6251    

  energy-cutoff  :      400.00
  volume of cell :     1205.20
      direct lattice vectors                 reciprocal lattice vectors
    10.641900000  0.000000000  0.000000000     0.093968182  0.000000000  0.000000000
     0.000000000 10.641900000  0.000000000     0.000000000  0.093968182  0.000000000
     0.000000000  0.000000000 10.641900000     0.000000000  0.000000000  0.093968182
     
Subroutine INISYM returns: Found  6 space group operations
 (whereof  6 operations are pure point group operations),
 and found     1 'primitive' translations

 
 
 KPOINTS: KPOINTS file                            

Automatic generation of k-mesh.
Space group operators:
 irot       det(A)        alpha          n_x          n_y          n_z        tau_x        tau_y        tau_z
    1     1.000000     0.000000     1.000000     0.000000     0.000000     0.000000     0.000000     0.000000
    2     1.000000   120.000000    -0.577350    -0.577350    -0.577350     0.000000     0.000000     0.000000
    3     1.000000   120.000000     0.577350     0.577350     0.577350     0.000000     0.000000     0.000000
    4    -1.000000   180.000000     0.000000    -0.707107     0.707107     0.000000     0.000000     0.000000
    5    -1.000000   180.000000     0.707107    -0.707107     0.000000     0.000000     0.000000     0.000000
    6    -1.000000   180.000000     0.707107     0.000000    -0.707107     0.000000     0.000000     0.000000
 
 Subroutine IBZKPT returns following result:
 ===========================================
 
"""


OUTCAR_v544_reclat_issue = """ vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Oct 26 2020 23:09:38) gamma-only       
  
 executed on             LinuxIFC date 2022.05.18  15:46:41
 running on   44 total cores
 distrk:  each k-point on   44 cores,    1 groups
 distr:  one band on NCORES_PER_BAND=  22 cores,    2 groups

 E-fermi :   5.6251  

  energy-cutoff  :      400.00
  volume of cell :     4246.89
      direct lattice vectors                 reciprocal lattice vectors
     8.753472000-15.161458247  0.000000000     0.057120192 -0.032978358  0.000000000
     8.753472000 15.161458247  0.000000000     0.057120192  0.032978358  0.000000000
     0.000000000  0.000000000 16.000000000     0.000000000  0.000000000  0.062500000

  length of vectors
    17.506944000 17.506944000 16.000000000     0.065956716  0.065956716  0.062500000

"""


OUTCAR_v642 = """ vasp.6.2.1 16May21 (build May 19 2022 15:04:39) complex                        
  
 executed on             LinuxIFC date 2022.12.04  09:57:01
 running on   40 total cores
 distrk:  each k-point on   40 cores,    1 groups
 distr:  one band on NCORE=  10 cores,    4 groups

 E-fermi :   5.6251    

  volume of cell :      11.4538

  direct lattice vectors                    reciprocal lattice vectors
     1.420026000  1.420026000  1.420026000     0.352106229  0.352106229  0.000000000
     1.420026000 -1.420026000 -1.420026000     0.352106229 -0.000000000 -0.352106229
    -1.420026000  1.420026000 -1.420026000    -0.000000000  0.352106229 -0.352106229

  length of vectors
     2.459557180  2.459557180  2.459557180     0.497953405  0.497953405  0.497953405

 Subroutine INISYM returns: Found 48 space group operations
 (whereof 48 operations are pure point group operations),
 and found     1 'primitive' translations



 irot  :   1
 --------------------------------------------------------------------
 isymop:   1   0   0
           0   1   0
           0   0   1

 gtrans:     0.0000000     0.0000000     0.0000000

 ptrans:     0.0000000     0.0000000     0.0000000
 
 rotmap:
 (   1->   1) 


 irot  :   2
 --------------------------------------------------------------------
 isymop:  -1   0   0
           0  -1   0
           0   0  -1

 gtrans:     0.0000000     0.0000000     0.0000000

 ptrans:     0.0000000     0.0000000     0.0000000
 
 rotmap:
 (   1->   1) 

"""

OUTCAR_v643 = """ vasp.6.4.3 19Mar24 (build Nov 30 2024 18:10:10) complex                        
  
 executed on             LinuxIFC date 2025.03.14  14:18:34
 running   40 mpi-ranks, on    1 nodes
 distrk:  each k-point on   10 cores,    4 groups
 distr:  one band on NCORE=   1 cores,   10 groups
 
 E-fermi :   5.6251    
 
energy-cutoff  :      600.00
  volume of cell :       56.91
      direct lattice vectors                 reciprocal lattice vectors
     3.846520000  0.000000000  0.000000000     0.259975250  0.000000000  0.000000000
     0.000000000  3.846520000  0.000000000     0.000000000  0.259975250  0.000000000
     0.000000000  0.000000000  3.846520000     0.000000000  0.000000000  0.259975250

  length of vectors
     3.846520000  3.846520000  3.846520000     0.259975250  0.259975250  0.259975250

 Subroutine INISYM returns: Found 48 space group operations
 (whereof 48 operations are pure point group operations),
 and found     1 'primitive' translations



 irot  :   1
 --------------------------------------------------------------------
 isymop:   1   0   0
           0   1   0
           0   0   1

 gtrans:     0.0000000     0.0000000     0.0000000

 ptrans:     0.0000000     0.0000000     0.0000000
 
 rotmap:
 (   1->   1)  (   2->   2)  (   3->   3)  (   4->   4)  (   5->   5) 


 irot  :   2
 --------------------------------------------------------------------
 isymop:  -1   0   0
           0  -1   0
           0   0  -1

 gtrans:     0.0000000     0.0000000     0.0000000

 ptrans:     0.0000000     0.0000000     0.0000000
 
 rotmap:
 (   1->   1)  (   2->   2)  (   3->   3)  (   4->   4)  (   5->   5) 



"""
@pytest.fixture
def outcar_v544(tmp_path):
    """Create a temporary PROJCAR file for testing."""
    outcar_file = tmp_path / "OUTCAR_v544"
    outcar_file.write_text(OUTCAR_v544)
    return outcar_file

@pytest.fixture
def outcar_v544_ibzkpt(tmp_path):
    """Create a temporary PROJCAR file for testing."""
    outcar_file = tmp_path / "OUTCAR_v544_ibzkpt"
    outcar_file.write_text(OUTCAR_v544_ibzkpt)
    return outcar_file

@pytest.fixture
def outcar_v544_reclat_issue(tmp_path):
    """Create a temporary PROJCAR file for testing."""
    outcar_file = tmp_path / "OUTCAR_v544_reclat_issue"
    outcar_file.write_text(OUTCAR_v544_reclat_issue)
    return outcar_file


@pytest.fixture
def outcar_v642(tmp_path):
    """Create a temporary PROJCAR file for testing."""
    outcar_file = tmp_path / "OUTCAR_v642"
    outcar_file.write_text(OUTCAR_v642)
    return outcar_file

@pytest.fixture
def outcar_v643(tmp_path):
    """Create a temporary PROJCAR file for testing."""
    outcar_file = tmp_path / "OUTCAR_v643"
    outcar_file.write_text(OUTCAR_v643)
    return outcar_file

class TestOutcar(BaseTest):
    
    def test_outcar_v544(self, outcar_v544):
        outcar = vasp.Outcar(outcar_v544)
        assert outcar.version == "5.4.4"
        assert outcar.version_tuple == (5, 4, 4)
        assert outcar.fermi == 5.6251
        assert outcar.get_symmetry_operations() is not None
        assert len(outcar.get_symmetry_operations()) == 24
        assert outcar.get_symmetry_operations()[0]["irot"] == 1
        assert outcar.get_symmetry_operations()[0]["rotation"].shape == (3, 3)
        assert outcar.get_symmetry_operations()[0]["gtrans"] is not None
        assert outcar.get_symmetry_operations()[0]["gtrans"].shape == (3,)
        assert outcar.get_symmetry_operations()[0]["gtrans"].dtype == float
        assert outcar.get_symmetry_operations()[0]["gtrans"].tolist() == [0.0, 0.0, 0.0]
        
    def test_outcar_v544(self, outcar_v544_ibzkpt):
        outcar = vasp.Outcar(outcar_v544_ibzkpt)
        assert outcar.version == "5.4.4"
        assert outcar.version_tuple == (5, 4, 4)
        assert outcar.fermi == 5.6251
        assert outcar.get_symmetry_operations() is not None
        assert len(outcar.get_symmetry_operations()) == 6
        assert outcar.get_symmetry_operations()[0]["irot"] == 1
        assert outcar.get_symmetry_operations()[0]["rotation"].shape == (3, 3)
        assert outcar.get_symmetry_operations()[0]["gtrans"] is not None
        assert outcar.get_symmetry_operations()[0]["gtrans"].shape == (3,)
        assert outcar.get_symmetry_operations()[0]["gtrans"].dtype == float
        
        
    def test_outcar_v544_reclat_issue_get_reciprocal_lattice(self, outcar_v544_reclat_issue):
        outcar = vasp.Outcar(outcar_v544_reclat_issue)
        assert outcar.version == "5.4.4"
        assert outcar.version_tuple == (5, 4, 4)
        assert outcar.fermi == 5.6251
        assert outcar.reciprocal_lattice is not None
        assert outcar.reciprocal_lattice.shape == (3, 3)
        assert outcar.reciprocal_lattice.dtype == float
        assert outcar.reciprocal_lattice.tolist() == [[0.057120192, -0.032978358, 0.0], [0.057120192, 0.032978358, 0.0], [0.0, 0.0, 0.062500000]]
        
    def test_outcar_v642(self, outcar_v642):
        outcar = vasp.Outcar(outcar_v642)
        assert outcar.version == "6.2.1"
        assert outcar.version_tuple == (6, 2, 1)
        assert outcar.fermi == 5.6251
        assert outcar.get_symmetry_operations() is not None
        assert len(outcar.get_symmetry_operations()) == 2
        assert outcar.get_symmetry_operations()[0]["irot"] == 1
        assert outcar.get_symmetry_operations()[0]["rotation"].shape == (3, 3)
        assert outcar.get_symmetry_operations()[0]["gtrans"] is not None
        assert outcar.get_symmetry_operations()[0]["gtrans"].shape == (3,)
        assert outcar.get_symmetry_operations()[0]["gtrans"].dtype == float
        
    def test_outcar_v643(self, outcar_v643):
        outcar = vasp.Outcar(outcar_v643)
        assert outcar.version == "6.4.3"
        assert outcar.version_tuple == (6, 4, 3)
        assert outcar.fermi == 5.6251
        assert outcar.get_symmetry_operations() is not None
        assert len(outcar.get_symmetry_operations()) == 2
        assert outcar.get_symmetry_operations()[0]["irot"] == 1
        assert outcar.get_symmetry_operations()[0]["rotation"].shape == (3, 3)
        assert outcar.get_symmetry_operations()[0]["gtrans"] is not None
        assert outcar.get_symmetry_operations()[0]["gtrans"].shape == (3,)
        assert outcar.get_symmetry_operations()[0]["gtrans"].dtype == float
        
    def test_outcar_from_str(self):
        outcar = vasp.Outcar.from_str(OUTCAR_v544)
        assert outcar.version == "5.4.4"
        assert outcar.version_tuple == (5, 4, 4)
        assert outcar.fermi == 5.6251
        assert outcar.get_symmetry_operations() is not None
        assert len(outcar.get_symmetry_operations()) == 24
        assert outcar.get_symmetry_operations()[0]["irot"] == 1
        assert outcar.get_symmetry_operations()[0]["rotation"].shape == (3, 3)
        assert outcar.get_symmetry_operations()[0]["gtrans"] is not None
        assert outcar.get_symmetry_operations()[0]["gtrans"].shape == (3,)
        assert outcar.get_symmetry_operations()[0]["gtrans"].dtype == float
        
