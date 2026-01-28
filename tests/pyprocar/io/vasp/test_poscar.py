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


POSCAR_STR = """ Sr V O
1.0
   3.8465199999999999   0.0000000000000000   0.0000000000000000
   0.0000000000000000   3.8465199999999999   0.0000000000000000
   0.0000000000000000   0.0000000000000000   3.8465199999999999
 Sr V O
 1 1 3
Direct
   0.0000000000000000   0.0000000000000000   0.0000000000000000
   0.5000000000000000   0.5000000000000000   0.5000000000000000
   0.5000000000000000   0.5000000000000000   0.0000000000000000
   0.5000000000000000   0.0000000000000000   0.5000000000000000
   0.0000000000000000   0.5000000000000000   0.5000000000000000
   
"""

POSCAR_STR_CARTESIAN = """Si
1.0
   5.4306975000000000   0.0000000000000000   0.0000000000000000
   0.0000000000000000   5.4306975000000000   0.0000000000000000
   0.0000000000000000   0.0000000000000000   5.4306975000000000
 2
Cartesian
   0.0000000000000000   0.0000000000000000   0.0000000000000000
   1.3576743750000000   1.3576743750000000   1.3576743750000000
"""

POSCAR_STR_SELECTIVE_DYNAMICS = """Sr V O
1.0
   3.8465199999999999   0.0000000000000000   0.0000000000000000
   0.0000000000000000   3.8465199999999999   0.0000000000000000
   0.0000000000000000   0.0000000000000000   3.8465199999999999
 Sr V O
 1 1 3
Selective Dynamics
Direct
   0.0000000000000000   0.0000000000000000   0.0000000000000000 T T T
   0.5000000000000000   0.5000000000000000   0.5000000000000000 F F F
   0.5000000000000000   0.5000000000000000   0.0000000000000000 T T T
   0.5000000000000000   0.0000000000000000   0.5000000000000000 T T T
   0.0000000000000000   0.5000000000000000   0.5000000000000000 T T T
"""


@pytest.fixture
def poscar_filepath(tmp_path: Path) -> Path:
    """Create a temporary POSCAR file for testing."""
    poscar_file = tmp_path / "POSCAR"
    poscar_file.write_text(POSCAR_STR)
    return poscar_file


@pytest.fixture
def poscar_filepath_cartesian(tmp_path: Path) -> Path:
    """Create a temporary POSCAR file with Cartesian coordinates."""
    poscar_file = tmp_path / "POSCAR_cartesian"
    poscar_file.write_text(POSCAR_STR_CARTESIAN)
    return poscar_file


@pytest.fixture
def poscar_filepath_selective_dynamics(tmp_path: Path) -> Path:
    """Create a temporary POSCAR file with Selective Dynamics."""
    poscar_file = tmp_path / "POSCAR_selective"
    poscar_file.write_text(POSCAR_STR_SELECTIVE_DYNAMICS)
    return poscar_file


class TestPoscar:
    def test_poscar_comment(self, poscar_filepath: Path) -> None:
        """Test that comment line is correctly parsed."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.comment.strip() == "Sr V O"

    def test_poscar_scale(self, poscar_filepath: Path) -> None:
        """Test that scale factor is correctly parsed."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.scale == 1.0

    def test_poscar_lattice(self, poscar_filepath: Path) -> None:
        """Test that lattice vectors are correctly parsed."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.lattice.shape == (3, 3)
        assert poscar.lattice.dtype == np.float64

        # Check that lattice is scaled correctly
        expected_lattice = np.array(
            [
                [3.8465199999999999, 0.0000000000000000, 0.0000000000000000],
                [0.0000000000000000, 3.8465199999999999, 0.0000000000000000],
                [0.0000000000000000, 0.0000000000000000, 3.8465199999999999],
            ]
        )
        assert np.allclose(poscar.lattice, expected_lattice)

    def test_poscar_has_species_names_line(self, poscar_filepath: Path) -> None:
        """Test detection of species names line."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.has_species_names_line is True

    def test_poscar_has_selective_dynamics_line(
        self, poscar_filepath: Path, poscar_filepath_selective_dynamics: Path
    ) -> None:
        """Test detection of Selective Dynamics line."""
        poscar = vasp.Poscar(poscar_filepath)
        print(poscar.file_str)
        assert poscar.has_selective_dynamics_line is False

        poscar_sd = vasp.Poscar(poscar_filepath_selective_dynamics)
        assert poscar_sd.has_selective_dynamics_line is True

    def test_poscar_species(self, poscar_filepath: Path) -> None:
        """Test that species are correctly parsed."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.species == ["Sr", "V", "O"]

    def test_poscar_composition(self, poscar_filepath: Path) -> None:
        """Test that composition is correctly parsed."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.composition == [1, 1, 3]

    def test_poscar_atoms(self, poscar_filepath: Path) -> None:
        """Test that atoms list is correctly generated."""
        poscar = vasp.Poscar(poscar_filepath)

        expected_atoms = ["Sr", "V", "O", "O", "O"]
        assert poscar.atoms == expected_atoms

    def test_poscar_n_atoms(self, poscar_filepath: Path) -> None:
        """Test that number of atoms is correctly calculated."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.n_atoms == 5
        assert poscar.n_atoms == sum(poscar.composition)

    def test_poscar_coord_system_direct(self, poscar_filepath: Path) -> None:
        """Test that Direct coordinate system is correctly detected."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.coord_system == "direct"

    def test_poscar_coord_system_cartesian(self, poscar_filepath_cartesian: Path) -> None:
        """Test that Cartesian coordinate system is correctly detected."""
        poscar = vasp.Poscar(poscar_filepath_cartesian)

        assert poscar.coord_system == "cartesian"

    def test_poscar_coord_system_line_number(self, poscar_filepath: Path) -> None:
        """Test that coordinate system line number is correctly found."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.coord_system_line_number == 7  # 0-indexed, line 8 in file

    def test_poscar_ion_positions(self, poscar_filepath: Path) -> None:
        """Test that ion positions are correctly parsed."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar.ion_positions.shape == (5, 3)
        assert poscar.ion_positions.dtype == np.float64

        # Check first position
        assert np.allclose(poscar.ion_positions[0], [0.0, 0.0, 0.0])

        # Check second position
        assert np.allclose(poscar.ion_positions[1], [0.5, 0.5, 0.5])

    def test_poscar_coordinates(self, poscar_filepath: Path) -> None:
        """Test that coordinates property is an alias for ion_positions."""
        poscar = vasp.Poscar(poscar_filepath)

        assert np.array_equal(poscar.coordinates, poscar.ion_positions)

    def test_poscar_from_str(self) -> None:
        """Test parsing from string."""
        poscar = vasp.Poscar.from_str(POSCAR_STR)

        assert poscar.comment.strip() == "Sr V O"
        assert poscar.scale == 1.0
        assert poscar.species == ["Sr", "V", "O"]
        assert poscar.composition == [1, 1, 3]
        assert poscar.n_atoms == 5
        assert poscar.coord_system == "direct"
        assert poscar.ion_positions.shape == (5, 3)

    def test_poscar_mapping_interface_getitem(self, poscar_filepath: Path) -> None:
        """Test Mapping interface __getitem__."""
        poscar = vasp.Poscar(poscar_filepath)

        assert poscar["comment"] == poscar.comment
        assert poscar["scale"] == poscar.scale
        assert poscar["species"] == poscar.species
        assert poscar["n_atoms"] == poscar.n_atoms

    def test_poscar_mapping_interface_iter(self, poscar_filepath: Path) -> None:
        """Test Mapping interface __iter__."""
        poscar = vasp.Poscar(poscar_filepath)

        atoms_list = list(poscar)
        assert atoms_list == poscar.atoms
        assert len(atoms_list) == 5

    def test_poscar_mapping_interface_len(self, poscar_filepath: Path) -> None:
        """Test Mapping interface __len__."""
        poscar = vasp.Poscar(poscar_filepath)

        assert len(poscar) == 5
        assert len(poscar) == poscar.n_atoms

    def test_poscar_real_file(self) -> None:
        """Test parsing a real POSCAR file from test data."""
        poscar_path = DATA_DIR / "examples" / "dos" / "non-spin-polarized" / "POSCAR"

        if not poscar_path.exists():
            pytest.skip(f"Real POSCAR test file not found at {poscar_path}")

        poscar = vasp.Poscar(poscar_path)

        # Verify that data was parsed
        assert poscar.n_atoms > 0
        assert len(poscar.species) > 0
        assert len(poscar.composition) > 0
        assert poscar.composition == [1, 1, 3]
        assert poscar.species == ["Sr", "V", "O"]

        # Verify array shapes
        assert poscar.lattice.shape == (3, 3)
        assert poscar.ion_positions.shape == (poscar.n_atoms, 3)
        assert len(poscar.atoms) == poscar.n_atoms

        # Verify data types
        assert poscar.lattice.dtype == np.float64
        assert poscar.ion_positions.dtype == np.float64

        # Verify coordinate system
        assert poscar.coord_system in ["direct", "cartesian"]
