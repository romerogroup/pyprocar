"""Tests for ElkGeometry extractor."""

import logging

import numpy as np
import pytest

from pyprocar.io.elk.geometry import ElkGeometry
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


# Example GEOMETRY.OUT content from SrVO3 calculation
GEOMETRY_OUT = """
scale
 1.0

scale1
 1.0

scale2
 1.0

scale3
 1.0

avec
   7.258900000       0.000000000       0.000000000
   0.000000000       7.258900000       0.000000000
   0.000000000       0.000000000       7.258900000

atoms
   3                                    : nspecies
'Sr.in'                                 : spfname
   1                                    : natoms; atpos, bfcmt below
    0.00000000    0.00000000    0.00000000    0.00000000  0.00000000  0.00000000
'V.in'                                  : spfname
   1                                    : natoms; atpos, bfcmt below
    0.50000000    0.50000000    0.50000000    0.00000000  0.00000000  1.00000000
'O.in'                                  : spfname
   3                                    : natoms; atpos, bfcmt below
    0.50000000    0.50000000    0.00000000    0.00000000  0.00000000  0.00000000
    0.50000000    0.00000000    0.50000000    0.00000000  0.00000000  0.00000000
    0.00000000    0.50000000    0.50000000    0.00000000  0.00000000  0.00000000
"""


@pytest.fixture
def geometry_file(tmp_path):
    """Create a temporary GEOMETRY.OUT file."""
    geom_file = tmp_path / "GEOMETRY.OUT"
    geom_file.write_text(GEOMETRY_OUT)
    return geom_file


class TestElkGeometryInit(BaseTest):
    def test_geometry_from_filepath(self, geometry_file):
        """Test loading ElkGeometry from GEOMETRY.OUT file."""
        geometry = ElkGeometry(geometry_file)
        assert geometry is not None
        assert geometry.filepath == geometry_file

    def test_geometry_from_str(self):
        """Test loading ElkGeometry from string content."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        assert geometry is not None
        assert geometry.filepath is None


class TestElkGeometryLattice(BaseTest):
    def test_lattice_shape(self):
        """Test lattice vectors shape is 3x3."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        assert geometry.lattice.shape == (3, 3)

    def test_lattice_values(self):
        """Test lattice vectors values."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        expected = np.array(
            [
                [7.2589, 0.0, 0.0],
                [0.0, 7.2589, 0.0],
                [0.0, 0.0, 7.2589],
            ]
        )
        np.testing.assert_allclose(geometry.lattice, expected, rtol=1e-5)

    def test_lattice_dtype(self):
        """Test lattice vectors are float."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        assert geometry.lattice.dtype == np.float64


class TestElkGeometryAtoms(BaseTest):
    def test_nspecies(self):
        """Test number of species."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        assert geometry.nspecies == 3

    def test_atoms_list(self):
        """Test atoms list."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        assert geometry.atoms == ["Sr", "V", "O", "O", "O"]

    def test_natoms(self):
        """Test number of atoms."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        assert geometry.natoms == 5


class TestElkGeometryCoordinates(BaseTest):
    def test_fractional_coordinates_shape(self):
        """Test fractional coordinates shape."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        assert geometry.fractional_coordinates.shape == (5, 3)

    def test_fractional_coordinates_values(self):
        """Test fractional coordinates values."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],  # Sr
                [0.5, 0.5, 0.5],  # V
                [0.5, 0.5, 0.0],  # O
                [0.5, 0.0, 0.5],  # O
                [0.0, 0.5, 0.5],  # O
            ]
        )
        np.testing.assert_allclose(geometry.fractional_coordinates, expected)

    def test_fractional_coordinates_dtype(self):
        """Test fractional coordinates are float."""
        geometry = ElkGeometry.from_str(GEOMETRY_OUT)
        assert geometry.fractional_coordinates.dtype == np.float64
