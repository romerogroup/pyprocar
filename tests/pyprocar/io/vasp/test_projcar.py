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

PROJCAR_DATA_DIR = DATA_DIR / "io" / "vasp" / "projcar"


class TestProjcarBase(BaseTest):
    def test_projcar_structure(self):
        """
        Test that Projcar class exists and has expected structure.
        """
        assert hasattr(vasp, "Projcar")
        assert callable(vasp.Projcar)

    def test_projcar_dict_schema(self, projcar_filepath):
        """
        Test that Projcar returns dictionary with correct schema.

        Schema:
        {
            "frac_coords": np.ndarray [n_atoms, 3],
            "radial_specs": list[dict{type: , params: {"N": int, "za": float}}],
            "projections": np.ndarray [n_k, n_bands, n_spins, n_atoms, n_orbitals] complex
        }
        """
        projcar = vasp.Projcar(projcar_filepath)
        data = projcar.to_dict()

        # Check that all required keys exist
        assert "frac_coords" in data
        assert "radial_specs" in data
        assert "projections" in data

        # Check frac_coords
        frac_coords = data["frac_coords"]
        assert isinstance(frac_coords, np.ndarray)
        assert frac_coords.ndim == 2
        assert frac_coords.shape[1] == 3
        n_atoms = frac_coords.shape[0]
        assert n_atoms > 0

        # Check radial_specs
        radial_specs = data["radial_specs"]
        assert isinstance(radial_specs, list)
        assert len(radial_specs) == n_atoms
        for spec in radial_specs:
            assert isinstance(spec, dict)
            assert "type" in spec
            assert "params" in spec
            assert isinstance(spec["params"], dict)

        # Check projections
        projections = data["projections"]
        assert isinstance(projections, np.ndarray)
        assert projections.ndim == 5
        assert projections.dtype == np.complex128 or projections.dtype == np.complex64
        assert projections.shape[3] == n_atoms

        # Verify shape consistency
        n_k, n_bands, n_spins, n_atoms_proj, n_orbitals = projections.shape
        assert n_atoms_proj == n_atoms
        assert n_k > 0
        assert n_bands > 0
        assert n_spins > 0
        assert n_orbitals > 0

    def test_projcar_attributes(self, projcar_filepath):
        """
        Test that Projcar has expected attributes.
        """
        projcar = vasp.Projcar(projcar_filepath)

        assert hasattr(projcar, "frac_coords")
        assert hasattr(projcar, "radial_specs")
        assert hasattr(projcar, "projections")
        assert hasattr(projcar, "n_atoms")
        assert hasattr(projcar, "n_k")
        assert hasattr(projcar, "n_bands")
        assert hasattr(projcar, "n_spins")
        assert hasattr(projcar, "n_orbitals")

        assert isinstance(projcar.frac_coords, np.ndarray)
        assert isinstance(projcar.radial_specs, list)
        assert isinstance(projcar.projections, np.ndarray)

    def test_projcar_mapping_interface(self, projcar_filepath):
        """
        Test that Projcar implements collections.abc.Mapping interface.
        """
        projcar = vasp.Projcar(projcar_filepath)

        # Test __contains__
        assert "frac_coords" in projcar
        assert "radial_specs" in projcar
        assert "projections" in projcar

        # Test __getitem__
        assert projcar["frac_coords"] is projcar.frac_coords
        assert projcar["radial_specs"] is projcar.radial_specs
        assert projcar["projections"] is projcar.projections

        # Test __iter__
        keys = list(projcar)
        assert "frac_coords" in keys
        assert "radial_specs" in keys
        assert "projections" in keys

        # Test __len__
        assert len(projcar) > 0

    def test_radial_specs_format(self, projcar_filepath):
        """
        Test that radial_specs have correct format.
        """
        projcar = vasp.Projcar(projcar_filepath)
        radial_specs = projcar.radial_specs

        for spec in radial_specs:
            assert "type" in spec
            assert isinstance(spec["type"], str)
            assert "params" in spec
            assert isinstance(spec["params"], dict)

            # Check for Hydrogen-like params
            if "Hydrogen-like" in spec["type"] or spec["type"] == "Hy":
                if "N" in spec["params"]:
                    assert isinstance(spec["params"]["N"], int)
                if "za" in spec["params"]:
                    assert isinstance(spec["params"]["za"], (int, float))

    def test_projections_complex_dtype(self, projcar_filepath):
        """
        Test that projections array has complex dtype.
        """
        projcar = vasp.Projcar(projcar_filepath)
        projections = projcar.projections

        assert np.iscomplexobj(projections)
        assert projections.dtype == np.complex128 or projections.dtype == np.complex64


def get_test_id(projcar_filepath: Path) -> str:
    """Creates a nice, readable ID for each test run."""
    return f"{projcar_filepath.stem}"


# Check if PROJCAR test data directory exists
projcar_files = []
if PROJCAR_DATA_DIR.exists():
    for filepath in PROJCAR_DATA_DIR.glob("PROJCAR*"):
        suffix = filepath.suffix
        if suffix in [".json", ".py"]:
            continue
        projcar_files.append(filepath)
else:
    # Create directory structure if it doesn't exist
    PROJCAR_DATA_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(
    params=projcar_files if projcar_files else [None],
    ids=get_test_id if projcar_files else lambda x: "no_file",
)
def projcar_filepath(request):
    """Fixture that provides PROJCAR file paths for testing."""
    if request.param is None:
        pytest.skip("No PROJCAR test files found")
    return request.param

