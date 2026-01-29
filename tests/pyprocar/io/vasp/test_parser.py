from pathlib import Path

from pyprocar.io.vasp.kpoints import Kpoints
from pyprocar.io.vasp.outcar import Outcar
from pyprocar.io.vasp.parser import VaspParser
from pyprocar.io.vasp.procar import Procar


class TestVaspParserInitialization:
    """Test VaspParser initialization with different input types."""

    def test_init_with_dirpath_only(self, tmp_path: Path) -> None:
        """Test initialization with only dirpath."""
        parser = VaspParser(dirpath=tmp_path)

        # All parsers should be None since no files exist
        assert parser.outcar is None
        assert parser.procar is None
        assert parser.kpoints is None
        assert parser.poscar is None
        assert parser.vasprun is None
        assert parser.doscar is None

    def test_init_with_parser_objects(self):
        """Test initialization with pre-instantiated parser objects."""
        # Create parser objects from test data
        test_data_dir = Path(__file__).parents[4] / "data" / "io" / "vasp"
        outcar_obj = Outcar(test_data_dir / "outcar" / "OUTCAR_v64")
        procar_obj = Procar(test_data_dir / "procar" / "PROCAR_spin-polarized")
        kpoints_obj = Kpoints(test_data_dir / "kpoints" / "KPOINTS_bands")

        # Initialize parser with objects
        parser = VaspParser(dirpath="", outcar=outcar_obj, procar=procar_obj, kpoints=kpoints_obj)

        # Check that the objects are stored correctly
        assert parser.outcar is outcar_obj
        assert parser.procar is procar_obj
        assert parser.kpoints is kpoints_obj
        assert parser.poscar is None
        assert parser.vasprun is None
        assert parser.doscar is None

    def test_init_with_file_paths(self):
        """Test initialization with file paths."""
        test_data_dir = Path(__file__).parents[4] / "data" / "io" / "vasp"

        # Initialize parser with file paths
        parser = VaspParser(
            dirpath="",
            outcar=test_data_dir / "outcar" / "OUTCAR_v64",
            procar=test_data_dir / "procar" / "PROCAR_spin-polarized",
            kpoints=test_data_dir / "kpoints" / "KPOINTS_bands",
        )

        # Check that parsers were created
        assert isinstance(parser.outcar, Outcar)
        assert isinstance(parser.procar, Procar)
        assert isinstance(parser.kpoints, Kpoints)
        assert parser.poscar is None

    def test_init_with_mixed_types(self):
        """Test initialization with mix of paths and objects."""
        test_data_dir = Path(__file__).parents[4] / "data" / "io" / "vasp"
        outcar_obj = Outcar(test_data_dir / "outcar" / "OUTCAR_v64")

        # Initialize with mix of object and path
        parser = VaspParser(
            dirpath="", outcar=outcar_obj, procar=test_data_dir / "procar" / "PROCAR_spin-polarized"
        )

        # Check that both types work together
        assert parser.outcar is outcar_obj
        assert isinstance(parser.procar, Procar)

    def test_init_with_none_values(self):
        """Test initialization with None values."""
        parser = VaspParser(
            dirpath="",
            outcar=None,
            procar=None,
            kpoints=None,
            poscar=None,
            vasprun=None,
            doscar=None,
        )

        # All parsers should be None
        assert parser.outcar is None
        assert parser.procar is None
        assert parser.kpoints is None
        assert parser.poscar is None
        assert parser.vasprun is None
        assert parser.doscar is None


class TestVaspParserFromStr:
    """Test VaspParser.from_str class method."""

    def test_from_str_with_outcar(self):
        """Test from_str with OUTCAR content."""
        test_data_dir = Path(__file__).parents[4] / "data" / "io" / "vasp"
        outcar_path = test_data_dir / "outcar" / "OUTCAR_v64"

        # Read the file content
        with open(outcar_path) as f:
            outcar_content = f.read()

        # Create parser from string
        parser = VaspParser.from_str(outcar=outcar_content)

        # Check that parser was created correctly
        assert isinstance(parser.outcar, Outcar)
        assert parser.outcar.filepath is None  # No filepath when created from string
        assert parser.procar is None

    def test_from_str_with_multiple_files(self):
        """Test from_str with multiple file contents."""
        test_data_dir = Path(__file__).parents[4] / "data" / "io" / "vasp"

        # Read file contents
        with open(test_data_dir / "outcar" / "OUTCAR_v64") as f:
            outcar_content = f.read()

        with open(test_data_dir / "kpoints" / "KPOINTS_bands") as f:
            kpoints_content = f.read()

        # Create parser from strings
        parser = VaspParser.from_str(outcar=outcar_content, kpoints=kpoints_content)

        # Check that parsers were created
        assert isinstance(parser.outcar, Outcar)
        assert isinstance(parser.kpoints, Kpoints)
        assert parser.procar is None
        assert parser.poscar is None

    def test_from_str_with_none_values(self):
        """Test from_str with None values."""
        parser = VaspParser.from_str()

        # All parsers should be None
        assert parser.outcar is None
        assert parser.procar is None
        assert parser.kpoints is None
        assert parser.poscar is None
        assert parser.vasprun is None
        assert parser.doscar is None

    def test_from_str_dirpath_is_empty(self):
        """Test that from_str sets dirpath to empty string (which resolves to cwd)."""
        test_data_dir = Path(__file__).parents[4] / "data" / "io" / "vasp"

        with open(test_data_dir / "outcar" / "OUTCAR_v64") as f:
            outcar_content = f.read()

        parser = VaspParser.from_str(outcar=outcar_content)

        # dirpath should resolve to current working directory (from BaseParser)
        assert parser.dirpath == Path().resolve()
