import logging
from pathlib import Path

import pytest

from tests.pyprocar.io.abinit import ABINIT_DATA_DIR, CALC_TYPES
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


def get_output_files() -> list[Path]:
    """Get all abinit.out files for testing."""
    files: list[Path] = []
    for calc_type in CALC_TYPES:
        for mode in ["bands", "dos", "fermi"]:
            filepath = ABINIT_DATA_DIR / calc_type / mode / "abinit.out"
            if filepath.exists():
                files.append(filepath)
    return files


@pytest.fixture(params=get_output_files(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
def output_filepath(request: pytest.FixtureRequest) -> Path:
    return request.param  # type: ignore[no-any-return]


class TestAbinitOutputIsFileOfType(BaseTest):
    def test_is_file_of_type_returns_true_for_abinit_output(self, output_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        assert AbinitOutput.is_file_of_type(output_filepath) is True

    def test_is_file_of_type_returns_false_for_non_abinit_file(self, tmp_path: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        fake_file = tmp_path / "fake.out"
        fake_file.write_text("This is not an ABINIT file")
        assert AbinitOutput.is_file_of_type(fake_file) is False


class TestAbinitOutputInit(BaseTest):
    def test_init_from_filepath(self, output_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        output = AbinitOutput(output_filepath)
        assert output._filepath is not None

    def test_from_str_creates_instance(self, output_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        content = output_filepath.read_text()
        output = AbinitOutput.from_str(content)
        assert output._file_str != ""


class TestAbinitOutputFermi(BaseTest):
    def test_fermi_is_float(self, output_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        output = AbinitOutput(output_filepath)
        assert isinstance(output.fermi, float)

    def test_fermi_in_reasonable_range(self, output_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        output = AbinitOutput(output_filepath)
        # Fermi energy typically between -50 and +50 eV
        assert -50 < output.fermi < 50


class TestAbinitOutputNspin(BaseTest):
    def test_nspin_for_non_spin_polarized(self) -> None:
        from pyprocar.io.abinit import AbinitOutput

        filepath = ABINIT_DATA_DIR / "non-spin-polarized" / "bands" / "abinit.out"
        output = AbinitOutput(filepath)
        assert output.nspin == 1

    def test_nspin_for_spin_polarized(self) -> None:
        from pyprocar.io.abinit import AbinitOutput

        filepath = ABINIT_DATA_DIR / "spin-polarized-colinear" / "bands" / "abinit.out"
        output = AbinitOutput(filepath)
        assert output.nspin == 2


class TestAbinitOutputStructure(BaseTest):
    def test_structure_returns_structure_object(self, output_filepath: Path) -> None:
        from pyprocar.core import Structure
        from pyprocar.io.abinit import AbinitOutput

        output = AbinitOutput(output_filepath)
        assert isinstance(output.structure, Structure)

    def test_lattice_shape_is_3x3(self, output_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        output = AbinitOutput(output_filepath)
        assert output.lattice.shape == (3, 3)

    def test_reclat_shape_is_3x3(self, output_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        output = AbinitOutput(output_filepath)
        assert output.reclat.shape == (3, 3)

    def test_atoms_is_non_empty_list(self, output_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitOutput

        output = AbinitOutput(output_filepath)
        assert len(output.atoms) > 0
