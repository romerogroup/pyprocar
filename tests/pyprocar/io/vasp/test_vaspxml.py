import math
from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.structure import Structure
from pyprocar.io.vasp.vasprun import VaspXML

dos_dir = Path("/home/lllang/Work/pyprocar/data/examples/dos")
bands_dir = Path("/home/lllang/Work/pyprocar/data/examples/bands")

bands_non_colinear_vasprun = bands_dir / "non-colinear" / "vasprun.xml"
bands_non_spin_polarized_vasprun = bands_dir / "non-spin-polarized" / "vasprun.xml"
bands_spin_polarized_vasprun = bands_dir / "spin-polarized" / "vasprun.xml"


dos_non_colinear_vasprun = dos_dir / "non-colinear" / "vasprun.xml"
dos_non_spin_polarized_vasprun = dos_dir / "non-spin-polarized" / "vasprun.xml"
dos_spin_polarized_vasprun = dos_dir / "spin-polarized" / "vasprun.xml"


# Fixtures
@pytest.fixture
def vasp_xml_non_spin_polarized():
    return VaspXML(filepath=bands_non_spin_polarized_vasprun)


@pytest.fixture
def vasp_xml_spin_polarized():
    return VaspXML(filepath=bands_spin_polarized_vasprun)


@pytest.fixture
def vasp_xml_non_colinear():
    return VaspXML(filepath=bands_non_colinear_vasprun)


@pytest.fixture
def vasp_xml_dos_non_spin_polarized():
    return VaspXML(filepath=dos_non_spin_polarized_vasprun)


@pytest.fixture
def vasp_xml_dos_spin_polarized():
    return VaspXML(filepath=dos_spin_polarized_vasprun)


@pytest.fixture
def vasp_xml_dos_non_colinear():
    return VaspXML(filepath=dos_non_colinear_vasprun)


# Test initialization
class TestVaspXMLInitialization:
    def test_init_with_filepath(self, vasp_xml_non_spin_polarized: VaspXML):
        assert vasp_xml_non_spin_polarized.filepath == bands_non_spin_polarized_vasprun

    def test_init_with_string(self, vasp_xml_non_spin_polarized: VaspXML):
        file_content = vasp_xml_non_spin_polarized.file_str
        vasp_xml_from_str = VaspXML.from_str(file_content)
        assert vasp_xml_from_str.filepath is None

    def test_filename_property(self, vasp_xml_non_spin_polarized: VaspXML):
        assert vasp_xml_non_spin_polarized.filename == "vasprun.xml"

    def test_file_str_property(self, vasp_xml_non_spin_polarized: VaspXML):
        file_str = vasp_xml_non_spin_polarized.file_str
        assert isinstance(file_str, str)

    def test_file_str_property_contains_xml(self, vasp_xml_non_spin_polarized: VaspXML):
        file_str = vasp_xml_non_spin_polarized.file_str
        assert "<?xml" in file_str


# Test spin-related properties
class TestVaspXMLSpinProperties:
    def test_is_spin_polarized_true(self, vasp_xml_spin_polarized: VaspXML):
        assert vasp_xml_spin_polarized.is_spin_polarized is True

    def test_is_spin_polarized_false(self, vasp_xml_non_spin_polarized: VaspXML):
        assert vasp_xml_non_spin_polarized.is_spin_polarized is False

    def test_is_noncolinear_true(self, vasp_xml_non_colinear: VaspXML):
        assert vasp_xml_non_colinear.is_noncolinear is True

    def test_is_noncolinear_false(self, vasp_xml_non_spin_polarized: VaspXML):
        assert vasp_xml_non_spin_polarized.is_noncolinear is False

    def test_has_dos_true(self, vasp_xml_dos_non_spin_polarized: VaspXML):
        assert vasp_xml_dos_non_spin_polarized.has_dos is True

    def test_spins_dict_colinear(self, vasp_xml_spin_polarized: VaspXML):
        spins_dict = vasp_xml_spin_polarized.spins_dict
        assert spins_dict == {"spin 1": "Spin-up", "spin 2": "Spin-down"}

    def test_spins_dict_non_colinear(self, vasp_xml_non_colinear: VaspXML):
        spins_dict = vasp_xml_non_colinear.spins_dict
        expected = {
            "spin 1": "Spin-Total",
            "spin 2": "Spin-x",
            "spin 3": "Spin-y",
            "spin 4": "Spin-z",
        }
        assert spins_dict == expected


# Test bands properties
class TestVaspXMLBands:
    def test_bands_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        bands = vasp_xml_non_spin_polarized.bands
        assert isinstance(bands, dict)

    def test_bands_has_spin_keys(self, vasp_xml_non_spin_polarized: VaspXML):
        bands = vasp_xml_non_spin_polarized.bands
        assert "spin 1" in bands

    def test_bands_eigen_values_shape(self, vasp_xml_non_spin_polarized: VaspXML):
        bands = vasp_xml_non_spin_polarized.bands
        eigen_values = bands["spin 1"]["eigen_values"]
        assert isinstance(eigen_values, np.ndarray)

    def test_bands_occupancies_shape(self, vasp_xml_non_spin_polarized: VaspXML):
        bands = vasp_xml_non_spin_polarized.bands
        occupancies = bands["spin 1"]["occupancies"]
        assert isinstance(occupancies, np.ndarray)

    def test_bands_spin_polarized_has_two_spins(self, vasp_xml_spin_polarized: VaspXML):
        bands = vasp_xml_spin_polarized.bands
        assert "spin 1" in bands
        assert "spin 2" in bands

    def test_bands_projected_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        bands_projected = vasp_xml_non_spin_polarized.bands_projected
        assert isinstance(bands_projected, dict)

    def test_bands_projected_has_labels(self, vasp_xml_non_spin_polarized: VaspXML):
        bands_projected = vasp_xml_non_spin_polarized.bands_projected
        assert "labels" in bands_projected

    def test_bands_projected_has_projection(self, vasp_xml_non_spin_polarized: VaspXML):
        bands_projected = vasp_xml_non_spin_polarized.bands_projected
        assert "projection" in bands_projected

    def test_bands_projected_projection_is_ndarray(self, vasp_xml_non_spin_polarized: VaspXML):
        bands_projected = vasp_xml_non_spin_polarized.bands_projected
        assert isinstance(bands_projected["projection"], np.ndarray)

    def test_bands_projected_projection_has_correct_dimensions(
        self, vasp_xml_non_spin_polarized: VaspXML
    ):
        bands_projected = vasp_xml_non_spin_polarized.bands_projected
        projection = bands_projected["projection"]
        assert projection.ndim == 6


# Test DOS properties
class TestVaspXMLDOS:
    def test_dos_total_returns_dict(self, vasp_xml_dos_non_spin_polarized: VaspXML):
        dos_total = vasp_xml_dos_non_spin_polarized.dos_total
        assert isinstance(dos_total, dict)

    def test_dos_total_has_energies(self, vasp_xml_dos_non_spin_polarized: VaspXML):
        dos_total = vasp_xml_dos_non_spin_polarized.dos_total
        assert "energies" in dos_total

    def test_dos_total_energies_is_ndarray(self, vasp_xml_dos_non_spin_polarized: VaspXML):
        dos_total = vasp_xml_dos_non_spin_polarized.dos_total
        assert isinstance(dos_total["energies"], np.ndarray)

    def test_dos_total_spin_polarized_has_spin_up(self, vasp_xml_dos_spin_polarized: VaspXML):
        dos_total = vasp_xml_dos_spin_polarized.dos_total
        assert "Spin-up" in dos_total

    def test_dos_total_spin_polarized_has_spin_down(self, vasp_xml_dos_spin_polarized: VaspXML):
        dos_total = vasp_xml_dos_spin_polarized.dos_total
        assert "Spin-down" in dos_total

    def test_dos_total_non_colinear_has_spin_total(self, vasp_xml_dos_non_colinear: VaspXML):
        dos_total = vasp_xml_dos_non_colinear.dos_total
        assert "Spin-Total" in dos_total

    def test_total_dos_returns_ndarray(self, vasp_xml_dos_non_spin_polarized: VaspXML):
        total_dos = vasp_xml_dos_non_spin_polarized.total_dos
        assert isinstance(total_dos, np.ndarray)

    def test_dos_projected_returns_ndarray_or_none(
        self, vasp_xml_dos_non_spin_polarized: VaspXML
    ):
        dos_projected = vasp_xml_dos_non_spin_polarized.dos_projected
        assert dos_projected is None or isinstance(dos_projected, np.ndarray)

    def test_dos_to_dict_returns_dict(self, vasp_xml_dos_non_spin_polarized: VaspXML):
        dos_dict = vasp_xml_dos_non_spin_polarized.dos_to_dict
        assert isinstance(dos_dict, dict)

    def test_dos_to_dict_has_total(self, vasp_xml_dos_non_spin_polarized: VaspXML):
        dos_dict = vasp_xml_dos_non_spin_polarized.dos_to_dict
        assert "total" in dos_dict


# Test structure properties
class TestVaspXMLStructure:
    def test_structures_returns_list(self, vasp_xml_non_spin_polarized: VaspXML):
        structures = vasp_xml_non_spin_polarized.structures
        assert isinstance(structures, list)

    def test_structures_contains_structure_objects(self, vasp_xml_non_spin_polarized: VaspXML):
        structures = vasp_xml_non_spin_polarized.structures
        assert all(isinstance(s, Structure) for s in structures)

    def test_initial_structure_is_structure(self, vasp_xml_non_spin_polarized: VaspXML):
        initial_structure = vasp_xml_non_spin_polarized.initial_structure
        assert isinstance(initial_structure, Structure)

    def test_final_structure_is_structure(self, vasp_xml_non_spin_polarized: VaspXML    ):
        final_structure = vasp_xml_non_spin_polarized.final_structure
        assert isinstance(final_structure, Structure)

    def test_structure_is_structure(self, vasp_xml_non_spin_polarized: VaspXML):
        structure = vasp_xml_non_spin_polarized.structure
        assert isinstance(structure, Structure)

    def test_structure_equals_final_structure(self, vasp_xml_non_spin_polarized: VaspXML):
        structure = vasp_xml_non_spin_polarized.structure
        final_structure = vasp_xml_non_spin_polarized.final_structure
        assert structure == final_structure

    def test_species_returns_list(self, vasp_xml_non_spin_polarized: VaspXML):
        species = vasp_xml_non_spin_polarized.species
        assert isinstance(species, list)

    def test_species_contains_strings(self, vasp_xml_non_spin_polarized: VaspXML):
        species = vasp_xml_non_spin_polarized.species
        assert all(isinstance(s, str) for s in species)


# Test kpoints properties
class TestVaspXMLKpoints:
    def test_kpoints_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        kpoints = vasp_xml_non_spin_polarized.kpoints
        assert isinstance(kpoints, dict)

    def test_kpoints_has_mode(self, vasp_xml_non_spin_polarized: VaspXML):
        kpoints = vasp_xml_non_spin_polarized.kpoints
        assert "mode" in kpoints

    def test_kpoints_list_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        kpoints_list = vasp_xml_non_spin_polarized.kpoints_list
        assert isinstance(kpoints_list, dict)

    def test_kpoints_list_has_kpoints_list(self, vasp_xml_non_spin_polarized: VaspXML):
        kpoints_list = vasp_xml_non_spin_polarized.kpoints_list
        assert "kpoints_list" in kpoints_list

    def test_kpoints_list_has_weights(self, vasp_xml_non_spin_polarized: VaspXML):
        kpoints_list = vasp_xml_non_spin_polarized.kpoints_list
        assert "weights" in kpoints_list


# Test energy and convergence properties
class TestVaspXMLEnergy:
    def test_fermi_returns_float_or_none(self, vasp_xml_dos_non_spin_polarized: VaspXML):
        fermi = vasp_xml_dos_non_spin_polarized.fermi
        assert fermi is None or isinstance(fermi, float)

    def test_energies_returns_list(self, vasp_xml_non_spin_polarized: VaspXML):
        energies = vasp_xml_non_spin_polarized.energies
        assert isinstance(energies, list)

    def test_energies_contains_lists(self, vasp_xml_non_spin_polarized: VaspXML):
        energies = vasp_xml_non_spin_polarized.energies
        assert all(isinstance(e, list) for e in energies)

    def test_last_energy_returns_float(self, vasp_xml_non_spin_polarized: VaspXML):
        last_energy = vasp_xml_non_spin_polarized.last_energy
        assert isinstance(last_energy, float)

    def test_energy_equals_last_energy(self, vasp_xml_non_spin_polarized: VaspXML):
        energy = vasp_xml_non_spin_polarized.energy
        last_energy = vasp_xml_non_spin_polarized.last_energy
        assert energy == last_energy

    def test_convergence_electronic_returns_bool(self, vasp_xml_non_spin_polarized: VaspXML):
        convergence = vasp_xml_non_spin_polarized.convergence_electronic
        assert isinstance(convergence, bool)

    def test_convergence_ionic_returns_bool(self, vasp_xml_non_spin_polarized: VaspXML):
        convergence = vasp_xml_non_spin_polarized.convergence_ionic
        assert isinstance(convergence, bool)

    def test_convergence_returns_bool(self, vasp_xml_non_spin_polarized: VaspXML):
        convergence = vasp_xml_non_spin_polarized.convergence
        assert isinstance(convergence, bool)

    def test_is_finished_returns_true(self, vasp_xml_non_spin_polarized: VaspXML):
        is_finished = vasp_xml_non_spin_polarized.is_finished
        assert is_finished is True


# Test metadata properties
class TestVaspXMLMetadata:
    def test_incar_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        incar = vasp_xml_non_spin_polarized.incar
        assert isinstance(incar, dict)

    def test_vasp_parameters_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        vasp_params = vasp_xml_non_spin_polarized.vasp_parameters
        assert isinstance(vasp_params, dict)

    def test_potcar_info_returns_list(self, vasp_xml_non_spin_polarized: VaspXML):
        potcar_info = vasp_xml_non_spin_polarized.potcar_info
        assert isinstance(potcar_info, dict)

    def test_forces_returns_list(self, vasp_xml_non_spin_polarized: VaspXML):
        forces = vasp_xml_non_spin_polarized.forces
        assert isinstance(forces, list)

    def test_iteration_data_returns_list(self, vasp_xml_non_spin_polarized: VaspXML):
        iteration_data = vasp_xml_non_spin_polarized.iteration_data
        assert isinstance(iteration_data, list)

    def test_calculation_returns_list(self, vasp_xml_non_spin_polarized: VaspXML):
        calculation = vasp_xml_non_spin_polarized.calculation
        assert isinstance(calculation, list)

    def test_run_info_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        run_info = vasp_xml_non_spin_polarized.run_info
        assert isinstance(run_info, dict)

    def test_general_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        general = vasp_xml_non_spin_polarized.general
        assert isinstance(general, dict)

    def test_kpoints_info_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        kpoints_info = vasp_xml_non_spin_polarized.kpoints_info
        assert isinstance(kpoints_info, dict)

    def test_vasp_params_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        vasp_params = vasp_xml_non_spin_polarized.vasp_params
        assert isinstance(vasp_params, dict)

    def test_atom_info_returns_dict(self, vasp_xml_non_spin_polarized: VaspXML):
        atom_info = vasp_xml_non_spin_polarized.atom_info
        assert isinstance(atom_info, dict)


# Test helper methods
class TestVaspXMLHelperMethods:
    def test_text_to_bool_true(self, vasp_xml_non_spin_polarized: VaspXML):
        assert vasp_xml_non_spin_polarized.text_to_bool("T") is True

    def test_text_to_bool_dot_true(self, vasp_xml_non_spin_polarized: VaspXML):
        assert vasp_xml_non_spin_polarized.text_to_bool(".True.") is True

    def test_text_to_bool_dot_true_upper(self, vasp_xml_non_spin_polarized: VaspXML):
        assert vasp_xml_non_spin_polarized.text_to_bool(".TRUE.") is True

    def test_text_to_bool_false(self, vasp_xml_non_spin_polarized: VaspXML):
        assert vasp_xml_non_spin_polarized.text_to_bool("F") is False

    def test_conv_string(self, vasp_xml_non_spin_polarized: VaspXML):
        result = vasp_xml_non_spin_polarized.conv("test", "string")
        assert result == "test"

    def test_conv_int(self, vasp_xml_non_spin_polarized: VaspXML):
        result = vasp_xml_non_spin_polarized.conv("42", "int")
        assert result == 42

    def test_conv_float(self, vasp_xml_non_spin_polarized: VaspXML):
        result = vasp_xml_non_spin_polarized.conv("3.14", "float")
        assert result == 3.14

    def test_conv_float_with_asterisk(self, vasp_xml_non_spin_polarized: VaspXML):
        result = vasp_xml_non_spin_polarized.conv("*****", "float")
        assert isinstance(result, float)
        assert math.isnan(result)

    def test_conv_logical_true(self, vasp_xml_non_spin_polarized: VaspXML):
        result = vasp_xml_non_spin_polarized.conv("T", "logical")
        assert result is True

    def test_conv_logical_false(self, vasp_xml_non_spin_polarized: VaspXML):
        result = vasp_xml_non_spin_polarized.conv("F", "logical")
        assert result is False


# Test mapping protocol
class TestVaspXMLMapping:
    def test_contains(self, vasp_xml_non_spin_polarized: VaspXML):
        assert "_filepath" in vasp_xml_non_spin_polarized

    def test_getitem(self, vasp_xml_non_spin_polarized: VaspXML):
        filepath = vasp_xml_non_spin_polarized["_filepath"]
        assert filepath is not None

    def test_len(self, vasp_xml_non_spin_polarized: VaspXML):
        length = len(vasp_xml_non_spin_polarized)
        assert length > 0

    def test_iter(self, vasp_xml_non_spin_polarized: VaspXML):
        keys = list(vasp_xml_non_spin_polarized)
        assert isinstance(keys, list)
