import logging

import numpy as np
import pytest

from pyprocar.core.dos import DensityOfStates
from pyprocar.core.property_store import Property
from pyprocar.core.structure import Structure
from tests.utils import DATA_DIR


@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def fractional_coordinates():
    return np.array([
                    [0.00,0.00,0.00],
                    [0.50,0.50,0.50],
                    [0.50,0.50,0.00],
                    [0.50,0.00,0.50],
                    [0.00,0.50,0.50]])
    
@pytest.fixture
def lattice():
    return np.eye(3)

@pytest.fixture
def atoms():
    return ["Sr", "V", "O", "O", "O"]

@pytest.fixture
def structure(fractional_coordinates, lattice, atoms):
    return Structure(
        atoms=atoms,
        fractional_coordinates=fractional_coordinates,
        lattice=lattice,
    )

def _make_random_dos(
    rng: np.random.Generator,
    n_energies: int = 32,
    n_spins: int = 2,
    n_atoms: int = 5,
    n_orbitals: int = 4,
    structure: Structure | None = None,
) -> DensityOfStates:
    if structure is not None:
        n_atoms = len(structure.atoms)
    energies = np.linspace(-5.0, 5.0, n_energies, dtype=float)
    total = rng.random((n_energies, n_spins)) + 0.5
    projected = rng.random((n_energies, n_spins, n_atoms, n_orbitals))
    return DensityOfStates(
        energies=energies,
        total=total,
        projected=projected,
        orbital_names=["s", "p", "d", "f"][:n_orbitals],
        structure=structure,
    )

@pytest.fixture
def dos(rng, structure):
    n_energies = 32
    n_atoms = len(structure.atoms)
    n_orbitals = 9
    n_spins=4
    
    energies = np.linspace(-5.0, 5.0, n_energies, dtype=float)
    total = rng.random((n_energies, n_spins)) + 0.5
    projected = rng.random((n_energies, n_spins, n_atoms, n_orbitals))
    return DensityOfStates(
        energies=energies,
        total=total,
        projected=projected,
        structure=structure,
    )

@pytest.fixture
def dos_non_spin_polarized(rng, structure):
    return _make_random_dos(rng, n_spins=1, structure=structure)


@pytest.fixture
def dos_spin_polarized(rng, structure):
    return _make_random_dos(rng, n_spins=2, structure=structure)


@pytest.fixture
def dos_non_collinear(rng, structure):
    return _make_random_dos(rng, n_spins=4, structure=structure)


@pytest.fixture
def non_spin_polarized_dir():
    return DATA_DIR / "examples" / "dos" / "non-spin-polarized"

#------------------------------------------------------------------
# Initialization and metadata tests
#------------------------------------------------------------------

def test_from_code(non_spin_polarized_dir):
    dos = DensityOfStates.from_code(code="vasp", dirpath=non_spin_polarized_dir)
    assert dos.energies.shape[0] > 0, f"Energies empty ({dos.energies.shape})"
    assert dos.total.to_array().shape[0] == dos.energies.shape[0], f"total does not match energies ({dos.total.to_array().shape} != {dos.energies.shape})"
    assert dos.projected is not None, f"Projected DOS is None"


def test_spin_metadata(dos_non_spin_polarized, dos_spin_polarized, dos_non_collinear):
    assert not dos_non_spin_polarized.is_spin_polarized, f"Non-spin-polarized DOS is spin-polarized ({dos_non_spin_polarized.is_spin_polarized})"
    assert dos_spin_polarized.is_spin_polarized, f"Spin-polarized DOS is not spin-polarized ({dos_spin_polarized.is_spin_polarized})"
    assert dos_non_collinear.is_non_collinear, f"Non-collinear DOS is not non-collinear ({dos_non_collinear.is_non_collinear})"
    assert dos_non_collinear.spin_projection_names == ["total", "x", "y", "z"], f"Non-collinear DOS has incorrect spin projection names ({dos_non_collinear.spin_projection_names})"

#------------------------------------------------------------------
# Compute methods tests
#------------------------------------------------------------------

def test_compute_projected_sum_spin_polarized(dos_spin_polarized):
    atoms = [0, 2]
    orbitals = [1, 3]

    projected_sum = dos_spin_polarized.compute_projected_sum(
        atoms=atoms,
        orbitals=orbitals,
        spins=[0],
        keepdims=False,
    )
    result = projected_sum.to_array()

    projected = dos_spin_polarized.projected.to_array()
    expected = np.sum(projected[..., orbitals], axis=-1)
    expected = np.sum(expected[..., atoms], axis=-1)
    manual = expected[..., [0]]

    assert result.shape == manual.shape, f"Projected sum does not match manual ({result.shape} != {manual.shape})"
    assert np.allclose(result, manual), f"Projected sum does not match manual ({result} != {manual})"
    assert projected_sum.metadata["atoms"] == atoms

def test_compute_projected_sum_non_collinear(dos_non_collinear):
    atoms = [0, 1]
    orbitals = [0, 2]
    spins = [1, 2, 3]

    summed_property = dos_non_collinear.compute_projected_sum(
        atoms=atoms,
        orbitals=orbitals,
        spins=spins,
        keepdims=False,
    )
    summed = summed_property.to_array()
    assert summed.shape == (dos_non_collinear.n_energies, 3)
    
    expanded_property = dos_non_collinear.compute_projected_sum(
        atoms=atoms,
        orbitals=orbitals,
        spins=spins,
        keepdims=True,
    )
    expanded = expanded_property.to_array()

    assert expanded.shape == (dos_non_collinear.n_energies, len(spins), 1, 1)
    assert len(expanded_property.metadata["label"]) == len(spins)
    assert expanded_property.metadata["spin_component_labels_latex"] == ["S_x", "S_y", "S_z"]
 
def test_compute_projected_sum_species_list(dos):
    atoms = None
    orbitals = [0]
    spins = [0]
    species = ["Sr", "V"]
    
    projected_sum = dos.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=spins, species=species)
    
    assert isinstance(projected_sum, Property), f"The result should be a Property instance, given a species as a list ({projected_sum})"

def test_compute_projected_sum_species_str(dos):
    atoms = None
    orbitals = [0]
    spins = [0]
    species = "Sr"
    
    projected_sum = dos.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=spins, species=species)
    
    assert isinstance(projected_sum, Property), f"The result should be a Property instance, given a species as a string ({projected_sum})"

def test_compute_projected_sum_orbitals_int(dos):
    atoms = None
    orbitals = 0
    spins = [0]
    species = None
    
    projected_sum = dos.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=spins, species=species)
    
    assert isinstance(projected_sum, Property), f"The result should be a Property instance, given a an orbital as an integer ({projected_sum})"

def test_compute_projected_sum_species_orbital_map(dos):
    spins = [0]
    species_orbital_map = {"Sr": [0], "V": [1,2,3]}
    
    projected_sum = dos.compute_projected_sum(spins=spins, species_orbital_map=species_orbital_map)
    
    assert isinstance(projected_sum, Property), f"The result should be a Property instance, given a species_orbital_map ({projected_sum})"
    
def test_compute_projected_sum_species_orbital_map_list(dos):
    spins = [0]
    species_orbital_map = [{"Sr": [0], "V": [4,5,6,7,8]}, {"O": [0,1,2]}]
    
    projected_sum = dos.compute_projected_sum(spins=spins, species_orbital_map=species_orbital_map)
    
    assert isinstance(projected_sum, list), f"The result should be a list given a list of species_orbital_map, ({projected_sum})"
    for projected_sum in projected_sum:
        print(projected_sum.metadata)
        assert isinstance(projected_sum, Property), f"The result should be a Property instance, ({projected_sum})"

def test_compute_projected_sum_atoms_orbital_map(dos):
    spins = [0]
    atoms_orbital_map = {(0): [0], (1): [4,5,6,7,8] , (2,3,4): [0,1,2]}
    
    projected_sum = dos.compute_projected_sum(spins=spins, atoms_orbital_map=atoms_orbital_map)
    
    assert isinstance(projected_sum, Property), f"The result should be a Property instance, given a atoms_orbital_map ({projected_sum})"
    
def test_compute_projected_sum_atoms_orbital_map_list(dos):
    spins = [0]
    atoms_orbital_map = [ {(0): [0], (1): [4,5,6,7,8] , (2): [0,1,2]}, {(0): [0]}]

    projected_sum = dos.compute_projected_sum(spins=spins, atoms_orbital_map=atoms_orbital_map)

   
    assert isinstance(projected_sum, list), f"The result should be a list given a list of atoms_orbital_map, ({projected_sum})"
    for projected_sum in projected_sum:
        assert isinstance(projected_sum, Property), f"The result should be a Property instance, ({projected_sum})"


def test_compute_projected_sum_metadata_contains_latex(dos_spin_polarized):
    prop = dos_spin_polarized.compute_projected_sum(
        atoms=[1],
        orbitals=[0],
        spins=[0],
        norm_mode="raw",
    )

    metadata = prop.metadata

    assert metadata["label"] == ["$\\mathrm{V}_{1}-(s)[\\uparrow]$"]
    assert metadata["label_plain"] == ["V_{1}-(s)[Spin-up]"]
    assert metadata["atom_label"] == "V_{1}"
    assert metadata["atom_label_latex"] == "\\mathrm{V}_{1}"
    assert metadata["spin_label_latex"] == "\\uparrow"
    assert metadata["spin_component_labels_latex"] == ["\\uparrow"]
    assert metadata["include_normal_label"] is False


def test_compute_projected_sum_metadata_without_normal_label(dos_spin_polarized):
    prop = dos_spin_polarized.compute_projected_sum(
        atoms=[1],
        orbitals=[0],
        spins=[0],
        norm_mode="raw",
        include_normal_label=False,
    )

    metadata = prop.metadata

    assert metadata["label"] == ["$\\mathrm{V}_{1}-(s)[\\uparrow]$"]
    assert metadata["label_plain"] == ["V_{1}-(s)[Spin-up]"]
    assert metadata["include_normal_label"] is False

def test_get_property_projected_sum_matches_dos_sum(dos_spin_polarized):
    atoms = [0, 1]
    orbitals = [0, 2]
    spins = [0]

    property_value = dos_spin_polarized.get_property(
        "projected_sum", atoms=atoms, orbitals=orbitals, spins=spins, keepdims=False
    ).to_array()
    projected = dos_spin_polarized.projected.to_array()
    manual = np.sum(projected[..., orbitals], axis=-1)
    manual = np.sum(manual[..., atoms], axis=-1)
    manual = manual[..., spins]
    
    assert property_value.shape == manual.shape

    assert np.allclose(property_value, manual)
    assert "projected_sum|atoms=0,1|orbitals=0,2|spins=0" in dos_spin_polarized.property_store


def test_get_property_projected_sum_caches_variants(dos_spin_polarized):
    initial_keys = set(dos_spin_polarized.property_store.keys())
    dos_spin_polarized.get_property("projected_sum", atoms=[0], orbitals=[0], spins=[0])
    after_first = set(dos_spin_polarized.property_store.keys())
    dos_spin_polarized.get_property("projected_sum", atoms=[1], orbitals=[1], spins=[0])
    after_second = set(dos_spin_polarized.property_store.keys())

    assert len(after_first - initial_keys) == 1
    assert len(after_second - after_first) == 1
    assert len(after_second - initial_keys) == 2


def _make_simple_structure() -> Structure:
    fractional_coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.5],
        ]
    )
    lattice = np.eye(3)
    atoms = ["Sr", "Sr", "O"]
    return Structure(atoms=atoms, fractional_coordinates=fractional_coordinates, lattice=lattice)

def test_compute_gradients_total(dos_spin_polarized):
    dos_spin_polarized.compute_gradients(gradient_order=2, names=["total"])
    grad = dos_spin_polarized.get_property(("total", "gradients", 1))
    hess = dos_spin_polarized.get_property(("total", "gradients", 2))
    assert grad.shape == dos_spin_polarized.total.to_array().shape
    assert hess.shape == dos_spin_polarized.total.to_array().shape

def test_compute_gradients_projected(dos_spin_polarized):
    dos_spin_polarized.compute_gradients(gradient_order=2, names=["projected"])
    grad = dos_spin_polarized.get_property(("projected", "gradients", 1))
    hess = dos_spin_polarized.get_property(("projected", "gradients", 2))
    assert grad.shape == dos_spin_polarized.projected.to_array().shape
    assert hess.shape == dos_spin_polarized.projected.to_array().shape


#------------------------------------------------------------------
# Get property tests
#------------------------------------------------------------------

def test_get_property_projected_sum_total(dos_spin_polarized):
    total_property = dos_spin_polarized.get_property(
        "projected_sum_total", spins=[0], keepdims=False
    ).to_array()
    manual_total = dos_spin_polarized.sum_projection_components(
        values_array=dos_spin_polarized.projected.to_array(),
        spins=[0],
        keepdims=False,
    )
    assert np.allclose(total_property, manual_total)

def test_get_property_normalized_total_max_normalization(dos_spin_polarized):
    normalized_array = dos_spin_polarized.get_property("normalized_total").to_array()
    total_array = dos_spin_polarized.total.to_array()
    assert normalized_array.shape == total_array.shape, f"Normalized total does not match total ({normalized_array.shape} != {total_array.shape})"
    assert np.allclose(np.max(np.abs(normalized_array), axis=0), np.ones(dos_spin_polarized.n_spin_channels))

def test_get_property_normalized_total_integral_normalization(dos_spin_polarized):
    normalized_integral = dos_spin_polarized.get_property("normalized_total", norm_mode="integral").to_array()
    integrals = np.trapezoid(normalized_integral, x=dos_spin_polarized.energies, axis=0)
    assert np.allclose(integrals, np.ones(dos_spin_polarized.n_spin_channels)), f"Integrals do not match ones ({integrals} != {np.ones(dos_spin_polarized.n_spin_channels)})"


def test_get_property_cumulative_total(dos_spin_polarized):
    cumulative = dos_spin_polarized.get_property("cumulative_total").to_array()
    assert cumulative.shape == dos_spin_polarized.total.to_array().shape, f"Cumulative total does not match total ({cumulative.shape} != {dos_spin_polarized.total.to_array().shape})"
    diff = np.diff(cumulative, axis=0)
    assert np.all(diff >= -1e-12)

#------------------------------------------------------------------
# Add property tests
#------------------------------------------------------------------


def test_add_property_from_array(dos_spin_polarized):
    custom = np.linspace(0, 1, dos_spin_polarized.n_energies)
    dos_spin_polarized.add_property(name="custom", value=custom)
    stored = dos_spin_polarized.get_property("custom")
    assert stored.name == "custom"
    assert np.allclose(stored.value, custom)


def test_add_property_from_property_instance(dos_spin_polarized):
    value = np.vstack(
        [
            np.linspace(0, 1, dos_spin_polarized.n_energies),
            np.linspace(1, 0, dos_spin_polarized.n_energies),
        ]
    ).T
    prop = Property(name="custom_vector", value=value)
    dos_spin_polarized.add_property(property=prop)
    stored = dos_spin_polarized.get_property("custom_vector")
    assert stored.name == "custom_vector"
    assert np.allclose(stored.value, value)


def test_add_property_invalid_shape_raises(dos_spin_polarized):
    with pytest.raises(ValueError):
        dos_spin_polarized.add_property(name="bad", value=np.ones((2,)))





def test_get_species_atom_map_str(dos):
    specie_atom_groups = dos.get_species_atom_map(species="Sr")
    
    assert len(specie_atom_groups) == 1
    assert set(specie_atom_groups.items()) == set([("Sr", tuple([0]))]), f"Specie atom groups do not match ({specie_atom_groups})"

def test_get_species_atom_map_none(dos):
    specie_atom_groups = dos.get_species_atom_map(species=None)
    
    assert len(specie_atom_groups) == len(dos.species), f"Specie atom groups do not match ({specie_atom_groups})"
    assert set(specie_atom_groups.items()) == set([("Sr", tuple([0])), ("V", tuple([1])), ("O", tuple([2, 3, 4]))]), f"Specie atom groups do not match ({specie_atom_groups})"

def test_get_species_atom_map_list(dos):
    specie_atom_groups = dos.get_species_atom_map(species=["Sr", "O"])
    
    assert len(specie_atom_groups) == 2
    assert set(specie_atom_groups.items()) == set([("Sr", tuple([0])), ("O", tuple([2, 3, 4]))]), f"Specie atom groups do not match ({specie_atom_groups})"
    
    
def test_compute_projected_sum_atom_groups(dos):
    atoms = [[0, 1], [2, 0, 1]]
    orbitals = [0]
    spins = [0]
    
    projected_sum_groups = dos.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=spins)
 
    assert len(projected_sum_groups) == 2
    
    for projected_sum in projected_sum_groups:
        print(repr(projected_sum))
    
    

    
