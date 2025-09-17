import numpy as np
import pytest

from pyprocar.core.dos import DensityOfStates
from pyprocar.core.property_store import Property
from tests.utils import DATA_DIR


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_random_dos(
    rng: np.random.Generator,
    n_energies: int = 32,
    n_spins: int = 2,
    n_atoms: int = 3,
    n_orbitals: int = 4,
) -> DensityOfStates:
    energies = np.linspace(-5.0, 5.0, n_energies, dtype=float)
    total = rng.random((n_energies, n_spins)) + 0.5
    projected = rng.random((n_energies, n_spins, n_atoms, n_orbitals))
    return DensityOfStates(
        energies=energies,
        total=total,
        projected=projected,
        orbital_names=["s", "p", "d", "f"][:n_orbitals],
    )


@pytest.fixture
def dos_non_spin_polarized(rng):
    return _make_random_dos(rng, n_spins=1)


@pytest.fixture
def dos_spin_polarized(rng):
    return _make_random_dos(rng, n_spins=2)


@pytest.fixture
def dos_non_collinear(rng):
    return _make_random_dos(rng, n_spins=4)


@pytest.fixture
def non_spin_polarized_dir():
    return DATA_DIR / "examples" / "dos" / "non-spin-polarized"


def test_dos_from_code(non_spin_polarized_dir):
    dos = DensityOfStates.from_code(code="vasp", dirpath=non_spin_polarized_dir)
    assert dos.energies.shape[0] > 0
    assert dos.total.shape[0] == dos.energies.shape[0]
    assert dos.projected is not None


def test_spin_metadata(dos_non_spin_polarized, dos_spin_polarized, dos_non_collinear):
    assert not dos_non_spin_polarized.is_spin_polarized
    assert dos_spin_polarized.is_spin_polarized
    assert dos_non_collinear.is_non_collinear
    assert dos_non_collinear.spin_projection_names == ["total", "x", "y", "z"]


def test_dos_sum_spin_polarized(dos_spin_polarized):
    atoms = [0, 2]
    orbitals = [1, 3]

    result = dos_spin_polarized.dos_sum(atoms=atoms, orbitals=orbitals, spins=[0])
    expected = np.sum(dos_spin_polarized.projected[..., orbitals], axis=-1)
    expected = np.sum(expected[..., atoms], axis=-1)
    manual = np.zeros_like(expected)
    manual[..., 0] = expected[..., 0]

    assert result.shape == expected.shape
    assert np.allclose(result, manual)


def test_dos_sum_non_collinear(dos_non_collinear):
    atoms = [0, 1]
    orbitals = [0, 2]
    spins = [1, 2, 3]

    summed = dos_non_collinear.dos_sum(atoms=atoms, orbitals=orbitals, spins=spins)
    manual = np.sum(dos_non_collinear.projected[..., orbitals], axis=-1)
    manual = np.sum(manual[..., atoms], axis=-1)
    manual = np.sum(manual[..., spins], axis=-1, keepdims=True)

    assert summed.shape == manual.shape == (dos_non_collinear.n_energies, 1)
    assert np.allclose(summed, manual)

    expanded = dos_non_collinear.dos_sum(
        atoms=atoms, orbitals=orbitals, spins=spins, sum_noncolinear=False
    )
    manual_expanded = np.sum(dos_non_collinear.projected[..., orbitals], axis=-1)
    manual_expanded = np.sum(manual_expanded[..., atoms], axis=-1)
    manual_expanded = manual_expanded[..., spins]

    assert expanded.shape == manual_expanded.shape == (dos_non_collinear.n_energies, len(spins))
    assert np.allclose(expanded, manual_expanded)


def test_compute_projected_sum_tuple(dos_spin_polarized):
    total_proj, projected = dos_spin_polarized.compute_projected_sum(
        atoms=[0], orbitals=[0], spins=[0]
    )
    assert total_proj.shape == dos_spin_polarized.total.shape
    assert projected.shape == dos_spin_polarized.total.shape


def test_projected_sum_property_matches_dos_sum(dos_spin_polarized):
    atoms = [0, 1]
    orbitals = [0, 2]
    spins = [0]

    property_value = dos_spin_polarized.get_property(
        "projected_sum", atoms=atoms, orbitals=orbitals, spins=spins
    ).value
    manual = dos_spin_polarized.dos_sum(atoms=atoms, orbitals=orbitals, spins=spins)

    assert np.allclose(property_value, manual)
    assert "projected_sum|atoms=0,1|orbitals=0,2|spins=0" in dos_spin_polarized.property_store


def test_projected_sum_property_caches_variants(dos_spin_polarized):
    initial_keys = set(dos_spin_polarized.property_store.keys())
    dos_spin_polarized.get_property("projected_sum", atoms=[0], orbitals=[0], spins=[0])
    after_first = set(dos_spin_polarized.property_store.keys())
    dos_spin_polarized.get_property("projected_sum", atoms=[1], orbitals=[1], spins=[0])
    after_second = set(dos_spin_polarized.property_store.keys())

    assert len(after_first - initial_keys) == 1
    assert len(after_second - after_first) == 1
    assert len(after_second - initial_keys) == 2


def test_projected_sum_total_property(dos_spin_polarized):
    total_property = dos_spin_polarized.get_property("projected_sum_total", spins=[0])
    manual_total = dos_spin_polarized.dos_sum()
    assert np.allclose(total_property.value, manual_total)


def test_build_parametric_dataset_matches_manual(dos_spin_polarized):
    dataset = dos_spin_polarized.build_parametric_dataset(atoms=[0], orbitals=[1], spins=[0])

    total_proj, projected = dos_spin_polarized.compute_projected_sum(
        atoms=[0], orbitals=[1], spins=[0]
    )

    manual_norm = np.divide(
        projected[:, 0],
        total_proj[:, 0],
        out=np.zeros_like(projected[:, 0]),
        where=total_proj[:, 0] != 0,
    )
    manual_norm = np.nan_to_num(manual_norm, nan=0.0, posinf=0.0, neginf=0.0)

    assert dataset.totals.shape == (1, dos_spin_polarized.n_energies)
    assert dataset.projected.shape == (1, dos_spin_polarized.n_energies)
    assert dataset.total_projected.shape == (1, dos_spin_polarized.n_energies)
    assert np.allclose(dataset.totals[0], dos_spin_polarized.total[:, 0])
    assert np.allclose(dataset.normalized()[0], manual_norm)


def test_build_parametric_dataset_spin_selection(dos_spin_polarized):
    dataset = dos_spin_polarized.build_parametric_dataset(spins=[1])
    assert dataset.totals.shape == (1, dos_spin_polarized.n_energies)
    assert dataset.spin_indices == (1,)
    assert dataset.spin_labels == (dos_spin_polarized.spin_projection_names[1],)


def test_build_parametric_dataset_invalid_spin_raises(dos_spin_polarized):
    with pytest.raises(IndexError):
        dos_spin_polarized.build_parametric_dataset(spins=[5])



def test_normalized_total_property(dos_spin_polarized):
    normalized = dos_spin_polarized.get_property("normalized_total").value
    assert normalized.shape == dos_spin_polarized.total.shape
    assert np.allclose(np.max(np.abs(normalized), axis=0), np.ones(dos_spin_polarized.n_spin_channels))

    normalized_integral = dos_spin_polarized.get_property(
        "normalized_total", mode="integral"
    ).value
    integrals = np.trapezoid(normalized_integral, x=dos_spin_polarized.energies, axis=0)
    assert np.allclose(integrals, np.ones(dos_spin_polarized.n_spin_channels))


def test_cumulative_total_property(dos_spin_polarized):
    cumulative = dos_spin_polarized.get_property("cumulative_total").value
    assert cumulative.shape == dos_spin_polarized.total.shape
    diff = np.diff(cumulative, axis=0)
    assert np.all(diff >= -1e-12)


def test_normalize_dos_creates_new_instance(dos_spin_polarized):
    normalized = dos_spin_polarized.normalize_dos(mode="max")
    assert normalized is not dos_spin_polarized
    assert np.allclose(
        np.max(np.abs(normalized.total), axis=0),
        np.ones(dos_spin_polarized.n_spin_channels),
    )
    if normalized.projected is not None:
        assert normalized.projected.shape == dos_spin_polarized.projected.shape


def test_compute_gradients(dos_spin_polarized):
    dos_spin_polarized.compute_gradients(gradient_order=2, names=["total"])
    grad = dos_spin_polarized.get_property(("total", "gradients", 1))
    hess = dos_spin_polarized.get_property(("total", "gradients", 2))
    assert grad.shape == dos_spin_polarized.total.shape
    assert hess.shape == dos_spin_polarized.total.shape


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

def test_compute_gradients_on_projected(dos_spin_polarized):
    dos_spin_polarized.compute_gradients(gradient_order=2, names=["projected"])
    grad = dos_spin_polarized.get_property(("projected", "gradients", 1))
    hess = dos_spin_polarized.get_property(("projected", "gradients", 2))
    assert grad.shape == dos_spin_polarized.projected.shape
    assert hess.shape == dos_spin_polarized.projected.shape
