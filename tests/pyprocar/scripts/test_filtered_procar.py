"""PROCARs written by pyprocar.filter load and plot through the one-call functions.

Expected values come from the unfiltered PROCAR of the same run: a row of an
atom-filtered PROCAR is the sum of the rows of its group of atoms, and a column
of an orbital-filtered PROCAR is the sum of its group of orbital columns.
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

import pyprocar
from pyprocar.core import ElectronicBandStructure
from pyprocar.io import get_parser
from tests.utils import DATA_DIR

pytestmark = pytest.mark.data

CALC = DATA_DIR / "examples/bands/non-spin-polarized"
FERMI = 5.3017


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _copy_calc(tmp_path: Path) -> Path:
    dst = tmp_path / "calc"
    shutil.copytree(CALC, dst, ignore=shutil.ignore_patterns("*.pkl", "CHG*", "WAVECAR", "*.h5"))
    return dst


def _filtered_calc(tmp_path: Path, **selection) -> Path:
    calc = _copy_calc(tmp_path)
    pyprocar.filter(str(calc / "PROCAR"), str(calc / "PROCAR-filtered"), **selection)
    (calc / "PROCAR-filtered").replace(calc / "PROCAR")
    return calc


def _ebs(calc: Path) -> ElectronicBandStructure:
    ebs = get_parser("vasp", calc).ebs
    assert ebs is not None
    return ebs


def _projected(ebs: ElectronicBandStructure) -> np.ndarray:
    projected = ebs.projected
    assert projected is not None
    return np.asarray(projected.value)


def _sum(ebs: ElectronicBandStructure, **selection) -> np.ndarray:
    prop = ebs.compute_projected_sum(**selection)
    return np.asarray(prop.value)


def test_atom_filtered_procar_plots_each_group_of_atoms(tmp_path: Path) -> None:
    calc = _filtered_calc(tmp_path, atoms=[[0], [1], [2, 3, 4]])
    unfiltered = _projected(_ebs(CALC))

    _, ax = pyprocar.bandsplot(code="vasp", dirname=calc, mode="plain", fermi=FERMI, show=False)
    assert len([line for line in ax.lines if len(line.get_xdata()) == 200]) == 20

    _, ax = pyprocar.bandsplot(
        code="vasp", dirname=calc, mode="parametric", fermi=FERMI, atoms=[2], show=False
    )
    assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 20

    ebs = _ebs(calc)
    np.testing.assert_allclose(
        _sum(ebs, atoms=[2])[..., 0], unfiltered[..., 0, 2:5, :].sum(axis=(-1, -2)), atol=2e-3
    )
    np.testing.assert_allclose(
        _sum(ebs, orbitals=[4, 5, 6, 7, 8])[..., 0],
        unfiltered[..., 0, :, 4:9].sum(axis=(-1, -2)),
        atol=5e-3,
    )


def test_atom_filtered_procar_labels_a_group_by_its_row_not_a_poscar_species(
    tmp_path: Path,
) -> None:
    ebs = _ebs(_filtered_calc(tmp_path, atoms=[[0, 1], [2, 3, 4]]))

    prop = ebs.compute_projected_sum(atoms=[0], orbitals=[0])

    assert prop.metadata["atom_label"] == "0"
    assert prop.metadata["label_plain"] == ["0-(s)"]


def test_atom_filtered_procar_refuses_species_selections(tmp_path: Path) -> None:
    calc = _filtered_calc(tmp_path, atoms=[[0, 1], [2, 3, 4]])

    with pytest.raises(ValueError, match="filtered by atoms"):
        pyprocar.bandsplot(code="vasp", dirname=calc, mode="overlay_species", show=False)
    with pytest.raises(ValueError, match="filtered by atoms"):
        pyprocar.bandsplot(code="vasp", dirname=calc, mode="parametric", atoms=["O"], show=False)


def test_orbital_filtered_procar_labels_each_column_by_its_header_name(tmp_path: Path) -> None:
    calc = _filtered_calc(tmp_path, orbitals=[[0], [1, 2, 3]])
    unfiltered = _projected(_ebs(CALC))

    _, ax = pyprocar.bandsplot(
        code="vasp", dirname=calc, mode="parametric", atoms=[1], orbitals=[1], show=False
    )
    assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 20

    ebs = _ebs(calc)
    s_column = ebs.compute_projected_sum(atoms=[1], orbitals=[0])
    p_column = ebs.compute_projected_sum(atoms=[1], orbitals=[1])

    assert s_column.metadata["orbital_label"] == "o0"
    assert p_column.metadata["orbital_label"] == "o1"
    np.testing.assert_allclose(
        np.asarray(p_column.value)[..., 0], unfiltered[..., 0, 1, 1:4].sum(axis=-1), atol=2e-3
    )


@pytest.mark.parametrize("header_name", ["x2-y2", "dx2"])
@pytest.mark.parametrize("selected", ["x2-y2", "dx2"])
def test_selecting_one_orbital_by_name_raises_whichever_name_the_header_uses(
    tmp_path: Path, header_name: str, selected: str
) -> None:
    calc = _copy_calc(tmp_path)
    procar = calc / "PROCAR"
    procar.write_text(procar.read_text().replace("x2-y2", header_name))

    with pytest.raises(ValueError, match=r"orbitals takes orbital indices or the shell names"):
        pyprocar.bandsplot(
            code="vasp", dirname=calc, mode="parametric", orbitals=[selected], show=False
        )
