"""PROCARs written by pyprocar.filter load and plot through the one-call functions.

Expected values come from the unfiltered PROCAR of the same run: a row of an
atom-filtered PROCAR is the sum of the rows of its group of atoms, and a column
of an orbital-filtered PROCAR is the sum of its group of orbital columns.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

import pyprocar
from pyprocar.core import ElectronicBandStructure
from pyprocar.io import get_parser
from tests.utils import DATA_DIR, writable_copy

pytestmark = pytest.mark.data

CALC = DATA_DIR / "examples/bands/non-spin-polarized"
FERMI2D_CALC = DATA_DIR / "examples/fermi2d/non-spin-polarized"
FERMI = 5.3017


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _copy_calc(tmp_path: Path, src: Path = CALC, extra: tuple[str, ...] = ()) -> Path:
    wanted = {"PROCAR", "OUTCAR", "POSCAR", "KPOINTS", *extra}
    return writable_copy(src, tmp_path / "calc", ignore=lambda _, names: set(names) - wanted)


def _filtered_calc(
    tmp_path: Path, src: Path = CALC, extra: tuple[str, ...] = (), **selection
) -> Path:
    calc = _copy_calc(tmp_path, src, extra)
    pyprocar.filter(str(calc / "PROCAR"), str(calc / "PROCAR-filtered"), **selection)
    (calc / "PROCAR-filtered").replace(calc / "PROCAR")
    return calc


def _ebs(calc: Path) -> ElectronicBandStructure:
    ebs = get_parser("vasp", calc).ebs
    assert ebs is not None
    return ebs


@pytest.fixture(scope="module")
def unfiltered() -> np.ndarray:
    projected = _ebs(CALC).projected
    assert projected is not None
    return np.asarray(projected.value)


def _sum(ebs: ElectronicBandStructure, **selection) -> np.ndarray:
    prop = ebs.compute_projected_sum(**selection)
    return np.asarray(prop.value)


def test_atom_filtered_procar_plots_each_group_of_atoms(
    tmp_path: Path, unfiltered: np.ndarray
) -> None:
    calc = _filtered_calc(tmp_path, atoms=[[0], [1], [2, 3, 4]])

    _, ax = pyprocar.bandsplot(
        code="vasp", dirname=str(calc), mode="plain", fermi=FERMI, show=False
    )
    assert len([line for line in ax.lines if len(line.get_xdata()) == 200]) == 20

    _, ax = pyprocar.bandsplot(
        code="vasp", dirname=str(calc), mode="parametric", fermi=FERMI, atoms=[2], show=False
    )
    assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 20

    ebs = _ebs(calc)
    np.testing.assert_allclose(
        _sum(ebs, atoms=[2])[..., 0], unfiltered[..., 0, 2:5, :].sum(axis=(-1, -2)), atol=1e-9
    )
    np.testing.assert_allclose(
        _sum(ebs, orbitals=[4, 5, 6, 7, 8])[..., 0],
        unfiltered[..., 0, :, 4:9].sum(axis=(-1, -2)),
        atol=1e-9,
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
        pyprocar.bandsplot(code="vasp", dirname=str(calc), mode="overlay_species", show=False)
    oxygen: Any = ["O"]
    with pytest.raises(ValueError, match="filtered by atoms"):
        pyprocar.bandsplot(
            code="vasp", dirname=str(calc), mode="parametric", atoms=oxygen, show=False
        )


def test_orbital_filtered_procar_labels_each_column_by_its_header_name(
    tmp_path: Path, unfiltered: np.ndarray
) -> None:
    calc = _filtered_calc(tmp_path, orbitals=[[0], [1, 2, 3]])

    _, ax = pyprocar.bandsplot(
        code="vasp", dirname=str(calc), mode="parametric", atoms=[1], orbitals=[1], show=False
    )
    assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 20

    ebs = _ebs(calc)
    s_column = ebs.compute_projected_sum(atoms=[1], orbitals=[0])
    p_column = ebs.compute_projected_sum(atoms=[1], orbitals=[1])

    assert s_column.metadata["orbital_label"] == "o0"
    assert p_column.metadata["orbital_label"] == "o1"
    np.testing.assert_allclose(
        np.asarray(p_column.value)[..., 0], unfiltered[..., 0, 1, 1:4].sum(axis=-1), atol=1e-9
    )


@pytest.mark.parametrize("header_name", ["x2-y2", "dx2"])
@pytest.mark.parametrize("selected", ["x2-y2", "dx2"])
def test_selecting_one_orbital_by_name_raises_whichever_name_the_header_uses(
    tmp_path: Path, header_name: str, selected: str
) -> None:
    calc = _copy_calc(tmp_path)
    procar = calc / "PROCAR"
    procar.write_text(procar.read_text().replace("x2-y2", header_name))
    by_name: Any = [selected]

    with pytest.raises(ValueError, match=r"orbitals takes orbital indices or the shell names"):
        pyprocar.bandsplot(
            code="vasp", dirname=str(calc), mode="parametric", orbitals=by_name, show=False
        )


def test_a_shell_name_selects_the_columns_its_header_names(tmp_path: Path) -> None:
    calc = _filtered_calc(tmp_path, orbitals=[[1], [2], [3]], orbital_names=["py", "pz", "px"])
    p_shell: Any = ["p"]

    def colors(orbitals) -> np.ndarray:
        _, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=str(calc),
            mode="parametric",
            atoms=[1],
            orbitals=orbitals,
            show=False,
        )
        return np.concatenate(
            [np.asarray(c.get_array()) for c in ax.collections if isinstance(c, LineCollection)]
        )

    np.testing.assert_array_equal(colors(p_shell), colors([0, 1, 2]))


def test_a_shell_name_selects_the_column_named_by_its_letter(tmp_path: Path) -> None:
    calc = _filtered_calc(
        tmp_path, orbitals=[[0], [1, 2, 3], [4, 5, 6, 7, 8]], orbital_names=["s", "p", "d"]
    )
    d_shell: Any = ["d"]

    def colors(orbitals) -> np.ndarray:
        _, ax = pyprocar.bandsplot(
            code="vasp",
            dirname=str(calc),
            mode="parametric",
            atoms=[1],
            orbitals=orbitals,
            show=False,
        )
        return np.concatenate(
            [np.asarray(c.get_array()) for c in ax.collections if isinstance(c, LineCollection)]
        )

    np.testing.assert_array_equal(colors(d_shell), colors([2]))
    _, ax = pyprocar.bandsplot(
        code="vasp", dirname=str(calc), mode="overlay_orbitals", atoms=[1], show=False
    )
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["s", "p", "d"]


def test_a_shell_name_raises_when_the_header_names_no_such_shell(tmp_path: Path) -> None:
    calc = _filtered_calc(tmp_path, orbitals=[[0], [1, 2, 3]])
    p_shell: Any = ["p"]

    with pytest.raises(ValueError, match="hold no whole p shell"):
        pyprocar.bandsplot(
            code="vasp", dirname=str(calc), mode="parametric", orbitals=p_shell, show=False
        )


def test_part_of_a_shell_is_not_that_shell(tmp_path: Path) -> None:
    calc = _filtered_calc(tmp_path, orbitals=[[0], [1], [2]], orbital_names=["s", "py", "pz"])
    p_shell: Any = ["p"]

    with pytest.raises(ValueError, match="hold no whole p shell"):
        pyprocar.bandsplot(
            code="vasp", dirname=str(calc), mode="parametric", orbitals=p_shell, show=False
        )


def test_fermi2d_refuses_to_unfold_an_atom_filtered_irreducible_procar(tmp_path: Path) -> None:
    """A symmetry operation moves O atoms between rows the filtered file no longer names."""
    calc = _filtered_calc(tmp_path, FERMI2D_CALC, ("vasprun.xml", "IBZKPT"), atoms=[[2], [3, 4]])

    with pytest.raises(ValueError, match="filtered by atoms"):
        pyprocar.fermi2D(
            code="vasp", dirname=str(calc), mode="plain", fermi=FERMI, k_z_plane=0.0, show=False
        )
