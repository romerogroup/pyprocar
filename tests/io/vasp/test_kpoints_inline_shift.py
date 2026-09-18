"""Regression tests for VASP KPOINTS files with grid and shift on one line."""

from pyprocar.io import vasp


def _write_kpoints(tmp_path, content):
    filepath = tmp_path / "KPOINTS"
    filepath.write_text(content)
    return filepath


def test_kpoints_grid_and_shift_on_same_line(tmp_path):
    filepath = _write_kpoints(
        tmp_path, "Automatic mesh\n0\nGamma\n60 60 1 0 0 0\n"
    )

    kpoints = vasp.Kpoints(filepath)

    assert kpoints.mode == "gamma"
    assert kpoints.automatic is True
    assert kpoints.kgrid == [60, 60, 1]
    assert kpoints.kshift == [0, 0, 0]


def test_kpoints_grid_and_shift_on_separate_lines(tmp_path):
    filepath = _write_kpoints(
        tmp_path, "Automatic mesh\n0\nMonkhorst-Pack\n4 4 2\n0 0 0\n"
    )

    kpoints = vasp.Kpoints(filepath)

    assert kpoints.mode == "monkhorst-pack"
    assert kpoints.kgrid == [4, 4, 2]
    assert kpoints.kshift == [0, 0, 0]
