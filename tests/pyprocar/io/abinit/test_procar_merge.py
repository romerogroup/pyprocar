import pytest

from pyprocar.io.abinit import AbinitProcar


@pytest.fixture
def parallel_dir(tmp_path):
    (tmp_path / "PROCAR_0000").write_text("head A\nhead B\nup\n")
    (tmp_path / "PROCAR_0001").write_text("head A\nhead B\ndown\n")
    return tmp_path


def test_merge_reads_parallel_files_without_writing(parallel_dir):
    procar = AbinitProcar(dirpath=parallel_dir, nspin=1).vasp_procar

    assert procar is not None
    assert procar.file_str == "head A\nhead B\nup\nhead A\nhead B\ndown\n"
    assert sorted(p.name for p in parallel_dir.iterdir()) == [
        "PROCAR_0000",
        "PROCAR_0001",
    ]


def test_spin_polarized_merge_appends_spin_down_block(parallel_dir):
    procar = AbinitProcar(dirpath=parallel_dir, nspin=2).vasp_procar

    assert procar is not None
    assert procar.file_str == "head A\nhead B\nup\n\nhead B\n\nhead A\nhead B\ndown\n"
    assert sorted(p.name for p in parallel_dir.iterdir()) == [
        "PROCAR_0000",
        "PROCAR_0001",
    ]
