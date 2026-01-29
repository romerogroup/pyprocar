import pytest
import vtk

collect_ignore_glob = [
    "**/test_bandstructure2d.py",
    "**/test_fermisurface.py",
    "**/test_procarunfold.py",
]


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    vtk.vtkObject.GlobalWarningDisplayOff()
