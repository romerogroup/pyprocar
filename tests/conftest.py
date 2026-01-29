import pytest
import vtk



collect_ignore_glob = [
    "**/test_bandstructure2d.py",
    "**/test_fermisurface.py",
    "**/test_procarunfold.py"
]


def pytest_configure(_config: pytest.Config) -> None:
    vtk.vtkObject.GlobalWarningDisplayOff()
