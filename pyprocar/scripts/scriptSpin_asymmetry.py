#!/usr/bin/env python3
"""
Created on Tue Aug 18 11:14:17 2020

@author: petavazohi
"""

from __future__ import annotations

from typing import NoReturn

import numpy.typing as npt


def spin_asymmetry(
    _procar: str = "PROCAR",
    _outcar: str = "OUTCAR",
    _fermi: float | None = None,
    _bands: list[int] | None = None,
    _interpolation_factor: int = 1,
    _mode: str = "plain",
    _supercell: list[int] | None = None,
    _colors: npt.ArrayLike | None = None,
    _background_color: str = "white",
    _save_colors: str | None = None,
    _cmap: str = "viridis",
    _atoms: list[int] | None = None,
    _orbitals: list[int] | None = None,
    _spin: list[int] | None = None,
    _spin_texture: bool = False,
    _arrow_color: str | None = None,
    _arrow_size: float = 0.015,
    _only_spin: bool = False,
    _fermi_shift: float = 0,
    _projection_accuracy: str = "normal",
    _code: str = "vasp",
    _vmin: float = 0,
    _vmax: float = 1,
    _savegif: str | None = None,
    _savemp4: str | None = None,
    _save3d: str | None = None,
    _perspective: bool = True,
    _save2d: bool = False,
    _camera_pos: list[float] | None = None,
    _widget: object = None,
    _show: bool = True,
) -> NoReturn:
    raise NotImplementedError(
        "The spin_asymmetry function has been deprecated. "
        "The underlying FermiSurface3D and boolean_add classes have been removed. "
        "Please use the FermiSurface class with the FermiHandler for Fermi surface "
        "visualization instead."
    )
