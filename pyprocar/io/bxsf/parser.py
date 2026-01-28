"""BXSF parser adapter."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

from pyprocar.core.ebs import ElectronicBandStructure, get_ebs_from_data
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from pyprocar.io.base import BaseParser
from pyprocar.io.bxsf.bxsf import Bxsf

if TYPE_CHECKING:
    from pyprocar.core.dos import DensityOfStates
    from pyprocar.core.kpoints import KPath
    from pyprocar.core.structure import Structure

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


class BxsfParser(BaseParser):
    _filepaths: list[Path]
    """Parser adapter for BXSF files.

    Parameters
    ----------
    dirpath : str | Path
        Directory containing BXSF file(s).
    filepaths : str | Path | list[Path]
        Path(s) to .bxsf file(s). Multiple files for spin-polarized data.
    """

    def __init__(
        self,
        dirpath: str | Path,
        filepaths: str | Path | list[Path] | None = None,
    ) -> None:
        super().__init__(dirpath)
        self._filepaths = self._normalize_filepaths(filepaths if filepaths is not None else Path("in.bxsf"))
        self._extractors: list[Bxsf] = []

        for filepath in self._filepaths:
            full_path = self.dirpath / filepath.name
            if full_path.exists():
                self._extractors.append(Bxsf(full_path))
            else:
                user_logger.warning(f"BXSF file not found: {full_path}")

    def _normalize_filepaths(self, filepaths: str | Path | list[Path]) -> list[Path]:
        """Normalize filepaths to list of Path objects."""
        if isinstance(filepaths, (str, Path)):
            return [Path(filepaths)]
        return [Path(p) for p in filepaths]

    @classmethod
    def from_str(cls, *file_strs: str) -> "BxsfParser":
        """Create parser from file content strings."""
        parser = cls.__new__(cls)
        parser.dirpath = Path("")
        parser._filepaths = []
        parser._extractors = [Bxsf.from_str(s) for s in file_strs]
        return parser

    @cached_property
    def kgrid_info(self) -> KGridInfo | None:
        """K-grid information for mesh-based EBS."""
        if not self._extractors:
            return None
        ext = self._extractors[0]
        return KGridInfo(
            kgrid=ext.nk_dim,
            kgrid_mode=KGRID_MODE.GAMMA,  # BXSF uses gamma-centered grids
            kshift=(0.0, 0.0, 0.0),
        )

    @property
    @override
    def ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure (mesh-based)."""
        if not self._extractors:
            user_logger.warning("No BXSF extractors available")
            return None

        try:
            ext = self._extractors[0]
            return get_ebs_from_data(
                kpoints=ext.kpoints,
                bands=ext.bands,
                projected=None,
                fermi=ext.fermi_energy,
                reciprocal_lattice=ext.reciprocal_lattice,
                kgrid_info=self.kgrid_info,
            )
        except Exception as e:
            user_logger.warning(f"Error creating EBS from BXSF: {e}")
            return None

    @property
    @override
    def kpath(self) -> KPath | None:
        return None

    @property
    @override
    def structure(self) -> Structure | None:
        return None

    @property
    @override
    def dos(self) -> DensityOfStates | None:
        return None
