"""FrmSrf parser adapter."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from pyprocar.core.ebs import ElectronicBandStructure, get_ebs_from_data
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from pyprocar.io.base import BaseParser
from pyprocar.io.frmsf.frmsf import Frmsf

if TYPE_CHECKING:
    from pyprocar.core.dos import DensityOfStates
    from pyprocar.core.kpoints import KPath
    from pyprocar.core.structure import Structure

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


class FrmsfParser(BaseParser):
    """Parser adapter for FrmSrf (FermiSurfer) files.

    Parameters
    ----------
    dirpath : str | Path
        Directory containing FrmSrf file.
    filepath : str | Path
        Path to .frmsf file relative to dirpath.
    """

    def __init__(
        self,
        dirpath: str | Path,
        filepath: str | Path | None = None,
    ) -> None:
        super().__init__(dirpath)
        self._frmsf: Frmsf | None = self._initialize_extractor(filepath if filepath is not None else Path("in.frmsf"))

    def _initialize_extractor(self, param: str | Path | Frmsf | None) -> Frmsf | None:
        """Initialize Frmsf extractor."""
        if isinstance(param, Frmsf):
            return param

        if param is None:
            return None

        filepath = self.dirpath / Path(param).name
        if filepath.exists():
            return Frmsf(filepath)

        user_logger.warning(f"FrmSrf file not found: {filepath}")
        return None

    @classmethod
    def from_str(cls, file_str: str) -> "FrmsfParser":
        """Create parser from file content string."""
        parser = cls.__new__(cls)
        parser.dirpath = Path("")
        parser._frmsf = Frmsf.from_str(file_str)
        return parser

    @cached_property
    def kgrid_info(self) -> KGridInfo | None:
        """K-grid information."""
        if self._frmsf is None:
            return None

        # Map FrmSrf generation methods to kgrid modes
        method = self._frmsf.kpoint_generation_method
        mode_map = {0: KGRID_MODE.MONKHORST, 1: KGRID_MODE.GAMMA, 2: KGRID_MODE.GAMMA}
        kgrid_mode = mode_map.get(method, KGRID_MODE.GAMMA)

        return KGridInfo(
            kgrid=self._frmsf.nk_dim,
            kgrid_mode=kgrid_mode,
            kshift=(0.0, 0.0, 0.0),
        )

    @property
    @override
    def ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure (mesh-based)."""
        if self._frmsf is None:
            user_logger.warning("No FrmSrf extractor available")
            return None

        try:
            # Add spin dimension if not present
            bands = self._frmsf.bands
            if bands.ndim == 2:
                bands = bands[..., np.newaxis]  # Add spin dimension

            return get_ebs_from_data(
                kpoints=self._frmsf.kpoints,
                bands=bands,
                projected=None,  # FrmSrf projections need format investigation
                fermi=0.0,  # FrmSrf doesn't provide Fermi energy
                reciprocal_lattice=self._frmsf.reciprocal_lattice,
                kgrid_info=self.kgrid_info,
            )
        except Exception as e:
            user_logger.warning(f"Error creating EBS from FrmSrf: {e}")
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
