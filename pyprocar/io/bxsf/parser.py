"""BXSF parser adapter."""

import logging
from functools import cached_property
from pathlib import Path

import numpy as np

from pyprocar.core.ebs import ElectronicBandStructure, get_ebs_from_data
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from pyprocar.io.base import BaseParser
from pyprocar.io.bxsf.bxsf import Bxsf, BxsfWriter
from pyprocar.io.qe.pw import PwOut
from pyprocar.utils.units import AU_TO_ANG

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


class BxsfParser(BaseParser):
    """Parser adapter for BXSF files.

    Parameters
    ----------
    dirpath : str | Path
        Directory containing BXSF file(s).
    filepaths : str | Path | list[Path] | None
        Path(s) to .bxsf file(s). Multiple files for spin-polarized data. ``None`` reads
        ``in.bxsf``, or else the first ``*.bxsf`` or ABINIT ``*_BXSF`` file in ``dirpath``.
    """

    def __init__(
        self,
        dirpath: str | Path,
        filepaths: str | Path | list[Path] | None = None,
    ):
        super().__init__(dirpath)
        self._filepaths = (
            self._find_bxsf_files() if filepaths is None else self._normalize_filepaths(filepaths)
        )
        self._extractors: list[Bxsf] = []

        for filepath in self._filepaths:
            full_path = self.dirpath / filepath.name
            if full_path.exists():
                self._extractors.append(Bxsf(full_path))
            else:
                user_logger.warning(f"BXSF file not found: {full_path}")

    def _find_bxsf_files(self) -> list[Path]:
        if (self.dirpath / "in.bxsf").exists():
            return [Path("in.bxsf")]
        found = sorted(
            Path(p.name)
            for p in self.dirpath.glob("*")
            if p.is_file() and p.name.lower().endswith((".bxsf", "_bxsf"))
        )
        if not found:
            user_logger.warning(f"No in.bxsf, *.bxsf or *_BXSF file found in {self.dirpath}")
        elif len(found) > 1:
            user_logger.warning(
                f"Found several BXSF files in {self.dirpath}: {[str(p) for p in found]}; "
                + f"reading {found[0]}. Pass filepaths to choose another."
            )
        return found[:1]

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

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray | None:
        """Reciprocal lattice in 1/Angstrom without the 2*pi factor.

        The unit in the file depends on its writer (see ``BxsfWriter``). A QE ``fs.x`` file
        carries no alat, so it is read from the QE output beside the file.
        """
        if not self._extractors:
            return None
        ext = self._extractors[0]
        b = ext.reciprocal_lattice
        match ext.writer:
            case BxsfWriter.WANNIER90:
                return b / (2 * np.pi)
            case BxsfWriter.ABINIT:
                return b
            case BxsfWriter.QE_FS:
                alat = _pw_alat_angstrom_beside(ext.filepath) if ext.filepath else None
                if alat is None:
                    user_logger.warning(
                        "QE fs.x BXSF stores b in units of 2*pi/alat and no QE output with "
                        + "alat was found beside it; b is left in units of 1/alat."
                    )
                    return b
                return b / alat
            case BxsfWriter.UNKNOWN:
                user_logger.warning(
                    "BXSF writer not recognised; assuming b includes the 2*pi in 1/Angstrom "
                    + "(the XCrySDen and Wannier90 convention)."
                )
                return b / (2 * np.pi)

    @property
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
                reciprocal_lattice=self.reciprocal_lattice,
                kgrid_info=self.kgrid_info,
            )
        except Exception as e:
            user_logger.warning(f"Error creating EBS from BXSF: {e}")
            return None


def _pw_alat_angstrom_beside(filepath: Path) -> float | None:
    for candidate in sorted(filepath.parent.iterdir()):
        if candidate.suffix.lower() in {".out", ".log"} and PwOut.is_file_of_type(candidate):
            alat = PwOut(candidate).alat
            if alat is not None:
                return alat * AU_TO_ANG
    return None
