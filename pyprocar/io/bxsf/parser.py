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
from pyprocar.utils.log_utils import warn_user

logger = logging.getLogger(__name__)


class BxsfParser(BaseParser):
    """Parser adapter for BXSF files.

    Parameters
    ----------
    dirpath : str | Path
        Directory containing BXSF file(s).
    filepaths : str | Path | list[Path] | None
        One BXSF file, or the QE ``fs.x`` spin pair ``<prefix>_fsup.bxsf`` and
        ``<prefix>_fsdw.bxsf``. ``None`` reads ``in.bxsf``, else that pair, else the first
        ``*.bxsf`` or ABINIT ``*_BXSF`` file in ``dirpath``.
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
                warn_user(f"BXSF file not found: {full_path}")
        _check_spin_files(self._extractors)

    def _find_bxsf_files(self) -> list[Path]:
        if (self.dirpath / "in.bxsf").exists():
            return [Path("in.bxsf")]
        found = sorted(
            Path(p.name)
            for p in self.dirpath.glob("*")
            if p.is_file() and p.name.lower().endswith((".bxsf", "_bxsf"))
        )
        chosen = _qe_fs_spin_pair(found) or found[:1]
        if not found:
            warn_user(f"No in.bxsf, *.bxsf or *_BXSF file found in {self.dirpath}")
        elif len(found) > len(chosen):
            warn_user(
                f"Found several BXSF files in {self.dirpath}: {[str(p) for p in found]}; "
                + f"reading {[str(p) for p in chosen]}. Pass filepaths to choose another."
            )
        return chosen

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
                    warn_user(
                        "QE fs.x BXSF stores b in units of 2*pi/alat and no QE output with "
                        + "alat was found beside it; b is left in units of 1/alat."
                    )
                    return b
                return b / alat
            case BxsfWriter.UNKNOWN:
                warn_user(
                    "BXSF writer not recognised; assuming b includes the 2*pi in 1/Angstrom "
                    + "(the XCrySDen and Wannier90 convention)."
                )
                return b / (2 * np.pi)

    @property
    def ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure (mesh-based)."""
        if not self._extractors:
            warn_user("No BXSF extractors available")
            return None

        try:
            ext = self._extractors[0]
            return get_ebs_from_data(
                kpoints=ext.kpoints,
                bands=self._bands(),
                projected=None,
                fermi=ext.fermi_energy,
                reciprocal_lattice=self.reciprocal_lattice,
                kgrid_info=self.kgrid_info,
            )
        except Exception as e:
            warn_user(f"Error creating EBS from BXSF: {e}")
            return None

    def _bands(self) -> np.ndarray:
        if len(self._extractors) == 1:
            return self._extractors[0].bands
        up, down = self._extractors
        return _stack_spin_pair(up, down)


def _stack_spin_pair(up: Bxsf, down: Bxsf) -> np.ndarray:
    """Stack the fs.x spin files on the union of their BAND labels.

    fs.x keeps, per spin, only the bands within deltaE of the Fermi level. A band
    missing from one spin lies wholly below or above that window there, so it is
    filled with a constant on that side, which adds no Fermi surface.
    """
    labels = sorted(set(up.band_labels) | set(down.band_labels))
    fermi = up.fermi_energy
    present = np.concatenate([up.bands, down.bands], axis=1)
    margin = float(present.max() - present.min()) + 1.0
    channels = []
    for spin in (up, down):
        columns = []
        for label in labels:
            if label in spin.band_labels:
                columns.append(spin.bands[:, spin.band_labels.index(label), 0])
            else:
                side = -1.0 if label < min(spin.band_labels) else 1.0
                columns.append(np.full(spin.bands.shape[0], fermi + side * margin))
        channels.append(np.stack(columns, axis=1))
    return np.stack(channels, axis=2)


def _check_spin_files(extractors: list[Bxsf]) -> None:
    paths = [ext.filepath for ext in extractors if ext.filepath is not None]
    if not extractors:
        return
    if len(extractors) == 1:
        name = paths[0].name if paths else ""
        if extractors[0].writer is BxsfWriter.QE_FS and name.endswith(("up.bxsf", "dw.bxsf")):
            warn_user(
                f"{name} holds one spin of a QE fs.x spin-polarized run, and its partner file "
                + "is missing; reading it as a single spin channel."
            )
        return
    is_pair = (
        len(extractors) == 2
        and all(ext.writer is BxsfWriter.QE_FS for ext in extractors)
        and extractors[0].nk_dim == extractors[1].nk_dim
        and _qe_fs_spin_pair(paths) == paths
    )
    if not is_pair:
        raise ValueError(
            f"BXSF files {[p.name for p in paths]} do not form a QE fs.x spin pair. Pass one "
            + "file, or the <prefix>_fsup.bxsf and <prefix>_fsdw.bxsf files of one fs.x run, "
            + "in that order."
        )


def _qe_fs_spin_pair(found: list[Path]) -> list[Path] | None:
    names = {p.name for p in found}
    for p in found:
        if p.name.endswith("up.bxsf") and p.name[: -len("up.bxsf")] + "dw.bxsf" in names:
            return [p, p.with_name(p.name[: -len("up.bxsf")] + "dw.bxsf")]
    return None


def _pw_alat_angstrom_beside(filepath: Path) -> float | None:
    for candidate in sorted(filepath.parent.iterdir()):
        if candidate.suffix.lower() in {".out", ".log"} and PwOut.is_file_of_type(candidate):
            alat = PwOut(candidate).alat
            if alat is not None:
                return alat * AU_TO_ANG
    return None
