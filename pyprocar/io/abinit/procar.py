"""Abinit PROCAR file parser with parallel merge support."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyprocar.io.vasp import Procar

if TYPE_CHECKING:
    from pyprocar.io.abinit.output import AbinitOutput

logger = logging.getLogger(__name__)


class AbinitProcar(Mapping[str, Any]):
    """Parse PROCAR files from Abinit with parallel merge support.
    
    Abinit generates separate PROCAR_* files in parallel runs that
    need to be merged before parsing. This class handles the merge
    process and then uses the VASP Procar parser.
    """

    def __init__(
        self,
        dirpath: str | Path | None = None,
        infilepaths: list[Path] | None = None,
        abinit_output: "AbinitOutput | None" = None,
        nspin: int | None = None,
    ):
        """Initialize AbinitProcar parser.
        
        Parameters
        ----------
        dirpath : str | Path | None
            Directory containing PROCAR files
        infilepaths : list[Path] | None
            Explicit list of PROCAR_* files to merge (optional)
        abinit_output : AbinitOutput | None
            AbinitOutput instance for nspin info (optional)
        nspin : int | None
            Number of spin channels (1 or 2), used if abinit_output not provided
        """
        self._dirpath = Path(dirpath) if dirpath else None
        self._infilepaths = infilepaths
        self._nspin = nspin
        
        # Get nspin from abinit_output if provided
        if abinit_output is not None:
            self._nspin = abinit_output.nspin

        # Auto-detect PROCAR files if not provided
        if self._infilepaths is None and self._dirpath is not None:
            self._infilepaths = sorted(self._dirpath.glob("PROCAR_*"))

        self._vasp_procar: Procar | None = None
        if self._infilepaths:
            self._vasp_procar = Procar.from_str(self._merge_parallel(self._infilepaths))
        elif self._dirpath is not None and (self._dirpath / "PROCAR").exists():
            self._vasp_procar = Procar(filepath=self._dirpath / "PROCAR")

    @property
    def vasp_procar(self) -> Procar | None:
        """Access to the underlying VASP Procar parser after merge."""
        return self._vasp_procar

    def _merge_parallel(self, infilepaths: list[Path]) -> str:
        """Merge PROCAR files from parallel Abinit runs."""
        if self._nspin is None:
            raise ValueError("nspin must be provided for merging PROCAR files")

        filepaths = sorted(infilepaths)
        logger.info(f"Merging {len(filepaths)} parallel PROCAR files...")

        if self._nspin != 2:
            return "".join(fp.read_text() for fp in filepaths)

        # Spin-polarized: first half is spin-up, second half (reversed) is spin-down
        spinup_filepaths = filepaths[: len(filepaths) // 2]
        spindown_filepaths = filepaths[len(filepaths) // 2 :][::-1]
        header2 = spinup_filepaths[0].read_text().splitlines(keepends=True)[1]
        return (
            "".join(fp.read_text() for fp in spinup_filepaths)
            + "\n"
            + header2
            + "\n"
            + "".join(fp.read_text() for fp in spindown_filepaths)
        )

    # Mapping protocol implementation
    def __contains__(self, key: object) -> bool:
        return key in self.__dict__

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return self.__dict__.__len__()
