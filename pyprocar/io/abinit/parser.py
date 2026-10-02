"""Abinit parser orchestrator with file auto-detection."""

import logging
import re
from functools import cached_property
from pathlib import Path
from typing import Any

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure
from pyprocar.io.abinit.dos import AbinitDOS
from pyprocar.io.abinit.kpoints import AbinitKpoints
from pyprocar.io.abinit.output import AbinitOutput
from pyprocar.io.abinit.procar import AbinitProcar
from pyprocar.io.base import BaseParser

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


class AbinitParser(BaseParser):
    """Auto-detects Abinit files in a directory and exposes
    lazy parser properties and computed objects (EBS, DOS, Structure).
    
    Example
    -------
    parser = AbinitParser("/path/to/abinit/calculation")
    ebs = parser.ebs
    dos = parser.dos
    structure = parser.structure
    """

    def __init__(
        self,
        dirpath: str | Path,
        abinit_output: str | Path | AbinitOutput | None = None,
        kpoints: str | Path | AbinitKpoints | None = None,
        **kwargs,
    ) -> None:
        super().__init__(dirpath=dirpath, **kwargs)
        
        self._detected: dict[str, Path | list[Path] | None] = {
            "output": None,
            "kpoints": None,
            "procar": None,
            "procar_parallel": [],
            "dos_total": None,
            "dos_atoms": [],
        }
        
        # Detect files automatically
        self.detect_files()
        
        # Override with explicit arguments if provided
        if abinit_output is not None:
            if isinstance(abinit_output, AbinitOutput):
                self._abinit_output = abinit_output
            else:
                self._detected["output"] = self.dirpath / Path(abinit_output).name
        
        if kpoints is not None:
            if isinstance(kpoints, AbinitKpoints):
                self._abinit_kpoints = kpoints
            else:
                self._detected["kpoints"] = self.dirpath / Path(kpoints).name

    def detect_files(self) -> None:
        """Auto-detect Abinit files in the directory."""
        if not self.dirpath.exists():
            user_logger.warning(f"Directory not found: {self.dirpath}")
            return

        files = list(self.dirpath.rglob("*"))
        
        # 1. Detect output files by content
        for f in files:
            if f.is_file() and (f.suffix == ".out" or f.name == "abinit.out"):
                if AbinitOutput.is_file_of_type(f):
                    self._detected["output"] = f
                    break

        # 2. Detect KPOINTS by filename
        kpoints_files = [f for f in files if f.is_file() and f.name == "KPOINTS"]
        if kpoints_files:
            self._detected["kpoints"] = kpoints_files[0]

        # 3. Detect PROCAR files
        procar_merged = [f for f in files if f.is_file() and f.name == "PROCAR"]
        procar_parallel = sorted([
            f for f in files 
            if f.is_file() and re.match(r"PROCAR_\d+", f.name)
        ])
        
        if procar_merged:
            self._detected["procar"] = procar_merged[0]
        self._detected["procar_parallel"] = procar_parallel

        # 4. Detect DOS files by name pattern
        dos_total = [f for f in files if f.is_file() and "DOS_TOTAL" in f.name]
        dos_atoms = sorted([
            f for f in files 
            if f.is_file() and re.match(r".*DOS_AT\d+", f.name)
        ])
        
        if dos_total:
            self._detected["dos_total"] = dos_total[0]
        self._detected["dos_atoms"] = dos_atoms

        logger.info(f"Detected files in {self.dirpath}: {self._detected}")

    def summary(self) -> dict[str, Any]:
        """Return summary of detected files and parsers."""
        def _p(v):
            if v is None:
                return None
            if isinstance(v, list):
                return [str(x) for x in v]
            return str(v)

        return {
            "dirpath": str(self.dirpath),
            "files": {k: _p(v) for k, v in self._detected.items()},
            "parsers": {
                "output": self._detected["output"] is not None,
                "kpoints": self._detected["kpoints"] is not None,
                "procar": (
                    self._detected["procar"] is not None or 
                    len(self._detected.get("procar_parallel", [])) > 0
                ),
                "dos": (
                    self._detected["dos_total"] is not None or
                    len(self._detected.get("dos_atoms", [])) > 0
                ),
            },
        }

    # -------- lazy parser properties --------
    @cached_property
    def abinit_output(self) -> AbinitOutput | None:
        if hasattr(self, "_abinit_output"):
            return self._abinit_output
        fp = self._detected.get("output")
        if not fp:
            return None
        try:
            return AbinitOutput(fp)
        except Exception as exc:
            logger.warning(f"Error parsing output file: {exc}")
            return None

    @cached_property
    def abinit_kpoints(self) -> AbinitKpoints | None:
        if hasattr(self, "_abinit_kpoints"):
            return self._abinit_kpoints
        fp = self._detected.get("kpoints")
        if not fp:
            return None
        try:
            return AbinitKpoints(fp)
        except Exception as exc:
            logger.debug(f"Error parsing KPOINTS file: {exc}")
            return None

    @cached_property
    def abinit_procar(self) -> AbinitProcar | None:
        try:
            return AbinitProcar(
                dirpath=self.dirpath,
                abinit_output=self.abinit_output,
            )
        except Exception as exc:
            logger.warning(f"Error parsing PROCAR: {exc}")
            return None

    @cached_property
    def abinit_dos(self) -> AbinitDOS | None:
        if not self._detected["dos_total"]:
            return None
        try:
            return AbinitDOS(self.dirpath)
        except Exception as exc:
            logger.warning(f"Error parsing DOS: {exc}")
            return None

    # -------- computed properties --------
    @cached_property
    def version(self) -> str | None:
        if self.abinit_output:
            return self.abinit_output.version
        return None

    @cached_property
    def version_tuple(self) -> tuple[int, ...] | None:
        if self.version:
            return tuple(int(x) for x in self.version.split("."))
        return None

    @property
    def ebs(self) -> ElectronicBandStructure | None:
        if self.abinit_procar is None or self.abinit_procar.vasp_procar is None:
            user_logger.warning("Cannot create EBS: PROCAR not available")
            return None
        if self.abinit_output is None:
            user_logger.warning("Cannot create EBS: output file not available")
            return None

        procar = self.abinit_procar.vasp_procar
        projected_phase = None
        if hasattr(procar, 'spd_phase') and procar.spd_phase is not None:
            projected_phase = procar._spd2projected(procar.spd_phase)
        
        return ElectronicBandStructure(
            kpoints=procar.kpoints,
            bands=procar.bands,
            projected=procar._spd2projected(procar.spd),
            fermi=self.abinit_output.fermi,
            projected_phase=projected_phase,
            orbital_names=procar.orbital_names_old[:-1],
            reciprocal_lattice=self.abinit_output.reclat,
            structure=self.structure,
        )

    @property
    def dos(self) -> DensityOfStates | None:
        if self.abinit_dos is None:
            return None
        return DensityOfStates(
            energies=self.abinit_dos.energies,
            total=self.abinit_dos.dos_total,
            fermi=self.abinit_dos.fermi,
            projected=self.abinit_dos.projected,
        )

    @property
    def structure(self) -> Structure | None:
        if self.abinit_output is None:
            return None
        return self.abinit_output.structure

    @property
    def kpath(self) -> KPath | None:
        if self.abinit_kpoints is None:
            return None
        if self.abinit_kpoints.knames is None:
            return None

        kpoints = None
        if self.abinit_procar and self.abinit_procar.vasp_procar:
            kpoints = self.abinit_procar.vasp_procar.kpoints

        return KPath(
            kpoints=kpoints,
            segment_names=self.abinit_kpoints.knames,
            n_grids=self.abinit_kpoints.ngrids,
            reciprocal_lattice=self.abinit_output.reclat if self.abinit_output else None,
        )

    @property
    def kgrid(self) -> tuple[int, int, int] | None:
        if self.abinit_kpoints is None:
            return None
        return self.abinit_kpoints.kgrid
