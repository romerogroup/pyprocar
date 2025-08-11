__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import ast
import copy
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure
from pyprocar.io.qe.projwfc import AtomicProjXML, ProjwfcDOS, ProjwfcIn, ProjwfcOut
from pyprocar.io.qe.pw import PwIn, PwOut, PwXML
from pyprocar.utils.units import AU_TO_ANG, HARTREE_TO_EV

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")



class QEParser:
    """The class is used to parse Quantum Expresso files.
    The most important objects that comes from this parser are the .ebs and .dos

    Parameters
    ----------
    dirpath : str, optional
        Directory path to where calculation took place, by default ""
    scf_in_filepath : str, optional
        The scf filename, by default "scf.in"
    bands_in_filepath : str, optional
        The bands filename in the case of a band structure calculation, by default "bands.in"
    pdos_in_filepath : str, optional
        The pdos filename in the case of a density ofstates calculation, by default "pdos.in"
    kpdos_in_filepath : str, optional
        The kpdos filename, by default "kpdos.in"
    atomic_proj_xml : str, optional
        The atomic projection xml name. This is located in the where the outdir is and in the {prefix}.save directory, by default "atomic_proj.xml"
    """

    def __init__(
        self,
        dirpath: Union[str, Path] = "",
        scf_in_filepath: Union[str, Path] = "scf.in",
        scf_out_filepath: Union[str, Path] = "scf.out",
        bands_in_filepath: Union[str, Path] = "bands.in",
        pdos_in_filepath: Union[str, Path] = "pdos.in",
        pdos_out_filepath: Union[str, Path] = "pdos.out",
        kpdos_in_filepath: Union[str, Path] = "kpdos.in",
        kpdos_out_filepath: Union[str, Path] = "kpdos.out",
        data_xml_filepath: Union[str, Path] = "data-file-schema.xml",
        atomic_proj_xml_filepath: Union[str, Path] = "atomic_proj.xml",
    ) -> None:
        self._scf_in_filepath = Path(scf_in_filepath)
        self._scf_out_filepath = Path(scf_out_filepath)
        self._bands_in_filepath = Path(bands_in_filepath)
        self._pdos_in_filepath = Path(pdos_in_filepath)
        self._kpdos_in_filepath = Path(kpdos_in_filepath)
        self._atomic_proj_xml_filepath = Path(atomic_proj_xml_filepath)
        self._pdos_out_filepath = Path(pdos_out_filepath)
        self._kpdos_out_filepath = Path(kpdos_out_filepath)
        self._data_xml_filepath = Path(data_xml_filepath)

    @property
    def scf_in_filepath(self) -> Path:
        return self._scf_in_filepath
    
    @property
    def scf_out_filepath(self) -> Path:
        return self._scf_out_filepath

    @property
    def bands_in_filepath(self) -> Path:
        return self._bands_in_filepath

    @property
    def pdos_in_filepath(self) -> Path:
        return self._pdos_in_filepath

    @property
    def pdos_out_filepath(self) -> Path:
        return self._pdos_out_filepath

    @property
    def kpdos_in_filepath(self) -> Path:
        return self._kpdos_in_filepath

    @property
    def kpdos_out_filepath(self) -> Path:
        return self._kpdos_out_filepath

    @property
    def atomic_proj_xml_filepath(self) -> Path:
        return self._atomic_proj_xml_filepath
    
    @cached_property
    def scf_in(self) -> PwIn:
        return PwIn(self.scf_in_filepath)
    
    
    
class QEParserAuto:
    """Auto-detects Quantum ESPRESSO files in a directory and exposes
    lazy parser properties and computed objects (EBS, DOS, Structure).

    Example
    -------
    parser = QEParserAuto("/path/to/qe/calculation")
    print(parser.summary())
    ebs = parser.ebs
    dos = parser.dos
    structure = parser.structure
    """

    def __init__(self, dirpath: Union[str, Path]) -> None:
        self._dirpath: Path = Path(dirpath)
        self._detected: Dict[str, Union[Path, List[Path], None]] = {
            "scf_in": None,
            "scf_out": None,
            "bands_in": None,
            "bands_out": None,
            "nscf_in": None,
            "nscf_out": None,
            "projwfc_in": None,
            "projwfc_out": None,
            "pdos_files": [],
            "atomic_proj_xml": None,
            "data_file_schema_xml": None,
            "data_xml": None,
            "pw_xml": None,
        }
        self.detect_files()

    # -------- file detection --------
    def detect_files(self) -> None:
        if not self._dirpath.exists():
            user_logger.warning(f"Directory not found: {self._dirpath}")
            return

        files = [p for p in self._dirpath.rglob("*") if p.is_file()]

        # XMLs
        atomic_proj_xml = [p for p in files if re.search(r"(?i)^atomic_proj\.xml$", p.name)]
        data_file_schema = [p for p in files if re.search(r"(?i)data-file-schema\.xml$", p.name)]
        data_xmls = [p for p in files if re.search(r"(?i)data-file\.xml$", p.name)]
        pw_xmls = [p for p in files if p.suffix.lower() == ".xml" and p.name.lower() not in {"atomic_proj.xml", "data-file-schema.xml", "data-file.xml"}]

        # Inputs
        # Detect inputs by peeking content
        in_files = [p for p in files if p.suffix.lower() == ".in"]
        scf_ins: List[Path] = []
        bands_ins: List[Path] = []
        nscf_ins: List[Path] = []
        projwfc_ins: List[Path] = []

        for inf in in_files:
            try:
                if ProjwfcIn.is_file_of_type(inf):
                    projwfc_ins.append(inf)
                elif PwIn.is_file_of_type(inf):
                    # Best-effort classification by filename hint
                    if re.search(r"(?i)\bbands", inf.name):
                        bands_ins.append(inf)
                    elif re.search(r"(?i)\bnscf", inf.name):
                        nscf_ins.append(inf)
                    else:
                        scf_ins.append(inf)
            except Exception:
                pass

        # Out files: detect program type by peeking first 5 lines
        out_files = [p for p in files if p.suffix.lower() == ".out"]
        pwscf_outs: List[Path] = []
        projwfc_outs: List[Path] = []
        for of in out_files:
            try:
                if PwOut.is_file_of_type(of):
                    pwscf_outs.append(of)
                elif ProjwfcOut.is_file_of_type(of):
                    projwfc_outs.append(of)
            except Exception:
                pass

        # Classify PWSCF out files by filename hints (best-effort)
        scf_outs = [p for p in pwscf_outs if re.search(r"(?i)\bscf", p.name)]
        bands_outs = [p for p in pwscf_outs if re.search(r"(?i)\bbands", p.name)]
        nscf_outs = [p for p in pwscf_outs if re.search(r"(?i)\bnscf", p.name)]

        # PDOS files
        pdos_files = [p for p in files if re.search(r"(?i)pdos_atm#|pdos_tot", p.name)]

        def sort_pref(fp_list: List[Path]) -> List[Path]:
            try:
                return sorted(
                    fp_list,
                    key=lambda p: (
                        len(p.relative_to(self._dirpath).parts),
                        -p.stat().st_mtime,
                    ),
                )
            except Exception:
                return fp_list

        self._detected["scf_in"] = sort_pref(scf_ins)[0] if scf_ins else None
        self._detected["scf_out"] = sort_pref(scf_outs)[0] if scf_outs else (sort_pref(pwscf_outs)[0] if pwscf_outs else None)
        self._detected["bands_in"] = sort_pref(bands_ins)[0] if bands_ins else None
        self._detected["bands_out"] = sort_pref(bands_outs)[0] if bands_outs else None
        self._detected["nscf_in"] = sort_pref(nscf_ins)[0] if nscf_ins else None
        self._detected["nscf_out"] = sort_pref(nscf_outs)[0] if nscf_outs else None
        self._detected["projwfc_in"] = sort_pref(projwfc_ins)[0] if projwfc_ins else None
        self._detected["projwfc_out"] = sort_pref(projwfc_outs)[0] if projwfc_outs else None
        self._detected["pdos_files"] = sort_pref(pdos_files)
        self._detected["atomic_proj_xml"] = sort_pref(atomic_proj_xml)[0] if atomic_proj_xml else None
        self._detected["data_file_schema_xml"] = sort_pref(data_file_schema)[0] if data_file_schema else None
        self._detected["data_xml"] = sort_pref(data_xmls)[0] if data_xmls else None
        self._detected["pw_xml"] = sort_pref(pw_xmls)[0] if pw_xmls else None

        logger.info(f"Detected files: {self.summary()}")

    def summary(self) -> Dict[str, Union[str, List[str], None]]:
        def _p(v: Optional[Path] | List[Path]):
            if v is None:
                return None
            if isinstance(v, list):
                return [str(x) for x in v]
            return str(v)

        return {
            "dirpath": str(self._dirpath),
            "files": {k: _p(v) for k, v in self._detected.items()},
            "parsers": {
                "scf_in": self._detected["scf_in"] is not None,
                "scf_out": self._detected["scf_out"] is not None,
                "bands_out": self._detected["bands_out"] is not None,
                "nscf_out": self._detected["nscf_out"] is not None,
                "projwfc_out": self._detected["projwfc_out"] is not None,
                "atomic_proj_xml": self._detected["atomic_proj_xml"] is not None,
                "pw_xml": self._detected["pw_xml"] is not None,
                "pdos": isinstance(self._detected.get("pdos_files"), list) and len(self._detected.get("pdos_files") or []) > 0,
            },
        }

    # -------- lazy parser properties --------
    @cached_property
    def scf_in(self) -> Optional[PwIn]:
        fp = self._detected.get("scf_in")
        if not fp:
            user_logger.warning("SCF input not found")
            return None
        try:
            return PwIn(fp)
        except Exception as exc:
            user_logger.warning(f"Error parsing SCF input: {exc}")
            return None

    @cached_property
    def scf_out(self) -> Optional[PwOut]:
        fp = self._detected.get("scf_out")
        if not fp:
            user_logger.warning("SCF output not found")
            return None
        try:
            return PwOut(fp)
        except Exception as exc:
            user_logger.warning(f"Error parsing SCF output: {exc}")
            return None
        
    @cached_property
    def bands_in(self) -> Optional[PwIn]:
        fp = self._detected.get("bands_in")
        if not fp:
            return None
        try:
            return PwIn(fp)
        except Exception:
            return None

    @cached_property
    def bands_out(self) -> Optional[PwOut]:
        fp = self._detected.get("bands_out")
        if not fp:
            return None
        try:
            return PwOut(fp)
        except Exception:
            return None

    @cached_property
    def nscf_in(self) -> Optional[PwIn]:
        fp = self._detected.get("nscf_in")
        if not fp:
            return None
        try:
            return PwIn(fp)
        except Exception:
            return None

    @cached_property
    def nscf_out(self) -> Optional[PwOut]:
        fp = self._detected.get("nscf_out")
        if not fp:
            return None
        try:
            return PwOut(fp)
        except Exception:
            return None

    @cached_property
    def projwfc_in(self) -> Optional[ProjwfcIn]:
        fp = self._detected.get("projwfc_in")
        if not fp:
            return None
        try:
            return ProjwfcIn(fp)
        except Exception:
            return None

    @cached_property
    def projwfc_out(self) -> Optional[ProjwfcOut]:
        fp = self._detected.get("projwfc_out")
        if not fp:
            return None
        try:
            return ProjwfcOut(fp)
        except Exception:
            return None

    @cached_property
    def projwfc_dos(self) -> Optional[ProjwfcDOS]:
        fps = self._detected.get("pdos_files")
        if not fps or not isinstance(fps, list) or len(fps) == 0:
            return None
        try:
            return ProjwfcDOS(self._dirpath)
        except Exception:
            return None

    @cached_property
    def atomic_proj_xml(self) -> Optional[AtomicProjXML]:
        fp = self._detected.get("atomic_proj_xml")
        if not fp:
            return None
        try:
            return AtomicProjXML(fp)
        except Exception:
            return None

    @cached_property
    def pw_xml(self) -> Optional[PwXML]:
        fp = self._detected.get("pw_xml")
        if not fp:
            return None
        try:
            return PwXML(fp)
        except Exception:
            return None

    # -------- computed properties --------
    @cached_property
    def ebs(self) -> Optional[ElectronicBandStructure]:
        # Fermi energy
        efermi = None
        if self.scf_out and self.scf_out.fermi_energy_ev is not None:
            efermi = self.scf_out.fermi_energy_ev
        elif self.nscf_out and self.nscf_out.fermi_energy_ev is not None:
            efermi = self.nscf_out.fermi_energy_ev
        if efermi is None:
            user_logger.warning("Fermi energy not found in SCF/NSCF outputs")
            return None

        # kpoints and bands
        if self.atomic_proj_xml and self.atomic_proj_xml.kpoints is not None and self.atomic_proj_xml.bands is not None:
            kpoints = self.atomic_proj_xml.kpoints
            bands = self.atomic_proj_xml.bands
            weights = self.atomic_proj_xml.weights
            logger.info("EBS: using atomic_proj.xml for kpoints/bands")
        elif self.projwfc_out and self.projwfc_out.kpoints is not None and self.projwfc_out.bands is not None:
            kpoints = self.projwfc_out.kpoints
            bands = self.projwfc_out.bands
            weights = None
            logger.info("EBS: using projwfc.out for kpoints/bands")
        else:
            user_logger.warning("No bands source found (atomic_proj.xml or projwfc.out)")
            return None

        reciprocal_lattice = None
        if self.pw_xml and self.pw_xml.reciprocal_lattice is not None:
            reciprocal_lattice = self.pw_xml.reciprocal_lattice
        elif self.scf_out and self.scf_out.reciprocal_axes is not None:
            reciprocal_lattice = self.scf_out.reciprocal_axes

        return ElectronicBandStructure(
            kpoints=kpoints,
            bands=bands,
            efermi=efermi,
            projected=None,
            projected_phase=None,
            weights=weights,
            kpath=None,
            labels=None,
            reciprocal_lattice=reciprocal_lattice,
        )

    @cached_property
    def dos(self) -> Optional[DensityOfStates]:
        if self.projwfc_dos is None:
            user_logger.warning("No PDOS files found for DOS construction")
            return None
        try:
            energies_grid = self.projwfc_dos.energies
            energies = energies_grid[0, :]
            total_dos = self.projwfc_dos.total_dos
            if total_dos is None:
                user_logger.warning("Total DOS not found in PDOS files")
                return None
            total = total_dos.T  # (n_spin, n_energies)
            efermi = self.scf_out.fermi_energy_ev if self.scf_out else 0.0
            return DensityOfStates(energies=energies, total=total, efermi=efermi)
        except Exception as exc:
            user_logger.warning(f"Error building DOS: {exc}")
            return None

    @cached_property
    def structure(self) -> Optional[Structure]:
        if self.pw_xml and self.pw_xml.direct_lattice is not None and self.pw_xml.atomic_species is not None and self.pw_xml.atomic_positions is not None:
            try:
                lattice = self.pw_xml.direct_lattice
                atoms = list(self.pw_xml.atomic_species)
                frac = self.pw_xml.atomic_positions
                logger.info("Structure: using PwXML")
                return Structure(atoms=atoms, fractional_coordinates=frac, lattice=lattice)
            except Exception as exc:
                user_logger.warning(f"Failed to build Structure from PwXML: {exc}")

        if self.scf_out and self.scf_out.crystal_axes is not None and self.scf_out.atomic_sites is not None and self.scf_out.alat is not None:
            try:
                lattice = np.array(self.scf_out.crystal_axes, dtype=float) * float(self.scf_out.alat) * AU_TO_ANG
                atoms = [site["symbol"] for site in self.scf_out.atomic_sites]
                frac = np.array([[site["x"], site["y"], site["z"]] for site in self.scf_out.atomic_sites], dtype=float)
                logger.info("Structure: using PwOut")
                return Structure(atoms=atoms, fractional_coordinates=frac, lattice=lattice)
            except Exception as exc:
                user_logger.warning(f"Failed to build Structure from PwOut: {exc}")

        user_logger.warning("Unable to build Structure from available files")
        return None

# Backward-compatible alias
QEParser = QEParserAuto
    