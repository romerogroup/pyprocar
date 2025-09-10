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

        # files: List[Path] = []
        # for root, dirs, filenames in os.walk(self._dirpath, followlinks=True):
        #     for name in filenames:
        #         try:
        #             files.append(Path(root) / name)
        #         except Exception:
        #             pass
        
        files = [p for p in self._dirpath.rglob("*", recurse_symlinks=True) if p.is_file()]


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
        out_files = [p for p in files if (p.suffix.lower() == ".out" or p.suffix.lower() == ".log")]
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

        detected_files = self.summary()
        log_msg = f"Detected files:\n"
        for k, v in detected_files.items():
            log_msg += f"{k}: "
            if isinstance(v, dict):
                log_msg += f"{k}:\n"
                for k2, v2 in v.items():
                    log_msg += f"  {k2}: {v2}\n"
            else:
                log_msg += f"{v}\n"
        logger.info(log_msg)

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
                "data_file_schema_xml": self._detected["data_file_schema_xml"] is not None,
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
        
    @cached_property
    def data_file_schema_xml(self) -> Optional[PwXML]:
        fp = self._detected.get("data_file_schema_xml")
        if not fp:
            return None
        try:
            return PwXML(fp)
        except Exception:
            return None
        
    @cached_property
    def alat(self) -> Optional[float]:
        if self.scf_out is not None and self.scf_out.alat is not None:
            logger.info("Parsing alat from scf.out")
            alat = self.scf_out.alat * AU_TO_ANG
        elif self.pw_xml is not None and self.pw_xml.alat is not None:
            logger.info("Parsing alat from pw.xml")
            alat = self.pw_xml.alat
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.alat is not None:
            logger.info("Parsing alat from data_file_schema.xml")
            alat = self.data_file_schema_xml.alat
        else:
            user_logger.warning("No alat found in scf.out or pw.xml")
            return None
        logger.debug(f"alat: {alat}")
        return alat
        
    @cached_property
    def _raw_kpoints(self) -> Optional[np.ndarray]:
        kpoints_cart = None
        if self.atomic_proj_xml is not None:
            logger.info("Parsing kpoints from atomic_proj.xml")
            kpoints_cart = self.atomic_proj_xml.kpoints
        elif self.projwfc_out is not None and self.projwfc_out.kpoints is not None:
            logger.info("Parsing kpoints from projwfc.out")
            kpoints_cart = self.projwfc_out.kpoints
        elif self.bands_in is not None and self.bands_in.kpoints_card is not None and self.bands_in.kpoints_card.kpoints is not None:
            logger.info("Parsing kpoints from bands.in")
            kpoints_cart = self.bands_in.kpoints_card.kpoints
        elif self.pw_xml is not None and self.pw_xml.kpoints is not None:
            logger.info("Parsing kpoints from pw.xml")
            kpoints_cart = self.pw_xml.kpoints
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.kpoints is not None:
            logger.info("Parsing kpoints from data_file_schema.xml")
            kpoints_cart = self.data_file_schema_xml.kpoints
        else:
            user_logger.warning("No kpoints found in atomic_proj.xml or projwfc.out or bands.in")
            return None
        
        scaled_kpoints_cart = kpoints_cart * (2*np.pi / (self.alat))
        
        kpoints = np.around(scaled_kpoints_cart.dot(np.linalg.inv(self.reciprocal_lattice)), decimals=8)
        
        return kpoints
    
    @cached_property
    def kpath(self) -> Optional[KPath]:
        if self.is_dos_calculation:
            logger.info("No kpath found for DOS calculation")
            return None
        
        if self.bands_in is None:
            logger.info("No bands.in file found, therefore not parsing kpath")
            return None
        
        if self.bands_in.kpoints_card is None:
            logger.info("No kpoints_card found in bands.in, therefore not parsing kpath")
            return None
        
        if self.bands_in.kpoints_card.modified_knames is None:
            logger.info("No modified_knames found in bands.in, therefore not parsing kpath")
            return None
        
        high_sym_points = self.bands_in.kpoints_card.high_symmetry_points
        
        kticks = find_high_symmetry_ticks(self._raw_kpoints, high_sym_points)
        self._kticks = kticks
        new_kpoints = insert_continuous_points(self._raw_kpoints, kticks)
        new_kpoints = np.array(new_kpoints)
        
        segment_names = self.bands_in.kpoints_card.modified_knames
        ngrids = [grid + 1 if i != len(self.bands_in.kpoints_card.ngrids) - 1 else grid for i, grid in enumerate(self.bands_in.kpoints_card.ngrids)]
        return KPath(
            knames= segment_names,
            kticks=kticks,
            special_kpoints=self.bands_in.kpoints_card.special_kpoints,
            ngrids=ngrids,
            has_time_reversal=True,
        )
        
    @cached_property
    def kticks(self) -> List[int]:
        if hasattr(self, "_kticks"):
            return self._kticks
        return []
    
    @property
    def kpoints(self) -> Optional[np.ndarray]:
        kpoints = self._raw_kpoints
        if self.kpath is not None:
            logger.info("Parsing kpoints from kpath")
            kpoints = insert_continuous_points(self._raw_kpoints, self.kticks)
        return kpoints
    
    @cached_property
    def nk1(self) -> Optional[int]:
        if self.nscf_in is not None and self.nscf_in.kpoints_card is not None and self.nscf_in.kpoints_card.nk1 is not None:
            return self.nscf_in.kpoints_card.nk1
        if self.pw_xml is not None and self.pw_xml.nk1 is not None:
            return self.pw_xml.nk1
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.nk1 is not None:
            logger.info("Parsing nk1 from data_file_schema.xml")
            return self.data_file_schema_xml.nk1
        return None
    
    @cached_property
    def nk2(self) -> Optional[int]:
        if self.nscf_in is not None and self.nscf_in.kpoints_card is not None and self.nscf_in.kpoints_card.nk2 is not None:
            return self.nscf_in.kpoints_card.nk2
        if self.pw_xml is not None and self.pw_xml.nk2 is not None:
            return self.pw_xml.nk2
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.nk2 is not None:
            logger.info("Parsing nk2 from data_file_schema.xml")
            return self.data_file_schema_xml.nk2
        return None
    
    @cached_property
    def nk3(self) -> Optional[int]:
        if self.nscf_in is not None and self.nscf_in.kpoints_card is not None and self.nscf_in.kpoints_card.nk3 is not None:
            return self.nscf_in.kpoints_card.nk3
        if self.pw_xml is not None and self.pw_xml.nk3 is not None:
            return self.pw_xml.nk3
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.nk3 is not None:
            logger.info("Parsing nk3 from data_file_schema.xml")
            return self.data_file_schema_xml.nk3
        return None
    
    @cached_property
    def sk1(self) -> Optional[int]:
        if self.nscf_in is not None and self.nscf_in.kpoints_card is not None and self.nscf_in.kpoints_card.sk1 is not None:
            return self.nscf_in.kpoints_card.sk1
        if self.pw_xml is not None and self.pw_xml.sk1 is not None:
            return self.pw_xml.sk1
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.sk1 is not None:
            logger.info("Parsing sk1 from data_file_schema.xml")
            return self.data_file_schema_xml.sk1
        return None
    
    @cached_property
    def sk2(self) -> Optional[int]:
        if self.nscf_in is not None and self.nscf_in.kpoints_card is not None and self.nscf_in.kpoints_card.sk2 is not None:
            return self.nscf_in.kpoints_card.sk2
        if self.pw_xml is not None and self.pw_xml.sk2 is not None:
            return self.pw_xml.sk2
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.sk2 is not None:
            logger.info("Parsing sk2 from data_file_schema.xml")
            return self.data_file_schema_xml.sk2
        return None
    
    @cached_property
    def sk3(self) -> Optional[int]:
        if self.nscf_in is not None and self.nscf_in.kpoints_card is not None and self.nscf_in.kpoints_card.sk3 is not None:
            return self.nscf_in.kpoints_card.sk3
        if self.pw_xml is not None and self.pw_xml.sk3 is not None:
            return self.pw_xml.sk3
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.sk3 is not None:
            logger.info("Parsing sk3 from data_file_schema.xml")
            return self.data_file_schema_xml.sk3
        return None
    
    @cached_property
    def reciprocal_lattice(self) -> Optional[np.ndarray]:
        if self.pw_xml is not None and self.pw_xml.reciprocal_lattice is not None:
            logger.info("Parsing reciprocal lattice from pw.xml")
            reciprocal_lattice = self.pw_xml.reciprocal_lattice
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.reciprocal_lattice is not None:
            logger.info("Parsing reciprocal lattice from data_file_schema.xml")
            reciprocal_lattice = self.data_file_schema_xml.reciprocal_lattice
        elif self.scf_out is not None and self.scf_out.reciprocal_axes is not None:
            logger.info("Parsing reciprocal lattice from scf.out")
            reciprocal_lattice = self.scf_out.reciprocal_axes
        elif self.bands_out is not None and self.bands_out.reciprocal_axes is not None:
            logger.info("Parsing reciprocal lattice from bands.out")
            reciprocal_lattice = self.bands_out.reciprocal_axes
        elif self.nscf_out is not None and self.nscf_out.reciprocal_axes is not None:
            logger.info("Parsing reciprocal lattice from nscf.out")
            reciprocal_lattice = self.nscf_out.reciprocal_axes
        else:
            logger.warning("No reciprocal lattice found in pw.xml or scf.out or bands.out or nscf.out")
            return None
        return  (2 * np.pi / self.alat) * reciprocal_lattice
    
    @cached_property
    def fermi(self) -> Optional[float]:
        if self.scf_out is not None and self.scf_out.fermi_energy_ev is not None:
            logger.debug(f"Fermi energy found in {self.scf_out.fermi_energy_ev}")
            return self.scf_out.fermi_energy_ev
        if self.pw_xml is not None and self.pw_xml.fermi is not None:
            return self.pw_xml.fermi
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.fermi is not None:
            return self.data_file_schema_xml.fermi
        
        return None
    
    @cached_property
    def bands(self) -> Optional[np.ndarray]:
        if self.atomic_proj_xml is not None and self.atomic_proj_xml.bands is not None:
            logger.info("Parsing bands from atomic_proj.xml")
            bands = self.atomic_proj_xml.bands
        elif self.projwfc_out is not None and self.projwfc_out.bands is not None:
            logger.info("Parsing bands from projwfc.out")
            bands = HARTREE_TO_EV * self.projwfc_out.bands
        elif self.pw_xml is not None and self.pw_xml.bands is not None:
            logger.info("Parsing bands from pw.xml")
            bands = HARTREE_TO_EV * self.pw_xml.bands
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.bands is not None:
            logger.info("Parsing bands from data_file_schema.xml")
            bands = HARTREE_TO_EV * self.data_file_schema_xml.bands
        else:
            user_logger.warning("No bands found in atomic_proj.xml or projwfc.out or pw.xml")
            return None
        
        if self.kpath is not None:
            bands = insert_continuous_points(bands, self.kticks)
        logger.debug(f"Bands: {bands.shape}")
        
        return bands
    
    @cached_property
    def spd_phase(self) -> Optional[np.ndarray]:
        if self.atomic_proj_xml is None and self.projwfc_out is None:
            return None
        
        wfc_mapping = self.projwfc_out.wfc_mapping
        projections = self.atomic_proj_xml.projections
        orbitals = self.projwfc_out.orbitals
        
        n_kpoints = self.atomic_proj_xml.n_kpoints
        n_bands = self.atomic_proj_xml.n_bands
        n_spin_channels = self.atomic_proj_xml.n_spin_channels
        n_atoms = self.projwfc_out.n_atoms
        n_orbitals = self.projwfc_out.n_orbitals

        
        pyprocar_projections_phase = np.zeros(
            shape=(n_kpoints, n_bands, n_spin_channels, n_atoms, n_orbitals),
            dtype=projections.dtype,
        )
        
        for state_num, wfc_info in wfc_mapping.items():
            element = wfc_info["element"]
            wfc_num = wfc_info["wfc_num"]
            atm_num = wfc_info["atm_num"]
            l = wfc_info["l"]
            j = wfc_info["j"]
            m_j = wfc_info["m_j"]
            m = wfc_info["m"]
            
            if m_j is not None:
                orbital_dict = {
                    "l": l,
                    "j": j,
                    "m_j": m_j,
                }
            else:
                orbital_dict = {
                    "l": l,
                    "m": m,
                }
            
            i_orbital = orbitals.index(orbital_dict)
            i_atom = atm_num - 1
            i_state = state_num - 1
            pyprocar_projections_phase[..., i_atom, i_orbital] += projections[..., i_state]
       
        n_kpoints = self.atomic_proj_xml.n_kpoints
        n_bands = self.atomic_proj_xml.n_bands
        n_spin_channels = self.atomic_proj_xml.n_spin_channels
        n_atoms = self.projwfc_out.n_atoms
        n_orbitals = self.projwfc_out.n_orbitals
        n_principals = 1
        
        # Move spin channels to the last axis. This is need to have the dimensionality to have the same shape as the format in pyproxcar
        pyprocar_projections_phase = np.moveaxis(pyprocar_projections_phase, 2, -1)
        pyprocar_projections_phase = pyprocar_projections_phase.reshape((n_kpoints, n_bands, n_atoms, n_principals, n_orbitals, n_spin_channels))
        
        if self.kpath is not None:
            pyprocar_projections_phase = insert_continuous_points(pyprocar_projections_phase, self.kticks)
        logger.debug(f"Spd Phase: {pyprocar_projections_phase.shape}")
        
        return pyprocar_projections_phase
    
    @cached_property
    def spd(self) -> Optional[np.ndarray]:
        if self.atomic_proj_xml is None and self.projwfc_out is None and self.spd_phase is None:
            return None
        
        logger.info(f"Parsing spd from spd phase")
        spd = np.absolute(self.spd_phase)**2
        
        n_kpoints = self.spd_phase.shape[0]
        if self.kpath is not None and n_kpoints != self.kpoints.shape[0]:
            spd = insert_continuous_points(spd, self.kticks)
        logger.debug(f"Spd: {spd.shape}")
        return spd
    
    @cached_property
    def orbitals(self) -> Optional[List[str]]:
        if self.projwfc_out is not None:
            logger.info("Parsing orbitals from projwfc.out")
            return self.projwfc_out.orbitals
        elif self.atomic_proj_xml is not None:
            logger.info("Parsing orbitals from atomic_proj.xml")
            return self.atomic_proj_xml.orbitals
        else:
            logger.info("No orbitals found in projwfc.out or atomic_proj.xml")
            return None
    
    # -------- computed properties --------
    @cached_property
    def ebs(self) -> Optional[ElectronicBandStructure]:
        return ElectronicBandStructure(
                kpoints=self.kpoints,
                n_kx=self.nk1,
                n_ky=self.nk2,
                n_kz=self.nk3,
                bands=self.bands,
                projected=self.spd,
                efermi=self.fermi,
                kpath=self.kpath,
                projected_phase=self.spd_phase,
                labels=self.orbitals,
                reciprocal_lattice=self.reciprocal_lattice,
            )
        
    @cached_property
    def projected_dos(self) -> Optional[np.ndarray]:
        if self.projwfc_dos is None:
            return None
        
        n_energies = self.projwfc_dos.n_energies
        n_spin_channels = self.projwfc_dos.n_spin_channels
        n_orbitals = self.projwfc_out.n_orbitals
        n_atoms = self.projwfc_dos.n_atoms

        # Reshaping to match what pyprocar expects
        n_principals = 1
        projected_dos = self.projwfc_dos.projected_dos    # with shape (n_energies, n_spin_channels, n_atoms, n_orbitals)
        projected_dos = np.moveaxis(projected_dos, 1, -1) # shape (n_energies, n_orbitals, n_atoms, n_spin_channels)
        projected_dos = np.moveaxis(projected_dos, 0, -1) # shape (n_atoms, n_orbitals, n_spin_channels, n_energies)
        projected_dos = projected_dos.reshape(n_atoms, n_principals, n_orbitals, n_spin_channels, n_energies)
        logger.debug(f"projected_dos: {projected_dos.shape}")
        return projected_dos
    
    @cached_property
    def total_dos(self) -> Optional[np.ndarray]:
        if self.projwfc_dos is None:
            return None
        n_spin_channels = self.projwfc_dos.n_spin_channels
        n_energies = self.projwfc_dos.n_energies
        logger.debug(f"total_dos: {self.projwfc_dos.total_dos.shape}")
        return self.projwfc_dos.total_dos.reshape((n_spin_channels, n_energies), order="C")
    
    @cached_property
    def energies(self) -> Optional[np.ndarray]:
        if self.projwfc_dos is None:
            return None
        return self.projwfc_dos.bands[0] - self.fermi
    
    @cached_property
    def is_dos_calculation(self) -> bool:
        logger.info("Checking if DOS calculation")
        is_dos_calculation = not self.projwfc_in.is_kresolved
        logger.info(f"Is DOS calculation: {is_dos_calculation}")
        return is_dos_calculation

    @cached_property
    def dos(self) -> Optional[DensityOfStates]:
        if self.projwfc_dos is None:
            user_logger.warning("No PDOS files found for DOS construction")
            return None
        
        if not self.is_dos_calculation:
            return None
        
        logger.debug(f"energies: {self.energies.shape}")
        logger.debug(f"total_dos: {self.total_dos.shape}")
        logger.debug(f"projected_dos: {self.fermi}")
        return DensityOfStates(
                energies=self.energies,
                total=self.total_dos,
                efermi=self.fermi,
                projected=self.projected_dos,
            )
        
    @cached_property
    def species(self) -> Optional[List[str]]:
        if self.pw_xml is not None and self.pw_xml.atomic_species is not None:
            return self.pw_xml.atomic_species
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.atomic_species is not None:
            return self.data_file_schema_xml.atomic_species
        else:
            user_logger.warning("No atomic species found in any input or output file")
            return None
        
    @cached_property
    def direct_lattice(self) -> Optional[np.ndarray]:
        if self.pw_xml is not None and self.pw_xml.direct_lattice is not None:
            return self.pw_xml.direct_lattice
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.direct_lattice is not None:
            return self.data_file_schema_xml.direct_lattice
        else:
            user_logger.warning("No direct lattice found in any input or output file")
            return None
        
    @cached_property
    def atomic_positions(self) -> Optional[np.ndarray]:
        if self.pw_xml is not None and self.pw_xml.atomic_positions is not None:
            return self.pw_xml.atomic_positions
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.atomic_positions is not None:
            return self.data_file_schema_xml.atomic_positions
        else:
            user_logger.warning("No atomic positions found in any input or output file")
            return None
        
    @cached_property
    def rotations(self) -> Optional[np.ndarray]:
        if self.pw_xml is not None and self.pw_xml.rotations is not None:
            return self.pw_xml.rotations
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.rotations is not None:
            return self.data_file_schema_xml.rotations
        else:
            user_logger.warning("No rotations found in any input or output file")
            return None

    @cached_property
    def structure(self) -> Optional[Structure]:
        return Structure(
            atoms=self.species,
            lattice=self.direct_lattice,
            fractional_coordinates=self.atomic_positions,
            rotations=self.rotations,
        )


def find_high_symmetry_ticks(raw_kpoints, high_sym_points, atol=1e-4):
    """
    Find indices of raw_kpoints that match high_sym_points within tolerance.
    Each high_sym_point is matched once, in order, to the first raw_kpoint
    within tolerance. Duplicates in high_sym_points are allowed.

    Parameters
    ----------
    raw_kpoints : (N, 3) ndarray
        List of kpoints along the path.
    high_sym_points : (M, 3) ndarray
        List of special kpoints to match, in order (duplicates allowed).
    atol : float
        Absolute tolerance for matching.

    Returns
    -------
    kticks : list[int]
        Indices in raw_kpoints corresponding to high_sym_points.
    """
    raw_kpoints = np.asarray(raw_kpoints)
    high_sym_points = np.asarray(high_sym_points)

    # Compute pairwise distances (N, M)
    dists = np.linalg.norm(
        raw_kpoints[:, None, :] - high_sym_points[None, :, :], axis=-1
    )

    kticks = []
    last_idx = -1  # ensure we move forward along raw_kpoints


    for j in range(dists.shape[1]):
        # Find matches *after* the last matched index
        matches = np.where((dists[:, j] < atol) & (np.arange(len(raw_kpoints)) > last_idx))[0]
        if len(matches) > 0:
            idx = matches[0]  # first valid match
            kticks.append(idx)
            last_idx = idx
        else:
            raise ValueError(
                f"No match found for high_sym_point {j}: {high_sym_points[j]}"
            )
            

    return kticks

def insert_continuous_points(arr: np.ndarray, tick_indices: np.ndarray) -> np.ndarray:
    """
    Insert duplicates at tick indices to enforce VASP-style repeated kpoints.

    Parameters
    ----------
    arr : np.ndarray
        Array with shape (nk, ...), where axis=0 corresponds to kpoints.
    tick_indices : array-like
        Indices of tick points (end of each segment).
        Continuous ticks will be duplicated.

    Returns
    -------
    np.ndarray
        New array with duplicated rows at continuous tick points.
    """
    tick_indices = np.asarray(tick_indices)

    # Continuous ticks are all except the very first one
    continuous_ticks = tick_indices[1:-1]  

    # Values to duplicate
    rows_to_insert = arr[continuous_ticks]

    # Insert them back at the right positions
    # np.insert shifts indices automatically, so we need to offset
    out = np.insert(arr, continuous_ticks + 1, rows_to_insert, axis=0)

    return out