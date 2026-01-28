from __future__ import annotations

__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllangWV@gmail.com"
__date__ = "September 6, 2025"

import contextlib
import logging
import os
import re
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from pyprocar.core import (
    DensityOfStates,
    ElectronicBandStructure,
    KPath,
    Structure,
    get_ebs_from_data,
)
from pyprocar.core import kpoints as k_utils
from pyprocar.io.base import BaseParser
from pyprocar.io.qe.projwfc import AtomicProjXML, ProjwfcDOS, ProjwfcIn, ProjwfcOut
from pyprocar.io.qe.pw import PwIn, PwOut, PwXML
from pyprocar.utils.units import AU_TO_ANG, HARTREE_TO_EV

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


class QEParser(BaseParser):
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

    def __init__(self, dirpath: str | Path) -> None:
        super().__init__(dirpath)
        self._dirpath: Path = Path(dirpath)
        self._detected: dict[str, Path | list[Path] | None] = {
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
        self._kticks: list[int] = []
        self.detect_files()

    # -------- file detection --------
    def detect_files(self) -> None:
        if not self._dirpath.exists():
            user_logger.warning(f"Directory not found: {self._dirpath}")
            return

        files: list[Path] = []
        for root, _dirs, filenames in os.walk(self._dirpath, followlinks=True):
            for name in filenames:
                with contextlib.suppress(Exception):
                    files.append(Path(root) / name)
        # Only works for pathlib==3.13 or python==3.13
        # files = [p for p in self._dirpath.rglob("*", recurse_symlinks=True) if p.is_file()]

        # XMLs
        atomic_proj_xml = [p for p in files if re.search(r"(?i)^atomic_proj\.xml$", p.name)]
        data_file_schema = [p for p in files if re.search(r"(?i)data-file-schema\.xml$", p.name)]
        data_xmls = [p for p in files if re.search(r"(?i)data-file\.xml$", p.name)]
        pw_xmls = [
            p
            for p in files
            if p.suffix.lower() == ".xml"
            and p.name.lower() not in {"atomic_proj.xml", "data-file-schema.xml", "data-file.xml"}
        ]

        # Inputs
        # Detect inputs by peeking content
        in_files = [p for p in files if p.suffix.lower() == ".in"]
        scf_ins: list[Path] = []
        bands_ins: list[Path] = []
        nscf_ins: list[Path] = []
        projwfc_ins: list[Path] = []

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
        pwscf_outs: list[Path] = []
        projwfc_outs: list[Path] = []
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

        def sort_pref(fp_list: list[Path]) -> list[Path]:
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
        self._detected["scf_out"] = (
            sort_pref(scf_outs)[0]
            if scf_outs
            else (sort_pref(pwscf_outs)[0] if pwscf_outs else None)
        )
        self._detected["bands_in"] = sort_pref(bands_ins)[0] if bands_ins else None
        self._detected["bands_out"] = sort_pref(bands_outs)[0] if bands_outs else None
        self._detected["nscf_in"] = sort_pref(nscf_ins)[0] if nscf_ins else None
        self._detected["nscf_out"] = sort_pref(nscf_outs)[0] if nscf_outs else None
        self._detected["projwfc_in"] = sort_pref(projwfc_ins)[0] if projwfc_ins else None
        self._detected["projwfc_out"] = sort_pref(projwfc_outs)[0] if projwfc_outs else None
        self._detected["pdos_files"] = sort_pref(pdos_files)
        self._detected["atomic_proj_xml"] = (
            sort_pref(atomic_proj_xml)[0] if atomic_proj_xml else None
        )
        self._detected["data_file_schema_xml"] = (
            sort_pref(data_file_schema)[0] if data_file_schema else None
        )
        self._detected["data_xml"] = sort_pref(data_xmls)[0] if data_xmls else None
        self._detected["pw_xml"] = sort_pref(pw_xmls)[0] if pw_xmls else None

        detected_files = self.summary()
        log_msg = "Detected files:\n"
        for k, v in detected_files.items():
            log_msg += f"{k}: "
            if isinstance(v, dict):
                log_msg += f"{k}:\n"
                for k2, v2 in v.items():  # pyright: ignore[reportUnknownVariableType]
                    log_msg += f"  {k2}: {v2}\n"
            else:
                log_msg += f"{v}\n"
        logger.info(log_msg)
        detected_files = self.summary()
        log_msg = "Detected files:\n"
        for k, v in detected_files.items():
            log_msg += f"{k}: "
            if isinstance(v, dict):
                log_msg += f"{k}:\n"
                for k2, v2 in v.items():  # pyright: ignore[reportUnknownVariableType]
                    log_msg += f"  {k2}: {v2}\n"
            else:
                log_msg += f"{v}\n"
        logger.info(log_msg)

    def summary(self) -> dict[str, Any]:
        def _p(v: Path | None | list[Path]) -> str | list[str] | None:
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
                "pdos": isinstance(pdos := self._detected.get("pdos_files"), list)
                and len(pdos) > 0,
            },
        }

    # -------- lazy parser properties --------
    @cached_property
    def scf_in(self) -> PwIn | None:
        fp = self._detected.get("scf_in")
        if not fp or isinstance(fp, list):
            user_logger.warning("SCF input not found")
            return None
        try:
            return PwIn(fp)
        except Exception as exc:
            user_logger.warning(f"Error parsing SCF input: {exc}")
            return None

    @cached_property
    def scf_out(self) -> PwOut | None:
        fp = self._detected.get("scf_out")
        if not fp or isinstance(fp, list):
            user_logger.warning("SCF output not found")
            return None
        try:
            return PwOut(fp)
        except Exception as exc:
            user_logger.warning(f"Error parsing SCF output: {exc}")
            return None

    @cached_property
    def bands_in(self) -> PwIn | None:
        fp = self._detected.get("bands_in")
        if not fp or isinstance(fp, list):
            return None
        try:
            return PwIn(fp)
        except Exception:
            return None

    @cached_property
    def bands_out(self) -> PwOut | None:
        fp = self._detected.get("bands_out")
        if not fp or isinstance(fp, list):
            return None
        try:
            return PwOut(fp)
        except Exception:
            return None

    @cached_property
    def nscf_in(self) -> PwIn | None:
        fp = self._detected.get("nscf_in")
        if not fp or isinstance(fp, list):
            return None
        try:
            return PwIn(fp)
        except Exception:
            return None

    @cached_property
    def nscf_out(self) -> PwOut | None:
        fp = self._detected.get("nscf_out")
        if not fp or isinstance(fp, list):
            return None
        try:
            return PwOut(fp)
        except Exception:
            return None

    @cached_property
    def projwfc_in(self) -> ProjwfcIn | None:
        fp = self._detected.get("projwfc_in")
        if not fp or isinstance(fp, list):
            return None
        try:
            return ProjwfcIn(fp)
        except Exception:
            return None

    @cached_property
    def projwfc_out(self) -> ProjwfcOut | None:
        fp = self._detected.get("projwfc_out")
        if not fp or isinstance(fp, list):
            return None
        try:
            return ProjwfcOut(fp)
        except Exception:
            return None

    @cached_property
    def projwfc_dos(self) -> ProjwfcDOS | None:
        fps = self._detected.get("pdos_files")
        if not fps or not isinstance(fps, list) or len(fps) == 0:
            return None
        try:
            return ProjwfcDOS(self._dirpath)
        except Exception:
            return None

    @cached_property
    def atomic_proj_xml(self) -> AtomicProjXML | None:
        fp = self._detected.get("atomic_proj_xml")
        if not fp or isinstance(fp, list):
            return None
        try:
            return AtomicProjXML(fp)
        except Exception:
            err_msg = "Error parsing atomic_proj.xml"
            logger.error(err_msg)
            return None

    @cached_property
    def pw_xml(self) -> PwXML | None:
        fp = self._detected.get("pw_xml")
        if not fp or isinstance(fp, list):
            return None
        try:
            return PwXML(fp)
        except Exception:
            return None

    @cached_property
    def data_file_schema_xml(self) -> PwXML | None:
        fp = self._detected.get("data_file_schema_xml")
        if not fp or isinstance(fp, list):
            return None
        try:
            return PwXML(fp)
        except Exception:
            return None

    @cached_property
    def alat(self) -> float | None:
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
    def _raw_kpoints(self) -> np.ndarray | None:
        kpoints_cart: np.ndarray | None = None
        if self.atomic_proj_xml is not None:
            logger.info("Parsing kpoints from atomic_proj.xml")
            kpoints_cart = self.atomic_proj_xml.kpoints
        elif self.projwfc_out is not None and self.projwfc_out.kpoints is not None:
            logger.info("Parsing kpoints from projwfc.out")
            kpoints_cart = self.projwfc_out.kpoints
        elif self.bands_in is not None and self.bands_in.kpoints_card.kpoints is not None:
            logger.info("Parsing kpoints from bands.in")
            kpoints_cart = self.bands_in.kpoints_card.kpoints
        elif self.pw_xml is not None and self.pw_xml.kpoints is not None:
            logger.info("Parsing kpoints from pw.xml")
            kpoints_cart = self.pw_xml.kpoints
        elif (
            self.data_file_schema_xml is not None and self.data_file_schema_xml.kpoints is not None
        ):
            logger.info("Parsing kpoints from data_file_schema.xml")
            kpoints_cart = self.data_file_schema_xml.kpoints

        if kpoints_cart is None:
            user_logger.warning("No kpoints found in atomic_proj.xml or projwfc.out or bands.in")
            return None

        if self.alat is None or self.reciprocal_lattice is None:
            user_logger.warning("Cannot compute kpoints without alat and reciprocal_lattice")
            return None

        scaled_kpoints_cart = kpoints_cart * (2 * np.pi / self.alat)

        kpoints: npt.NDArray[np.float64] = np.around(  # pyright: ignore[reportUnknownVariableType]
            scaled_kpoints_cart.dot(np.linalg.inv(self.reciprocal_lattice)), decimals=8
        )

        return kpoints  # pyright: ignore[reportUnknownVariableType]

    @cached_property
    @override
    def kpath(self) -> KPath | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.is_dos_calculation:
            logger.info("No kpath found for DOS calculation")
            return None

        if self.bands_in is None:
            logger.info("No bands.in file found, therefore not parsing kpath")
            return None

        kpoints_card = self.bands_in.kpoints_card
 
        modified_knames = kpoints_card.modified_knames

        if self._raw_kpoints is None:
            logger.info("No kpoints found, therefore not parsing kpath")
            return None

        high_sym_points = kpoints_card.high_symmetry_points

        if high_sym_points is None:
            logger.info("No high symmetry points found")
            return None

        kticks: list[int] = find_high_symmetry_ticks(self._raw_kpoints, high_sym_points)
        self._kticks = kticks
        new_kpoints = insert_continuous_points(self._raw_kpoints, kticks)
        new_kpoints = np.array(new_kpoints)

        # Convert list[list[str]] to list[tuple[str, str]]
        segment_names: list[tuple[str, str]] = [(names[0], names[1]) for names in modified_knames]
        return KPath(
            kpoints=new_kpoints,
            segment_names=segment_names,
            reciprocal_lattice=self.reciprocal_lattice,
        )

    @cached_property
    def kticks(self) -> list[int]:
        if hasattr(self, "_kticks"):
            return self._kticks
        return []

    @property
    def kpoints(self) -> np.ndarray | None:
        kpoints: np.ndarray | None = self._raw_kpoints
        if self.kpath is not None:
            logger.info("Parsing kpoints from kpath")
            kpoints = np.asarray(self.kpath.kpoints)
        return kpoints

    @cached_property
    def kgrid_info(self) -> k_utils.KGridInfo | None:
        if self.kpath is not None:
            return None

        nk1, nk2, nk3 = self.nk1, self.nk2, self.nk3
        sk1, sk2, sk3 = self.sk1, self.sk2, self.sk3

        if nk1 is None or nk2 is None or nk3 is None:
            return None
        if sk1 is None or sk2 is None or sk3 is None:
            return None

        return k_utils.KGridInfo(
            kgrid=(nk1, nk2, nk3),
            kgrid_mode=k_utils.KGRID_MODE.MONKHORST,
            kshift=(float(sk1), float(sk2), float(sk3)),
        )

    @cached_property
    def nk1(self) -> int | None:
        if self.nscf_in is not None and self.nscf_in.kpoints_card.nk1 is not None:
            return self.nscf_in.kpoints_card.nk1
        if self.pw_xml is not None:
            return self.pw_xml.nk1
        elif self.data_file_schema_xml is not None:
            logger.info("Parsing nk1 from data_file_schema.xml")
            return self.data_file_schema_xml.nk1
        return None

    @cached_property
    def nk2(self) -> int | None:
        if self.nscf_in is not None and self.nscf_in.kpoints_card.nk2 is not None:
            return self.nscf_in.kpoints_card.nk2
        if self.pw_xml is not None:
            return self.pw_xml.nk2
        if self.data_file_schema_xml is not None:
            logger.info("Parsing nk2 from data_file_schema.xml")
            return self.data_file_schema_xml.nk2
        return None

    @cached_property
    def nk3(self) -> int | None:
        if self.nscf_in is not None and self.nscf_in.kpoints_card.nk3 is not None:
            return self.nscf_in.kpoints_card.nk3
        if self.pw_xml is not None:
            return self.pw_xml.nk3
        if self.data_file_schema_xml is not None:
            logger.info("Parsing nk3 from data_file_schema.xml")
            return self.data_file_schema_xml.nk3
        return None

    @cached_property
    def sk1(self) -> int | None:
        if self.nscf_in is not None and self.nscf_in.kpoints_card.sk1 is not None:
            return self.nscf_in.kpoints_card.sk1
        if self.pw_xml is not None:
            return self.pw_xml.sk1
        elif self.data_file_schema_xml is not None:
            logger.info("Parsing sk1 from data_file_schema.xml")
            return self.data_file_schema_xml.sk1
        return None

    @cached_property
    def sk2(self) -> int | None:
        if self.nscf_in is not None and self.nscf_in.kpoints_card.sk2 is not None:
            return self.nscf_in.kpoints_card.sk2
        if self.pw_xml is not None:
            return self.pw_xml.sk2
        elif self.data_file_schema_xml is not None:
            logger.info("Parsing sk2 from data_file_schema.xml")
            return self.data_file_schema_xml.sk2
        return None

    @cached_property
    def sk3(self) -> int | None:
        if self.nscf_in is not None and self.nscf_in.kpoints_card.sk3 is not None:
            return self.nscf_in.kpoints_card.sk3
        if self.pw_xml is not None:
            return self.pw_xml.sk3
        if self.data_file_schema_xml is not None:
            logger.info("Parsing sk3 from data_file_schema.xml")
            return self.data_file_schema_xml.sk3
        return None

    @cached_property
    @override
    def reciprocal_lattice(self) -> npt.NDArray[np.float64] | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        reciprocal_lattice: np.ndarray | None = None
        if self.pw_xml is not None and self.pw_xml.reciprocal_lattice is not None:
            logger.info("Parsing reciprocal lattice from pw.xml")
            reciprocal_lattice = self.pw_xml.reciprocal_lattice
        elif (
            self.data_file_schema_xml is not None
            and self.data_file_schema_xml.reciprocal_lattice is not None
        ):
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

        if reciprocal_lattice is None:
            logger.warning(
                "No reciprocal lattice found in pw.xml or scf.out or bands.out or nscf.out"
            )
            return None

        if self.alat is None:
            logger.warning("Cannot compute reciprocal lattice without alat")
            return None

        return (2 * np.pi / self.alat) * reciprocal_lattice

    @cached_property
    def fermi(self) -> float | None:
        if self.scf_out is not None and self.scf_out.fermi_energy_ev is not None:
            logger.debug(f"Fermi energy found in {self.scf_out.fermi_energy_ev}")
            return self.scf_out.fermi_energy_ev
        if self.pw_xml is not None and self.pw_xml.fermi is not None:
            return self.pw_xml.fermi
        elif self.data_file_schema_xml is not None and self.data_file_schema_xml.fermi is not None:
            return self.data_file_schema_xml.fermi

        return None

    @cached_property
    def bands(self) -> np.ndarray | None:
        if self.atomic_proj_xml is not None and self.atomic_proj_xml.bands is not None:
            logger.info("Parsing bands from atomic_proj.xml")
            bands = self.atomic_proj_xml.bands
            logger.info("Parsing bands from atomic_proj.xml")
            bands = self.atomic_proj_xml.bands
        elif self.projwfc_out is not None and self.projwfc_out.bands is not None:
            logger.info("Parsing bands from projwfc.out")
            bands = HARTREE_TO_EV * self.projwfc_out.bands
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
    def spd_phase(self) -> np.ndarray | None:
        if self.atomic_proj_xml is None or self.projwfc_out is None:
            return None
        logger.info("Parsing spd phase from atomic_proj.xml and projwfc.out")

        wfc_mapping: dict[int, dict[str, Any]] = self.projwfc_out.wfc_mapping
        projections = self.atomic_proj_xml.projections
        orbitals = self.projwfc_out.orbitals

        if projections is None:
            return None

        n_kpoints = self.atomic_proj_xml.n_kpoints
        n_bands = self.atomic_proj_xml.n_bands
        n_spin_channels = self.atomic_proj_xml.n_spin_channels
        n_atoms = self.projwfc_out.n_atoms
        n_orbitals = self.projwfc_out.n_orbitals

        if n_atoms is None:
            return None

        pyprocar_projections_phase = np.zeros(
            shape=(n_kpoints, n_bands, n_spin_channels, n_atoms, n_orbitals),
            dtype=projections.dtype,
        )

        for state_num, wfc_info in wfc_mapping.items():
            atm_num: int = wfc_info["atm_num"]
            orbital_l: int = wfc_info["l"]
            j: float | None = wfc_info["j"]
            m_j: float | None = wfc_info["m_j"]
            m: int | None = wfc_info["m"]

            orbital_dict: dict[str, int | float | None] = (
                {"l": orbital_l, "j": j, "m_j": m_j}
                if m_j is not None
                else {"l": orbital_l, "m": m}
            )

            i_orbital = orbitals.index(orbital_dict)  # pyright: ignore[reportArgumentType]
            i_atom = atm_num - 1
            i_state = state_num - 1
            pyprocar_projections_phase[..., i_atom, i_orbital] += projections[..., i_state]

        if self.kpath is not None:
            pyprocar_projections_phase = insert_continuous_points(
                pyprocar_projections_phase, self.kticks
            )
        logger.debug(f"Spd Phase: {pyprocar_projections_phase.shape}")

        return pyprocar_projections_phase

    @cached_property
    def spd(self) -> np.ndarray | None:
        if self.spd_phase is None:
            return None
        logger.info("Parsing spd from spd phase")
        spd = np.absolute(self.spd_phase) ** 2

        n_kpoints = self.spd_phase.shape[0]
        if self.kpath is not None and n_kpoints != self.kpath.n_kpoints:
            spd = insert_continuous_points(spd, self.kticks)
        logger.debug(f"Spd: {spd.shape}")
        return spd

    @cached_property
    def orbitals(self) -> list[dict[str, int | float]] | None:
        if self.projwfc_out is not None:
            logger.info("Parsing orbitals from projwfc.out")
            result: list[dict[str, int | float]] = []
            for orb in self.projwfc_out.orbitals:
                result.append(dict(orb))
            return result
        else:
            logger.info("No orbitals found in projwfc.out")
            return None

    # -------- computed properties --------
    @cached_property
    @override
    def ebs(self) -> ElectronicBandStructure | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.fermi is None:
            user_logger.warning("Cannot create EBS without fermi energy")
            return None

        # TODO: orbitals is list[dict] but get_ebs_from_data expects list[str]
        # Need to convert quantum numbers to orbital names
        orbital_names: list[str] | None = None

        return get_ebs_from_data(
            kpoints=self.kpoints,
            bands=self.bands,
            projected=self.spd,
            projected_phase=self.spd_phase,
            fermi=self.fermi,
            reciprocal_lattice=self.reciprocal_lattice,
            orbital_names=orbital_names,
            structure=self.structure,
            kpath=self.kpath,
            kgrid_info=self.kgrid_info,
        )

    @cached_property
    def projected_dos(self) -> np.ndarray | None:
        if self.projwfc_dos is None or self.projwfc_out is None:
            return None

        n_energies = self.projwfc_dos.n_energies
        n_spin_channels = self.projwfc_dos.n_spin_channels
        n_orbitals = self.projwfc_out.n_orbitals
        n_atoms = self.projwfc_dos.n_atoms

        # Reshaping to match what pyprocar expects
        n_principals = 1
        projected_dos = (
            self.projwfc_dos.projected_dos
        )  # with shape (n_energies, n_spin_channels, n_atoms, n_orbitals)
        projected_dos = np.moveaxis(
            projected_dos, 1, -1
        )  # shape (n_energies, n_orbitals, n_atoms, n_spin_channels)
        projected_dos = np.moveaxis(
            projected_dos, 0, -1
        )  # shape (n_atoms, n_orbitals, n_spin_channels, n_energies)
        projected_dos = projected_dos.reshape(
            n_atoms, n_principals, n_orbitals, n_spin_channels, n_energies
        )
        logger.debug(f"projected_dos: {projected_dos.shape}")
        return projected_dos

    @cached_property
    def total_dos(self) -> np.ndarray | None:
        if self.projwfc_dos is None:
            return None
        total_dos = self.projwfc_dos.total_dos
        if total_dos is None:
            return None
        n_spin_channels = self.projwfc_dos.n_spin_channels
        n_energies = self.projwfc_dos.n_energies
        logger.debug(f"total_dos: {total_dos.shape}")
        return total_dos.reshape((n_spin_channels, n_energies), order="C")

    @cached_property
    def energies(self) -> np.ndarray | None:
        if self.projwfc_dos is None or self.fermi is None:
            return None
        return self.projwfc_dos.bands[0] - self.fermi

    @cached_property
    def is_dos_calculation(self) -> bool:
        logger.info("Checking if DOS calculation")
        if self.projwfc_in is None:
            return False
        is_dos_calculation = not self.projwfc_in.is_kresolved
        logger.info(f"Is DOS calculation: {is_dos_calculation}")
        return is_dos_calculation

    @cached_property
    @override
    def dos(self) -> DensityOfStates | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.projwfc_dos is None:
            user_logger.warning("No PDOS files found for DOS construction")
            return None

        if not self.is_dos_calculation:
            return None

        if self.energies is None or self.total_dos is None:
            return None

        logger.debug(f"energies: {self.energies.shape}")
        logger.debug(f"total_dos: {self.total_dos.shape}")
        logger.debug(f"fermi: {self.fermi}")

        fermi = self.fermi if self.fermi is not None else 0.0
        return DensityOfStates(
            energies=self.energies,
            total=self.total_dos,
            fermi=fermi,
            projected=self.projected_dos,
        )

    @cached_property
    def species(self) -> list[str] | None:
        if self.pw_xml is not None and self.pw_xml.atomic_species is not None:
            return self.pw_xml.atomic_species
        elif (
            self.data_file_schema_xml is not None
            and self.data_file_schema_xml.atomic_species is not None
        ):
            return self.data_file_schema_xml.atomic_species
        else:
            user_logger.warning("No atomic species found in any input or output file")
            return None

    @cached_property
    def direct_lattice(self) -> np.ndarray | None:
        if self.pw_xml is not None and self.pw_xml.direct_lattice is not None:
            return self.pw_xml.direct_lattice
        elif (
            self.data_file_schema_xml is not None
            and self.data_file_schema_xml.direct_lattice is not None
        ):
            return self.data_file_schema_xml.direct_lattice
        else:
            user_logger.warning("No direct lattice found in any input or output file")
            return None

    @cached_property
    def atomic_positions(self) -> np.ndarray | None:
        if self.pw_xml is not None and self.pw_xml.atomic_positions is not None:
            return self.pw_xml.atomic_positions
        elif (
            self.data_file_schema_xml is not None
            and self.data_file_schema_xml.atomic_positions is not None
        ):
            return self.data_file_schema_xml.atomic_positions
        else:
            user_logger.warning("No atomic positions found in any input or output file")
            return None

    @cached_property
    def rotations(self) -> np.ndarray | None:
        if self.pw_xml is not None and self.pw_xml.rotations is not None:
            return self.pw_xml.rotations
        elif (
            self.data_file_schema_xml is not None
            and self.data_file_schema_xml.rotations is not None
        ):
            return self.data_file_schema_xml.rotations
        else:
            user_logger.warning("No rotations found in any input or output file")
            return None

    @cached_property
    @override
    def structure(self) -> Structure | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return Structure(
            atoms=self.species,
            lattice=self.direct_lattice,
            fractional_coordinates=self.atomic_positions,
            rotations=self.rotations,
        )


def find_high_symmetry_ticks(
    raw_kpoints: npt.NDArray[np.float64] | None,
    high_sym_points: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]],
    atol: float = 1e-4,
) -> list[int]:
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
    dists = np.linalg.norm(raw_kpoints[:, None, :] - high_sym_points[None, :, :], axis=-1)

    kticks: list[int] = []
    last_idx = -1  # ensure we move forward along raw_kpoints

    for j in range(dists.shape[1]):
        # Find matches *after* the last matched index
        matches = np.where((dists[:, j] < atol) & (np.arange(len(raw_kpoints)) > last_idx))[0]
        if len(matches) > 0:
            idx = matches[0]  # first valid match
            kticks.append(idx)
            last_idx = idx
        else:
            raise ValueError(f"No match found for high_sym_point {j}: {high_sym_points[j]}")

    return kticks


def insert_continuous_points(arr: np.ndarray, tick_indices: list[int] | np.ndarray) -> np.ndarray:
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
    tick_indices_arr = np.asarray(tick_indices)

    # Continuous ticks are all except the very first one
    continuous_ticks = tick_indices_arr[1:-1]

    # Values to duplicate
    rows_to_insert = arr[continuous_ticks]

    # Insert them back at the right positions
    # np.insert shifts indices automatically, so we need to offset
    out = np.insert(arr, continuous_ticks + 1, rows_to_insert, axis=0)

    return out
