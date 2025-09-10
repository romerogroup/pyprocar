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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyvista import row_array
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure
from pyprocar.io.qe.utils import parse_qe_input_cards
from pyprocar.utils import np_utils
from pyprocar.utils.info import OrbitalOrdering
from pyprocar.utils.units import AU_TO_ANG, HARTREE_TO_EV, RYDBERG_TO_EV

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

FLOAT_PATTERN = r"[-+]?\d+(?:\.\d+)?"
COORDS_PATTERN = rf"\s*({FLOAT_PATTERN})\s*({FLOAT_PATTERN})\s*({FLOAT_PATTERN})\s*"

ORBITAL_ORDERING = OrbitalOrdering()



def convert_lorbnum_to_letter(lorbnum):
        """A helper method to convert the lorb number to the letter format

        Parameters
        ----------
        lorbnum : int
            The number of the l orbital

        Returns
        -------
        str
            The l orbital name
        """
        lorb_mapping = {0: "s", 1: "p", 2: "d", 3: "f"}
        return lorb_mapping[lorbnum]
    
    
class ProjwfcIn:
    """Holds the projwfc input file."""
    
    @classmethod
    def is_file_of_type(cls, filepath: Union[str, Path]) -> bool:
        """Quickly determine if an input file looks like a projwfc.x input.

        Checks the beginning of the file for a 'projwfc' token (case-insensitive)
        in comments or the file body. Also accepts presence of typical projwfc
        variables like 'outdir' and 'prefix' in a minimal input.
        """
        try:
            p = Path(filepath)
            with p.open("r", errors="ignore") as f:
                head_lines = [f.readline() for _ in range(50)]
            head = "".join(head_lines)
            if not head:
                return False
            if re.search(r"projwfc", head, re.IGNORECASE):
                return True
            return False
        except Exception:
            return False

    _filepath: Optional[Path]
    _text: Optional[str]
    
    def __init__(self, filepath: Union[str, Path]) -> None:
        self._filepath = Path(filepath)
        self._text = self._read()
        
    def _read(self) -> str:
        with open(self.filepath, "r") as f:
            text = f.read()
        return text
    
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @property
    def text(self) -> str:
        return self._text
    
    @cached_property
    def data(self) -> dict[str, Any]:
        data = parse_qe_input_cards(self.text)
        logger.info(f"PROJWFC INPUT: {data}")
        return data
    
    @cached_property
    def is_kresolved(self) -> bool:
        if "kresolveddos" in self.data:
            return self.data["kresolveddos"] == True
        else:
            return False
    
    

class ProjwfcOut:
    """Holds the projwfc/kpdos/pdos output for wfc mapping purposes."""

    @classmethod
    def is_file_of_type(cls, filepath: Union[str, Path]) -> bool:
        """Quickly determine if a .out file belongs to PROJWFC.

        Reads up to the first 5 lines and checks for 'PROJWFC' (case-insensitive).
        """
        try:
            p = Path(filepath)
            with p.open("r", errors="ignore") as f:
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    if re.search(r"PROJWFC", line, re.IGNORECASE):
                        return True
        except Exception:
            pass
        return False

    _filepath: Optional[Path]
    _text: Optional[str]
    
    def __init__(self, filepath: Union[str, Path]) -> None:
        self._filepath = Path(filepath)
        self._text = self._filepath.read_text()
    
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @property
    def text(self) -> str:
        return self._text
    
    @cached_property
    def version_tuple(self) -> tuple[int, int, int] | None:
        pattern = re.compile(r"^\s*Program\s+PWSCF\s+v\.?\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", re.I | re.M)
        m = pattern.search(self.text)
        if m:
            version_tuple = list(int(g) for g in m.groups() if g is not None)
            if len(version_tuple) == 2:
                version_tuple.append(0)
            version_tuple = tuple(version_tuple)
        else:
            version_tuple = None
            
        return version_tuple
    
    @cached_property
    def version(self) -> str | None:
        if self.version_tuple:
            return f"{self.version_tuple[0]}.{self.version_tuple[1]}.{self.version_tuple[2]}"
        else:
            return None
    
    @cached_property
    def is_non_colinear(self) -> bool:
        if self.atm_wfcs:
            atm_wfc = self.atm_wfcs[0]
            return atm_wfc['j'] is not None
        return False
    
    @cached_property
    def is_spin_polarized(self) -> bool:
        """Whether the calculation is spin polarized."""
        n_kpoints = self.kpoints.shape[0]
        return n_kpoints != self.nkstot
        
    @cached_property
    def n_spin_channels(self) -> int:
        """Number of spin channels."""
        if self.is_spin_polarized:
            return 2
        else:
            return 1
        
    @cached_property
    def l_orbital_map(self) -> dict[str, int]:
        return ORBITAL_ORDERING.l_orbital_map
    
    @cached_property
    def orbital_map(self) -> dict[str, int]:
        return ORBITAL_ORDERING.az_to_flat_index
    
    @cached_property
    def n_l_orbitals(self) -> int:
        return len(self.l_orbital_map)
    
    @cached_property
    def n_atm_wfcs(self) -> int:
        return len(self.atm_wfcs)
    
    @cached_property
    def atm_wfcs(self):

        state_pattern = re.compile(
            r"""^\s*state\s+\#\s*(?P<state_num>\d+):\s*
                atom\s+(?P<atom_num>\d+)\s*\((?P<element>[A-Za-z]+)\s*\),\s*
                wfc\s+(?P<wfc_num>\d+)\s*
                \(
                l\s*=\s*(?P<l>\d+)
                (?:\s+j\s*=\s*(?P<j>[-+]?\d+(?:\.\d+)?))?
                (?:\s+m_j\s*=\s*(?P<mj>[-+]?\d+(?:\.\d+)?))?
                (?:\s+m\s*=\s*(?P<m>[-+]?\d+(?:\.\d+)?))?
                \)
            """,
            re.VERBOSE | re.MULTILINE
        )
        
        results = []
        for match in state_pattern.finditer(self.text):
            results.append({
                "state_num": int(match.group("state_num")),
                "atom_num": int(match.group("atom_num")),
                "element": match.group("element"),
                "wfc_num": int(match.group("wfc_num")),
                "l": int(match.group("l")),
                "j": float(match.group("j")) if match.group("j") else None,
                "m_j": float(match.group("mj")) if match.group("mj") else None,
                "m": int(match.group("m")) if match.group("m") else None
            })

        return results

    @cached_property
    def header_numbers(self) -> Dict[str, int]:
        """Parse key integer parameters from projwfc output header.

        Extracts values for the following keys when present: ``natomwfc``,
        ``nx``, ``nbnd``, ``nkstot``, ``npwx``, ``nkb``.

        Returns
        -------
        dict[str, int]
            Mapping from key name to parsed integer value.
        """
        pattern = re.compile(
            r"^\s*(natomwfc|nx|nbnd|nkstot|npwx|nkb)\s*=\s*(\d+)\s*$",
            re.MULTILINE,
        )
        numbers: Dict[str, int] = {}
        for key, value in pattern.findall(self.text):
            # Prefer the first occurrence if duplicated
            numbers.setdefault(key, int(value))
        return numbers

    @cached_property
    def natomwfc(self) -> Optional[int]:
        """Number of atomic wavefunctions reported in ``projwfc.out``."""
        return self.header_numbers.get("natomwfc")

    @cached_property
    def nx(self) -> Optional[int]:
        """Auxiliary ``nx`` value reported in ``projwfc.out``."""
        return self.header_numbers.get("nx")

    @cached_property
    def nbnd(self) -> Optional[int]:
        """Number of bands ``nbnd`` reported in ``projwfc.out``."""
        return self.header_numbers.get("nbnd")

    @cached_property
    def nkstot(self) -> Optional[int]:
        """Total number of k-points ``nkstot`` reported in ``projwfc.out``."""
        return self.header_numbers.get("nkstot")

    @cached_property
    def npwx(self) -> Optional[int]:
        """Plane-wave cutoff related ``npwx`` reported in ``projwfc.out``."""
        return self.header_numbers.get("npwx")

    @cached_property
    def nkb(self) -> Optional[int]:
        """Number of projectors ``nkb`` reported in ``projwfc.out``."""
        return self.header_numbers.get("nkb")
    
    @cached_property
    def n_atoms(self) -> Optional[int]:
        """Number of atoms ``nat`` reported in ``projwfc.out``."""
        n_atoms = 0
        for atm_wfc in self.atm_wfcs:
            n_atoms = max(n_atoms, atm_wfc["atom_num"])
        return n_atoms

    @cached_property
    def non_colinear_orbitals(self):
        return ORBITAL_ORDERING.flat_soc_order
    
    @cached_property
    def colinear_orbitals(self):
        return ORBITAL_ORDERING.az_to_lm_records
    
    @cached_property
    def orbitals(self):
        if self.is_non_colinear:
            return self.non_colinear_orbitals
        else:
            return self.colinear_orbitals
        
    @cached_property
    def n_orbitals(self) -> int:
        return len(self.orbitals)
    
    @cached_property
    def wfc_mapping(self):
        wfc_mapping = {}
        for atm_wfc in self.atm_wfcs:
            wfc_mapping[atm_wfc["state_num"]] = {
                "element": atm_wfc["element"],
                "wfc_num": atm_wfc["wfc_num"],
                "atm_num": atm_wfc["atom_num"],
                "l": atm_wfc["l"],
                "j": atm_wfc["j"],
                "m_j": atm_wfc["m_j"],
                "m": atm_wfc["m"]
            }

        return wfc_mapping
    
    @cached_property
    def _parallel_info(self) -> dict[str, Any]:
        """Parse parallel execution information from ``projwfc.out``."""
        info: dict[str, Any] = {
            "parallel_version": None,
            "n_cores": None,
            "mpi_processes": None,
            "n_threads": None,
            "n_nodes": None,
            "n_pool": None,
            "proc_nbgrp_npool_nimage": None,
            "available_mem": None,
        }
        text = self.text

        # Example: Parallel version (MPI & OpenMP), running on     112 processor cores
        m = re.search(
            r"^\s*Parallel\s+version\s*\(\s*([^)]+?)\s*\)\s*,\s*running\s+on\s+(\d+).*",
            text, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            info["parallel_version"] = m.group(1).strip()
            info["n_cores"] = int(m.group(2))

        # Example: Number of MPI processes:               112
        m = re.search(r"^\s*Number\s+of\s+MPI\s+processes:\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["mpi_processes"] = int(m.group(1))

        # Example: Threads/MPI process:                     1
        m = re.search(r"^\s*Threads/MPI\s+process:\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["n_threads"] = int(m.group(1))

        # Example: MPI processes distributed on     1 nodes
        m = re.search(r"^\s*MPI\s+processes\s+distributed\s+on\s+(\d+)\s+nodes", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["n_nodes"] = int(m.group(1))

        # Example: K-points division:     npool     =       7
        m = re.search(r"^\s*K-points\s+division:\s*npool\s*=\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["n_pool"] = int(m.group(1))

        # Example (variants):
        #   R & G space division:  proc/nbgrp/npool/nimage =  16/  1/  8/  1
        #   R & G space division:  proc/nbgrp/npool/nimage =      16
        m = re.search(
            r"^\s*R\s*&\s*G\s*space\s*division:\s*proc/nbgrp/npool/nimage\s*=\s*([0-9\s/]+)",
            text, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
            info["proc_nbgrp_npool_nimage"] = nums  # length can be 1 or 4

        # Example: 469343 MiB available memory ...
        m = re.search(r"^\s*(\d+)\s*MiB\s+available\s+memory", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["available_mem"] = int(m.group(1))

        return info
    
    @cached_property
    def parallel_version(self) -> str | None:
        """Parallel build description, e.g., 'MPI & OpenMP'."""
        return self._parallel_info["parallel_version"]

    @cached_property
    def n_cores(self) -> int | None:
        """Number of processor cores reported."""
        return self._parallel_info["n_cores"]

    @cached_property
    def mpi_processes(self) -> int | None:
        """Number of MPI processes."""
        return self._parallel_info["mpi_processes"]

    @cached_property
    def n_threads(self) -> int | None:
        """Threads per MPI process."""
        return self._parallel_info["n_threads"]

    @cached_property
    def n_nodes(self) -> int | None:
        """Number of nodes used."""
        return self._parallel_info["n_nodes"]

    @cached_property
    def n_pool(self) -> int | None:
        """npool for k-point division."""
        return self._parallel_info["n_pool"]

    @cached_property
    def proc_nbgrp_npool_nimage(self) -> list[int] | None:
        """proc/nbgrp/npool/nimage numbers as a list (length 1 or 4)."""
        return self._parallel_info["proc_nbgrp_npool_nimage"]


    @cached_property
    def _parallelization_details(self) -> dict[str, Any] | None:
        """Parse 'Parallelization info' table from QE outputs.

        Returns
        ----
        dict or None
            A dict with keys 'sticks' and 'gvecs', each mapping to a dict
            of per-channel 'dense'/'smooth'/'pw' stats, with 'min'/'max'/'sum'.

            Example:
            {
              "sticks": {
                "dense": {"min": 48, "max": 49, "sum": 969},
                "smooth": {"min": 24, "max": 25, "sum": 483},
                "pw": {"min": 8, "max": 9, "sum": 173},
              },
              "gvecs": {
                "dense": {"min": 956, "max": 958, "sum": 19141},
                "smooth": {"min": 340, "max": 343, "sum": 6819},
                "pw": {"min": 73, "max": 77, "sum": 1505},
              },
            }
        """
        text = self.text
        # Anchor on the section header
        header = re.search(r"^\s*Parallelization\s+info\s*$", text, re.IGNORECASE | re.MULTILINE)
        if not header:
            return None

        # Collect the next few lines to find Min/Max/Sum rows
        after = text[header.end():]

        # Each row has three numbers for sticks and three for G-vecs
        row_re = re.compile(
            r"^\s*(Min|Max|Sum)\s+"
            r"(\d+)\s+(\d+)\s+(\d+).*?"        # sticks: dense, smooth, pw
            r"(\d+)\s+(\d+)\s+(\d+)\s*$",      # g-vecs: dense, smooth, pw
            re.IGNORECASE | re.MULTILINE,
        )

        stats: dict[str, dict[str, dict[str, int]]] = {
            "sticks": {"dense": {}, "smooth": {}, "pw": {}},
            "gvecs": {"dense": {}, "smooth": {}, "pw": {}},
        }

        found = False
        for m in row_re.finditer(after):
            found = True
            label = m.group(1).lower()  # 'min' | 'max' | 'sum'
            s_dense, s_smooth, s_pw = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            g_dense, g_smooth, g_pw = (int(m.group(5)), int(m.group(6)), int(m.group(7)))

            stats["sticks"]["dense"][label] = s_dense
            stats["sticks"]["smooth"][label] = s_smooth
            stats["sticks"]["pw"][label] = s_pw

            stats["gvecs"]["dense"][label] = g_dense
            stats["gvecs"]["smooth"][label] = g_smooth
            stats["gvecs"]["pw"][label] = g_pw

        return stats if found else None

    @cached_property
    def parallelization_table(self) -> dict[str, Any] | None:
        """Structured 'Parallelization info' stats if present."""
        return self._parallelization_details

    @cached_property
    def sticks_min(self) -> tuple[int, int, int] | None:
        """Min sticks counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["sticks"]["dense"].get("min"),
            d["sticks"]["smooth"].get("min"),
            d["sticks"]["pw"].get("min"),
        )

    @cached_property
    def sticks_max(self) -> tuple[int, int, int] | None:
        """Max sticks counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["sticks"]["dense"].get("max"),
            d["sticks"]["smooth"].get("max"),
            d["sticks"]["pw"].get("max"),
        )

    @cached_property
    def sticks_sum(self) -> tuple[int, int, int] | None:
        """Sum sticks counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["sticks"]["dense"].get("sum"),
            d["sticks"]["smooth"].get("sum"),
            d["sticks"]["pw"].get("sum"),
        )

    @cached_property
    def gvecs_min(self) -> tuple[int, int, int] | None:
        """Min G-vecs counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["gvecs"]["dense"].get("min"),
            d["gvecs"]["smooth"].get("min"),
            d["gvecs"]["pw"].get("min"),
        )

    @cached_property
    def gvecs_max(self) -> tuple[int, int, int] | None:
        """Max G-vecs counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["gvecs"]["dense"].get("max"),
            d["gvecs"]["smooth"].get("max"),
            d["gvecs"]["pw"].get("max"),
        )

    @cached_property
    def gvecs_sum(self) -> tuple[int, int, int] | None:
        """Sum G-vecs counts as (dense, smooth, pw)."""
        d = self._parallelization_details
        if not d:
            return None
        return (
            d["gvecs"]["dense"].get("sum"),
            d["gvecs"]["smooth"].get("sum"),
            d["gvecs"]["pw"].get("sum"),
        )

    @cached_property
    def using_slab_decomposition(self) -> bool:
        """Whether 'Using Slab Decomposition' appears in the output."""
        return bool(re.search(r"^\s*Using\s+Slab\s+Decomposition", self.text, re.IGNORECASE | re.MULTILINE))

    @cached_property
    def negative_core_charge(self) -> float | None:
        """Parsed value from 'Check: negative core charge= ...' if present."""
        m = re.search(
            r"negative\s+core\s+charge\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
            self.text,
            re.IGNORECASE,
        )
        return float(m.group(1)) if m else None

    @cached_property
    def gaussian_broadening(self) -> dict[str, Any] | None:
        """Gaussian broadening parameters read from input, if present.

        Returns
        ----
        dict or None
            {'ngauss': int, 'degauss': float} when available.
        """
        m = re.search(
            r"Gaussian\s+broadening.*?ngauss,degauss=\s*([+-]?\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
            self.text,
            re.IGNORECASE,
        )
        if not m:
            return None
        return {"ngauss": int(m.group(1)), "degauss": float(m.group(2))}

    @cached_property
    def ngauss(self) -> int | None:
        """Convenience accessor for Gaussian broadening 'ngauss'."""
        gb = self.gaussian_broadening
        return gb["ngauss"] if gb else None

    @cached_property
    def degauss(self) -> float | None:
        """Convenience accessor for Gaussian broadening 'degauss'."""
        gb = self.gaussian_broadening
        return gb["degauss"] if gb else None

    @cached_property
    def _xc_info(self) -> dict[str, Any]:
        """Parse XC/HSE-related info from ``projwfc.out``."""
        info: dict[str, Any] = {
            "exx_fraction": None,
            "exx_screening_param": None,
            "xc_functional": None,
            "xc_functional_params": None,
            "xc_exx_fraction": None,
        }
        text = self.text

        # Example: EXX fraction changed:   0.25
        m = re.search(r"^\s*EXX\s+fraction\s+changed:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["exx_fraction"] = float(m.group(1))

        # Example: EXX Screening parameter changed:    0.1060000
        m = re.search(
            r"^\s*EXX\s+Screening\s+parameter\s+changed:\s*([0-9]*\.?[0-9]+)",
            text, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            info["exx_screening_param"] = float(m.group(1))

        # Example: EXX-fraction              =        0.25
        m = re.search(r"^\s*EXX-fraction\s*=\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["xc_exx_fraction"] = float(m.group(1))

        # Example:
        #   Exchange-correlation= HSE
        #                         (   1   4  12   4   0   0   0)
        m = re.search(r"^\s*Exchange-correlation\s*=\s*([^\n]+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["xc_functional"] = m.group(1).strip()
            after = text[m.end():]
            mp = re.search(r"^\s*\(\s*([0-9\s]+)\s*\)\s*$", after, re.MULTILINE)
            if mp:
                params = [int(x) for x in mp.group(1).split()]
                info["xc_functional_params"] = params

        return info

    @cached_property
    def exx_fraction(self) -> float | None:
        """'EXX fraction changed' value."""
        return self._xc_info["exx_fraction"]

    @cached_property
    def exx_screening_param(self) -> float | None:
        """'EXX Screening parameter changed' value."""
        return self._xc_info["exx_screening_param"]

    @cached_property
    def xc_functional(self) -> str | None:
        """XC functional name, e.g., 'HSE'."""
        return self._xc_info["xc_functional"]

    @cached_property
    def xc_functional_params(self) -> list[int] | None:
        """XC functional integer parameters list."""
        return self._xc_info["xc_functional_params"]

    @cached_property
    def xc_exx_fraction(self) -> float | None:
        """'EXX-fraction =' value inside the XC functional block."""
        return self._xc_info["xc_exx_fraction"]
    
    @cached_property
    def kpoints(self) -> int:
        """Number of spin channels."""
        kpoint_re = re.compile(rf"^\s*k\s*=\s*{COORDS_PATTERN}$", re.IGNORECASE | re.MULTILINE)
        k_re = kpoint_re.findall(self.text)
        if not k_re:
            return None
        
        kpoints = np.array(k_re, dtype=float)
        if self.kpoints_is_spin_polarized(kpoints):
            kpoints = kpoints[:self.nkstot//2]

        return kpoints
    
    def kpoints_is_spin_polarized(self, kpoints: np.ndarray) -> bool:
        """Whether the kpoints are spin polarized."""
        n_kpoints = kpoints.shape[0]
        if n_kpoints % 2 != 0:
            return False
        
        kpoints_up = kpoints[:n_kpoints//2]
        kpoints_down = kpoints[n_kpoints//2:]
        
        if np.allclose(kpoints_up, kpoints_down):
            return True
        else:
            return False
        
    @cached_property
    def bands(self) -> np.ndarray | None:
        """Parse bands into shape (nkstot, nbnd, n_spin_channels)."""
        band_re = re.compile(
            rf"^\s*====\s*e\(\s*\d+\s*\)\s*=\s*({FLOAT_PATTERN})\s*eV\s*====",
            re.IGNORECASE | re.MULTILINE
        )
        bands = band_re.findall(self.text)
        if not bands:
            return None
        
        bands = np.array(bands, dtype=float).reshape(self.nkstot, self.nbnd)
        
        if self.is_spin_polarized:
            bands = bands.reshape(self.nkstot // 2, self.nbnd, self.n_spin_channels)
        else:
            bands = bands.reshape(self.nkstot, self.nbnd, self.n_spin_channels)
            
        return bands
        
    @cached_property
    def psi2(self) -> np.ndarray | None:
        """Parse psi2 into shape (nkstot, nbnd, n_spin_channels)."""
        psi2_re = re.compile(
            rf"^\s*\|psi\|\^2\s*=\s*({FLOAT_PATTERN})\s*$",
            re.IGNORECASE | re.MULTILINE
        )
        psi2 = psi2_re.findall(self.text)
        if not psi2:
            return None
        
        psi2 = np.array(psi2, dtype=float).reshape(self.nkstot, self.nbnd)
        
        if self.is_spin_polarized:
            psi2 = psi2.reshape(self.nkstot // 2, self.nbnd, self.n_spin_channels)
        else:
            psi2 = psi2.reshape(self.nkstot, self.nbnd, self.n_spin_channels)
        
        return psi2
        
        
    @cached_property
    def psi_coeffs(self) -> np.ndarray | None:
        """
        Parse psi coefficients into shape (nkstot, nbnd, n_spin_channels, natomwfc).
        Missing coefficients are filled with 0.0.
        """
        # Match each psi line and capture the whole coefficient list
        psi_line_re = re.compile(
            r"psi\s*=\s*([^\n\r]+)",
            re.IGNORECASE
        )
        
        # Match individual coefficient/index pairs
        coeff_re = re.compile(
            rf"({FLOAT_PATTERN})\s*\*\s*\[#\s*(\d+)\]"
        )

        matches = psi_line_re.findall(self.text)
        if not matches:
            return None

        # Prepare output array
        psi_array = np.zeros((self.nkstot, self.nbnd, self.natomwfc), dtype=float)

        if len(matches) != self.nkstot * self.nbnd:
            raise ValueError(
                f"Expected {self.nkstot * self.nbnd} psi lines, found {len(matches)}"
            )

        for ikb, line in enumerate(matches):
            coeffs = coeff_re.findall(line)
            for coeff_str, idx_str in coeffs:
                coeff_val = float(coeff_str)
                idx_val = int(idx_str) - 1  # convert to 0-based index
                psi_array[
                    ikb // self.nbnd,  # k-point index
                    ikb % self.nbnd,   # band index
                    idx_val            # atomic wfc index
                ] = coeff_val

        if self.is_spin_polarized:
            psi_array = psi_array.reshape(self.nkstot // 2, self.nbnd, self.n_spin_channels, self.natomwfc)
        else:
            psi_array = psi_array.reshape(self.nkstot, self.nbnd, self.n_spin_channels, self.natomwfc)
            
        return psi_array
    
    @cached_property
    def lowdin_charges(self):
        lowdin_per_orb = np.zeros((self.n_atoms, self.n_orbitals, self.n_spin_channels))
        lowdin_per_l = np.zeros((self.n_atoms, self.n_l_orbitals, self.n_spin_channels))
        total_charges = np.zeros((self.n_atoms, self.n_l_orbitals))
        polarization = np.zeros((self.n_atoms, self.n_l_orbitals))
        spilling_parameter = None

        # Extract the Lowdin Charges block
        block_re = re.compile(
            r"Lowdin Charges:(.*?)(?:Spilling Parameter:\s*([-+]?\d*\.\d+))",
            re.S | re.I
        )
        m = block_re.search(self.text)
        if not m:
            return None
        block_text, spill_str = m.groups()
        spilling_parameter = float(spill_str)

        # Split into lines and parse
        lines = [l.strip() for l in block_text.strip().splitlines() if l.strip()]

        atom_idx = -1
        spin_channel = 0

        if not self.is_spin_polarized:
            # Non-spin-polarized parsing
            for line in lines:
                if line.startswith("Atom #"):
                    atom_idx = int(re.search(r"Atom #\s+(\d+)", line).group(1)) - 1
                    # detect which l-orbital this line is for
                    l_match = re.search(r",\s*([spdf])\s*=", line)
                    if not l_match:
                        continue
                    l_name = l_match.group(1)
                    l_idx = self.l_orbital_map[l_name]
                    total_val = float(re.search(r"total charge\s*=\s*([-+]?\d*\.\d+)", line).group(1))
                    total_charges[atom_idx, l_idx] = total_val
                    # parse all orbitals in this line
                    for orb, val in re.findall(r"([a-z0-9\-\+]+)\s*=\s*([-+]?\d*\.\d+)", line):
                        orb = orb.lower()
                        if orb in self.orbital_map:
                            lowdin_per_orb[atom_idx, self.orbital_map[orb], 0] = float(val)
                        if orb in self.l_orbital_map:
                            lowdin_per_l[atom_idx, self.l_orbital_map[orb], 0] = float(val)

        else:
            # Spin-polarized parsing
            for line in lines:
                if line.startswith("Atom #"):
                    atom_idx = int(re.search(r"Atom #\s+(\d+)", line).group(1)) - 1
                    # total charge line
                    for orb, val in re.findall(r"([spdf])\s*=\s*([-+]?\d*\.\d+)", line):
                        orb = orb.lower()
                        if orb in self.l_orbital_map:
                            total_charges[atom_idx, self.l_orbital_map[orb]] = float(val)

                elif line.startswith("spin up"):
                    spin_channel = 0
                    for orb, val in re.findall(r"([a-z0-9\-\+]+)\s*=\s*([-+]?\d*\.\d+)", line):
                        orb = orb.lower()
                        if orb in self.orbital_map:
                            lowdin_per_orb[atom_idx, self.orbital_map[orb], spin_channel] = float(val)
                        if orb in self.l_orbital_map:
                            lowdin_per_l[atom_idx, self.l_orbital_map[orb], spin_channel] = float(val)

                elif line.startswith("spin down"):
                    spin_channel = 1
                    for orb, val in re.findall(r"([a-z0-9\-\+]+)\s*=\s*([-+]?\d*\.\d+)", line):
                        orb = orb.lower()
                        if orb in self.orbital_map:
                            lowdin_per_orb[atom_idx, self.orbital_map[orb], spin_channel] = float(val)
                        if orb in self.l_orbital_map:
                            lowdin_per_l[atom_idx, self.l_orbital_map[orb], spin_channel] = float(val)

                elif line.startswith("polarization"):
                    for orb, val in re.findall(r"([spdf])\s*=\s*([-+]?\d*\.\d+)", line):
                        orb = orb.lower()
                        if orb in self.l_orbital_map:
                            polarization[atom_idx, self.l_orbital_map[orb]] = float(val)

        return {
            "lowdin_charges_per_orbital": lowdin_per_orb,
            "lowdin_charges_per_l_orbital": lowdin_per_l,
            "total_charges": total_charges,
            "polarization": polarization,
            "spilling_parameter": spilling_parameter
        }
        
    @cached_property
    def lowdin_charges_per_orbital(self) -> np.ndarray | None:
        if not self.lowdin_charges:
            return None
        
        return self.lowdin_charges["lowdin_charges_per_orbital"]
    
    @cached_property
    def lowdin_charges_per_l_orbital(self) -> np.ndarray | None:
        if not self.lowdin_charges:
            return None
        
        return self.lowdin_charges["lowdin_charges_per_l_orbital"]
    
    @cached_property
    def total_charges(self) -> np.ndarray | None:
        if not self.lowdin_charges:
            return None
        
        return self.lowdin_charges["total_charges"]
    
    @cached_property
    def polarization(self) -> np.ndarray | None:
        if not self.lowdin_charges:
            return None
        
        return self.lowdin_charges["polarization"]
    
    @cached_property
    def spilling_parameter(self) -> float | None:
        if not self.lowdin_charges:
            return None
        
        return self.lowdin_charges["spilling_parameter"]
    
    


class AtomicProjXML:
    def __init__(self, filepath: Union[str, Path]) -> None:
        self._filepath = Path(filepath)
        self._tree = ET.parse(self._filepath)
        self._root = self._tree.getroot()

    
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @cached_property
    def root(self) -> ET.Element:
        return self._root
    
    @cached_property
    def tree(self) -> ET.ElementTree:
        return self._tree
    
    @cached_property
    def n_bands(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0
        
        return int(header_match[0].get("NUMBER_OF_BANDS"))
    
    @cached_property
    def n_kpoints(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0
        
        return int(header_match[0].get("NUMBER_OF_K-POINTS"))
    
    @cached_property
    def n_spin_channels(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0
        
        return int(header_match[0].get("NUMBER_OF_SPIN_COMPONENTS"))
    
    @cached_property
    def n_atm_wfc(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0
        
        return int(header_match[0].get("NUMBER_OF_ATOMIC_WFC"))
    
    @cached_property
    def n_electrons(self) -> int:
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0
        
        return int(header_match[0].get("NUMBER_OF_ELECTRONS"))
    
    @cached_property
    def is_noncolinear(self) -> bool:
        header_match = self.root.findall(".//ATOMIC_SIGMA_PHI")
        if header_match:
            return True
        return False
    
    @cached_property
    def fermi(self) -> float:
        """Fermi energy in eV"""
        header_match = self.root.findall(".//HEADER")
        if not header_match:
            return 0
        
        return float(header_match[0].get("FERMI_ENERGY")) * RYDBERG_TO_EV
    
    @cached_property
    def n_spin_projections(self) -> int:
        if self.is_noncolinear:
            return 4
        else:
            return self.n_spin_channels
        
    @cached_property
    def eigen_states(self) -> dict[str, Any]:
        """
        Parses the atomic_proj.xml file and returns the bands, projections, kpoints, and weights.
         - Energies are in Rydberg.
         - Bands are subtracted by the Fermi energy.
        Returns:
            dict[str, Any]: A dictionary containing the bands, projections, kpoints, and weights.
                - bands: np.ndarray of shape (n_kpoints, n_bands, n_spin_channels) # In Rydberg
                - projections: np.ndarray of shape (n_kpoints, n_bands, n_spin_projections, n_atm_wfc)
                - kpoints: np.ndarray of shape (n_kpoints, 3)
                - weights: np.ndarray of shape (n_kpoints)
        """
        eigen_states = {"bands": [], "projections": [], "kpoints": [], "weights": []}
        
        eigen_states_match = self.root.findall(".//EIGENSTATES")
        if not eigen_states_match:
            return None
        
        eigen_state_element = eigen_states_match[0]
        
        kpoints_match = eigen_state_element.findall(".//K-POINT")
        bands_by_kpoint = eigen_state_element.findall(".//E")
        projections_by_kpoint = eigen_state_element.findall(".//PROJS")

        n_all_kpoints = len(bands_by_kpoint)
        raw_kpoints = []
        raw_weights = []
        raw_bands = []
        raw_projections = []
        logger.debug(f"Parsing kpoints: {n_all_kpoints}")
        logger.debug(f"Parsing projections: {len(projections_by_kpoint)}")
        for i_kpoint in range(n_all_kpoints):
            band_element = bands_by_kpoint[i_kpoint]
            raw_bands.append(band_element.text.strip().split())
            
            kpoint_element = kpoints_match[i_kpoint]
            weight = float(kpoint_element.attrib["Weight"])
            raw_weights.append(weight)
            
            kpoint = np.array(kpoint_element.text.strip().split(), dtype=float)
            raw_kpoints.append(kpoint)
            
        
            atomic_wfc_projections = projections_by_kpoint[i_kpoint].findall(".//ATOMIC_WFC")
            if not atomic_wfc_projections:
                continue
            projection = np.zeros(shape=(self.n_bands, self.n_atm_wfc), dtype=np_utils.COMPLEX_DTYPE)
            for atomic_wfc_projection in atomic_wfc_projections:
                i_atm_wfc = int(atomic_wfc_projection.attrib["index"]) - 1
                i_spin_projection = int(atomic_wfc_projection.attrib["spin"]) - 1

                atomic_band_projections = atomic_wfc_projection.text.strip().split("\n")
                
                for i_band, atomic_band_projection in enumerate(atomic_band_projections):
                    real, imag = atomic_band_projection.strip().split()
                    
                    projection[i_band, i_atm_wfc] += complex(float(real), float(imag))
            raw_projections.append(projection)
                
        raw_bands = np.array(raw_bands, dtype=float)
        raw_projections = np.array(raw_projections, dtype=np_utils.COMPLEX_DTYPE)
        raw_kpoints = np.array(raw_kpoints, dtype=float)
        raw_weights = np.array(raw_weights, dtype=float)
        
        logger.debug(f"raw_bands: {raw_bands.shape}")
        logger.debug(f"raw_projections: {raw_projections.shape}")
        logger.debug(f"raw_kpoints: {raw_kpoints.shape}")
        logger.debug(f"raw_weights: {raw_weights.shape}")
        
        bands = np.zeros(shape=(self.n_kpoints, self.n_bands, self.n_spin_channels), dtype=float)
        projections = np.zeros(shape=(self.n_kpoints, self.n_bands, self.n_spin_projections, self.n_atm_wfc), dtype=np_utils.COMPLEX_DTYPE)
        
        
        if self.n_spin_channels == 2:
            kpoints = raw_kpoints[:self.n_kpoints]
            weights = raw_weights[:self.n_kpoints]
            bands[..., 0] = raw_bands[:self.n_kpoints]
            bands[..., 1] = raw_bands[self.n_kpoints:]
            
            logger.debug(f"raw_projections spin-up: {raw_projections[:self.n_kpoints].shape}")
            logger.debug(f"raw_projections spin-down: {raw_projections[self.n_kpoints:].shape}")
            projections[:, :, 0, :] = raw_projections[:self.n_kpoints]
            projections[:, :, 1, :] = raw_projections[self.n_kpoints:]
        else:
            kpoints = raw_kpoints
            weights = raw_weights
            bands[..., 0] = raw_bands
            projections[:, :, 0, :] = raw_projections
            
            
        logger.info(f"bands: {bands.shape}")
        logger.info(f"projections: {projections.shape}")
        logger.info(f"kpoints: {kpoints.shape}")
        logger.info(f"weights: {weights.shape}")
        
  
        eigen_states["bands"] = bands
        eigen_states["projections"] = projections
        eigen_states["weights"] = weights
        eigen_states["kpoints"] = kpoints
        
        return eigen_states
    
    @cached_property
    def kpoints(self) -> np.ndarray | None:
        if self.eigen_states:
            return self.eigen_states["kpoints"]
        else:
            return None
        
    @cached_property
    def bands(self) -> np.ndarray | None:
        """"""
        if self.eigen_states:
            bands = self.eigen_states["bands"] * RYDBERG_TO_EV
            return bands
        else:
            return None
        
    @cached_property
    def projections(self) -> np.ndarray | None:
        if self.eigen_states:
            return self.eigen_states["projections"]
        else:
            return None
        
    @cached_property
    def weights(self) -> np.ndarray | None:
        if self.eigen_states:
            return self.eigen_states["weights"]
        else:
            return None
class ProjwfcPDOSFile:
    
    FILE_PATTERN = re.compile(
    r"pdos_atm#(\d+)\(([^)]+)\)_wfc#(\d+)\(([^)_]+)(?:_j([\d.]+))?\)"
    )

    def __init__(self, filepath: Union[str, Path]) -> None:
        self._filepath = Path(filepath)
        self._text = self._read()
        
    def _read(self) -> str:
        with open(self._filepath, "r") as f:
            return f.read()
        
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @property
    def text(self) -> str:
        return self._text
    
    def _first_data_line(self) -> str | None:
        for line in self.text.splitlines():
            if line.strip() and not line.strip().startswith("#"):
                return line
        return None
    
    @cached_property
    def filename_info(self) -> dict[str, Any]:
        m = self.FILE_PATTERN.search(self._filepath.name)
        if not m:
            return {}
        return {
            "atom_index": int(m.group(1)),
            "atom_symbol": m.group(2),
            "wfc_index": int(m.group(3)),
            "orbital": m.group(4),
            "j_value": float(m.group(5)) if m.group(5) else None,
        }
        
    @cached_property
    def lines(self) -> list[str]:
        return self.text.splitlines()
    
    @cached_property
    def columns(self) -> list[str]:
        """
        Extracts column names from the header line starting with '#'
        """
        header_line = self.lines[0]
        columns = header_line.strip("# ").split()
        if columns[1] == "(eV)":
            str_val=columns.pop(1)
            # columns[0] = columns[0] + str_val
        return columns
    
    @property
    def data(self) -> pd.DataFrame:
        cols = self.columns
        data_lines = []
        for line in self.lines[1:]:
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = line.split()
            try:
                row = [int(parts[0])] + [float(x.replace("E", "e")) for x in parts[1:]] \
                    if cols and cols[0].lower() == "ik" else [float(x.replace("E", "e")) for x in parts]
                data_lines.append(row)
            except ValueError:
                continue
        
        data_array = np.array(data_lines)
        logger.debug(f"data_array: {data_array.shape}")
        # Adjust header length to match actual data length
        if data_lines:
            n_data_cols = len(data_lines[0])
            if len(cols) > n_data_cols:
                cols = cols[:n_data_cols]
            elif len(cols) < n_data_cols:
                # Add generic names for missing columns
                extra_cols = [f"col{i}" for i in range(len(cols) + 1, n_data_cols + 1)]
                cols = cols + extra_cols

        return pd.DataFrame(data_lines, columns=cols)
    

class ProjwfcDOS:
    """
    Parser for a collection of projwfc.x PDOS files.
    Can be initialized with:
      - A list of file paths
      - A directory path (auto-detects PDOS files)
    """

    FILE_PATTERN = re.compile(
        r"pdos_atm#(\d+)\(([^)]+)\)_wfc#(\d+)\(([^)]+)\)(?:_j([\d.]+))?"
    )

    def __init__(self, paths: Union[str, Path, List[Union[str, Path]]]) -> None:
        self.total_dos_path = None  # store .pdos_tot file path if found

        if isinstance(paths, (str, Path)):
            p = Path(paths)
            if p.is_dir():
                # Find all pdos_atm files
                self.filepaths = sorted(
                    f for f in p.iterdir() if self.FILE_PATTERN.search(f.name)
                )
                # Look for .pdos_tot file
                tot_files = list(p.glob("*.pdos_tot"))
                if tot_files:
                    self.total_dos_path = tot_files[0]
            elif p.is_file():
                self.filepaths = [p]
            else:
                raise ValueError(f"Invalid path: {paths}")
        elif isinstance(paths, list):
            self.filepaths = [Path(fp) for fp in paths]
        else:
            raise TypeError("paths must be a directory path, file path, or list of file paths")


    def _parse_filename(self, filename: str) -> dict[str, Any]:
        m = self.FILE_PATTERN.search(filename)
        if not m:
            return {}
        return {
            "atom_index": int(m.group(1)),
            "atom_symbol": m.group(2),
            "wfc_index": int(m.group(3)),
            "orbital": m.group(4),
            "j_value": float(m.group(5)) if m.group(5) else None,
        }

    @cached_property
    def files_metadata(self) -> list[dict[str, Any]]:
        files_metadata = []
        for fp in self.pdos_files:
            files_metadata.append(fp.filename_info)
        return files_metadata
    
    @cached_property
    def n_atoms(self) -> int:
        n_atoms = 0
        for file_metadata in self.files_metadata:
            if file_metadata["atom_index"] is not None:
                n_atoms = max(n_atoms, file_metadata["atom_index"])
        return n_atoms
    
    @cached_property
    def total_dos_filepath(self) -> Path | None:
        if not self.total_dos_path:
            return None
        return ProjwfcPDOSFile(self.total_dos_path)
    
    @cached_property
    def total_dos(self) -> Optional[pd.DataFrame]:
        """
        Parse the total DOS from the .pdos_tot file if available.
        Returns a pandas DataFrame or None if no file found.
        """
        if not self.total_dos_path:
            return None
        
        dos_array = np.zeros((self.n_energies, self.n_spin_channels))
        

        df = self.total_dos_filepath.data
        # Determine spin channels
        if self.is_spin_polarized:
            # Spin-polarized: use dosup(E) and dosdw(E) columns
            dos_up = df["dosup(E)"].to_numpy()
            dos_down = df["dosdw(E)"].to_numpy()
            dos_array = np.hstack((dos_up, dos_down))  # shape (n_energies, 2)
        else:
            # Non-spin-polarized: only one DOS column
            dos_total = df["dos(E)"].to_numpy()
            dos_array = dos_total[:, np.newaxis]  # shape (n_energies, 1)

        return dos_array

    @cached_property
    def pdos_files(self) -> list[ProjwfcPDOSFile]:
        return [ProjwfcPDOSFile(filepath) for filepath in self.filepaths]
            
    @cached_property
    def data(self) -> list[dict[str, Any]]:
        dataframes = [pdos_file.data for pdos_file in self.pdos_files]
        dataframes = self._regularize_projections(dataframes)
        return dataframes
    
    @cached_property
    def is_non_colinear(self) -> bool:
        return self.files_metadata[0]["j_value"] is not None
    
    
    @cached_property
    def is_spin_polarized(self) -> bool:
        for column_name in self.pdos_files[0].columns:
            if "pdosup" in column_name.lower():
                return True
        return False
    
    @cached_property
    def n_spin_channels(self) -> int:
        if self.is_spin_polarized:
            return 2
        else:
            return 1
        
    @cached_property
    def is_kresolved(self) -> bool:
        df = self.data[0]  # first file's DataFrame

        if "ik" in df.columns:
            return True
        else:
            return False

    # -------------------------
    # Public access: energies
    # -------------------------
    @cached_property
    def bands(self) -> np.ndarray:
        """
        Returns the energy grid in shape (n_kpoints, n_energies).
        - If k-resolved: n_kpoints > 1
        - If not k-resolved: n_kpoints == 1
        """
        df = self.data[0]  # first file's DataFrame
        
        
        if "ik" in df.columns:
            # K-resolved case
            ik_values = df["ik"].unique()
            n_kpoints = len(ik_values)

            # Extract energies for each k-point
            bands = []
            for ik in ik_values:
                e_k = df[df["ik"] == ik]["E"].to_numpy()
                bands.append(e_k)
            bands = np.array(bands)

            # Validation check across all files
            for ifile, df_other in enumerate(self.data[1:], start=1):
                for i, ik in enumerate(ik_values):
                    e_k_other = df_other[df_other["ik"] == ik]["E"].to_numpy()
                    if not np.allclose(e_k_other, bands[i]):
                        raise ValueError(
                            f"Energies differ in file {self.files_metadata[ifile]['filepath']} "
                            f"for k-point {ik}"
                        )
            return bands

        else:
            ref_energies = df["E"].to_numpy()
            
            # Validation check across all files
            for ifile, df_other in enumerate(self.data[1:], start=1):
                e_other = df_other["E"].to_numpy()
                if e_other.shape != ref_energies.shape:
                    raise ValueError(f"Energies differ in file {self.pdos_files[ifile].filepath}")
                
            return ref_energies[np.newaxis, :] # shape (1, n_energies)

    
    @cached_property
    def energies(self) -> np.ndarray:
        return self.bands
    
    @cached_property
    def n_energies(self) -> int:
        return self.bands.shape[1]
    
    
    def _regularize_projections(self, dfs: list[pd.DataFrame]):
        ref_energies = dfs[0]["E"].to_numpy()
        for ifile, df_other in enumerate(dfs[1:], start=1):
            df_values = df_other.to_numpy()
            new_df_values = np.zeros((ref_energies.shape[0], df_values.shape[1]))
            
            energies= df_values[:, 0]
            scalar_values = df_values[:, 1:]
            
            new_df_values[:, 0] = ref_energies
            for i in range(1,scalar_values.shape[1]):
                new_df_values[:, i] = np.interp(ref_energies, energies, scalar_values[:, i])
            
            
            dfs[ifile] = pd.DataFrame(new_df_values, columns=df_other.columns)
        return dfs
        
    # -------------------------
    # Cached property: projected_dos
    # -------------------------
    @cached_property
    def projected_dos(self) -> np.ndarray:
        """
        Returns PDOS array with shape:
            (n_energies, n_spin_channels, n_atoms, n_orbitals)
        Orbitals are ordered according to self.orbitals (from ORBITAL_ORDERING).
        """
        atom_indices = sorted(set(f["atom_index"] for f in self.files_metadata))
        n_atoms = len(atom_indices)
        n_orbitals = self.n_orbitals

        # Initialize array
        dos_array = np.zeros((self.n_energies, self.n_spin_channels, n_atoms, n_orbitals))

        # Group files by atom
        atom_to_files: Dict[int, List[dict]] = {a: [] for a in atom_indices}
        for f in self.files_metadata:
            atom_to_files[f["atom_index"]].append(f)

        for ai, atom in enumerate(atom_indices):
            for f in atom_to_files[atom]:
                file_index = self.files_metadata.index(f)
                df = self.data[file_index]

                if self.is_non_colinear:
                    # SOC case: multiple m-components in one file
                    l = ORBITAL_ORDERING.l_orbital_map[f["orbital"][0].lower()]
                    j = f["j_value"]
                    n_m = df.shape[1] - 2  # E, LDOS, then PDOS_m...
                    pdos_m = df.iloc[:, 2:].to_numpy()
                    m_values = np.linspace(-j, j, int(2 * j + 1))
                    for mi, m in enumerate(m_values):
                        target_idx = ORBITAL_ORDERING.get_soc_index(l, j, m)
                        dos_array[:, 0, ai, target_idx] = pdos_m[:, mi]

                else:
                    # Collinear case
                    l = ORBITAL_ORDERING.l_orbital_map[f["orbital"][0].lower()]

                    if f["orbital"] in ("s", "p", "d", "f"):
                        # Multiple m-components in one file
                        n_pdos_cols = df.shape[1] - 2
                        if self.is_spin_polarized and self.n_spin_channels == 2:
                            n_m = n_pdos_cols // 2
                        else:
                            n_m = n_pdos_cols

                        pdos_m = df.iloc[:, 2:].to_numpy()
                        orb_names_for_l = ORBITAL_ORDERING.azimuthal_order[
                            list(ORBITAL_ORDERING.l_orbital_map.keys())[l]
                        ]
                        n_m_expected = len(orb_names_for_l)

                        if n_m != n_m_expected:
                            raise ValueError(
                                f"Mismatch: orbital {f['metadata']['orbital']} has {n_m} m-components in file "
                                f"but {n_m_expected} in ORBITAL_ORDERING"
                        )

                        for mi, orb_name in enumerate(orb_names_for_l):
                            target_idx = ORBITAL_ORDERING.az_to_flat_index[orb_name]
                            if self.is_spin_polarized and self.n_spin_channels == 2:
                                dos_array[:, 0, ai, target_idx] = pdos_m[:, mi * 2]    # spin-up
                                dos_array[:, 1, ai, target_idx] =  pdos_m[:, mi * 2 + 1] # spin-down
                            else:
                                dos_array[:, 0, ai, target_idx] = pdos_m[:, mi]

                    else:
                        # Single m-component per file
                        m_idx = ORBITAL_ORDERING.az_to_flat_index[f["orbital"]]
                        target_idx = m_idx
                        if self.is_spin_polarized and self.n_spin_channels == 2:
                            dos_array[:, 0, ai, target_idx] = df.iloc[:, -2].to_numpy()
                            dos_array[:, 1, ai, target_idx] = df.iloc[:, -1].to_numpy()
                        else:
                            dos_array[:, 0, ai, target_idx] = df.iloc[:, -1].to_numpy()

        return dos_array

    def _l_from_orbital(self, orb: str) -> int:
        """Map orbital letter to l quantum number."""
        return ORBITAL_ORDERING.l_orbital_map[orb[0].lower()]

    def _m_from_orbital_name(self, orb: str) -> int:
        """Map orbital name to m index for colinear case."""
        return ORBITAL_ORDERING.az_to_flat_index[orb]

    @cached_property
    def colinear_orbitals(self):
        return ORBITAL_ORDERING.az_to_lm_records

    @cached_property
    def non_colinear_orbitals(self):
        return ORBITAL_ORDERING.flat_soc_order
    
    @cached_property
    def orbitals(self):
        if self.is_non_colinear:
            return self.non_colinear_orbitals
        else:
            return self.colinear_orbitals
        
    @cached_property
    def n_orbitals(self) -> int:
        return len(self.orbitals)
