__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import re
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from pyprocar.core.atomic_orbital_index import (
    OrbitalIndexer,
)

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

FLOAT_PATTERN = r"[-+]?\d+(?:\.\d+)?"
COORDS_PATTERN = rf"\s*({FLOAT_PATTERN})\s*({FLOAT_PATTERN})\s*({FLOAT_PATTERN})\s*"

ORBITAL_ORDERING = OrbitalIndexer()


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


class ProjwfcOut:
    """Holds the projwfc/kpdos/pdos output for wfc mapping purposes."""

    @classmethod
    def is_file_of_type(cls, filepath: str | Path) -> bool:
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

    _filepath: Path
    _text: str

    def __init__(self, filepath: str | Path) -> None:
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
        pattern = re.compile(
            r"^\s*Program\s+PWSCF\s+v\.?\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", re.I | re.M
        )
        m = pattern.search(self.text)
        if not m:
            return None

        version_list = [int(g) for g in m.groups() if g is not None]
        if len(version_list) == 1:
            version_list.extend([0, 0])
        elif len(version_list) == 2:
            version_list.append(0)

        return (version_list[0], version_list[1], version_list[2])

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
            return atm_wfc["j"] is not None
        return False

    @cached_property
    def is_spin_polarized(self) -> bool:
        """Whether the calculation is spin polarized."""
        if self.kpoints is None or self.nkstot is None:
            return False
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
            re.VERBOSE | re.MULTILINE,
        )

        results = []
        for match in state_pattern.finditer(self.text):
            results.append(
                {
                    "state_num": int(match.group("state_num")),
                    "atom_num": int(match.group("atom_num")),
                    "element": match.group("element"),
                    "wfc_num": int(match.group("wfc_num")),
                    "l": int(match.group("l")),
                    "j": float(match.group("j")) if match.group("j") else None,
                    "m_j": float(match.group("mj")) if match.group("mj") else None,
                    "m": int(match.group("m")) if match.group("m") else None,
                }
            )

        return results

    @cached_property
    def header_numbers(self) -> dict[str, int]:
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
        numbers: dict[str, int] = {}
        for key, value in pattern.findall(self.text):
            # Prefer the first occurrence if duplicated
            numbers.setdefault(key, int(value))
        return numbers

    @cached_property
    def natomwfc(self) -> int | None:
        """Number of atomic wavefunctions reported in ``projwfc.out``."""
        return self.header_numbers.get("natomwfc")

    @cached_property
    def nx(self) -> int | None:
        """Auxiliary ``nx`` value reported in ``projwfc.out``."""
        return self.header_numbers.get("nx")

    @cached_property
    def nbnd(self) -> int | None:
        """Number of bands ``nbnd`` reported in ``projwfc.out``."""
        return self.header_numbers.get("nbnd")

    @cached_property
    def nkstot(self) -> int | None:
        """Total number of k-points ``nkstot`` reported in ``projwfc.out``."""
        return self.header_numbers.get("nkstot")

    @cached_property
    def npwx(self) -> int | None:
        """Plane-wave cutoff related ``npwx`` reported in ``projwfc.out``."""
        return self.header_numbers.get("npwx")

    @cached_property
    def nkb(self) -> int | None:
        """Number of projectors ``nkb`` reported in ``projwfc.out``."""
        return self.header_numbers.get("nkb")

    @cached_property
    def n_atoms(self) -> int | None:
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
                "m": atm_wfc["m"],
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
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            info["parallel_version"] = m.group(1).strip()
            info["n_cores"] = int(m.group(2))

        # Example: Number of MPI processes:               112
        m = re.search(
            r"^\s*Number\s+of\s+MPI\s+processes:\s*(\d+)", text, re.IGNORECASE | re.MULTILINE
        )
        if m:
            info["mpi_processes"] = int(m.group(1))

        # Example: Threads/MPI process:                     1
        m = re.search(r"^\s*Threads/MPI\s+process:\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["n_threads"] = int(m.group(1))

        # Example: MPI processes distributed on     1 nodes
        m = re.search(
            r"^\s*MPI\s+processes\s+distributed\s+on\s+(\d+)\s+nodes",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            info["n_nodes"] = int(m.group(1))

        # Example: K-points division:     npool     =       7
        m = re.search(
            r"^\s*K-points\s+division:\s*npool\s*=\s*(\d+)", text, re.IGNORECASE | re.MULTILINE
        )
        if m:
            info["n_pool"] = int(m.group(1))

        # Example (variants):
        #   R & G space division:  proc/nbgrp/npool/nimage =  16/  1/  8/  1
        #   R & G space division:  proc/nbgrp/npool/nimage =      16
        m = re.search(
            r"^\s*R\s*&\s*G\s*space\s*division:\s*proc/nbgrp/npool/nimage\s*=\s*([0-9\s/]+)",
            text,
            re.IGNORECASE | re.MULTILINE,
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
        after = text[header.end() :]

        # Each row has three numbers for sticks and three for G-vecs
        row_pattern = (
            r"^\s*(Min|Max|Sum)\s+"
            r"(\d+)\s+(\d+)\s+(\d+).*?"  # sticks: dense, smooth, pw
            r"(\d+)\s+(\d+)\s+(\d+)\s*$"  # g-vecs: dense, smooth, pw
        )
        row_re = re.compile(row_pattern, re.IGNORECASE | re.MULTILINE)

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
        return bool(
            re.search(r"^\s*Using\s+Slab\s+Decomposition", self.text, re.IGNORECASE | re.MULTILINE)
        )

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
        m = re.search(
            r"^\s*EXX\s+fraction\s+changed:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE | re.MULTILINE
        )
        if m:
            info["exx_fraction"] = float(m.group(1))

        # Example: EXX Screening parameter changed:    0.1060000
        m = re.search(
            r"^\s*EXX\s+Screening\s+parameter\s+changed:\s*([0-9]*\.?[0-9]+)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            info["exx_screening_param"] = float(m.group(1))

        # Example: EXX-fraction              =        0.25
        m = re.search(
            r"^\s*EXX-fraction\s*=\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE | re.MULTILINE
        )
        if m:
            info["xc_exx_fraction"] = float(m.group(1))

        # Example:
        #   Exchange-correlation= HSE
        #                         (   1   4  12   4   0   0   0)
        m = re.search(
            r"^\s*Exchange-correlation\s*=\s*([^\n]+)", text, re.IGNORECASE | re.MULTILINE
        )
        if m:
            info["xc_functional"] = m.group(1).strip()
            after = text[m.end() :]
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
    def kpoints(self) -> np.ndarray | None:
        """Parse k-points from projwfc.out."""
        kpoint_re = re.compile(rf"^\s*k\s*=\s*{COORDS_PATTERN}$", re.IGNORECASE | re.MULTILINE)
        k_re = kpoint_re.findall(self.text)
        if not k_re:
            return None

        kpoints = np.array(k_re, dtype=float)
        if self.nkstot is not None and self.kpoints_is_spin_polarized(kpoints):
            kpoints = kpoints[: self.nkstot // 2]

        return kpoints

    def kpoints_is_spin_polarized(self, kpoints: np.ndarray) -> bool:
        """Whether the kpoints are spin polarized."""
        n_kpoints = kpoints.shape[0]
        if n_kpoints % 2 != 0:
            return False

        kpoints_up = kpoints[: n_kpoints // 2]
        kpoints_down = kpoints[n_kpoints // 2 :]

        return bool(np.allclose(kpoints_up, kpoints_down))

    @cached_property
    def bands(self) -> np.ndarray | None:
        """Parse bands into shape (nkstot, nbnd, n_spin_channels)."""
        if self.nkstot is None or self.nbnd is None:
            return None

        band_re = re.compile(
            rf"^\s*====\s*e\(\s*\d+\s*\)\s*=\s*({FLOAT_PATTERN})\s*eV\s*====",
            re.IGNORECASE | re.MULTILINE,
        )
        bands_list = band_re.findall(self.text)
        if not bands_list:
            return None

        bands = np.array(bands_list, dtype=float).reshape(self.nkstot, self.nbnd)

        if self.is_spin_polarized:
            bands = bands.reshape(self.nkstot // 2, self.nbnd, self.n_spin_channels)
        else:
            bands = bands.reshape(self.nkstot, self.nbnd, self.n_spin_channels)

        return bands

    @cached_property
    def psi2(self) -> np.ndarray | None:
        """Parse psi2 into shape (nkstot, nbnd, n_spin_channels)."""
        if self.nkstot is None or self.nbnd is None:
            return None

        psi2_re = re.compile(
            rf"^\s*\|psi\|\^2\s*=\s*({FLOAT_PATTERN})\s*$", re.IGNORECASE | re.MULTILINE
        )
        psi2_list = psi2_re.findall(self.text)
        if not psi2_list:
            return None

        psi2 = np.array(psi2_list, dtype=float).reshape(self.nkstot, self.nbnd)

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
        if self.nkstot is None or self.nbnd is None or self.natomwfc is None:
            return None

        # Match each psi line and capture the whole coefficient list
        psi_line_re = re.compile(r"psi\s*=\s*([^\n\r]+)", re.IGNORECASE)

        # Match individual coefficient/index pairs
        coeff_re = re.compile(rf"({FLOAT_PATTERN})\s*\*\s*\[#\s*(\d+)\]")

        matches = psi_line_re.findall(self.text)
        if not matches:
            return None

        # Prepare output array
        psi_array = np.zeros((self.nkstot, self.nbnd, self.natomwfc), dtype=float)

        if len(matches) != self.nkstot * self.nbnd:
            msg = f"Expected {self.nkstot * self.nbnd} psi lines, found {len(matches)}"
            raise ValueError(msg)

        for ikb, line in enumerate(matches):
            coeffs = coeff_re.findall(line)
            for coeff_str, idx_str in coeffs:
                coeff_val = float(coeff_str)
                idx_val = int(idx_str) - 1  # convert to 0-based index
                psi_array[
                    ikb // self.nbnd,  # k-point index
                    ikb % self.nbnd,  # band index
                    idx_val,  # atomic wfc index
                ] = coeff_val

        if self.is_spin_polarized:
            psi_array = psi_array.reshape(
                self.nkstot // 2, self.nbnd, self.n_spin_channels, self.natomwfc
            )
        else:
            psi_array = psi_array.reshape(
                self.nkstot, self.nbnd, self.n_spin_channels, self.natomwfc
            )

        return psi_array

    @cached_property
    def lowdin_charges(self) -> dict[str, np.ndarray | float] | None:
        if self.n_atoms is None:
            return None

        n_atoms = self.n_atoms
        lowdin_per_orb = np.zeros((n_atoms, self.n_orbitals, self.n_spin_channels))
        lowdin_per_l = np.zeros((n_atoms, self.n_l_orbitals, self.n_spin_channels))
        total_charges_arr = np.zeros((n_atoms, self.n_l_orbitals))
        polarization_arr = np.zeros((n_atoms, self.n_l_orbitals))
        spilling_parameter: float | None = None

        # Extract the Lowdin Charges block
        block_re = re.compile(
            r"Lowdin Charges:(.*?)(?:Spilling Parameter:\s*([-+]?\d*\.\d+))", re.S | re.I
        )
        m = block_re.search(self.text)
        if not m:
            return None
        block_text, spill_str = m.groups()
        spilling_parameter = float(spill_str)

        # Split into lines and parse
        lines = [ln.strip() for ln in block_text.strip().splitlines() if ln.strip()]

        atom_idx = -1
        spin_channel = 0

        if not self.is_spin_polarized:
            # Non-spin-polarized parsing
            for line in lines:
                if line.startswith("Atom #"):
                    atom_match = re.search(r"Atom #\s+(\d+)", line)
                    if not atom_match:
                        continue
                    atom_idx = int(atom_match.group(1)) - 1
                    # detect which l-orbital this line is for
                    l_match = re.search(r",\s*([spdf])\s*=", line)
                    if not l_match:
                        continue
                    l_name = l_match.group(1)
                    l_idx = self.l_orbital_map[l_name]
                    total_match = re.search(r"total charge\s*=\s*([-+]?\d*\.\d+)", line)
                    if total_match:
                        total_val = float(total_match.group(1))
                        total_charges_arr[atom_idx, l_idx] = total_val
                    # parse all orbitals in this line
                    orb_pattern = r"([a-z0-9\-\+]+)\s*=\s*([-+]?\d*\.\d+)"
                    for orb, val in re.findall(orb_pattern, line):
                        orb = orb.lower()
                        if orb in self.orbital_map:
                            lowdin_per_orb[atom_idx, self.orbital_map[orb], 0] = float(val)
                        if orb in self.l_orbital_map:
                            lowdin_per_l[atom_idx, self.l_orbital_map[orb], 0] = float(val)

        else:
            # Spin-polarized parsing
            orb_pattern = r"([a-z0-9\-\+]+)\s*=\s*([-+]?\d*\.\d+)"
            for line in lines:
                if line.startswith("Atom #"):
                    atom_match = re.search(r"Atom #\s+(\d+)", line)
                    if not atom_match:
                        continue
                    atom_idx = int(atom_match.group(1)) - 1
                    # total charge line
                    for orb, val in re.findall(r"([spdf])\s*=\s*([-+]?\d*\.\d+)", line):
                        orb = orb.lower()
                        if orb in self.l_orbital_map:
                            total_charges_arr[atom_idx, self.l_orbital_map[orb]] = float(val)

                elif line.startswith("spin up"):
                    spin_channel = 0
                    for orb, val in re.findall(orb_pattern, line):
                        orb = orb.lower()
                        if orb in self.orbital_map:
                            lowdin_per_orb[atom_idx, self.orbital_map[orb], spin_channel] = float(
                                val
                            )
                        if orb in self.l_orbital_map:
                            lowdin_per_l[atom_idx, self.l_orbital_map[orb], spin_channel] = float(
                                val
                            )

                elif line.startswith("spin down"):
                    spin_channel = 1
                    for orb, val in re.findall(orb_pattern, line):
                        orb = orb.lower()
                        if orb in self.orbital_map:
                            lowdin_per_orb[atom_idx, self.orbital_map[orb], spin_channel] = float(
                                val
                            )
                        if orb in self.l_orbital_map:
                            lowdin_per_l[atom_idx, self.l_orbital_map[orb], spin_channel] = float(
                                val
                            )

                elif line.startswith("polarization"):
                    for orb, val in re.findall(r"([spdf])\s*=\s*([-+]?\d*\.\d+)", line):
                        orb = orb.lower()
                        if orb in self.l_orbital_map:
                            polarization_arr[atom_idx, self.l_orbital_map[orb]] = float(val)

        return {
            "lowdin_charges_per_orbital": lowdin_per_orb,
            "lowdin_charges_per_l_orbital": lowdin_per_l,
            "total_charges": total_charges_arr,
            "polarization": polarization_arr,
            "spilling_parameter": spilling_parameter,
        }

    @cached_property
    def lowdin_charges_per_orbital(self) -> np.ndarray | None:
        if not self.lowdin_charges:
            return None

        result = self.lowdin_charges["lowdin_charges_per_orbital"]
        return result if isinstance(result, np.ndarray) else None

    @cached_property
    def lowdin_charges_per_l_orbital(self) -> np.ndarray | None:
        if not self.lowdin_charges:
            return None

        result = self.lowdin_charges["lowdin_charges_per_l_orbital"]
        return result if isinstance(result, np.ndarray) else None

    @cached_property
    def total_lowdin_charges(self) -> np.ndarray | None:
        if not self.lowdin_charges:
            return None

        result = self.lowdin_charges["total_charges"]
        return result if isinstance(result, np.ndarray) else None

    @cached_property
    def lowdin_polarization(self) -> np.ndarray | None:
        if not self.lowdin_charges:
            return None

        result = self.lowdin_charges["polarization"]
        return result if isinstance(result, np.ndarray) else None

    @cached_property
    def spilling_parameter(self) -> float | None:
        if not self.lowdin_charges:
            return None

        result = self.lowdin_charges["spilling_parameter"]
        return result if isinstance(result, float) else None
