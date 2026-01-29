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
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


class PwOut:
    """Lightweight parser for the QE scf.out file."""

    @classmethod
    def is_file_of_type(cls, filepath: str | Path) -> bool:
        """Quickly determine if a .out file belongs to PWSCF.

        Reads up to the first 5 lines and checks for 'PWSCF' (case-insensitive).
        """
        try:
            p = Path(filepath)
            with p.open("r", errors="ignore") as f:
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    if re.search(r"PWSCF", line, re.IGNORECASE):
                        return True
        except Exception:
            pass
        return False

    def __init__(self, filepath: str | Path) -> None:
        self._filepath: Path = Path(filepath)
        self._text: str = self._read()

    def _read(self) -> str:
        with open(self.filepath) as f:
            return f.read()

    @property
    def filepath(self) -> Path:
        return self._filepath

    @cached_property
    def text(self) -> str:
        return self._text

    @cached_property
    def version_tuple(self) -> tuple[int, int, int] | None:
        pattern = re.compile(
            r"^\s*Program\s+PROJWFC\s+v\.?\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", re.I | re.M
        )
        m = pattern.search(self.text)
        if m:
            version_list = [int(g) for g in m.groups() if g is not None]
            if len(version_list) == 2:
                version_list.append(0)
            return (version_list[0], version_list[1], version_list[2])
        return None

    @cached_property
    def version(self) -> str | None:
        if self.version_tuple:
            return f"{self.version_tuple[0]}.{self.version_tuple[1]}.{self.version_tuple[2]}"
        else:
            return None

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
        row_re = re.compile(
            r"^\s*(Min|Max|Sum)\s+"
            + r"(\d+)\s+(\d+)\s+(\d+).*?"  # sticks: dense, smooth, pw
            + r"(\d+)\s+(\d+)\s+(\d+)\s*$",  # g-vecs: dense, smooth, pw
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
        return bool(
            re.search(r"^\s*Using\s+Slab\s+Decomposition", self.text, re.IGNORECASE | re.MULTILINE)
        )

    # -------------------------
    # System info
    # -------------------------
    @cached_property
    def _system_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        text = self.text

        patterns = {
            "bravais_index": r"bravais-lattice index\s*=\s*(\d+)",
            "alat": r"lattice parameter \(alat\)\s*=\s*([\d.]+)",
            "cell_volume": r"unit-cell volume\s*=\s*([\d.]+)",
            "natoms": r"number of atoms/cell\s*=\s*(\d+)",
            "ntyp": r"number of atomic types\s*=\s*(\d+)",
            "nelectrons": r"number of electrons\s*=\s*([\d.]+)",
            "nkohn_sham": r"number of Kohn-Sham states\s*=\s*(\d+)",
            "ecutwfc": r"kinetic-energy cutoff\s*=\s*([\d.]+)",
            "ecutrho": r"charge density cutoff\s*=\s*([\d.]+)",
            "conv_thr": r"scf convergence threshold\s*=\s*([\d.Ee+-]+)",
            "mixing_beta": r"mixing beta\s*=\s*([\d.]+)",
            "n_scf_steps": r"number of iterations used\s*=\s*(\d+)",
        }

        for key, pat in patterns.items():
            m = re.search(pat, text, re.I)
            if m:
                val = m.group(1)
                try:
                    val = int(val)
                except ValueError:
                    val = float(val)
                info[key] = val
            else:
                info[key] = None

        return info

    @cached_property
    def bravais_index(self) -> int | None:
        return self._system_info["bravais_index"]

    @cached_property
    def alat(self) -> float | None:
        return self._system_info["alat"]

    @cached_property
    def cell_volume(self) -> float | None:
        return self._system_info["cell_volume"]

    @cached_property
    def natoms(self) -> int | None:
        return self._system_info["natoms"]

    @cached_property
    def ntyp(self) -> int | None:
        return self._system_info["ntyp"]

    @cached_property
    def nelectrons(self) -> float | None:
        return self._system_info["nelectrons"]

    @cached_property
    def nkohn_sham(self) -> int | None:
        return self._system_info["nkohn_sham"]

    @cached_property
    def ecutwfc(self) -> float | None:
        return self._system_info["ecutwfc"]

    @cached_property
    def ecutrho(self) -> float | None:
        return self._system_info["ecutrho"]

    @cached_property
    def conv_thr(self) -> float | None:
        return self._system_info["conv_thr"]

    @cached_property
    def mixing_beta(self) -> float | None:
        return self._system_info["mixing_beta"]

    @cached_property
    def n_scf_steps(self) -> int | None:
        return self._system_info["n_scf_steps"]

    # -------------------------
    # Exchange-correlation
    # -------------------------
    @cached_property
    def exchange_correlation(self) -> dict[str, Any] | None:
        """
        Returns:
            {
                "functional": "SLA  PW   PBX  PBC",
                "params": [1, 4, 3, 4, 0, 0, 0]
            }
        """
        m = re.search(
            r"Exchange-correlation=\s*(.+?)\n\s*\(([\d\s]+)\)",
            self.text,
            re.I,
        )
        if m:
            func = m.group(1).strip()
            params = [int(x) for x in m.group(2).split()]
            return {"functional": func, "params": params}
        return None

    # -------------------------
    # celldm values
    # -------------------------
    @cached_property
    def celldm(self) -> dict[str, float] | None:
        """
        Returns:
            {"celldm1": 7.26885, "celldm2": 0.0, ..., "celldm6": 0.0}
        """
        m = re.search(
            r"celldm\(1\)=\s*([\d.]+)\s*celldm\(2\)=\s*([\d.]+)\s*celldm\(3\)=\s*([\d.]+)\s*"
            + r"celldm\(4\)=\s*([\d.]+)\s*celldm\(5\)=\s*([\d.]+)\s*celldm\(6\)=\s*([\d.]+)",
            self.text,
            re.I,
        )
        if m:
            vals = [float(x) for x in m.groups()]
            return {f"celldm{i + 1}": vals[i] for i in range(6)}
        return None

    # -------------------------
    # Crystal axes
    # -------------------------
    @cached_property
    def crystal_axes(self) -> NDArray[np.floating[Any]] | None:
        """
        Returns:
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]]
        """
        pattern = (
            r"crystal axes:.*?\n"
            r"\s*a\(1\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)\s*\n"
            r"\s*a\(2\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)\s*\n"
            r"\s*a\(3\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)"
        )
        m = re.search(pattern, self.text, re.I | re.S)
        if m:
            nums = [float(x) for x in m.groups()]
            return np.array([nums[0:3], nums[3:6], nums[6:9]], dtype=float)
        return None

    # -------------------------
    # Reciprocal axes
    # -------------------------
    @cached_property
    def reciprocal_axes(self) -> NDArray[np.floating[Any]] | None:
        pattern = (
            r"reciprocal axes:.*?\n"
            r"\s*b\(1\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)\s*\n"
            r"\s*b\(2\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)\s*\n"
            r"\s*b\(3\)\s*=\s*\(\s*([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s*\)"
        )
        m = re.search(pattern, self.text, re.I | re.S)
        if m:
            nums = [float(x) for x in m.groups()]
            return np.array([nums[0:3], nums[3:6], nums[6:9]], dtype=float)
        return None

    # -------------------------
    # Pseudopotential details
    # -------------------------
    @cached_property
    def pseudopotentials(self) -> list[dict[str, Any]] | None:
        """
        Returns a list of dicts with pseudopotential details.
        """
        pseudos: list[dict[str, Any]] = []
        pattern = re.compile(
            r"PseudoPot\.\s*#\s*(\d+)\s*for\s+(\S+)\s+read from file:\s*(\S+)\s*"
            + r"MD5 check sum:\s*([a-f0-9]+)\s*"
            + r"Pseudo is\s*(.+?),\s*Zval\s*=\s*([\d.]+)\s*"
            + r"Generated using\s*(.+?)\s*"
            + r"Shape of augmentation charge:\s*(\S+)\s*"
            + r"Using radial grid of\s*(\d+)\s*points,\s*(\d+)\s*beta functions.*?:\s*"
            + r"((?:\s*l\(\d+\)\s*=\s*\d+\s*)+)"
            + r"Q\(r\)\s*pseudized",
            re.I | re.S,
        )
        for m in pattern.finditer(self.text):
            l_block = m.group(11)
            l_values = [int(x) for x in re.findall(r"l\(\d+\)\s*=\s*(\d+)", l_block)]
            pseudos.append(
                {
                    "index": int(m.group(1)),
                    "symbol": m.group(2),
                    "file": m.group(3),
                    "md5": m.group(4),
                    "description": m.group(5).strip(),
                    "z_val": float(m.group(6)),
                    "generated_using": m.group(7).strip(),
                    "augmentation_shape": m.group(8),
                    "radial_grid_points": int(m.group(9)),
                    "n_beta": int(m.group(10)),
                    "l_values": l_values,
                }
            )
        return pseudos or None

    # -------------------------
    # Atomic species
    # -------------------------
    @cached_property
    def atomic_species(self) -> list[dict[str, Any]] | None:
        """
        Parses the 'atomic species' table.
        Returns:
            [
                {'symbol': 'Sr', 'valence': 10.0, 'mass': 87.62, 'pseudo': 'Sr( 1.00)'},
                {'symbol': 'V', 'valence': 13.0, 'mass': 50.9415, 'pseudo': 'V ( 1.00)'},
                {'symbol': 'O', 'valence': 6.0, 'mass': 15.9994, 'pseudo': 'O ( 1.00)'}
            ]
        """
        species: list[dict[str, Any]] = []
        m = re.search(
            r"atomic species\s+valence\s+mass\s+pseudopotential\n(.*?)\n\s*\n",
            self.text,
            re.S | re.I,
        )
        if m:
            for line in m.group(1).strip().splitlines():
                match = re.match(r"^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+(.+?)\s*$", line)
                if match:
                    species.append(
                        {
                            "symbol": match.group(1),
                            "valence": float(match.group(2)),
                            "mass": float(match.group(3)),
                            "pseudo": match.group(4),
                        }
                    )
        return species or None

    # -------------------------
    # Atomic positions
    # -------------------------
    @cached_property
    def atomic_sites(self) -> list[dict[str, Any]] | None:
        """
        Parses the atomic positions block.
        Returns:
            [
                {'site': 1, 'symbol': 'Sr', 'tau_index': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                ...
            ]
        """
        atoms: list[dict[str, Any]] = []
        # Match lines like:
        #  1           Sr  tau(   1) = (   0.0000000   0.0000000   0.0000000  )
        pattern = re.compile(
            r"^\s*(\d+)\s+(\S+)\s+tau\(\s*(\d+)\s*\)\s*=\s*\(\s*([-\d.Ee]+)\s+([-\d.Ee]+)\s+([-\d.Ee]+)\s*\)",
            re.M,
        )
        for m in pattern.finditer(self.text):
            atoms.append(
                {
                    "site": int(m.group(1)),
                    "symbol": m.group(2),
                    "tau_index": int(m.group(3)),
                    "x": float(m.group(4)),
                    "y": float(m.group(5)),
                    "z": float(m.group(6)),
                }
            )
        return atoms or None

    @cached_property
    def n_atoms(self) -> int | None:
        return self._system_info["nat"]

    @cached_property
    def atomic_positions(self) -> NDArray[np.floating[Any]] | None:
        if self.n_atoms is None or self.atomic_sites is None:
            return None

        atomic_position = np.zeros((self.n_atoms, 3), dtype=float)

        for atom in self.atomic_sites:
            atomic_position[atom["site"] - 1] = [atom["x"], atom["y"], atom["z"]]
        return atomic_position

    @cached_property
    def atomic_symbols(self) -> list[str] | None:
        if self.atomic_sites is None:
            return None
        atomic_symbols: list[str] = []
        for atom in self.atomic_sites:
            atomic_symbols.append(str(atom["symbol"]))
        return atomic_symbols

    @cached_property
    def _kpoints_fft_mem_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "n_kpoints": None,
            "smearing_type": None,
            "smearing_width_ry": None,
            "dense_gvecs": None,
            "dense_fft_dims": None,
            "smooth_gvecs": None,
            "smooth_fft_dims": None,
            "max_ram_per_proc_mb": None,
            "total_ram_mb": None,
            "negative_core_charge": None,
            "starting_charge": None,
            "renormalized_charge": None,
            "n_starting_wfcs": None,
            "cpu_time_so_far_s": None,
        }
        text = self.text

        # number of k points and smearing
        m = re.search(
            r"number of k points=\s*(\d+)\s+(\w+)\s+smearing,\s*width\s*\(Ry\)=\s*([\d.]+)",
            text,
            re.I,
        )
        if m:
            info["n_kpoints"] = int(m.group(1))
            info["smearing_type"] = m.group(2)
            info["smearing_width_ry"] = float(m.group(3))

        # dense grid
        dense_pattern = (
            r"Dense\s+grid:\s*(\d+)\s+G-vectors\s+"
            r"FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)"
        )
        m = re.search(dense_pattern, text, re.I)
        if m:
            info["dense_gvecs"] = int(m.group(1))
            info["dense_fft_dims"] = [int(m.group(2)), int(m.group(3)), int(m.group(4))]

        # smooth grid
        smooth_pattern = (
            r"Smooth\s+grid:\s*(\d+)\s+G-vectors\s+"
            r"FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)"
        )
        m = re.search(smooth_pattern, text, re.I)
        if m:
            info["smooth_gvecs"] = int(m.group(1))
            info["smooth_fft_dims"] = [int(m.group(2)), int(m.group(3)), int(m.group(4))]

        # RAM estimates
        m = re.search(
            r"Estimated max dynamical RAM per process >\s*([\d.]+)\s*MB",
            text,
            re.I,
        )
        if m:
            info["max_ram_per_proc_mb"] = float(m.group(1))

        m = re.search(
            r"Estimated total dynamical RAM >\s*([\d.]+)\s*MB",
            text,
            re.I,
        )
        if m:
            info["total_ram_mb"] = float(m.group(1))

        # negative core charge
        m = re.search(
            r"negative core charge=\s*([-\d.Ee]+)",
            text,
            re.I,
        )
        if m:
            info["negative_core_charge"] = float(m.group(1))

        # starting charge and renormalized
        m = re.search(
            r"starting charge\s*([\d.Ee+-]+),\s*renormalised to\s*([\d.Ee+-]+)",
            text,
            re.I,
        )
        if m:
            info["starting_charge"] = float(m.group(1))
            info["renormalized_charge"] = float(m.group(2))

        # starting wfcs
        m = re.search(
            r"Starting wfcs are\s*(\d+)",
            text,
            re.I,
        )
        if m:
            info["n_starting_wfcs"] = int(m.group(1))

        # total cpu time so far
        m = re.search(
            r"total cpu time spent up to now is\s*([\d.]+)\s*secs",
            text,
            re.I,
        )
        if m:
            info["cpu_time_so_far_s"] = float(m.group(1))

        return info

    # Public accessors
    @cached_property
    def n_kpoints(self) -> int | None:
        return self._kpoints_fft_mem_info["n_kpoints"]

    @cached_property
    def smearing_type(self) -> str | None:
        return self._kpoints_fft_mem_info["smearing_type"]

    @cached_property
    def smearing_width_ry(self) -> float | None:
        return self._kpoints_fft_mem_info["smearing_width_ry"]

    @cached_property
    def dense_gvecs(self) -> int | None:
        return self._kpoints_fft_mem_info["dense_gvecs"]

    @cached_property
    def dense_fft_dims(self) -> list[int] | None:
        return self._kpoints_fft_mem_info["dense_fft_dims"]

    @cached_property
    def smooth_gvecs(self) -> int | None:
        return self._kpoints_fft_mem_info["smooth_gvecs"]

    @cached_property
    def smooth_fft_dims(self) -> list[int] | None:
        return self._kpoints_fft_mem_info["smooth_fft_dims"]

    @cached_property
    def max_ram_per_proc_mb(self) -> float | None:
        return self._kpoints_fft_mem_info["max_ram_per_proc_mb"]

    @cached_property
    def total_ram_mb(self) -> float | None:
        return self._kpoints_fft_mem_info["total_ram_mb"]

    @cached_property
    def negative_core_charge(self) -> float | None:
        return self._kpoints_fft_mem_info["negative_core_charge"]

    @cached_property
    def starting_charge(self) -> float | None:
        return self._kpoints_fft_mem_info["starting_charge"]

    @cached_property
    def renormalized_charge(self) -> float | None:
        return self._kpoints_fft_mem_info["renormalized_charge"]

    @cached_property
    def n_starting_wfcs(self) -> int | None:
        return self._kpoints_fft_mem_info["n_starting_wfcs"]

    @cached_property
    def cpu_time_so_far_s(self) -> float | None:
        return self._kpoints_fft_mem_info["cpu_time_so_far_s"]

    @cached_property
    def band_structure_info(self) -> dict[str, Any] | None:
        """
        Parses the 'Band Structure Calculation' section.
        Returns:
            {
                'method': str,
                'ethr': float,
                'avg_iterations': float,
                'cpu_time_so_far_s': float
            }
        """
        text = self.text

        # Match the block between "Band Structure Calculation" and "End of ..."
        m = re.search(
            r"Band Structure Calculation\s*(.*?)End of band structure calculation",
            text,
            re.I | re.S,
        )
        if not m:
            return None

        block = m.group(1)

        info: dict[str, Any] = {
            "method": None,
            "ethr": None,
            "avg_iterations": None,
            "cpu_time_so_far_s": None,
        }

        # Method line
        m2 = re.search(r"^\s*(.+diagonalization.+)$", block, re.I | re.M)
        if m2:
            info["method"] = m2.group(1).strip()

        # ethr and avg iterations
        m2 = re.search(
            r"ethr\s*=\s*([\d.Ee+-]+),\s*avg\s+#\s*of\s*iterations\s*=\s*([\d.]+)",
            block,
            re.I,
        )
        if m2:
            info["ethr"] = float(m2.group(1))
            info["avg_iterations"] = float(m2.group(2))

        # CPU time so far
        m2 = re.search(
            r"total cpu time spent up to now is\s*([\d.]+)\s*secs",
            block,
            re.I,
        )
        if m2:
            info["cpu_time_so_far_s"] = float(m2.group(1))

        return info

    @cached_property
    def is_scf_calculation(self) -> bool:
        return self.scf_iterations is not None

    @cached_property
    def is_bands_calculation(self) -> bool:
        return self.band_structure_info is not None

    @cached_property
    def scf_iterations(self) -> list[dict[str, Any]] | None:
        """
        Parses the 'Self-consistent Calculation' section into a list of dicts.
        Each dict contains:
            iteration, ecut, beta, ethr, avg_iterations,
            negative_rho_up, negative_rho_down,
            cpu_time_so_far_s, total_energy, estimated_scf_accuracy
        """
        iterations: list[dict[str, Any]] = []

        # Regex to match each iteration block
        pattern = re.compile(
            r"iteration\s*#\s*(\d+)\s+ecut=\s*([\d.]+)\s*Ry\s+beta=\s*([\d.]+).*?"
            + r"ethr\s*=\s*([\d.Ee+-]+),\s*avg\s+#\s*of\s*iterations\s*=\s*([\d.]+).*?"
            + r"(?:negative rho\s*\(up,\s*down\):\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s*)?"
            + r"total cpu time spent up to now is\s*([\d.]+)\s*secs.*?"
            + r"total energy\s*=\s*([-\d.]+)\s*Ry\s*"
            + r"estimated scf accuracy\s*<\s*([\d.Ee+-]+)\s*Ry",
            re.I | re.S,
        )

        for m in pattern.finditer(self.text):
            iterations.append(
                {
                    "iteration": int(m.group(1)),
                    "ecut": float(m.group(2)),
                    "beta": float(m.group(3)),
                    "ethr": float(m.group(4)),
                    "avg_iterations": float(m.group(5)),
                    "negative_rho_up": float(m.group(6)) if m.group(6) else None,
                    "negative_rho_down": float(m.group(7)) if m.group(7) else None,
                    "cpu_time_so_far_s": float(m.group(8)),
                    "total_energy": float(m.group(9)),
                    "estimated_scf_accuracy": float(m.group(10)),
                }
            )

        return iterations or None

    @cached_property
    def final_results(self) -> dict[str, Any] | None:
        """
        Parses the final results section of the QE SCF output.
        Returns:
            {
                'fermi_energy_ev': float,
                'total_energy_final_ry': float,
                'total_all_electron_energy_ry': float,
                'estimated_scf_accuracy_ry': float,
                'smearing_contrib_ry': float,
                'internal_energy_ry': float,
                'energy_terms': {
                    'one_electron': float,
                    'hartree': float,
                    'xc': float,
                    'ewald': float,
                    'one_center_paw': float
                },
                'n_iterations_to_converge': int
            }
        """
        info: dict[str, Any] = {
            "fermi_energy_ev": None,
            "total_energy_final_ry": None,
            "total_all_electron_energy_ry": None,
            "estimated_scf_accuracy_ry": None,
            "smearing_contrib_ry": None,
            "internal_energy_ry": None,
            "energy_terms": {},
            "n_iterations_to_converge": None,
        }
        text = self.text

        # Fermi energy
        m = re.search(r"the Fermi energy is\s*([-\d.]+)\s*ev", text, re.I)
        if m:
            info["fermi_energy_ev"] = float(m.group(1))

        # Final total energy (with !)
        m = re.search(r"!\s*total energy\s*=\s*([-\d.]+)\s*Ry", text, re.I)
        if m:
            info["total_energy_final_ry"] = float(m.group(1))

        # Total all-electron energy
        m = re.search(r"total all-electron energy\s*=\s*([-\d.]+)\s*Ry", text, re.I)
        if m:
            info["total_all_electron_energy_ry"] = float(m.group(1))

        # Estimated SCF accuracy
        m = re.search(r"estimated scf accuracy\s*<\s*([-\d.Ee+]+)\s*Ry", text, re.I)
        if m:
            info["estimated_scf_accuracy_ry"] = float(m.group(1))

        # Smearing contribution (-TS)
        m = re.search(r"smearing contrib\.\s*\(-TS\)\s*=\s*([-\d.]+)\s*Ry", text, re.I)
        if m:
            info["smearing_contrib_ry"] = float(m.group(1))

        # Internal energy
        m = re.search(r"internal energy E=F\+TS\s*=\s*([-\d.]+)\s*Ry", text, re.I)
        if m:
            info["internal_energy_ry"] = float(m.group(1))

        # Energy terms
        terms_patterns = {
            "one_electron": r"one-electron contribution\s*=\s*([-\d.]+)\s*Ry",
            "hartree": r"hartree contribution\s*=\s*([-\d.]+)\s*Ry",
            "xc": r"xc contribution\s*=\s*([-\d.]+)\s*Ry",
            "ewald": r"ewald contribution\s*=\s*([-\d.]+)\s*Ry",
            "one_center_paw": r"one-center paw contrib\.\s*=\s*([-\d.]+)\s*Ry",
        }
        for key, pat in terms_patterns.items():
            m = re.search(pat, text, re.I)
            if m:
                info["energy_terms"][key] = float(m.group(1))

        # Number of iterations to converge
        m = re.search(
            r"convergence has been achieved in\s*(\d+)\s*iterations",
            text,
            re.I,
        )
        if m:
            info["n_iterations_to_converge"] = int(m.group(1))

        return info

    @cached_property
    def fermi_energy_ev(self) -> float | None:
        if self.final_results is None:
            return None
        return self.final_results["fermi_energy_ev"]

    @cached_property
    def total_energy_final_ry(self) -> float | None:
        if self.final_results is None:
            return None
        return self.final_results["total_energy_final_ry"]

    @cached_property
    def total_all_electron_energy_ry(self) -> float | None:
        if self.final_results is None:
            return None
        return self.final_results["total_all_electron_energy_ry"]

    @cached_property
    def estimated_scf_accuracy_ry(self) -> float | None:
        if self.final_results is None:
            return None
        return self.final_results["estimated_scf_accuracy_ry"]

    @cached_property
    def smearing_contrib_ry(self) -> float | None:
        if self.final_results is None:
            return None
        return self.final_results["smearing_contrib_ry"]

    @cached_property
    def internal_energy_ry(self) -> float | None:
        if self.final_results is None:
            return None
        return self.final_results["internal_energy_ry"]

    @cached_property
    def energy_terms(self) -> dict[str, float] | None:
        if self.final_results is None:
            return None
        return self.final_results["energy_terms"]

    @cached_property
    def n_iterations_to_converge(self) -> int | None:
        if self.final_results is None:
            return None
        return self.final_results["n_iterations_to_converge"]

    @cached_property
    def timing_info(self) -> dict[str, Any] | None:
        """
        Parses the timing breakdown section of QE output.
        Returns:
            {
                "output_data_dir": str,
                "timings": [
                    {"name": str, "cpu_s": float, "wall_s": float,
                     "calls": int, "section": str|None},
                    ...
                ],
                "total_pwscf_cpu_s": float,
                "total_pwscf_wall_s": float,
                "terminated_on": str
            }
        """
        text = self.text
        info: dict[str, Any] = {
            "output_data_dir": None,
            "timings": [],
            "total_pwscf_cpu_s": None,
            "total_pwscf_wall_s": None,
            "terminated_on": None,
        }

        # Output data dir
        m = re.search(r"Writing all to output data dir\s+(\S+)", text, re.I)
        if m:
            info["output_data_dir"] = m.group(1)

        # Regex for timing lines
        timing_line = re.compile(
            r"^\s*([A-Za-z0-9_:*]+)\s*:\s*([\d.]+)s CPU\s+([\d.]+)s WALL\s*\(\s*(\d+)\s*calls?\)",
            re.M,
        )

        current_section = None
        for line in text.splitlines():
            # Section headers
            sec_match = re.match(r"^\s*Called by\s+(.+?):", line, re.I)
            if sec_match:
                current_section = sec_match.group(1).strip()
                continue
            if re.match(r"^\s*General routines", line, re.I):
                current_section = "General routines"
                continue
            if re.match(r"^\s*Parallel routines", line, re.I):
                current_section = "Parallel routines"
                continue

            # Timing lines
            m = timing_line.match(line)
            if m:
                info["timings"].append(
                    {
                        "name": m.group(1),
                        "cpu_s": float(m.group(2)),
                        "wall_s": float(m.group(3)),
                        "calls": int(m.group(4)),
                        "section": current_section,
                    }
                )

        # Total PWSCF time
        m = re.search(r"PWSCF\s*:\s*([\d.]+)s CPU\s+([\d.]+)s WALL", text, re.I)
        if m:
            info["total_pwscf_cpu_s"] = float(m.group(1))
            info["total_pwscf_wall_s"] = float(m.group(2))

        # Termination time
        m = re.search(r"This run was terminated on:\s*(.+)", text, re.I)
        if m:
            info["terminated_on"] = m.group(1).strip()

        return info

    @cached_property
    def output_data_dir(self) -> str | None:
        if self.timing_info is None:
            return None
        return self.timing_info["output_data_dir"]

    @cached_property
    def timings(self) -> list[dict[str, Any]] | None:
        if self.timing_info is None:
            return None
        return self.timing_info["timings"]

    @cached_property
    def total_pwscf_cpu_s(self) -> float | None:
        if self.timing_info is None:
            return None
        return self.timing_info["total_pwscf_cpu_s"]

    @cached_property
    def total_pwscf_wall_s(self) -> float | None:
        if self.timing_info is None:
            return None
        return self.timing_info["total_pwscf_wall_s"]

    @cached_property
    def terminated_on(self) -> str | None:
        if self.timing_info is None:
            return None
        return self.timing_info["terminated_on"]
