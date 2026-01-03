__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import re
from functools import cached_property
from pathlib import Path
from re import Pattern
from typing import Any, ClassVar

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

from pyprocar.core.atomic_orbital_index import OrbitalIndexer

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


class ProjwfcPDOSFile:
    FILE_PATTERN: ClassVar[Pattern[str]] = re.compile(
        r"pdos_atm#(\d+)\(([^)]+)\)_wfc#(\d+)\(([^)_]+)(?:_j([\d.]+))?\)"
    )

    _filepath: Path
    _text: str

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = Path(filepath)
        self._text = self._read()

    def _read(self) -> str:
        with open(self._filepath) as f:
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
            columns.pop(1)
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
                row = (
                    [int(parts[0])] + [float(x.replace("E", "e")) for x in parts[1:]]
                    if cols and cols[0].lower() == "ik"
                    else [float(x.replace("E", "e")) for x in parts]
                )
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

        return pd.DataFrame(data_lines, columns=pd.Index(cols))


class ProjwfcDOS:
    """
    Parser for a collection of projwfc.x PDOS files.
    Can be initialized with:
      - A list of file paths
      - A directory path (auto-detects PDOS files)
    """

    FILE_PATTERN: ClassVar[Pattern[str]] = re.compile(
        r"pdos_atm#(\d+)\(([^)]+)\)_wfc#(\d+)\(([^)]+)\)(?:_j([\d.]+))?"
    )

    total_dos_path: Path | None
    filepaths: list[Path]

    def __init__(self, paths: str | Path | list[str | Path]) -> None:
        self.total_dos_path = None  # store .pdos_tot file path if found

        if isinstance(paths, (str, Path)):
            p = Path(paths)
            if p.is_dir():
                # Find all pdos_atm files
                self.filepaths = sorted(f for f in p.iterdir() if self.FILE_PATTERN.search(f.name))
                # Look for .pdos_tot file
                tot_files = list(p.glob("*.pdos_tot"))
                if tot_files:
                    self.total_dos_path = tot_files[0]
            elif p.is_file():
                self.filepaths = [p]
            else:
                raise ValueError(f"Invalid path: {paths}")
        else:  # isinstance(paths, list)
            self.filepaths = [Path(fp) for fp in paths]

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
    def total_dos_filepath(self) -> ProjwfcPDOSFile | None:
        if not self.total_dos_path:
            return None
        return ProjwfcPDOSFile(self.total_dos_path)

    @cached_property
    def total_dos(self) -> np.ndarray | None:
        """
        Parse the total DOS from the .pdos_tot file if available.
        Returns a numpy array or None if no file found.
        """
        total_dos_file = self.total_dos_filepath
        if total_dos_file is None:
            return None

        dos_array = np.zeros((self.n_energies, self.n_spin_channels))

        df = total_dos_file.data
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
    def data(self) -> list[pd.DataFrame]:
        dataframes = [pdos_file.data for pdos_file in self.pdos_files]
        dataframes = self._regularize_projections(dataframes)
        return dataframes

    @cached_property
    def is_non_colinear(self) -> bool:
        return self.files_metadata[0]["j_value"] is not None

    @cached_property
    def is_spin_polarized(self) -> bool:
        return any("pdosup" in column_name.lower() for column_name in self.pdos_files[0].columns)

    @cached_property
    def n_spin_channels(self) -> int:
        if self.is_spin_polarized:
            return 2
        else:
            return 1

    @cached_property
    def is_kresolved(self) -> bool:
        df = self.data[0]  # first file's DataFrame
        return "ik" in df.columns

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

            # Extract energies for each k-point
            bands = []
            for ik in ik_values:
                e_k = df.loc[df["ik"] == ik, "E"].to_numpy()
                bands.append(e_k)
            bands = np.array(bands)

            # Validation check across all files
            for ifile, df_other in enumerate(self.data[1:], start=1):
                for i, ik in enumerate(ik_values):
                    e_k_other = df_other.loc[df_other["ik"] == ik, "E"].to_numpy()
                    if not np.allclose(e_k_other, bands[i]):
                        filepath = self.files_metadata[ifile]["filepath"]
                        msg = f"Energies differ in file {filepath} for k-point {ik}"
                        raise ValueError(msg)
            return bands

        else:
            ref_energies = df["E"].to_numpy()

            # Validation check across all files
            for ifile, df_other in enumerate(self.data[1:], start=1):
                e_other = df_other["E"].to_numpy()
                if e_other.shape != ref_energies.shape:
                    raise ValueError(f"Energies differ in file {self.pdos_files[ifile].filepath}")

            return ref_energies[np.newaxis, :]  # shape (1, n_energies)

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

            energies = df_values[:, 0]
            scalar_values = df_values[:, 1:]

            new_df_values[:, 0] = ref_energies
            for i in range(1, scalar_values.shape[1]):
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
        atom_to_files: dict[int, list[dict]] = {a: [] for a in atom_indices}
        for f in self.files_metadata:
            atom_to_files[f["atom_index"]].append(f)

        for ai, atom in enumerate(atom_indices):
            for f in atom_to_files[atom]:
                file_index = self.files_metadata.index(f)
                df = self.data[file_index]

                if self.is_non_colinear:
                    # SOC case: multiple m-components in one file
                    orbital_l = ORBITAL_ORDERING.l_orbital_map[f["orbital"][0].lower()]
                    j = f["j_value"]
                    n_m = df.shape[1] - 2  # E, LDOS, then PDOS_m...
                    pdos_m = df.iloc[:, 2:].to_numpy()
                    m_values = np.linspace(-j, j, int(2 * j + 1))
                    for mi, m in enumerate(m_values):
                        target_idx = ORBITAL_ORDERING.get_soc_index(orbital_l, j, m)
                        dos_array[:, 0, ai, target_idx] = pdos_m[:, mi]

                else:
                    # Collinear case
                    orbital_l = ORBITAL_ORDERING.l_orbital_map[f["orbital"][0].lower()]

                    if f["orbital"] in ("s", "p", "d", "f"):
                        # Multiple m-components in one file
                        n_pdos_cols = df.shape[1] - 2
                        if self.is_spin_polarized and self.n_spin_channels == 2:
                            n_m = n_pdos_cols // 2
                        else:
                            n_m = n_pdos_cols

                        pdos_m = df.iloc[:, 2:].to_numpy()
                        orb_names_for_l = ORBITAL_ORDERING.azimuthal_order[
                            list(ORBITAL_ORDERING.l_orbital_map.keys())[orbital_l]
                        ]
                        n_m_expected = len(orb_names_for_l)

                        if n_m != n_m_expected:
                            orb = f["metadata"]["orbital"]
                            msg = (
                                f"Mismatch: orbital {orb} has {n_m} m-components "
                                f"in file but {n_m_expected} in ORBITAL_ORDERING"
                            )
                            raise ValueError(msg)

                        for mi, orb_name in enumerate(orb_names_for_l):
                            target_idx = ORBITAL_ORDERING.az_to_flat_index[orb_name]
                            if self.is_spin_polarized and self.n_spin_channels == 2:
                                dos_array[:, 0, ai, target_idx] = pdos_m[:, mi * 2]  # spin-up
                                dos_array[:, 1, ai, target_idx] = pdos_m[:, mi * 2 + 1]  # spin-down
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
