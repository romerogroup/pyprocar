import gzip
import logging
import re
from collections.abc import Iterator, Mapping
from functools import cached_property
from io import TextIOWrapper
from pathlib import Path
from typing import Any, override

import numpy as np

# from pyprocar.utils import np_utils

logger = logging.getLogger(__name__)


class Procar(Mapping[str, Any]):
    """
    A class to parse the PROJCAR file from VASP.

    The PROJCAR file contains information about the projections of the
    Kohn-Sham orbitals onto localized orbitals specified with the LOCPROJ tag.

    Parameters
    ----------
    filepath : Union[str, Path]
        The PROJCAR filepath
    file_str : str | None, optional
        The PROJCAR file content as string

    Returns
    -------
    dict
        Dictionary with keys:
        - "frac_coords": np.ndarray of shape [n_atoms, 3]
        - "radial_specs": list[dict] with keys "type" and "params"
        - "projections": np.ndarray of shape [n_k, n_bands, n_spins, n_atoms, n_orbitals]
          with complex dtype
    """

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        logger.info(f"Initializing Locproj parser for {filepath}")
        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str or ""

    @classmethod
    def from_str(cls, input: str) -> "Procar":
        return cls(file_str=input)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            file_stream = self._open_file(self.filepath)
            self._file_str = file_stream.read()
            file_stream.close()
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str

    @property
    def filename(self) -> str:
        if self.filepath is None:
            return ""
        return self.filepath.name

    @property
    def orbital_names(self) -> list[str]:
        return [
            "s",
            "py",
            "pz",
            "px",
            "dxy",
            "dyz",
            "dz2",
            "dxz",
            "x2-y2",
            "fy3x2",
            "fxyz",
            "fyz2",
            "fz3",
            "fxz2",
            "fzx2",
            "fx3",
            "tot",
        ]

    @property
    def orbital_names_old(self) -> list[str]:
        return ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2", "tot"]

    @property
    def orbital_names_short(self) -> list[str]:
        return ["s", "p", "d", "f", "tot"]

    @property
    def labels(self) -> list[str]:
        return self.orbital_names_old[:-1]

    @cached_property
    def n_kpoints(self) -> int:
        tmp = re.findall(r"# of k-points:\s+(\d+)", self.file_str)
        if len(tmp) == 0:
            raise ValueError("No n_kpoints found in PROJCAR file. Issue with regex parsing.")
        n_kpoints = int(tmp[0])
        return n_kpoints

    @cached_property
    def n_bands(self) -> int:
        tmp = re.findall(r"# of bands:\s+(\d+)", self.file_str)
        if len(tmp) == 0:
            raise ValueError("No n_bands found in PROJCAR file. Issue with regex parsing.")
        n_bands = int(tmp[0])
        return n_bands

    @cached_property
    def n_atoms(self) -> int:
        tmp = re.findall(r"# of ions:\s+(\d+)", self.file_str)
        if len(tmp) == 0:
            raise ValueError("No n_atoms found in PROCAR file. Issue with regex parsing.")
        n_atoms = int(tmp[0])
        return n_atoms

    @cached_property
    def n_spins(self) -> int:
        if self.is_non_colinear:
            return 4
        elif self.is_spin_polarized:
            return 2
        else:
            return 1

    @cached_property
    def _raw_kpoints(self) -> np.ndarray:
        regex = r"k-point\s+\d+\s*:\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+"
        kpoints = re.findall(regex, self.file_str)
        if len(kpoints) == 0:
            raise ValueError("No kpoints found in PROJCAR file. Issue with regex parsing.")
        return np.array(kpoints, dtype=float)

    @cached_property
    def _n_raw_kpoints(self) -> int:
        return len(self._raw_kpoints)

    @cached_property
    def kpoints(self) -> np.ndarray:
        return self._raw_kpoints[: self.n_kpoints]

    @cached_property
    def is_spin_polarized(self) -> bool:
        return self._n_raw_kpoints == self.n_kpoints * 2

    @cached_property
    def is_non_colinear(self) -> bool:
        """Check if this is a non-colinear calculation (4 spin channels)"""
        # For non-colinear, there's one "ion" header per band, but 4 "tot" lines
        # (one per spin component)
        # Count the number of "tot" lines in projection blocks
        tot_lines = re.findall(r"^tot\s+", self.file_str, re.MULTILINE)
        if len(tot_lines) == 0:
            return False
        # Don't use self.n_spins here to avoid circular dependency
        expected_base = self.n_kpoints * self.n_bands
        actual_count = len(tot_lines)
        # Non-colinear has 4 "tot" lines per k-point/band
        # but is_spin_polarized is False (no duplicate k-points)
        return (not self.is_spin_polarized) and (actual_count == expected_base * 4)

    @cached_property
    def bands(self) -> np.ndarray:
        bands_info = re.findall(r"band\s*(\d+)\s*#\s*energy\s*([-.\d\s]+)", self.file_str)
        if len(bands_info) == 0:
            raise ValueError("No bands found in PROJCAR file. Issue with regex parsing.")
        raw_bands = np.array(bands_info, dtype=float)[:, 1]

        # Get actual number of bands found (might be less than expected for incomplete test data)
        n_bands_found = len(raw_bands)

        if self.is_spin_polarized:
            # Take only what we have, up to the expected amount
            n_bands_to_use = min(n_bands_found, self.n_kpoints * self.n_bands * 2)
            spin_up_bands = raw_bands[: n_bands_to_use // 2]
            spin_down_bands = raw_bands[n_bands_to_use // 2 : n_bands_to_use]

            # Determine actual shape from data
            actual_n_kpoints = len(spin_up_bands) // self.n_bands if self.n_bands > 0 else 0
            spin_up_bands = spin_up_bands[: actual_n_kpoints * self.n_bands]
            spin_up_bands = spin_up_bands.reshape(actual_n_kpoints, self.n_bands)
            spin_down_bands = spin_down_bands[: actual_n_kpoints * self.n_bands]
            spin_down_bands = spin_down_bands.reshape(actual_n_kpoints, self.n_bands)
            bands = np.stack((spin_up_bands, spin_down_bands), axis=-1)
        elif self.is_non_colinear:
            # For non-colinear, we only use the first spin component (total charge density)
            # Take only n_kpoints * n_bands entries (one per k-point/band, the first of 4)
            n_entries_per_kband = 4  # non-colinear has 4 energy entries per k/band
            n_bands_to_use = min(n_bands_found, self.n_kpoints * self.n_bands * n_entries_per_kband)
            # Extract every 4th entry (or just use the first n_kpoints*n_bands)
            bands = raw_bands[: self.n_kpoints * self.n_bands].reshape(
                self.n_kpoints, self.n_bands, 1
            )
            # Expand to 4 spin channels
            bands = np.repeat(bands, 4, axis=2)
        else:
            # Non-polarized case
            n_bands_to_use = min(n_bands_found, self.n_kpoints * self.n_bands)
            bands = raw_bands[:n_bands_to_use].reshape(-1, self.n_bands, 1)

        return bands

    @cached_property
    def has_phase(self) -> bool:
        return "phase" in self.file_str

    @cached_property
    def n_orbitals(self) -> int:
        """Get the number of orbitals from the first orbital header"""
        orbital_headers = re.findall(r"ion(.+)", self.file_str)
        if len(orbital_headers) == 0:
            raise ValueError("No orbital headers found in PROCAR file")

        # Look for a header with "tot" in it (the regular projection header)
        # Phase headers don't have "tot"
        for header in orbital_headers:
            found_orbs = header.split()
            if "tot" in found_orbs:
                # Subtract 1 for "tot" column
                return len(found_orbs) - 1

        # If no header with "tot" found, use the first header
        found_orbs = orbital_headers[0].split()
        return len(found_orbs)

    @cached_property
    def spd(self) -> np.ndarray | None:
        """Projection data array with shape [n_kpoints, n_bands, n_spins, n_atoms, n_orbitals]"""
        return self._read_projections()

    @cached_property
    def projected(self) -> np.ndarray | None:
        """Projected data array with shape [n_kpoints, n_bands, n_spins, n_atoms, n_orbitals]"""
        return self._spd2projected(self.spd)

    @cached_property
    def projected_phase(self) -> np.ndarray | None:
        """Projected phase data array with shape
        [n_kpoints, n_bands, n_spins, n_atoms, n_orbitals]"""
        if not self.has_phase:
            return None
        return self._read_phases()

    def _read_projections(self) -> np.ndarray | None:
        """
        Reads all the spd-projected data. A typical/expected block is:

            ion      s     py     pz     px    dxy    dyz    dz2    dxz    dx2    tot
            1  0.079  0.000  0.001  0.000  0.000  0.000  0.000  0.000  0.000  0.079
            2  0.152  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.152
            3  0.079  0.000  0.001  0.000  0.000  0.000  0.000  0.000  0.000  0.079
            4  0.188  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.188
            5  0.188  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.188
            tot  0.686  0.000  0.002  0.000  0.000  0.000  0.000  0.000  0.000  0.688
            (x2 for spin-polarized -akwardkly formatted-, x4 non-collinear -nicely
             formatted-).

        """

        # finding all orbital headers
        header_lines = re.findall(r"(ion.+tot)", self.file_str)

        # testing if the orbital names are known (the standard ones)
        orbital_in_procar = header_lines[0].split()
        n_spd_columns = len(orbital_in_procar)

        # only the first 'size' orbital
        standard_orbitals = self.orbital_names[: n_spd_columns - 1] + self.orbital_names[-1:]
        standard_orbitals_short = (
            self.orbital_names_short[: n_spd_columns - 1] + self.orbital_names_short[-1:]
        )
        standard_orbitals_old = self.orbital_names_old[: n_spd_columns - 1] + [
            self.orbital_names_old[-1:]
        ]
        if (
            orbital_in_procar != standard_orbitals
            and orbital_in_procar != standard_orbitals_short
            and orbital_in_procar != standard_orbitals_old
        ):
            logger.warning(
                f"{n_spd_columns} orbitals. (Some of) "
                + "They are unknow (if you did 'filter' them it is OK)."
            )

        # Vasp format different for 1 atom
        n_spd_rows = self.n_atoms + 1
        n_projection_rows = self.n_atoms + 1
        if self.is_non_colinear:
            n_spd_rows *= 4

        line_pattern = [r".+\n" for _ in range(n_spd_rows)]
        pattern = r"ion.*\n(" + "".join(line_pattern) + ")"

        projection_blocks = re.findall(pattern, self.file_str)

        spd = []
        for block in projection_blocks:
            if "charge" in block:
                continue
            projection_lines = block.replace("tot", "0").strip().split("\n")
            for line in projection_lines:
                spd.append(line.strip().split())  # pyright: ignore[reportUnknownMemberType]

        spd = np.array(spd, dtype=float)

        # handling collinear polarized case
        if self.is_spin_polarized:
            # splitting both spin components, now they are along k-points
            # axis (1st axis) but, then should be concatenated along the
            # bands.
            up, down = np.vsplit(spd, 2)

            up = up.reshape(
                self.n_kpoints,
                self.n_bands,
                1,
                n_projection_rows,
                n_spd_columns,  # +1 for ion number, +1 for tot
            )
            down = down.reshape(
                self.n_kpoints,
                self.n_bands,
                1,
                n_projection_rows,
                n_spd_columns,
            )
            # concatenating bandwise. Density and magntization, their
            # meaning is obvious, and do uses 2 times more memory than
            # required, but I *WANT* to keep it as close as possible to the
            # non-collinear or non-polarized case
            density = np.concatenate((up, down), axis=1)
            magnet = np.concatenate((up, -down), axis=1)

            # concatenated along 'n_spins` axis
            spd = np.concatenate((density, magnet), axis=2)

        # otherwise, just a reshaping suffices
        else:
            spd = spd.reshape(
                self.n_kpoints,
                self.n_bands,
                self.n_spins,
                n_projection_rows,
                n_spd_columns,
            )

        if self.n_atoms == 1:
            spd = np.pad(
                spd,
                ((0, 0), (0, 0), (0, 0), (0, 1), (0, 0)),
                "constant",
                constant_values=(0),
            )
            spd[:, :, :, 1, :] = spd[:, :, :, 0, :]

        return spd

    def _read_phases(self) -> np.ndarray | None:
        """
        Helped method to parse the projection phases
        """
        # finding all orbital headers
        header_lines = re.findall(r"(ion.+tot)", self.file_str)

        # testing if the orbital names are known (the standard ones)
        orbital_in_procar = header_lines[0].split()
        n_spd_columns = len(orbital_in_procar)

        # only the first 'size' orbital
        standard_orbitals = self.orbital_names[: n_spd_columns - 1] + self.orbital_names[-1:]
        standard_orbitals_short = (
            self.orbital_names_short[: n_spd_columns - 1] + self.orbital_names_short[-1:]
        )
        standard_orbitals_old = self.orbital_names_old[: n_spd_columns - 1] + [
            self.orbital_names_old[-1:]
        ]
        if (
            orbital_in_procar != standard_orbitals
            and orbital_in_procar != standard_orbitals_short
            and orbital_in_procar != standard_orbitals_old
        ):
            logger.warning(
                f"{n_spd_columns} orbitals. (Some of) "
                + "They are unknow (if you did 'filter' them it is OK)."
            )

        # Vasp format different for 1 atom
        n_spd_rows = self.n_atoms + 1
        # n_projection_rows = self.n_atoms + 1
        if self.is_non_colinear:
            n_spd_rows *= 4

        line_pattern = [r".+\n" for _ in range(n_spd_rows)]
        pattern = r"ion.*\n(" + "".join(line_pattern) + ")"

        projection_blocks = re.findall(pattern, self.file_str)

        spd_phase = []
        real_parts = []
        imaginary_parts = []
        for block in projection_blocks:
            if "charge" not in block:
                continue
            projection_lines = block.replace("tot", "0").strip().split("\n")
            for line in projection_lines:
                if "charge" in line:
                    continue
                projection_phases_with_tot_with_ion = line.strip().split()
                projection_phases = projection_phases_with_tot_with_ion[1:-1]

                real_parts.append(projection_phases[::2])  # pyright: ignore[reportUnknownMemberType]
                imaginary_parts.append(projection_phases[1::2])  # pyright: ignore[reportUnknownMemberType]

        real_parts = np.array(real_parts, dtype=float)
        imaginary_parts = np.array(imaginary_parts, dtype=float)
        spd_phase = real_parts + 1j * imaginary_parts

        # handling collinear polarized case
        if self.is_spin_polarized:
            # splitting both spin components, now they are along k-points
            # axis (1st axis) but, then should be concatenated along the
            # bands.
            up, down = np.vsplit(spd_phase, 2)
            # n_spins = 1 for a while, we will made the distinction
            up = up.reshape(
                self.n_kpoints,
                self.n_bands,
                1,
                self.n_atoms,
                self.n_orbitals,
            )
            down = down.reshape(
                self.n_kpoints,
                self.n_bands,
                1,
                self.n_atoms,
                self.n_orbitals,
            )
            # concatenating bandwise. Density and magntization, their
            # meaning is obvious, and do uses 2 times more memory than
            # required, but I *WANT* to keep it as close as possible to the
            # non-collinear or non-polarized case
            density = np.concatenate((up, down), axis=1)
            magnet = np.concatenate((up, -down), axis=1)
            # concatenated along 'n_spins axis'
            spd_phase = np.concatenate((density, magnet), axis=2)

        # otherwise, just a reshaping suffices
        else:
            spd_phase = spd_phase.reshape(
                self.n_kpoints,
                self.n_bands,
                self.n_spins,
                self.n_atoms,
                self.n_orbitals,
            )

        return spd_phase

    def _validate_file(self, filepath: str | Path) -> Path:
        """
        Validate and resolve the PROJCAR file path.

        Parameters
        ----------
        filepath : str | Path
            Path to PROJCAR file

        Returns
        -------
        Path
            Validated file path
        """
        filepath = Path(filepath)
        if filepath.is_dir():
            filepath = filepath / "PROJCAR"
        elif not filepath.is_file():
            # Try with .gz extension
            gz_path = filepath.with_suffix(".gz")
            if gz_path.is_file():
                return gz_path
            raise OSError(f"PROJCAR file not found: {filepath}")

        return filepath

    def _open_file(self, filepath: str | Path) -> TextIOWrapper:
        """
        Open PROJCAR file, handling gzipped files.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to file

        Returns
        -------
        file-like object
            Opened file stream
        """
        filepath = Path(filepath)
        if filepath.suffix == ".gz" or filepath.with_suffix(".gz").is_file():
            gz_path = filepath.with_suffix(".gz") if filepath.suffix != ".gz" else filepath
            if gz_path.is_file():
                return gzip.open(gz_path, mode="rt")

        if filepath.is_file():
            return open(filepath)

        raise OSError(f"PROCAR file not found: {filepath}")

    # def _repair(self):
    #     """
    #     It Tries to repair some stupid problems due the stupid fixed
    #     format of the stupid fortran.

    #     Up to now it only separes k-points as the following:
    #     k-point    61 :    0.00000000-0.50000000 0.00000000 ...
    #     to
    #     k-point    61 :    0.00000000 -0.50000000 0.00000000 ...

    #     But as I found new stupid errors they should be fixed here.
    #     """
    #     if (
    #         len(re.findall(r"(band\s)(\*\*\*)", self.file_str)) != 0
    #         or len(re.findall(r"(\.\d{8})(\d{2}\.)", self.file_str)) != 0
    #         or len(re.findall(r"(\d)-(\d)", self.file_str)) != 0
    #         or len(re.findall(r"\*+", self.file_str)) != 0
    #     ):
    #         logger.info("PROCAR needs repairing")
    #         # Fixing bands issues (when there are more than 999 bands)
    #         # band *** # energy    6.49554019 # occ.  0.00000000
    #         self.file_str = re.sub(r"(band\s)(\*\*\*)", r"\1 1000", self.file_str)
    #         # Fixing k-point issues

    #         self.file_str = re.sub(r"(\.\d{8})(\d{2}\.)", r"\1 \2", self.file_str)
    #         self.file_str = re.sub(r"(\d)-(\d)", r"\1 -\2", self.file_str)

    #         self.file_str = re.sub(r"\*+", r" -10.0000 ", self.file_str)

    #         outfile = open(self.filename + "-repaired", "w")
    #         for iline in self.meta_lines:
    #             outfile.write(iline)
    #         outfile.write(self.file_str)
    #         outfile.close()
    #         print("Repaired PROCAR is written at {}-repaired".format(self.filename))
    #         print(
    #             "Please use {}-repaired next time for better efficiency".format(
    #                 self.filename
    #             )
    #         )
    #     return

    def _spd2projected(self, spd: np.ndarray | None) -> np.ndarray | None:
        """
        Helpermethod to project the spd array to the projected array
        which will be fed into pyprocar.coreElectronicBandStructure object

        Parameters
        ----------
        spd : np.ndarray
            The spd array from the earlier parse. This has a structure simlar to the
            PROCAR output in vasp
            Has the shape [n_kpoints,n_band,n_spins,n_orbital,n_atoms]
        nprinciples : int, optional
            The prinicipal quantum numbers, by default 1

        Returns
        -------
        np.ndarray
            The projected array. Has the shape [n_kpoints,n_band,n_spin, n_atom,n_orbital]
        """
        # This function is for VASP
        # non-pol and colinear
        # spd is formed as (nkpoints,nbands, nspin, natom+1, norbital+2)
        # natom+1 > last column is total
        # norbital+2 > 1st column is the number of atom last is total
        # non-colinear
        # spd is formed as (nkpoints,nbands, nspin +1 , natom+1, norbital+2)
        # natom+1 > last column is total
        # norbital+2 > 1st column is the number of atom last is total
        # nspin +1 > last column is total
        if spd is None:
            return None
        natoms = spd.shape[3] - 1

        nkpoints = spd.shape[0]

        nbands = spd.shape[1]
        norbitals = spd.shape[4] - 2
        nspins = 4 if spd.shape[2] == 4 else spd.shape[2]
        nbands = int(spd.shape[1] / 2) if nspins == 2 else spd.shape[1]

        projected = np.zeros(
            shape=(nkpoints, nbands, nspins, natoms, norbitals),
            dtype=spd.dtype,
        )

        if nspins == 2:
            # For spin-polarized, the density channel (spin 0) has up in first half,
            # down in second half
            projected[:, :, 0, :, :] = spd[:, :nbands, 0, :-1, 1:-1]  # spin up from density
            projected[:, :, 1, :, :] = spd[:, nbands:, 0, :-1, 1:-1]  # spin down from density
        else:
            projected[:, :, :, :, :] = spd[:, :, :, :-1, 1:-1]

        return projected

    @override
    def __contains__(self, key: object) -> bool:
        """Check if key exists as an attribute."""
        return key in self.__dict__

    @override
    def __getitem__(self, key: str) -> Any:
        """Get attribute value."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[str]:
        """Iterate over public attributes."""
        # Return all public attributes (not starting with _)
        for key in dir(self):
            if not key.startswith("_") and not callable(getattr(self, key)):
                yield key

    @override
    def __len__(self) -> int:
        """Return number of public attributes."""
        return sum(1 for _ in self)
