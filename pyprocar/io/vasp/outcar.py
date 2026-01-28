import logging
import re
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, override

import numpy as np

logger = logging.getLogger(__name__)


class Outcar(Mapping[str, Any]):
    """
    A class to parse the OUTCAR file from a VASP run and extract electronic structure data.

    The OUTCAR file provides detailed output of a VASP run,
    including a summary of used input parameters,
    information about electronic steps and KS-eigenvalues,
    stress tensors, forces on atoms, local charges
    and magnetic moments, and dielectric properties. The amount of output written onto the
    OUTCAR file can
    be chosen by modifying the NWRITE tag in the INCAR file.

    The Outcar class acts as a Mapping, providing key-value access to the variables parsed from
    the OUTCAR file.
    """

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        """
        Constructor method to initialize an Outcar object. Reads the file specified
        by filename and stores its content.

        Parameters
        ----------
        filename : Union[str, Path], optional
            The OUTCAR filename. If not provided, defaults to "OUTCAR".
        """
        self._file_str: str = file_str
        self._filepath: str | Path | None = filepath

    @classmethod
    def from_str(cls, input: str) -> "Outcar":
        return cls(file_str=input)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(self.filepath) as file_stream:
                self._file_str = file_stream.read()
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str

    def _get_axes_nk(self):
        """
        n_kx

        Returns
        -------
        n_kx
            n_kx
        """

        raw_text = re.findall(r"generate\s*k-points\s*for:\s*(.*)", self.file_str)
        n_kx = None
        n_ky = None
        n_kz = None
        if len(raw_text) > 0:
            raw_text = raw_text[-1]
            n_kx = int(raw_text.split()[0])
            n_ky = int(raw_text.split()[1])
            n_kz = int(raw_text.split()[2])
        return n_kx, n_ky, n_kz

    @cached_property
    def kgrid(self):
        return self._get_axes_nk()

    @cached_property
    def n_kx(self):
        return self.kgrid[0]

    @cached_property
    def n_ky(self):
        return self.kgrid[1]

    @cached_property
    def n_kz(self):
        return self.kgrid[2]

    @cached_property
    def version(self):
        """
        Returns the version of the OUTCAR file.
        """
        return re.findall(r"vasp\.\d+\.\d+\.\d+", self.file_str)[-1].strip("vasp.")

    @cached_property
    def version_tuple(self):
        """
        Returns the version of the OUTCAR file as a tuple.
        """
        return tuple(int(x) for x in self.version.split("."))

    @cached_property
    def fermi(self):
        """
        Just finds all E-fermi fields in the outcar file and keeps the
        last one (if more than one found).

        Returns
        -------
        fermi
            the fermi energy
        """
        return float(re.findall(r"E-fermi\s*:\s*(-?\d+.\d+)", self.file_str)[-1])

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray:
        """
        Finds and return the reciprocal lattice vectors, if more than
        one set present, it return just the last one.

        Returns
        -------
        np.ndaaray
            return the reciprocal lattice vectors
        """

        match = re.search(r"reciprocal lattice vectors[\s\S]+?(?=\n\s?\n\s?)", self.file_str)
        if match is None:
            raise ValueError("No reciprocal lattice vectors found")
        numbers = re.findall(r"[-]?\d+\.\d+", match.group(0))

        # Create a NumPy array from the found numbers, reshape it to 3 rows and 6 columns
        all_vectors = np.array(numbers, dtype=float).reshape(3, 6)

        # Slice the array to get the last 3 columns,
        reciprocal_lattice = all_vectors[:, 3:]
        return reciprocal_lattice

    @cached_property
    def rotations(self) -> np.ndarray:
        """
        Finds the point symmetry operations included in the OUTCAR file
        and returns them in matrix form.

        Returns
        -------
        np.ndarray
            The rotation matrices
        """

        sym_ops = self.get_symmetry_operations()

        if not sym_ops:
            return np.empty((0, 3, 3))

        rotations: list[np.ndarray] = []
        for sym_op in sym_ops:
            rotations.append(sym_op["rotation"])

        return np.array(rotations)

    def get_symmetry_operations(self) -> list[dict[str, Any]]:
        raw_spg_ops = re.search(r"Found\s+(\d+)\s+space group operations", self.file_str)

        if raw_spg_ops is None:
            return []

        n_spg_operations = int(raw_spg_ops.group(1))

        logger.debug(f"n_spg_operations: {n_spg_operations}")

        vasp54_block_match = re.search(
            r"Space group operators:\s*\n"  # header line
            + r"([ \t]*irot[\s\S]+?)"  # from the column headers …
            + r"(?=\n\s?\n\s?)",  # … up to the next blank/non-indented line
            self.file_str,
            flags=re.IGNORECASE,
        )

        vasp60_block_pattern = re.compile(
            r"""
            irot\s*:\s*(?P<irot>\d+)          # 1) operator index
            .*?                               # skip down to isymop
            isymop:\s*
            (?P<r1>-?\d+\s+-?\d+\s+-?\d+)\s*  # 2) first row of 3 ints
            (?P<r2>-?\d+\s+-?\d+\s+-?\d+)\s*  # 3) second row
            (?P<r3>-?\d+\s+-?\d+\s+-?\d+)     # 4) third row
            .*?                               # skip to gtrans
            gtrans:\s*
            (?P<g1>-?\d+\.\d+)\s+
            (?P<g2>-?\d+\.\d+)\s+
            (?P<g3>-?\d+\.\d+)
            .*?                               # skip to ptrans
            ptrans:\s*
            (?P<p1>-?\d+\.\d+)\s+
            (?P<p2>-?\d+\.\d+)\s+
            (?P<p3>-?\d+\.\d+)
            .*?                               # skip to rotmap
            rotmap:\s*\n                      # the “rotmap:” line might stand alone
            (?P<rotmap>                       # now capture one or more “( a-> b )” entries
            (?:[ \t]*\(\s*\d+\s*->\s*\d+\s*\)\s*)+
            )
            (?=\n\n)                       # stop at double new line
        """,
            re.VERBOSE | re.DOTALL,
        )

        vasp60_block_matches = vasp60_block_pattern.findall(self.file_str)

        if vasp54_block_match:
            logger.info("Detected Space Group Operators in a format from VASP 5.4")
            spg_operators: list[dict[str, Any]] = []
            block = vasp54_block_match.group(1).rstrip()
            logger.debug(f"space group operators block: \n{block}")

            headers = None
            for i, line in enumerate(block.splitlines()):
                values = line.split()
                if i == 0:
                    headers = values
                    continue

                assert headers is not None

                spg_operator: dict[str, Any] = {}
                for ispg, value in enumerate(values):
                    column_name: str = headers[ispg]
                    value = int(value) if column_name == "irot" else float(value)
                    spg_operator[column_name] = value

                spg_operators.append(spg_operator)

            sym_ops: list[dict[str, Any]] = []
            for operator in spg_operators:
                sym_op: dict[str, Any] = {}
                irot = operator["irot"]
                det_A = operator["det(A)"]
                # convert alpha to radians
                alpha = np.pi * operator["alpha"] / 180.0
                # get rotation axis
                x = operator["n_x"]
                y = operator["n_y"]
                z = operator["n_z"]
                tau_x = operator["tau_x"]
                tau_y = operator["tau_y"]
                tau_z = operator["tau_z"]

                gtrans = np.array([tau_x, tau_y, tau_z])

                R = (
                    np.array(
                        [
                            [
                                np.cos(alpha) + x**2 * (1 - np.cos(alpha)),
                                x * y * (1 - np.cos(alpha)) - z * np.sin(alpha),
                                x * z * (1 - np.cos(alpha)) + y * np.sin(alpha),
                            ],
                            [
                                y * x * (1 - np.cos(alpha)) + z * np.sin(alpha),
                                np.cos(alpha) + y**2 * (1 - np.cos(alpha)),
                                y * z * (1 - np.cos(alpha)) - x * np.sin(alpha),
                            ],
                            [
                                z * x * (1 - np.cos(alpha)) - y * np.sin(alpha),
                                z * y * (1 - np.cos(alpha)) + x * np.sin(alpha),
                                np.cos(alpha) + z**2 * (1 - np.cos(alpha)),
                            ],
                        ]
                    )
                    * det_A
                )

                R = (  # pyright: ignore[reportConstantRedefinition]
                    np.linalg.inv(self.reciprocal_lattice.T).dot(R).dot(self.reciprocal_lattice.T)
                )
                R = np.round(R, decimals=3)  # pyright: ignore[reportConstantRedefinition, reportUnknownVariableType]

                sym_op["irot"] = irot
                sym_op["rotation"] = R
                sym_op["gtrans"] = gtrans
                sym_op["ptrans"] = None
                sym_op["rotmap"] = None

                sym_ops.append(sym_op)

            return sym_ops

        elif vasp60_block_matches:
            logger.info("Detected Space Group Operators in a format from VASP 6.*")
            sym_ops = []
            for vasp60_block_match in vasp60_block_matches:
                sym_op = {}
                irot = int(vasp60_block_match[0])
                r1 = np.array([int(val) for val in vasp60_block_match[1].split()])
                r2 = np.array([int(val) for val in vasp60_block_match[2].split()])
                r3 = np.array([int(val) for val in vasp60_block_match[3].split()])
                g1 = float(vasp60_block_match[4])
                g2 = float(vasp60_block_match[5])
                g3 = float(vasp60_block_match[6])
                gtrans = np.array([g1, g2, g3])
                p1 = float(vasp60_block_match[7])
                p2 = float(vasp60_block_match[8])
                p3 = float(vasp60_block_match[9])
                ptrans = np.array([p1, p2, p3])
                rotmap = vasp60_block_match[10].strip()

                rotation = np.array([r1, r2, r3])
                rotation = np.array(rotation)

                sym_op["irot"] = irot
                sym_op["rotation"] = rotation.T
                sym_op["gtrans"] = gtrans
                sym_op["ptrans"] = ptrans
                sym_op["rotmap"] = rotmap

                sym_ops.append(sym_op)

            return sym_ops

        else:
            logger.info("No space group operators block found")
            return []

    def to_dict(self) -> dict[str, Any]:
        tmp_dict: dict[str, Any] = {}
        symops = self.get_symmetry_operations()
        for symop in symops:
            for key in symop:
                if isinstance(symop[key], np.ndarray):
                    symop[key] = symop[key].tolist()
        tmp_dict["symops"] = symops
        tmp_dict["fermi"] = self.fermi
        tmp_dict["reciprocal_lattice"] = self.reciprocal_lattice.tolist()
        tmp_dict["rotations"] = self.rotations.tolist()
        return tmp_dict

    @override
    def __contains__(self, key: str) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return key in self.__dict__

    @override
    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    @override
    def __iter__(self):
        return self.__dict__.__iter__()

    @override
    def __len__(self):
        return len(self.__dict__)
