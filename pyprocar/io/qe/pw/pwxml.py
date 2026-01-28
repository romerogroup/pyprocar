from __future__ import annotations

__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np

from pyprocar.utils.units import AU_TO_ANG, HARTREE_TO_EV

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


def str2bool(v: str) -> bool:
    """Converts a string of a boolean to an actual boolean

    Parameters
    ----------
    v : str
        The string of the boolean value

    Returns
    -------
    boolean
        The boolean value
    """
    return v.lower() in ("true")


class PwXML:
    """Parser for the XML output of the PW module in Quantum ESPRESSO."""

    _filepath: Path
    _tree: ET.ElementTree[ET.Element]
    _root: ET.Element

    def __init__(self, filepath: str | Path) -> None:
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
    def tree(self) -> ET.ElementTree[ET.Element]:
        return self._tree

    @cached_property
    def is_non_colinear(self) -> bool | None:
        match = self.root.findall(".//output/magnetization/noncolin")
        if match and match[0].text is not None:
            return str2bool(match[0].text)
        return None

    @cached_property
    def is_spin_calc(self) -> bool | None:
        match = self.root.findall(".//output/magnetization/lsda")
        if match and match[0].text is not None:
            return str2bool(match[0].text)
        return None

    @cached_property
    def is_spin_orbit_calc(self) -> bool | None:
        match = self.root.findall(".//output/magnetization/spinorbit")
        if match and match[0].text is not None:
            return str2bool(match[0].text)
        return None

    def _parse_magnetization(self, _main_xml_root: ET.Element) -> None:
        """A helper method to parse the magnetization tag of the main xml file

        Parameters
        ----------
        _main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """

    @cached_property
    def spin_orbit_orbitals(self) -> list[dict[str, str | float]]:
        return [
            {"l": "s", "j": 0.5, "m": -0.5},
            {"l": "s", "j": 0.5, "m": 0.5},
            {"l": "p", "j": 0.5, "m": -0.5},
            {"l": "p", "j": 0.5, "m": 0.5},
            {"l": "p", "j": 1.5, "m": -1.5},
            {"l": "p", "j": 1.5, "m": -0.5},
            {"l": "p", "j": 1.5, "m": -0.5},
            {"l": "p", "j": 1.5, "m": 1.5},
            {"l": "d", "j": 1.5, "m": -1.5},
            {"l": "d", "j": 1.5, "m": -0.5},
            {"l": "d", "j": 1.5, "m": -0.5},
            {"l": "d", "j": 1.5, "m": 1.5},
            {"l": "d", "j": 2.5, "m": -2.5},
            {"l": "d", "j": 2.5, "m": -1.5},
            {"l": "d", "j": 2.5, "m": -0.5},
            {"l": "d", "j": 2.5, "m": 0.5},
            {"l": "d", "j": 2.5, "m": 1.5},
            {"l": "d", "j": 2.5, "m": 2.5},
        ]

    @cached_property
    def colinear_orbitals(self) -> list[dict[str, int]]:
        orbitals: list[dict[str, int]] = [
            {"l": 0, "m": 1},
            {"l": 1, "m": 3},
            {"l": 1, "m": 1},
            {"l": 1, "m": 2},
            {"l": 2, "m": 5},
            {"l": 2, "m": 3},
            {"l": 2, "m": 1},
            {"l": 2, "m": 2},
            {"l": 2, "m": 4},
        ]
        return orbitals

    @cached_property
    def colinear_orbital_names(self) -> list[str]:
        return [
            "s",
            "py",
            "pz",
            "px",
            "dxy",
            "dyz",
            "dz2",
            "dxz",
            "dx2",
            "tot",
        ]

    @cached_property
    def orbitals(self) -> list[dict[str, str | float]] | list[dict[str, int]]:
        if self.is_non_colinear:
            return self.spin_orbit_orbitals
        else:
            return self.colinear_orbitals

    @cached_property
    def orbital_names(self) -> list[str]:
        orbital_names: list[str] = []
        if self.is_non_colinear:
            for orbital in self.spin_orbit_orbitals:
                tmp_name: str = ""
                for key, value in orbital.items():
                    # print(key,value)
                    if key != "l":
                        tmp_name = tmp_name + key + str(value)
                    else:
                        tmp_name = tmp_name + str(value) + "_"
                orbital_names.append(tmp_name)
            return orbital_names
        else:
            return self.colinear_orbital_names

    @cached_property
    def n_orbitals(self) -> int:
        return len(self.orbitals)

    @cached_property
    def n_spin(self) -> int:
        if self.is_non_colinear:
            return 4
        elif self.is_spin_calc:
            return 2
        else:
            return 1

    @cached_property
    def atm_wfc(self) -> int:
        match = self.root.findall(".//output/band_structure/num_of_atomic_wfc")
        if match and match[0].text is not None:
            return int(match[0].text)
        return 0

    @cached_property
    def n_electrons(self) -> float | None:
        match = self.root.findall(".//output/band_structure/nelec")
        if match and match[0].text is not None:
            return float(match[0].text)
        return None

    @cached_property
    def n_bands(self) -> int:
        match = self.root.findall(".//output/band_structure/nbnd_up")
        if match and match[0].text is not None:
            return int(match[0].text)

        match = self.root.findall(".//output/band_structure/nbnd")
        if match and match[0].text is not None:
            return int(match[0].text)
        return 0

    @cached_property
    def n_bands_up(self) -> int:
        match = self.root.findall(".//output/band_structure/nbnd_up")
        if match and match[0].text is not None:
            return int(match[0].text)
        return 0

    @cached_property
    def n_bands_down(self) -> int:
        match = self.root.findall(".//output/band_structure/nbnd_dw")
        if match and match[0].text is not None:
            return int(match[0].text)
        return 0

    @cached_property
    def n_kpoints(self) -> int:
        match = self.root.findall(".//output/band_structure/nks")
        if match and match[0].text is not None:
            return int(match[0].text)
        return 0

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray | None:
        match = self.root.findall(".//output/basis_set/reciprocal_lattice")
        if match:
            lattice_vectors: list[np.ndarray] = []
            for acell in match[0]:
                if acell.text is not None:
                    lattice_vectors.append(np.array(acell.text.split(), dtype=float))
            return np.array(lattice_vectors, dtype=float)
        return None

    @cached_property
    def direct_lattice(self) -> np.ndarray | None:
        match = self.root.findall(".//output/atomic_structure/cell")
        if match:
            lattice_vectors: list[np.ndarray] = []
            for acell in match[0]:
                if acell.text is not None:
                    lattice_vectors.append(np.array(acell.text.split(), dtype=float))
            return np.array(lattice_vectors, dtype=float) * AU_TO_ANG
        return None

    @cached_property
    def atomic_sites(self) -> dict[str, Any] | None:
        match = self.root.findall(".//output/atomic_structure/atomic_positions")
        if match:
            atomic_sites: dict[str, Any] = {"species": [], "positions": [], "index": []}
            for ion in match[0]:
                atomic_sites["species"].append(ion.attrib["name"])
                if ion.text is not None:
                    atomic_sites["positions"].append(ion.text.split())
                atomic_sites["index"].append(int(ion.attrib["index"]))

            atomic_sites["positions"] = np.array(atomic_sites["positions"], dtype=float)
            return atomic_sites
        return None

    @cached_property
    def atomic_positions(self) -> np.ndarray | None:
        if self.atomic_sites:
            return self.atomic_sites["positions"]
        return None

    @cached_property
    def atomic_species(self) -> list[str] | None:
        if self.atomic_sites:
            return self.atomic_sites["species"]
        return None

    @cached_property
    def atomic_indices(self) -> list[int] | None:
        if self.atomic_sites:
            return self.atomic_sites["index"]
        return None

    @cached_property
    def specie_types(self) -> dict[str, Any] | None:
        match = self.root.findall(".//output/atomic_structure/atomic_species")
        if match:
            atomic_sites: dict[str, Any] = {
                "species": [],
                "mass": [],
                "pseudo_file": [],
                "starting_magnetization": [],
                "spin_teta": [],
            }
            for ion in match[0]:
                atomic_sites["species"].append(ion.attrib["name"])
                atomic_sites["mass"].append(float(ion.attrib["mass"]))
                atomic_sites["pseudo_file"].append(ion.attrib["pseudo_file"])
                atomic_sites["starting_magnetization"].append(
                    float(ion.attrib["starting_magnetization"])
                )
                atomic_sites["spin_teta"].append(float(ion.attrib["spin_teta"]))

            atomic_sites["mass"] = np.array(atomic_sites["mass"], dtype=float)
            atomic_sites["starting_magnetization"] = np.array(
                atomic_sites["starting_magnetization"], dtype=float
            )
            atomic_sites["spin_teta"] = np.array(atomic_sites["spin_teta"], dtype=float)
            return atomic_sites
        return None

    @cached_property
    def n_species_types(self) -> int:
        if self.specie_types:
            return len(self.specie_types["species"])
        return 0

    @cached_property
    def compositions(self) -> dict[str, int]:
        if self.atomic_species and self.specie_types:
            species_types = self.specie_types["species"]
            compositions = {specie_name: 0 for specie_name in species_types}

            for specie_name in self.atomic_species:
                compositions[specie_name] += 1
            return compositions
        return {}

    @cached_property
    def n_atoms(self) -> int:
        if self.atomic_species:
            return len(self.atomic_species)
        return 0

    @cached_property
    def alat(self) -> float | None:
        match = self.root.findall(".//output/atomic_structure")
        if match:
            return float(match[0].attrib["alat"]) * AU_TO_ANG
        return None

    @cached_property
    def ks_energies(self) -> dict[str, np.ndarray] | None:
        ks_energies_match = self.root.findall(".//output/band_structure/ks_energies")
        if not ks_energies_match:
            return None

        logger.debug(f"n_ks_energies: {len(ks_energies_match)}")

        raw_n_bands = self.n_bands if not self.is_spin_calc else self.n_bands * 2

        raw_bands = np.zeros(shape=(self.n_kpoints, raw_n_bands))
        raw_occupations = np.zeros(shape=(self.n_kpoints, raw_n_bands))
        kpoints: np.ndarray = np.zeros(shape=(self.n_kpoints, 3))
        weights = np.zeros(shape=(self.n_kpoints))
        bands = np.zeros(shape=(self.n_kpoints, self.n_bands, self.n_spin))
        occupations = np.zeros(shape=(self.n_kpoints, self.n_bands, self.n_spin))
        npws = np.zeros(shape=(self.n_kpoints, self.n_bands))

        for ikpoint, kpoint_element in enumerate(ks_energies_match):
            kpoints_match = kpoint_element.findall(".//k_point")
            if kpoints_match and kpoints_match[0].text is not None:
                kpoints[ikpoint, :] = np.array(kpoints_match[0].text.split(), dtype=float)

            weight_match = kpoint_element.findall(".//k_point")
            if weight_match:
                weights[ikpoint] = np.array(weight_match[0].attrib["weight"], dtype=float)

            eigenvalues_match = kpoint_element.findall(".//eigenvalues")
            if eigenvalues_match and eigenvalues_match[0].text is not None:
                raw_bands[ikpoint, :] = np.array(eigenvalues_match[0].text.split(), dtype=float)

            occupations_match = kpoint_element.findall(".//occupations")
            if occupations_match and occupations_match[0].text is not None:
                raw_occupations[ikpoint, :] = np.array(
                    occupations_match[0].text.split(), dtype=float
                )

            npws_match = kpoint_element.findall(".//npw")
            if npws_match and npws_match[0].text is not None:
                npws[ikpoint, :] = np.array(npws_match[0].text.split(), dtype=int)

            if self.is_spin_calc:
                bands[ikpoint, :, 0] = raw_bands[ikpoint, : self.n_bands_up]
                bands[ikpoint, :, 1] = raw_bands[ikpoint, self.n_bands_up :]
                occupations[ikpoint, :, 0] = raw_occupations[ikpoint, : self.n_bands_up]
                occupations[ikpoint, :, 1] = raw_occupations[ikpoint, self.n_bands_up :]
            else:
                bands[ikpoint, :, 0] = raw_bands[ikpoint, :]
                occupations[ikpoint, :, 0] = raw_occupations[ikpoint, :]

        if self.alat is not None and self.reciprocal_lattice is not None:
            kpoints = kpoints * (2 * np.pi / self.alat)
            # Converting back to crystal basis - cast to ensure type is known
            kpoints = cast(
                np.ndarray,
                np.around(kpoints.dot(np.linalg.inv(self.reciprocal_lattice)), decimals=8),
            )
        # print(bands[:,:,0].shape)
        # print(bands[:,self.n_bands:,1].shape)
        # print(np.allclose(bands[...,0], bands[...,1]))

        ks_energies_dict: dict[str, np.ndarray] = {
            "bands": bands,
            "occupations": occupations,
            "kpoints": kpoints,
            "weights": weights,
        }
        return ks_energies_dict

    @cached_property
    def kpoints(self) -> np.ndarray | None:
        if self.ks_energies:
            return self.ks_energies["kpoints"]
        return None

    @cached_property
    def weights(self) -> np.ndarray | None:
        if self.ks_energies:
            return self.ks_energies["weights"]
        return None

    @cached_property
    def bands(self) -> np.ndarray | None:
        if self.ks_energies:
            return self.ks_energies["bands"]
        return None

    @cached_property
    def occupations(self) -> np.ndarray | None:
        if self.ks_energies:
            return self.ks_energies["occupations"]
        return None

    @cached_property
    def symmetries_element(self) -> ET.Element | None:
        match = self.root.findall(".//output/symmetries")
        if match:
            return match[0]
        return None

    @cached_property
    def n_symmetries(self) -> int:
        if self.symmetries_element is None:
            return 0
        match = self.symmetries_element.findall(".//nsym")
        if match and match[0].text is not None:
            return int(match[0].text)
        return 0

    @cached_property
    def n_rotations(self) -> int:
        if self.symmetries_element is None:
            return 0
        match = self.symmetries_element.findall(".//nrot")
        if match and match[0].text is not None:
            return int(match[0].text)
        return 0

    @cached_property
    def spg(self) -> int:
        if self.symmetries_element is None:
            return 0
        match = self.symmetries_element.findall(".//space_group")
        if match and match[0].text is not None:
            return int(match[0].text)
        return 0

    @cached_property
    def symmetry_operations_elements(self) -> ET.Element | None:
        if self.symmetries_element is None:
            return None
        match = self.symmetries_element.findall(".//symmetries")
        if match:
            return match[0]
        return None

    @cached_property
    def n_sym_ops(self) -> int:
        if self.symmetries_element is None:
            return 0
        n_sym_match = self.symmetries_element.findall(".//nsym")
        if n_sym_match and n_sym_match[0].text is not None:
            return int(n_sym_match[0].text)
        return 0

    @cached_property
    def n_rot(self) -> int:
        if self.symmetries_element is None:
            return 0
        n_rot_match = self.symmetries_element.findall(".//nrot")
        if n_rot_match and n_rot_match[0].text is not None:
            return int(n_rot_match[0].text)
        return 0

    @cached_property
    def sym_ops(self) -> dict[str, Any] | None:
        if self.symmetries_element is None:
            return None
        sym_ops_match = self.symmetries_element.findall(".//symmetry")
        if sym_ops_match:
            sym_ops: dict[str, Any] = {
                "rotations": [],
                "translations": [],
                "equivalent_atoms": [],
            }

            for symmetry_operation in sym_ops_match:
                rotation_match = symmetry_operation.findall(".//rotation")
                if rotation_match and rotation_match[0].text is not None:
                    rotation = np.array(rotation_match[0].text.split(), dtype=float)
                else:
                    rotation = np.eye(3).flatten()
                rotation = rotation.reshape(3, 3).T

                sym_ops["rotations"].append(rotation)

                fractional_translation = symmetry_operation.findall(".//fractional_translation")
                if fractional_translation and fractional_translation[0].text is not None:
                    sym_ops["translations"].append(
                        np.array(fractional_translation[0].text.split(), dtype=float)
                    )
                else:
                    sym_ops["translations"].append(np.zeros(3))

                equivalent_atoms = symmetry_operation.findall(".//equivalent_atoms")
                if equivalent_atoms and equivalent_atoms[0].text is not None:
                    sym_ops["equivalent_atoms"].append(
                        np.array(equivalent_atoms[0].text.split(), dtype=int)
                    )
                else:
                    sym_ops["equivalent_atoms"].append(np.zeros(5))

            sym_ops["rotations"] = np.array(sym_ops["rotations"], dtype=float)
            sym_ops["translations"] = np.array(sym_ops["translations"], dtype=float)
            return sym_ops
        return None

    @cached_property
    def rotations(self) -> np.ndarray | None:
        if self.sym_ops:
            return self.sym_ops["rotations"]
        return None

    @cached_property
    def fermi(self) -> float | None:
        match = self.root.findall(".//output/band_structure/fermi_energy")
        if match and match[0].text is not None:
            return float(match[0].text) * HARTREE_TO_EV
        return None

    @cached_property
    def kmesh_mode(self) -> str | None:
        monkhorst_pack_match = self.root.findall(
            ".//output/band_structure/starting_k_points/monkhorst_pack"
        )
        gamma_point_match = self.root.findall(
            ".//output/band_structure/starting_k_points/gamma_point"
        )
        if monkhorst_pack_match:
            return "monkhorst_pack"
        elif gamma_point_match:
            return "gamma_point"
        return None

    @cached_property
    def nk1(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["nk1"])
        return 0

    @cached_property
    def nk2(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["nk2"])
        return 0

    @cached_property
    def nk3(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["nk3"])
        return 0

    @cached_property
    def sk1(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["k1"])
        return 0

    @cached_property
    def sk2(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["k2"])
        return 0

    @cached_property
    def sk3(self) -> int:
        match = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack")
        if match:
            return int(match[0].attrib["k3"])
        return 0

    @cached_property
    def functional(self) -> str | None:
        """Return the DFT functional used in the calculation."""
        match = self.root.findall(".//output/dft/functional")
        if match and match[0].text is not None:
            return match[0].text.strip()
        return None

    @cached_property
    def total_energy(self) -> dict[str, float] | None:
        """Return the total energy components from the calculation.

        Returns a dictionary with keys: etot, eband, ehart, vtxc, etxc, ewald, demet.
        Values are in Hartree units as stored in the XML.
        """
        match = self.root.findall(".//output/total_energy")
        if not match:
            return None

        energy_dict: dict[str, float] = {}
        energy_tags = ["etot", "eband", "ehart", "vtxc", "etxc", "ewald", "demet"]

        for tag in energy_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                energy_dict[tag] = float(tag_match[0].text)

        return energy_dict if energy_dict else None

    @cached_property
    def etot(self) -> float | None:
        """Return the total energy (etot) in Hartree."""
        if self.total_energy and "etot" in self.total_energy:
            return self.total_energy["etot"]
        return None

    @cached_property
    def eband(self) -> float | None:
        """Return the band energy (eband) in Hartree."""
        if self.total_energy and "eband" in self.total_energy:
            return self.total_energy["eband"]
        return None

    @cached_property
    def ehart(self) -> float | None:
        """Return the Hartree energy (ehart) in Hartree."""
        if self.total_energy and "ehart" in self.total_energy:
            return self.total_energy["ehart"]
        return None

    @cached_property
    def vtxc(self) -> float | None:
        """Return the exchange-correlation potential energy (vtxc) in Hartree."""
        if self.total_energy and "vtxc" in self.total_energy:
            return self.total_energy["vtxc"]
        return None

    @cached_property
    def etxc(self) -> float | None:
        """Return the exchange-correlation energy (etxc) in Hartree."""
        if self.total_energy and "etxc" in self.total_energy:
            return self.total_energy["etxc"]
        return None

    @cached_property
    def ewald(self) -> float | None:
        """Return the Ewald energy (ewald) in Hartree."""
        if self.total_energy and "ewald" in self.total_energy:
            return self.total_energy["ewald"]
        return None

    @cached_property
    def demet(self) -> float | None:
        """Return the smearing energy correction (demet) in Hartree."""
        if self.total_energy and "demet" in self.total_energy:
            return self.total_energy["demet"]
        return None

    @cached_property
    def exit_status(self) -> int | None:
        """Return the exit status of the calculation.

        Returns 0 for successful completion, non-zero for errors.
        """
        match = self.root.findall(".//exit_status")
        if match and match[0].text is not None:
            return int(match[0].text)
        return None

    @cached_property
    def timing_info(self) -> dict[str, Any] | None:
        """Return timing information from the calculation.

        Returns a dictionary with keys:
        - 'total': dict with label, cpu, and wall time for total timing
        - 'partial': list of dicts with label, calls, cpu, and wall time for partial timings
        """
        match = self.root.findall(".//timing_info")
        if not match:
            return None

        timing_dict: dict[str, Any] = {}

        # Parse total timing
        total_match = match[0].findall(".//total")
        if total_match:
            total_element = total_match[0]
            total_dict: dict[str, Any] = {}

            if "label" in total_element.attrib:
                total_dict["label"] = total_element.attrib["label"]

            cpu_match = total_element.findall(".//cpu")
            if cpu_match and cpu_match[0].text is not None:
                total_dict["cpu"] = float(cpu_match[0].text)

            wall_match = total_element.findall(".//wall")
            if wall_match and wall_match[0].text is not None:
                total_dict["wall"] = float(wall_match[0].text)

            timing_dict["total"] = total_dict

        # Parse partial timings
        partial_match = match[0].findall(".//partial")
        if partial_match:
            partial_list: list[dict[str, Any]] = []

            for partial_element in partial_match:
                partial_dict: dict[str, Any] = {}

                if "label" in partial_element.attrib:
                    partial_dict["label"] = partial_element.attrib["label"]

                if "calls" in partial_element.attrib:
                    partial_dict["calls"] = int(partial_element.attrib["calls"])

                cpu_match = partial_element.findall(".//cpu")
                if cpu_match and cpu_match[0].text is not None:
                    partial_dict["cpu"] = float(cpu_match[0].text)

                wall_match = partial_element.findall(".//wall")
                if wall_match and wall_match[0].text is not None:
                    partial_dict["wall"] = float(wall_match[0].text)

                partial_list.append(partial_dict)

            timing_dict["partial"] = partial_list

        return timing_dict if timing_dict else None

    @cached_property
    def total_cpu_time(self) -> float | None:
        """Return the total CPU time in seconds."""
        if self.timing_info and "total" in self.timing_info:
            return self.timing_info["total"].get("cpu")
        return None

    @cached_property
    def total_wall_time(self) -> float | None:
        """Return the total wall time in seconds."""
        if self.timing_info and "total" in self.timing_info:
            return self.timing_info["total"].get("wall")
        return None

    @cached_property
    def closed_info(self) -> dict[str, str] | None:
        """Return information from the closed tag.

        Returns a dictionary with keys:
        - 'date': The date when the calculation closed
        - 'time': The time when the calculation closed
        """
        match = self.root.findall(".//closed")
        if not match:
            return None

        closed_dict: dict[str, str] = {}

        if "DATE" in match[0].attrib:
            closed_dict["date"] = match[0].attrib["DATE"]

        if "TIME" in match[0].attrib:
            closed_dict["time"] = match[0].attrib["TIME"]

        return closed_dict if closed_dict else None

    @cached_property
    def closed_date(self) -> str | None:
        """Return the date when the calculation closed."""
        if self.closed_info and "date" in self.closed_info:
            return self.closed_info["date"]
        return None

    @cached_property
    def closed_time(self) -> str | None:
        """Return the time when the calculation closed."""
        if self.closed_info and "time" in self.closed_info:
            return self.closed_info["time"]
        return None

    @cached_property
    def general_info(self) -> dict[str, Any] | None:
        """Return general information from the calculation.

        Returns a dictionary with keys:
        - 'xml_format': dict with name, version, and text
        - 'creator': dict with name, version, and text
        - 'created': dict with date, time, and text
        - 'job': string with job name (empty string if not set)
        """
        match = self.root.findall(".//general_info")
        if not match:
            return None

        general_dict: dict[str, Any] = {}

        # Parse xml_format
        xml_format_match = match[0].findall(".//xml_format")
        if xml_format_match:
            xml_format_dict: dict[str, str] = {}
            element = xml_format_match[0]

            if "NAME" in element.attrib:
                xml_format_dict["name"] = element.attrib["NAME"]
            if "VERSION" in element.attrib:
                xml_format_dict["version"] = element.attrib["VERSION"]
            if element.text is not None:
                xml_format_dict["text"] = element.text.strip()

            general_dict["xml_format"] = xml_format_dict

        # Parse creator
        creator_match = match[0].findall(".//creator")
        if creator_match:
            creator_dict: dict[str, str] = {}
            element = creator_match[0]

            if "NAME" in element.attrib:
                creator_dict["name"] = element.attrib["NAME"]
            if "VERSION" in element.attrib:
                creator_dict["version"] = element.attrib["VERSION"]
            if element.text is not None:
                creator_dict["text"] = element.text.strip()

            general_dict["creator"] = creator_dict

        # Parse created
        created_match = match[0].findall(".//created")
        if created_match:
            created_dict: dict[str, str] = {}
            element = created_match[0]

            if "DATE" in element.attrib:
                created_dict["date"] = element.attrib["DATE"]
            if "TIME" in element.attrib:
                created_dict["time"] = element.attrib["TIME"]
            if element.text is not None:
                created_dict["text"] = element.text.strip()

            general_dict["created"] = created_dict

        # Parse job
        job_match = match[0].findall(".//job")
        if job_match:
            general_dict["job"] = job_match[0].text.strip() if job_match[0].text else ""

        return general_dict if general_dict else None

    @cached_property
    def xml_format_name(self) -> str | None:
        """Return the XML format name."""
        if self.general_info and "xml_format" in self.general_info:
            return self.general_info["xml_format"].get("name")
        return None

    @cached_property
    def xml_format_version(self) -> str | None:
        """Return the XML format version."""
        if self.general_info and "xml_format" in self.general_info:
            return self.general_info["xml_format"].get("version")
        return None

    @cached_property
    def creator_name(self) -> str | None:
        """Return the creator name (e.g., PWSCF)."""
        if self.general_info and "creator" in self.general_info:
            return self.general_info["creator"].get("name")
        return None

    @cached_property
    def creator_version(self) -> str | None:
        """Return the creator version (e.g., 7.2)."""
        if self.general_info and "creator" in self.general_info:
            return self.general_info["creator"].get("version")
        return None

    @cached_property
    def created_date(self) -> str | None:
        """Return the date when the file was created."""
        if self.general_info and "created" in self.general_info:
            return self.general_info["created"].get("date")
        return None

    @cached_property
    def created_time(self) -> str | None:
        """Return the time when the file was created."""
        if self.general_info and "created" in self.general_info:
            return self.general_info["created"].get("time")
        return None

    @cached_property
    def job(self) -> str | None:
        """Return the job name."""
        if self.general_info and "job" in self.general_info:
            return self.general_info["job"]
        return None

    @cached_property
    def parallel_info(self) -> dict[str, int] | None:
        """Return parallel execution information from the calculation.

        Returns a dictionary with keys:
        - 'nprocs': number of MPI processes
        - 'nthreads': number of threads per process
        - 'ntasks': number of tasks
        - 'nbgrp': number of band groups
        - 'npool': number of pools
        - 'ndiag': number of processors for diagonalization
        """
        match = self.root.findall(".//parallel_info")
        if not match:
            return None

        parallel_dict: dict[str, int] = {}
        tags = ["nprocs", "nthreads", "ntasks", "nbgrp", "npool", "ndiag"]

        for tag in tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                parallel_dict[tag] = int(tag_match[0].text)

        return parallel_dict if parallel_dict else None

    @cached_property
    def nprocs(self) -> int | None:
        """Return the number of MPI processes."""
        if self.parallel_info and "nprocs" in self.parallel_info:
            return self.parallel_info["nprocs"]
        return None

    @cached_property
    def nthreads(self) -> int | None:
        """Return the number of threads per process."""
        if self.parallel_info and "nthreads" in self.parallel_info:
            return self.parallel_info["nthreads"]
        return None

    @cached_property
    def ntasks(self) -> int | None:
        """Return the number of tasks."""
        if self.parallel_info and "ntasks" in self.parallel_info:
            return self.parallel_info["ntasks"]
        return None

    @cached_property
    def nbgrp(self) -> int | None:
        """Return the number of band groups."""
        if self.parallel_info and "nbgrp" in self.parallel_info:
            return self.parallel_info["nbgrp"]
        return None

    @cached_property
    def npool(self) -> int | None:
        """Return the number of pools."""
        if self.parallel_info and "npool" in self.parallel_info:
            return self.parallel_info["npool"]
        return None

    @cached_property
    def ndiag(self) -> int | None:
        """Return the number of processors for diagonalization."""
        if self.parallel_info and "ndiag" in self.parallel_info:
            return self.parallel_info["ndiag"]
        return None

    # =========================================================================
    # Input Section Properties
    # =========================================================================

    @cached_property
    def control_variables(self) -> dict[str, Any] | None:
        """Return control variables from input section.

        Returns a dictionary with control settings like calculation type,
        prefix, pseudo_dir, outdir, etc.
        """
        match = self.root.findall(".//input/control_variables")
        if not match:
            return None

        control_dict: dict[str, Any] = {}

        # String tags
        string_tags = [
            "title",
            "calculation",
            "restart_mode",
            "prefix",
            "pseudo_dir",
            "outdir",
            "disk_io",
            "verbosity",
        ]
        for tag in string_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                control_dict[tag] = tag_match[0].text.strip()

        # Boolean tags
        bool_tags = ["stress", "forces", "wf_collect", "fcp", "rism"]
        for tag in bool_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                control_dict[tag] = str2bool(tag_match[0].text)

        # Integer tags
        int_tags = ["max_seconds", "nstep", "print_every"]
        for tag in int_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                control_dict[tag] = int(tag_match[0].text)

        # Float tags
        float_tags = ["etot_conv_thr", "forc_conv_thr", "press_conv_thr"]
        for tag in float_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                control_dict[tag] = float(tag_match[0].text)

        return control_dict if control_dict else None

    @cached_property
    def calculation(self) -> str | None:
        """Return the calculation type (scf, bands, relax, etc.)."""
        if self.control_variables and "calculation" in self.control_variables:
            return self.control_variables["calculation"]
        return None

    @cached_property
    def prefix(self) -> str | None:
        """Return the calculation prefix."""
        if self.control_variables and "prefix" in self.control_variables:
            return self.control_variables["prefix"]
        return None

    @cached_property
    def input_atomic_species(self) -> dict[str, Any] | None:
        """Return atomic species information from input section.

        Returns a dictionary with:
        - 'ntyp': number of species types
        - 'species': list of dicts with name, mass, pseudo_file, starting_magnetization
        """
        match = self.root.findall(".//input/atomic_species")
        if not match:
            return None

        species_dict: dict[str, Any] = {}

        if "ntyp" in match[0].attrib:
            species_dict["ntyp"] = int(match[0].attrib["ntyp"])

        species_list: list[dict[str, Any]] = []
        species_match = match[0].findall(".//species")
        for species_elem in species_match:
            sp: dict[str, Any] = {}

            if "name" in species_elem.attrib:
                sp["name"] = species_elem.attrib["name"]

            mass_match = species_elem.findall(".//mass")
            if mass_match and mass_match[0].text is not None:
                sp["mass"] = float(mass_match[0].text)

            pseudo_match = species_elem.findall(".//pseudo_file")
            if pseudo_match and pseudo_match[0].text is not None:
                sp["pseudo_file"] = pseudo_match[0].text.strip()

            mag_match = species_elem.findall(".//starting_magnetization")
            if mag_match and mag_match[0].text is not None:
                sp["starting_magnetization"] = float(mag_match[0].text)

            species_list.append(sp)

        species_dict["species"] = species_list
        return species_dict if species_dict else None

    @cached_property
    def input_spin(self) -> dict[str, bool] | None:
        """Return spin settings from input section.

        Returns a dictionary with:
        - 'lsda': spin-polarized calculation
        - 'noncolin': non-collinear calculation
        - 'spinorbit': spin-orbit coupling
        """
        match = self.root.findall(".//input/spin")
        if not match:
            return None

        spin_dict: dict[str, bool] = {}

        bool_tags = ["lsda", "noncolin", "spinorbit"]
        for tag in bool_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                spin_dict[tag] = str2bool(tag_match[0].text)

        return spin_dict if spin_dict else None

    @cached_property
    def input_bands(self) -> dict[str, Any] | None:
        """Return bands settings from input section.

        Returns a dictionary with:
        - 'occupations': occupation method
        - 'smearing': dict with type and degauss
        - 'tot_charge': total charge
        """
        match = self.root.findall(".//input/bands")
        if not match:
            return None

        bands_dict: dict[str, Any] = {}

        # Occupations
        occ_match = match[0].findall(".//occupations")
        if occ_match and occ_match[0].text is not None:
            bands_dict["occupations"] = occ_match[0].text.strip()

        # Smearing
        smearing_match = match[0].findall(".//smearing")
        if smearing_match:
            smearing_dict: dict[str, Any] = {}
            if smearing_match[0].text is not None:
                smearing_dict["type"] = smearing_match[0].text.strip()
            if "degauss" in smearing_match[0].attrib:
                smearing_dict["degauss"] = float(smearing_match[0].attrib["degauss"])
            bands_dict["smearing"] = smearing_dict

        # Total charge
        charge_match = match[0].findall(".//tot_charge")
        if charge_match and charge_match[0].text is not None:
            bands_dict["tot_charge"] = float(charge_match[0].text)

        return bands_dict if bands_dict else None

    @cached_property
    def input_basis(self) -> dict[str, Any] | None:
        """Return basis settings from input section.

        Returns a dictionary with:
        - 'gamma_only': gamma-point only calculation
        - 'ecutwfc': wavefunction cutoff (Ry)
        - 'ecutrho': charge density cutoff (Ry)
        """
        match = self.root.findall(".//input/basis")
        if not match:
            return None

        basis_dict: dict[str, Any] = {}

        # Gamma only
        gamma_match = match[0].findall(".//gamma_only")
        if gamma_match and gamma_match[0].text is not None:
            basis_dict["gamma_only"] = str2bool(gamma_match[0].text)

        # Cutoffs
        ecutwfc_match = match[0].findall(".//ecutwfc")
        if ecutwfc_match and ecutwfc_match[0].text is not None:
            basis_dict["ecutwfc"] = float(ecutwfc_match[0].text)

        ecutrho_match = match[0].findall(".//ecutrho")
        if ecutrho_match and ecutrho_match[0].text is not None:
            basis_dict["ecutrho"] = float(ecutrho_match[0].text)

        return basis_dict if basis_dict else None

    @cached_property
    def ecutwfc(self) -> float | None:
        """Return the wavefunction cutoff energy (Ry)."""
        if self.input_basis and "ecutwfc" in self.input_basis:
            return self.input_basis["ecutwfc"]
        return None

    @cached_property
    def ecutrho(self) -> float | None:
        """Return the charge density cutoff energy (Ry)."""
        if self.input_basis and "ecutrho" in self.input_basis:
            return self.input_basis["ecutrho"]
        return None

    @cached_property
    def electron_control(self) -> dict[str, Any] | None:
        """Return electron control settings from input section.

        Returns a dictionary with diagonalization, mixing, and convergence settings.
        """
        match = self.root.findall(".//input/electron_control")
        if not match:
            return None

        electron_dict: dict[str, Any] = {}

        # String tags
        string_tags = ["diagonalization", "mixing_mode"]
        for tag in string_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                electron_dict[tag] = tag_match[0].text.strip()

        # Float tags
        float_tags = ["mixing_beta", "conv_thr", "diago_thr_init"]
        for tag in float_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                electron_dict[tag] = float(tag_match[0].text)

        # Integer tags
        int_tags = [
            "mixing_ndim",
            "max_nstep",
            "exx_nstep",
            "diago_cg_maxiter",
            "diago_ppcg_maxiter",
            "diago_rmm_ndim",
            "diago_gs_nblock",
        ]
        for tag in int_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                electron_dict[tag] = int(tag_match[0].text)

        # Boolean tags
        bool_tags = [
            "real_space_q",
            "real_space_beta",
            "tq_smoothing",
            "tbeta_smoothing",
            "diago_full_acc",
            "diago_rmm_conv",
        ]
        for tag in bool_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                electron_dict[tag] = str2bool(tag_match[0].text)

        return electron_dict if electron_dict else None

    @cached_property
    def k_points_ibz(self) -> dict[str, Any] | None:
        """Return k-points in the irreducible Brillouin zone from input section.

        Returns a dictionary with:
        - 'nk': number of k-points
        - 'k_points': list of dicts with weight and coordinates
        """
        match = self.root.findall(".//input/k_points_IBZ")
        if not match:
            return None

        kpoints_dict: dict[str, Any] = {}

        # Number of k-points
        nk_match = match[0].findall(".//nk")
        if nk_match and nk_match[0].text is not None:
            kpoints_dict["nk"] = int(nk_match[0].text)

        # K-points list
        kpoint_list: list[dict[str, Any]] = []
        kpoint_match = match[0].findall(".//k_point")
        for kpoint_elem in kpoint_match:
            kp: dict[str, Any] = {}

            if "weight" in kpoint_elem.attrib:
                kp["weight"] = float(kpoint_elem.attrib["weight"])

            if kpoint_elem.text is not None:
                kp["coordinates"] = np.array(kpoint_elem.text.split(), dtype=float)

            kpoint_list.append(kp)

        kpoints_dict["k_points"] = kpoint_list
        return kpoints_dict if kpoints_dict else None

    @cached_property
    def ion_control(self) -> dict[str, Any] | None:
        """Return ion control settings from input section.

        Returns a dictionary with ion dynamics settings.
        """
        match = self.root.findall(".//input/ion_control")
        if not match:
            return None

        ion_dict: dict[str, Any] = {}

        # String tags
        string_tags = ["ion_dynamics"]
        for tag in string_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                ion_dict[tag] = tag_match[0].text.strip()

        # Float tags
        float_tags = ["upscale"]
        for tag in float_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                ion_dict[tag] = float(tag_match[0].text)

        # Boolean tags
        bool_tags = ["remove_rigid_rot", "refold_pos"]
        for tag in bool_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                ion_dict[tag] = str2bool(tag_match[0].text)

        return ion_dict if ion_dict else None

    @cached_property
    def cell_control(self) -> dict[str, Any] | None:
        """Return cell control settings from input section.

        Returns a dictionary with cell dynamics settings.
        """
        match = self.root.findall(".//input/cell_control")
        if not match:
            return None

        cell_dict: dict[str, Any] = {}

        # String tags
        string_tags = ["cell_dynamics", "cell_do_free"]
        for tag in string_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                cell_dict[tag] = tag_match[0].text.strip()

        # Float tags
        float_tags = ["pressure", "wmass"]
        for tag in float_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                cell_dict[tag] = float(tag_match[0].text)

        return cell_dict if cell_dict else None

    @cached_property
    def symmetry_flags(self) -> dict[str, bool] | None:
        """Return symmetry flags from input section.

        Returns a dictionary with symmetry settings.
        """
        match = self.root.findall(".//input/symmetry_flags")
        if not match:
            return None

        sym_dict: dict[str, bool] = {}

        bool_tags = [
            "nosym",
            "nosym_evc",
            "noinv",
            "no_t_rev",
            "force_symmorphic",
            "use_all_frac",
        ]
        for tag in bool_tags:
            tag_match = match[0].findall(f".//{tag}")
            if tag_match and tag_match[0].text is not None:
                sym_dict[tag] = str2bool(tag_match[0].text)

        return sym_dict if sym_dict else None
