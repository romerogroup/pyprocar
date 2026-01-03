from __future__ import annotations

__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Any

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
    def spin_orbit_orbitals(self) -> list[dict]:
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
    def colinear_orbitals(self) -> list[dict]:
        orbitals = [
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
    def orbitals(self) -> list[dict]:
        if self.is_non_colinear:
            return self.spin_orbit_orbitals
        else:
            return self.colinear_orbitals

    @cached_property
    def orbital_names(self) -> list[str]:
        orbital_names = []
        if self.is_non_colinear:
            for orbital in self.spin_orbit_orbitals:
                tmp_name = ""
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
        match = self.root.findall(".//output/band_structure/nbnd_down")
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
            lattice_vectors = []
            for acell in match[0]:
                if acell.text is not None:
                    lattice_vectors.append(np.array(acell.text.split(), dtype=float))
            return np.array(lattice_vectors, dtype=float)
        return None

    @cached_property
    def direct_lattice(self) -> np.ndarray | None:
        match = self.root.findall(".//output/atomic_structure/cell")
        if match:
            lattice_vectors = []
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
        kpoints = np.zeros(shape=(self.n_kpoints, 3))
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
            # Converting back to crystal basis
            kpoints = np.around(kpoints.dot(np.linalg.inv(self.reciprocal_lattice)), decimals=8)
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
