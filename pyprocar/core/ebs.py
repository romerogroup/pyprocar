# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: Logan Lang
@author: Pedram Tavadze
@author: Freddy Farah

"""

import copy
import itertools
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pyvista as pv

from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.core.kpath import KPath
from pyprocar.core.serializer import get_serializer
from pyprocar.core.structure import Structure
from pyprocar.utils import mathematics
from pyprocar.utils.physics_constants import (
    EV_TO_J,
    FREE_ELECTRON_MASS,
    HBAR_EV,
    HBAR_J,
    METER_ANGSTROM,
)
from pyprocar.utils.unfolder import Unfolder

logger = logging.getLogger(__name__)

NUMERICAL_STABILITY_FACTOR = 0.0001


def reduced_to_cartesian(kpoints, reciprocal_lattice):
    if reciprocal_lattice is not None:
        return np.dot(kpoints, reciprocal_lattice)
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return


def cartesian_to_reduced(cartesian, reciprocal_lattice):
    """Converts cartesian coordinates to fractional coordinates

    Parameters
    ----------
    cartesian : np.ndarray
        The cartesian coordinates. shape = [N,3]
    reciprocal_lattice : np.ndarray
        The reciprocal lattice vector matrix. Will have the shape (3, 3), defaults to None

    Returns
    -------
    np.ndarray
        The fractional coordinates. shape = [N,3]
    """
    if reciprocal_lattice is not None:
        return np.dot(cartesian, np.linalg.inv(reciprocal_lattice))
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return


def calculate_avg_inv_effective_mass(hessian):
    # Calculate the trace of each 3x3 matrix along the last two axes
    m_inv = (np.trace(hessian, axis1=-2, axis2=-1) * EV_TO_J / HBAR_J**2) / 3
    # Calculate the harmonic average effective mass for each element
    e_mass = FREE_ELECTRON_MASS * m_inv
    return e_mass


def _deepcopy_dict(d):
    """Performs a deep copy of a dictionary containing NumPy arrays."""
    if d is None:
        return None
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in d.items()}


class ElectronicBandStructure:
    """This object stores electronic band structure informomration.

    Parameters
    ----------
    kpoints : np.ndarray
        The kpoints array. Will have the shape (n_kpoints, 3)
    bands : np.ndarray
        The bands array. Will have the shape (n_kpoints, n_bands)
    efermi : float
        The fermi energy
    projected : np.ndarray, optional
        The projections array. Will have the shape (n_kpoints, n_bands, n_spins, norbitals,n_atoms), defaults to None
    projected_phase : np.ndarray, optional
        The full projections array that incudes the complex part. Will have the shape (n_kpoints, n_bands, n_spins, norbitals,n_atoms), defaults to None
    weights : np.ndarray, optional
        The weights of the kpoints. Will have the shape (n_kpoints, 1), defaults to None
    orbital_names : List, optional
        The names of the orbitals. Defaults to None
    reciprocal_lattice : np.ndarray, optional
        The reciprocal lattice vector matrix. Will have the shape (3, 3), defaults to None
    shifted_to_efermi : bool, optional
         Boolean to determine if the fermi energy is shifted, defaults to False
    """

    def __init__(
        self,
        kpoints: np.ndarray,
        bands: np.ndarray,
        efermi: float = 0.0,
        projected: np.ndarray = None,
        projected_phase: np.ndarray = None,
        weights: np.ndarray = None,
        orbital_names: List = None,
        reciprocal_lattice: np.ndarray = None,
        shifted_to_efermi: bool = False,
        structure: Structure = None,
        properties: Dict[str, np.ndarray] = None,
        gradients: Dict[str, np.ndarray] = None,
        hessians: Dict[str, np.ndarray] = None,
    ):
        self._kpoints = np.array(kpoints, dtype=float).copy()
        # We store energies both in its own attribute and in the properties dict
        # to allow for a unified way of handling derivatives.

        self._efermi = efermi
        self._orbital_names = orbital_names
        self._reciprocal_lattice = reciprocal_lattice
        self._shifted_to_efermi = shifted_to_efermi

        properties = {} if properties is None else properties
        gradients = {} if gradients is None else gradients
        hessians = {} if hessians is None else hessians

        self._properties = _deepcopy_dict(properties)

        self._properties["bands"] = np.array(bands, dtype=float).copy()

        if projected is not None:
            self._properties["projected"] = np.array(projected, dtype=float).copy()
        if projected_phase is not None:
            self._properties["projected_phase"] = np.array(
                projected_phase, dtype=float
            ).copy()
        if weights is not None:
            self._properties["weights"] = np.array(weights, dtype=float).copy()

        self._gradients = _deepcopy_dict(gradients)
        self._hessians = _deepcopy_dict(hessians)
        self._structure = structure

        # Make attributes read-only
        # self._kpoints.flags.writeable = False
        # for prop in self._properties.values():
        #     if isinstance(prop, np.ndarray):
        #         prop.flags.writeable = False
        # for prop in self._gradients.values():
        #     if isinstance(prop, np.ndarray):
        #         prop.flags.writeable = False
        # for prop in self._hessians.values():
        #     if isinstance(prop, np.ndarray):
        #         prop.flags.writeable = False

    def __str__(self):
        ret = "\n Electronic Band Structure     \n"
        ret += "============================\n"
        ret += "Total number of kpoints   = {}\n".format(self.n_kpoints)
        ret += "Total number of bands    = {}\n".format(self.n_bands)
        ret += "Total number of atoms    = {}\n".format(self.n_atoms)
        ret += "Total number of orbitals = {}\n".format(self.n_orbitals)
        ret += "Total number of spin channels = {}\n".format(self.n_spin_channels)
        ret += "Total number of spin projections = {}\n".format(self.n_spins)

        ret += "\nArray shapes: \n"
        ret += "------------------------     \n"
        k_properties = self.property_names
        for prop in k_properties:
            ret += f"{prop} shape = {self.get_property(prop).shape}\n"

        ret += "\nGradients: \n"
        ret += "------------------------     \n"
        for prop in self._gradients.keys():
            ret += f"{prop} shape = {self.get_gradient(prop).shape}\n"

        ret += "\nHessians: \n"
        ret += "------------------------     \n"
        for prop in self._hessians.keys():
            ret += f"{prop} shape = {self.get_hessian(prop).shape}\n"

        ret += "\nAdditional information: \n"
        if self.orbital_names is not None:
            ret += "Orbital Names = {}\n".format(self.orbital_names)
        ret += f"Spin Projection Names = {self.spin_projection_names}\n"
        ret += f"Non-colinear = {self.is_non_collinear}\n"
        if self.reciprocal_lattice is not None:
            ret += "Reciprocal Lattice = \n {}\n".format(self.reciprocal_lattice)
        ret += "Fermi Energy = {}\n".format(self.efermi)
        ret += "Has Phase = {}\n".format(self.has_phase)
        if self.structure is not None:
            ret += "\nStructure: \n"
            ret += "------------------------     \n"
            ret += "Structure = \n {}\n".format(self.structure)

        return ret

    @property
    def kpoints(self):
        return self._kpoints

    @property
    def bands(self):
        return self.get_property("bands")

    @property
    def projected(self):
        return self.get_property("projected")

    @property
    def projected_phase(self):
        return self.get_property("projected_phase")

    @property
    def weights(self):
        return self.get_property("weights")

    @property
    def orbital_names(self):
        return self._orbital_names

    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice

    @property
    def efermi(self):
        return self._efermi

    @property
    def structure(self):
        return self._structure

    @property
    def n_kpoints(self):
        """The number of k points

        Returns
        -------
        int
            The number of k points
        """
        return self.kpoints.shape[0]

    @property
    def n_bands(self):
        """The number of bands

        Returns
        -------
        int
            The number of bands
        """
        return self.bands.shape[1]

    @property
    def n_spins(self):
        """The number of spin projections

        Returns
        -------
        int
            The number of spin projections
        """
        return self.projected.shape[2]

    @property
    def n_atoms(self):
        """The number of atoms

        Returns
        -------
        int
            The number of atoms
        """
        return self.projected.shape[3]

    @property
    def n_orbitals(self):
        """The number of orbitals

        Returns
        -------
        int
            The number of orbitals
        """
        return self.projected.shape[4]

    @property
    def n_spin_channels(self):
        """The number of spin channels

        Returns
        -------
        int
            The number of spin channels
        """
        return self.bands.shape[2]

    @property
    def spin_channels(self):
        """The number of spin channels

        Returns
        -------
        int
            The number of spin channels
        """

        return np.arange(self.n_spin_channels)

    @property
    def spin_projection_names(self):
        spin_projection_names = ["Spin-up", "Spin-down"]
        if self.is_non_collinear:
            return ["total", "x", "y", "z"]
        elif self.n_spins == 2:
            return spin_projection_names
        else:
            return spin_projection_names[:1]

    @property
    def kpoints_cartesian(self):
        return reduced_to_cartesian(self.kpoints, self.reciprocal_lattice)

    @property
    def inv_reciprocal_lattice(self):
        """Returns the inverse of the reciprocal lattice"""
        if self.reciprocal_lattice is not None:
            return np.linalg.inv(self.reciprocal_lattice)
        else:
            print(
                "Please provide a reciprocal lattice when initiating the Procar class"
            )
            return None

    @property
    def is_non_collinear(self):
        """Boolean to determine if this is a non-colinear calculation

        Returns
        -------
        bool
            Boolean to determine if this is a non-colinear calculation
        """
        if self.n_spins == 4:
            return True
        else:
            return False

    @property
    def has_phase(self):
        """Boolean to determine if this is a phase calculation

        Returns
        -------
        bool
            Boolean to determine if this is a phase calculation
        """
        return self.projected_phase is not None

    @property
    def band_property_names(self):
        names = []
        for property_name in self._properties.keys():
            property_value = self.get_property(property_name)
            if not isinstance(property_value, np.ndarray):
                continue
            if property_value.shape[1] == self.n_bands:
                names.append(property_name)
        return names

    @property
    def property_names(self):
        names = []
        for property_name in self._properties.keys():
            property_value = self.get_property(property_name)
            if not isinstance(property_value, np.ndarray):
                continue
            names.append(property_name)
        return names

    @property
    def bands_velocity(self):
        if "bands_velocity" in self._properties:
            return self._properties.get("bands_velocity", None)

        bands_gradient = self.get_gradient("bands")
        band_vel = bands_gradient / HBAR_EV
        self._properties["bands_velocity"] = band_vel
        return band_vel

    @property
    def bands_speed(self):
        if "band_speed" in self._properties:
            return self._properties.get("band_speed", None)

        band_velocity = self.get_property("bands_velocity")
        band_speed = np.linalg.norm(band_velocity, axis=-1)
        self._properties["band_speed"] = band_speed
        return band_speed

    @property
    def avg_inv_effective_mass(self):

        if "avg_inv_effective_mass" in self._properties:
            return self._properties.get("avg_inv_effective_mass", None)

        bands_hessian = self.get_hessian("bands")
        avg_inv_effective_mass = calculate_avg_inv_effective_mass(bands_hessian)
        self._properties["avg_inv_effective_mass"] = avg_inv_effective_mass
        return avg_inv_effective_mass

    @property
    def ebs_ipr(self):
        if "ebs_ipr" in self._properties:
            return self._properties.get("ebs_ipr", None)

        orbitals = np.arange(self.n_orbitals, dtype=int)
        # sum over orbitals
        proj = np.sum(self.projected[:, :, :, :, orbitals], axis=-1)
        # keeping only the last principal quantum number
        # selecting all atoms:
        atoms = np.arange(self.n_atoms, dtype=int)
        # the ipr is \frac{\sum_i |c_i|^4}{(\sum_i |c_i^2|)^2}
        # mind, every c_i is c_{i,n,k} with n,k the band and k-point indexes
        num = np.absolute(proj) ** 2

        num = np.sum(num[:, :, :, atoms], axis=-1)
        den = np.absolute(proj) ** 1 + NUMERICAL_STABILITY_FACTOR  # avoiding zero
        den = np.sum(den[:, :, :, atoms], axis=-1) ** 2
        IPR = num / den

        self._properties["ebs_ipr"] = IPR
        return IPR

    @property
    def spin_texture(self):
        if not self.is_non_collinear:
            raise ValueError(
                "Spin texture is only available for non-collinear calculations"
            )

        if "spin_texture" in self._properties:
            return self._properties.get("spin_texture", None)

        spin_texture = self.projected[:, :, 1:, :, :]

        spin_texture = np.moveaxis(spin_texture, 2, -1)

        self._properties["spin_texture"] = spin_texture
        return spin_texture

    def get_atomic_orbital_spin_texture(
        self, atoms: List[int] = None, orbitals: List[int] = None
    ):
        if not self.is_non_collinear:
            raise ValueError(
                "Spin texture is only available for non-collinear calculations"
            )

        if atoms is None:
            atoms = np.arange(self.n_atoms, dtype=int)
        if orbitals is None:
            orbitals = np.arange(self.n_orbitals, dtype=int)

        summed_projection = self.ebs_sum(
            atoms=atoms, orbitals=orbitals, sum_noncolinear=False
        )

        orbital_resolved_spin_texture = summed_projection[..., 1:]
        temp_shape = list(orbital_resolved_spin_texture.shape)
        temp_shape.insert(2, 1)
        orbital_resolved_spin_texture = orbital_resolved_spin_texture.reshape(
            temp_shape, order="F"
        )

        label = ElectronicBandStructure.get_atomic_orbital_label(atoms, orbitals)

        label = f"spin_texture__{label}"

        # self._properties[label] = orbital_resolved_spin_texture

        self._properties["spin_texture"] = orbital_resolved_spin_texture
        return self._properties["spin_texture"]

    def get_projected_sum(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        spins: List[int] = None,
    ):
        atomic_orbital_label = ElectronicBandStructure.get_atomic_orbital_label(
            atoms, orbitals
        )
        if spins is None:
            spins = list(np.arange(self.n_spins, dtype=int))

        spin_project_label = ElectronicBandStructure.get_spin_projection_label(spins)
        label = f"projected__{atomic_orbital_label}|{spin_project_label}"

        projected_sum = self.ebs_sum(
            atoms=atoms, orbitals=orbitals, spins=spins, sum_noncolinear=True
        )

        # self._properties[label] = projected_sum
        self._properties["projected_sum"] = projected_sum
        return self._properties["projected_sum"]

    @staticmethod
    def get_atomic_orbital_label(atoms: List[int], orbitals: List[int]):
        atom_label = ElectronicBandStructure.get_atom_label(atoms)
        orbital_label = ElectronicBandStructure.get_orbital_label(orbitals)
        return f"{atom_label}|{orbital_label}"

    @staticmethod
    def get_orbital_label(orbitals: List[int]):
        orbitals_label = ",".join([str(orbital) for orbital in orbitals])
        return f"orbitals-({orbitals_label})"

    @staticmethod
    def get_atom_label(atoms: List[int]):
        atoms_label = ",".join([str(atom) for atom in atoms])
        return f"atoms-({atoms_label})"

    @staticmethod
    def get_band_label(bands: Union[List[int], int], spins: Union[List[int], int]):
        if isinstance(bands, int):
            bands = [bands]
        if isinstance(spins, int):
            spins = [spins]

        bands_label = ",".join([str(band) for band in bands])
        spins_label = ",".join([str(spin) for spin in spins])
        return f"(bands|spins)-({bands_label}|{spins_label})"

    @staticmethod
    def extract_band_index(label: str):
        raw_text = re.findall(r"\(bands\|spins\)-\((.*)\)", label)
        bands, spins = raw_text[0].split("|")
        bands = [int(band) for band in bands.split(",")]
        spins = [int(spin) for spin in spins.split(",")]

        if len(bands) == 1:
            bands = bands[0]
        if len(spins) == 1:
            spins = spins[0]

        return bands, spins

    @staticmethod
    def extract_property_label(label: str):
        property_name = label.split("__")[0]
        return property_name

    @staticmethod
    def get_spin_projection_label(spin_projections: List[int]):
        spin_projection_names_label = ",".join(
            [str(spin_projection_name) for spin_projection_name in spin_projections]
        )
        return f"spin_projections-({spin_projection_names_label})"

    @staticmethod
    def get_band_property_label(property_name, band_index: int, spin_index: int):
        band_label = ElectronicBandStructure.get_band_label(band_index, spin_index)
        return f"{property_name}__{band_label}"

    @staticmethod
    def get_property_gradient_label(property_name: str):
        return f"{property_name}_gradient"

    @staticmethod
    def get_property_hessian_label(property_name: str):
        return f"{property_name}_hessian"

    def compute_gradients(
        self,
        property_names: List[str] = None,
        first_order: bool = True,
        second_order: bool = False,
        recalculate: bool = False,
        order="F",
        **kwargs,
    ):
        raise NotImplementedError()

    def get_property(self, property_name: str, order="F", **kwargs):
        # Special case when we define the property as a property of the class
        if property_name in [
            "bands_velocity",
            "bands_speed",
            "avg_inv_effective_mass",
            "ebs_ipr",
            "spin_texture",
        ]:
            property_value = getattr(self, property_name, None)
            return property_value

        property_value = self._properties.get(property_name, None)
        return property_value

    def get_gradient(
        self, property_name: str, compute: bool = True, order="F", **kwargs
    ):
        self.get_property(property_name)

        try:
            if compute:
                self.compute_gradients(
                    property_names=[property_name],
                    first_order=True,
                    second_order=False,
                    order=order,
                    **kwargs,
                )
        except NotImplementedError:
            pass
        gradient = self._gradients.get(property_name, None)
        return gradient

    def get_hessian(
        self, property_name: str, compute: bool = True, order="F", **kwargs
    ):
        self.get_gradient(property_name, compute=compute, order=order, **kwargs)

        try:
            if compute:
                self.compute_gradients(
                    property_names=[property_name],
                    first_order=False,
                    second_order=True,
                    order=order,
                    **kwargs,
                )
        except NotImplementedError:
            pass
        hessian = self._hessians.get(property_name, None)
        return hessian

    def add_property(self, property_name: str, property_value: np.ndarray):
        self._properties[property_name] = property_value
        return self

    def ebs_sum(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        spins: List[int] = None,
        sum_noncolinear: bool = True,
    ):
        """_summary_

        Parameters
        ----------
        atoms : List[int], optional
            List of atoms to be summed over, by default None
        orbitals : List[int], optional
            List of orbitals to be summed over, by default None
        spins : List[int], optional
            List of spins to be summed over, by default None
        sum_noncolinear : bool, optional
            Determines if the projection should be summed in a non-colinear calculation, by default True

        Returns
        -------
        ret : list float
            The summed projections
        """

        if atoms is None:
            atoms = np.arange(self.n_atoms, dtype=int)
        if spins is None:
            spins = np.arange(self.n_spins, dtype=int)
        if orbitals is None:
            orbitals = np.arange(self.n_orbitals, dtype=int)
        # sum over orbitals
        ret = np.sum(self.projected[:, :, :, :, orbitals], axis=-1)
        # sum over atoms
        ret = np.sum(ret[:, :, :, atoms], axis=-1)
        # sum over spins only in non collinear and reshaping for consistency (nkpoints, nbands, nspins)
        # in non-mag, non-colin nspin=1, in colin nspin=2
        if self.is_non_collinear and sum_noncolinear:
            ret = np.sum(ret[:, :, spins], axis=-1).reshape(
                self.n_kpoints, self.n_bands, 1
            )

        return ret

    def save(self, path: Path):
        serializer = get_serializer(path)
        serializer.save(self, path)

    @classmethod
    def load(cls, path: Path):
        serializer = get_serializer(path)
        return serializer.load(path)

    @classmethod
    def from_data(cls, **kwargs):
        if kwargs.get("kpath", None) is not None:
            return ElectronicBandStructurePath(
                kpoints=kwargs.get("kpoints", None),
                bands=kwargs.get("bands", None),
                projected=kwargs.get("projected", None),
                projected_phase=kwargs.get("projected_phase", None),
                weights=kwargs.get("weights", None),
                efermi=kwargs.get("efermi", None),
                reciprocal_lattice=kwargs.get("reciprocal_lattice", None),
                orbital_names=kwargs.get("orbital_names", None),
                structure=kwargs.get("structure", None),
                kpath=kwargs.get("kpath", None),
                properties=kwargs.get("properties", None),
                gradients=kwargs.get("gradients", None),
                hessians=kwargs.get("hessians", None),
            )
        elif kwargs.get("kgrid", None) is not None:
            return ElectronicBandStructureMesh(
                kpoints=kwargs.get("kpoints", None),
                bands=kwargs.get("bands", None),
                projected=kwargs.get("projected", None),
                projected_phase=kwargs.get("projected_phase", None),
                weights=kwargs.get("weights", None),
                efermi=kwargs.get("efermi", None),
                reciprocal_lattice=kwargs.get("reciprocal_lattice", None),
                orbital_names=kwargs.get("orbital_names", None),
                structure=kwargs.get("structure", None),
                kgrid=kwargs.get("kgrid", None),
                properties=kwargs.get("properties", None),
                gradients=kwargs.get("gradients", None),
                hessians=kwargs.get("hessians", None),
            )
        else:
            return ElectronicBandStructure(
                kpoints=kwargs.get("kpoints", None),
                bands=kwargs.get("bands", None),
                projected=kwargs.get("projected", None),
                projected_phase=kwargs.get("projected_phase", None),
                weights=kwargs.get("weights", None),
                properties=kwargs.get("properties", None),
                gradients=kwargs.get("gradients", None),
                hessians=kwargs.get("hessians", None),
            )

    def reduce_bands_near_energy(
        self, energy: float, tolerance: float = 0.7, inplace=True
    ):
        """
        Reduces the bands to those near the fermi energy
        """
        logger.info("____Reducing bands near fermi energy____")
        full_band_index = []
        bands_spin_index = {}

        for ispin in self.spin_channels:
            bands_spin_index[ispin] = []
            for iband in range(self.n_bands):
                fermi_surface_test = len(
                    np.where(
                        np.logical_and(
                            self.bands[:, iband, ispin] >= energy - tolerance,
                            self.bands[:, iband, ispin] <= energy + tolerance,
                        )
                    )[0]
                )
                if fermi_surface_test != 0:
                    bands_spin_index[ispin].append(iband)

                    if iband not in full_band_index:  # Avoid duplicates
                        full_band_index.append(iband)

        band_dependent_properties = self.band_property_names
        property_dict = {}
        gradient_dict = {}
        hessian_dict = {}
        for prop in self.property_names:
            original_value = self.get_property(prop)
            original_gradient = self.get_gradient(prop, compute=False)
            original_hessian = self.get_hessian(prop, compute=False)

            if prop in band_dependent_properties:
                if original_value is not None:
                    property_dict[prop] = original_value[:, full_band_index, ...]

                if original_gradient is not None:
                    gradient_dict[prop] = original_gradient[:, full_band_index, ...]

                if original_hessian is not None:
                    hessian_dict[prop] = original_hessian[:, full_band_index, ...]
            else:
                property_dict[prop] = original_value
                gradient_dict[prop] = original_gradient
                hessian_dict[prop] = original_hessian

        debug_message = f"Bands near energy {energy}. "
        debug_message += f"Spin-0 {bands_spin_index[0]} |"
        if self.n_spin_channels > 1 and not self.is_non_collinear:
            debug_message += f" Spin-1 {bands_spin_index[1]}"
        logger.debug(debug_message)

        if inplace:
            self._properties = property_dict
            self._gradients = gradient_dict
            self._hessians = hessian_dict
            return self
        else:
            new_bands = property_dict.pop("bands")
            return ElectronicBandStructure.from_data(
                kpoints=self._kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=property_dict,
                gradients=gradient_dict,
                hessians=hessian_dict,
                kpath=self.__dict__.get("kpath", None),
                kgrid=self.__dict__.get("kgrid", None),
            )

    def reduce_bands_near_fermi(self, tolerance=0.7, inplace=True):
        """
        Reduces the bands to those near the fermi energy
        """
        return self.reduce_bands_near_energy(self.efermi, tolerance, inplace=inplace)

    def reduce_bands_by_index(self, bands, inplace=True):
        """
        Reduces the bands to those near the fermi energy
        """

        new_properties = {}
        new_gradients = {}
        new_hessians = {}
        for prop in self.property_names:
            original_value = self.get_property(prop)
            original_gradient = self.get_gradient(prop, compute=False)
            original_hessian = self.get_hessian(prop, compute=False)

            if prop in self.band_property_names:
                if original_value is not None:
                    new_properties[prop] = original_value[:, bands, ...]

                if original_gradient is not None:
                    new_gradients[prop] = original_gradient[:, bands, ...]

                if original_hessian is not None:
                    new_hessians[prop] = original_hessian[:, bands, ...]
            else:
                new_properties[prop] = original_value
                new_gradients[prop] = original_gradient
                new_hessians[prop] = original_hessian

        if inplace:
            self._properties = new_properties
            self._gradients = new_gradients
            self._hessians = new_hessians
            return self
        else:
            new_bands = new_properties.pop("bands")
            return ElectronicBandStructure.from_data(
                kpoints=self._kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kpath=self.__dict__.get("kpath", None),
                kgrid=self.__dict__.get("kgrid", None),
            )

    def fix_collinear_spin(self, inplace=True):
        """
        Converts data from two spin channels to a single channel, adjusting the spin down values to negatives. This is typically used for plotting the Density of States (DOS).

        Parameters
        ----------
        No parameters are required for this function.

        Returns
        -------
        bool
            Returns True if the function changed the data, False otherwise.
        """
        if self.n_spin_channels != 2:
            return False

        new_properties = {}
        new_gradients = {}
        new_hessians = {}
        for prop in self.property_names:
            original_value = self.get_property(prop)
            original_gradient = self.get_gradient(prop, compute=False)
            original_hessian = self.get_hessian(prop, compute=False)

            if prop in self.band_property_names:
                if original_value is not None:
                    original_value_shape = list(original_value.shape)
                    band_dim = original_value_shape[1]
                    original_value_shape[1] = 2 * band_dim
                    original_value_shape[2] = 1
                    original_value = original_value.reshape(original_value_shape)
                    new_properties[prop] = original_value

                if original_gradient is not None:
                    original_gradient_shape = list(original_gradient.shape)
                    original_gradient_shape[1] = 2 * original_gradient_shape[1]
                    original_gradient_shape[2] = 1
                    original_gradient = original_gradient.reshape(
                        original_gradient_shape
                    )
                    new_gradients[prop] = original_gradient

                if original_hessian is not None:
                    original_hessian_shape = list(original_hessian.shape)
                    original_hessian_shape[1] = 2 * original_hessian_shape[1]
                    original_hessian_shape[2] = 1
                    original_hessian = original_hessian.reshape(original_hessian_shape)
                    new_hessians[prop] = original_hessian
            else:
                new_properties[prop] = original_value
                new_gradients[prop] = original_gradient
                new_hessians[prop] = original_hessian

        if inplace:
            self._properties = new_properties
            self._gradients = new_gradients
            self._hessians = new_hessians
            return self
        else:
            new_bands = new_properties.pop("bands")
            return ElectronicBandStructure.from_data(
                kpoints=self._kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kpath=self.__dict__.get("kpath", None),
                kgrid=self.__dict__.get("kgrid", None),
            )

    def shift_bands(self, shift_value, inplace=False):

        new_gradients = self._gradients
        new_hessians = self._hessians
        new_properties = self._properties
        new_bands = new_properties.pop("bands")
        new_bands += shift_value

        if inplace:
            self._properties["bands"] = new_bands
            return self
        else:
            return ElectronicBandStructure.from_data(
                kpoints=self._kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kpath=self.__dict__.get("kpath", None),
                kgrid=self.__dict__.get("kgrid", None),
            )

    def unfold(self, transformation_matrix=None, structure=None, inplace=False):
        """The method helps unfold the bands. This is done by using the unfolder to find the new kpoint weights.
        The current weights are then updated

        Parameters
        ----------
        transformation_matrix : np.ndarray, optional
            The transformation matrix to transform the basis. Expected size is (3,3), by default None
        structure : pyprocar.core.Structure, optional
            The structure of a material, by default None
        inplace : bool, optional
            If True, the method will modify the current instance, by default False

        Returns
        -------
        None
            None
        """
        uf = Unfolder(
            ebs=self,
            transformation_matrix=transformation_matrix,
            structure=structure,
        )

        new_gradients = self._gradients
        new_hessians = self._hessians
        new_properties = self._properties

        new_properties["weights"] = uf.weights

        if inplace:
            self._properties = new_properties
            self._gradients = new_gradients
            self._hessians = new_hessians
            return self
        else:
            new_bands = new_properties.pop("bands")

            return ElectronicBandStructure.from_data(
                kpoints=self._kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kpath=self.__dict__.get("kpath", None),
                kgrid=self.__dict__.get("kgrid", None),
            )


class ElectronicBandStructurePath(ElectronicBandStructure):

    def __init__(self, kpath: KPath, **kwargs):
        super().__init__(**kwargs)
        self._kpath = kpath

    def __str__(self):
        ret = super().__str__()
        ret += "\nKPath: \n"
        ret += "------------------------     \n"
        ret += "KPath = \n {}\n".format(self.kpath)
        return ret

    @property
    def kpath(self):
        return self._kpath

    @property
    def knames(self):
        return self.kpath.knames

    @property
    def n_segments(self):
        return self.kpath.nsegments

    @property
    def tick_positions(self):
        return self.kpath.tick_positions

    @property
    def tick_names(self):
        return self.kpath.tick_names

    def get_kpath_segments(self, isegments: List[int] = None, cartesian: bool = False):
        if isegments is None:
            isegments = list(range(self.n_segments))
        kpath_segments = []
        tick_position_segments = []
        for isegment in isegments:
            kpath_segment, tick_position_segment = self.get_kpath_segment(isegment)
            kpath_segments.append(kpath_segment)
            tick_position_segments.append(tick_position_segment)
        return kpath_segments, tick_position_segments

    def get_kpath_segment(self, isegment: List[int], cartesian: bool = False):
        kpoints = self.kpoints_cartesian if cartesian else self.kpoints

        tick_position_segment = np.arange(
            self.tick_positions[isegment],
            self.tick_positions[isegment + 1] + 1,
            step=1,
        )
        kpath_segment = kpoints[tick_position_segment]

        return kpath_segment, tick_position_segment

    def get_continous_segments(self):
        """
        Gets the indices of the continuous segments in the kpath
        """

        kpath_segments, tick_position_segments = self.get_kpath_segments()

        continuous_segments = []

        for isegment, tick_position_segment in enumerate(tick_position_segments):
            if isegment == 0:
                continuous_segments.append(tick_position_segment)
                continue

            current_start_index = tick_position_segment[0]
            prev_end_index = tick_position_segments[isegment - 1][-1]
            if current_start_index == prev_end_index:
                continuous_segments[-1] = np.concatenate(
                    (continuous_segments[-1], tick_position_segment[1:])
                )
            else:
                continuous_segments.append(tick_position_segment)
        return continuous_segments

    def get_kpath_distances(
        self,
        isegments: List[int] = None,
        return_kpath_segments: bool = False,
        return_continuous_segments: bool = False,
        cartesian: bool = False,
    ):
        if return_kpath_segments and return_continuous_segments:
            raise ValueError(
                "Cannot return both kpath segments and continuous segments"
            )

        if isegments is None:
            isegments = list(range(self.n_segments))
        kpath_distances = []

        k_path_segments, tick_position_segments = self.get_kpath_segments(
            isegments=isegments, cartesian=cartesian
        )

        continuous_segments = self.get_continous_segments()

        segment_max = 0
        k_total = None
        for k_indices in continuous_segments:
            k_path_segment = self.kpoints_cartesian[k_indices]
            k_diffs = np.gradient(k_path_segment, axis=0)
            k_diffs = np.linalg.norm(k_diffs, axis=1)

            k_distances = np.cumsum(k_diffs) + segment_max

            segment_max = k_distances[-1]

            if k_total is None:
                k_total = k_distances
            else:
                k_total = np.concatenate((k_total, k_distances))

        if return_kpath_segments:
            k_segment_distances = []
            for k_indices in tick_position_segments:
                k_segment_distances.append(k_total[k_indices])
            return k_segment_distances
        elif return_continuous_segments:
            k_segment_distances = []
            for k_indices in continuous_segments:
                k_segment_distances.append(k_total[k_indices])
            return k_segment_distances
        else:
            return k_total

    def compute_gradients(
        self,
        property_names: List[str] = None,
        first_order: bool = True,
        second_order: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):

        property_names = set(property_names) if property_names else set()
        current_properties = set(self.property_names)

        if property_names:

            # Filter out properties that don't exist in the current property store
            # Keep properties that exist in current_properties
            existing_properties = property_names.intersection(current_properties)
            # Keep properties that don't exist in current_properties
            missing_properties = property_names - current_properties

            if missing_properties:
                raise ValueError(
                    f"Properties {missing_properties} not found in the property store. Use compute_gradients to compute the gradients."
                )
        else:
            property_names = current_properties

        kpoints_cartesian = self.kpoints_cartesian
        if first_order:
            for prop_name in property_names:
                if prop_name in self._gradients and not recalculate:
                    continue

                continuous_segments = self.get_continous_segments()

                val_array = self.get_property(prop_name, as_mesh=False)
                gradients = np.zeros(val_array.shape)
                for k_indices in continuous_segments:
                    kpath_segment = kpoints_cartesian[k_indices]
                    delta_k = np.gradient(kpath_segment, axis=0)
                    delta_k = np.linalg.norm(delta_k, axis=1)

                    gradients[k_indices, ...] = np.gradient(
                        val_array[k_indices, ...],
                        delta_k,
                        axis=0,
                        edge_order=2,
                    )

                self._gradients[prop_name] = gradients * METER_ANGSTROM

        if second_order:
            for prop_name in property_names:
                if prop_name in self._hessians and not recalculate:
                    continue

                continuous_segments = self.get_continous_segments()

                val_gradients = self.get_gradient(prop_name, as_mesh=False)
                hessians = np.zeros(val_gradients.shape)
                for k_indices in continuous_segments:
                    kpath_segment = kpoints_cartesian[k_indices]
                    delta_k = np.gradient(kpath_segment, axis=0)
                    delta_k = np.linalg.norm(delta_k, axis=1)

                    hessians[k_indices, ...] = np.gradient(
                        val_gradients[k_indices, ...],
                        delta_k,
                        axis=0,
                        edge_order=2,
                    )
                self._hessians[prop_name] = hessians * METER_ANGSTROM


class ElectronicBandStructureMesh(ElectronicBandStructure):

    def __init__(self, kgrid: Tuple[int, int, int] = None, **kwargs):
        super().__init__(**kwargs)
        self._kgrid = kgrid

        if self.is_fbz:
            self.sort_by_kpoints(inplace=True)

    def __str__(self):
        ret = super().__str__()
        ret += "\nKGrid: \n"
        ret += "------------------------     \n"
        ret += "(nkx, nky, nkz) = \n {}\n".format(self.kgrid)
        return ret

    @property
    def kgrid(self):
        return self._kgrid

    @property
    def n_kx(self):
        return self.kgrid[0]

    @property
    def n_ky(self):
        return self.kgrid[1]

    @property
    def n_kz(self):
        return self.kgrid[2]

    @property
    def is_ibz(self):
        return self.n_kpoints != self.n_kx * self.n_ky * self.n_kz

    @property
    def is_fbz(self):
        return self.n_kpoints == self.n_kx * self.n_ky * self.n_kz

    def get_kpoints(self, as_mesh=False, **kwargs):
        kpoints = self.kpoints
        if as_mesh:
            return mathematics.array_to_mesh(
                array=kpoints,
                nkx=self.n_kx,
                nky=self.n_ky,
                nkz=self.n_kz,
                **kwargs,
            )
        else:
            return kpoints

    def compute_gradients(
        self,
        property_names: List[str] = None,
        first_order: bool = True,
        second_order: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):
        property_names = set(property_names) if property_names else set()
        current_properties = set(self.property_names)

        if property_names:

            # Filter out properties that don't exist in the current property store
            # Keep properties that exist in current_properties
            existing_properties = property_names.intersection(current_properties)
            # Keep properties that don't exist in current_properties
            missing_properties = property_names - current_properties

            if missing_properties:
                raise ValueError(
                    f"Properties {missing_properties} not found in the property store. Use compute_gradients to compute the gradients."
                )
        else:
            property_names = current_properties

        if self.is_ibz:
            self.ibz2fbz(inplace=True)

        if first_order:
            for prop_name in property_names:
                if prop_name in self._gradients and not recalculate:
                    continue

                val_mesh = self.get_property(prop_name, as_mesh=True, **kwargs)

                gradients_mesh = mathematics.calculate_3d_mesh_scalar_gradients(
                    val_mesh, self.reciprocal_lattice
                )
                gradients_mesh *= METER_ANGSTROM

                self._gradients[prop_name] = mathematics.mesh_to_array(
                    mesh=gradients_mesh, **kwargs
                )

        if second_order:
            for prop_name in property_names:
                if prop_name in self._hessians and not recalculate:
                    continue

                val_mesh = self.get_gradient(prop_name, as_mesh=True, **kwargs)

                hessians_mesh = mathematics.calculate_3d_mesh_scalar_gradients(
                    val_mesh, self.reciprocal_lattice
                )
                hessians_mesh *= METER_ANGSTROM
                self._hessians[prop_name] = mathematics.mesh_to_array(
                    mesh=hessians_mesh, **kwargs
                )

    def get_property(
        self, property_name: str, as_mesh=False, compute: bool = True, **kwargs
    ):
        property_value = super().get_property(property_name, compute=compute, **kwargs)

        if as_mesh:
            return mathematics.array_to_mesh(
                array=property_value,
                nkx=self.n_kx,
                nky=self.n_ky,
                nkz=self.n_kz,
                **kwargs,
            )
        else:
            return property_value

    def get_gradient(
        self, property_name: str, as_mesh=False, compute: bool = True, **kwargs
    ):

        gradient = super().get_gradient(property_name, compute=compute, **kwargs)

        if gradient is None:
            return None

        if as_mesh:
            return mathematics.array_to_mesh(
                array=gradient,
                nkx=self.n_kx,
                nky=self.n_ky,
                nkz=self.n_kz,
                **kwargs,
            )
        else:
            return gradient

    def get_hessian(
        self, property_name: str, as_mesh=False, compute: bool = True, **kwargs
    ):
        hessian = super().get_hessian(property_name, compute=compute, **kwargs)

        if hessian is None:
            return None

        if as_mesh:
            return mathematics.array_to_mesh(
                array=hessian,
                nkx=self.n_kx,
                nky=self.n_ky,
                nkz=self.n_kz,
                **kwargs,
            )
        else:
            return hessian

    def sort_by_kpoints(self, inplace=True, order="F"):
        """Sorts the bands and projected arrays by kpoints"""

        if order == "C":
            sorted_indices = np.lexsort(
                (self.kpoints[:, 2], self.kpoints[:, 1], self.kpoints[:, 0])
            )
        elif order == "F":
            sorted_indices = np.lexsort(
                (self.kpoints[:, 0], self.kpoints[:, 1], self.kpoints[:, 2])
            )

        new_properties = {}
        new_gradients = {}
        new_hessians = {}
        new_kpoints = self.kpoints[sorted_indices, ...]

        for prop in self.property_names:
            original_value = self.get_property(prop)
            original_gradient = self.get_gradient(prop, compute=False)
            original_hessian = self.get_hessian(prop, compute=False)
            if original_value is not None:
                new_properties[prop] = original_value[sorted_indices, ...]
            if original_gradient is not None:
                new_gradients[prop] = original_gradient[sorted_indices, ...]
            if original_hessian is not None:
                new_hessians[prop] = original_hessian[sorted_indices, ...]
        if inplace:
            self._kpoints = new_kpoints
            self._properties = new_properties
            self._gradients = new_gradients
            self._hessians = new_hessians
            self._kgrid = self.kgrid
            return self
        else:
            new_bands = new_properties.pop("bands")

            return ElectronicBandStructure.from_data(
                kpoints=new_kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kgrid=self.kgrid,
            )

    def ibz2fbz(self, rotations=None, decimals=4, inplace=True):
        """Applys symmetry operations to the kpoints, bands, and projections

        Parameters
        ----------
        rotations : np.ndarray
            The point symmetry operations of the lattice
        decimals : int
            The number of decimals to round the kpoints
            to when checking for uniqueness
        """
        rotations = []
        if self.is_fbz:
            logger.warning("Band structure is already in the FBZ, skipping ibz2fbz")
            return self

        if len(rotations) == 0 and self.structure is not None:
            rotations = self.structure.rotations
        if len(rotations) == 0:
            logger.warning("No rotations provided, skipping ibz2fbz")
            return self

        n_kpoints = self.n_kpoints

        # Add initial values before applying rotations
        new_properties = {}
        new_kpoints = self.kpoints.copy()
        for prop in self.property_names:
            original_value = self.get_property(prop)
            if original_value is not None:
                new_properties[prop] = original_value.copy()

        # Apply rotations and copy properties
        for i, rotation in enumerate(rotations):
            start_idx = i * n_kpoints
            end_idx = start_idx + n_kpoints

            # Rotate kpoints
            new_values = self.kpoints.dot(rotation.T)
            new_kpoints = np.concatenate([new_kpoints, new_values], axis=0)
            # Update properties
            for prop in self.property_names:
                original_value = self.get_property(prop)
                if original_value is not None:
                    new_properties[prop] = np.concatenate(
                        [new_properties[prop], original_value], axis=0
                    )

        # Apply boundary conditions to kpoints
        new_kpoints = -np.fmod(new_kpoints + 6.5, 1) + 0.5

        # Floating point error can cause the kpoints to be off by 0.000001 or so
        # causing the unique indices to misidentify the kpoints
        new_kpoints = new_kpoints.round(decimals=decimals)
        _, unique_indices = np.unique(new_kpoints, axis=0, return_index=True)

        new_kpoints = new_kpoints[unique_indices, ...]
        for prop in new_properties.keys():
            new_properties[prop] = new_properties[prop][unique_indices, ...]

        if inplace:
            self._kpoints = new_kpoints
            self._properties = new_properties
            self._gradients = self._gradients
            self._hessians = self._hessians
            self._kgrid = self.kgrid
            return self.sort_by_kpoints(inplace=inplace)
        else:
            new_bands = new_properties.pop("bands")
            ebs = ElectronicBandStructure.from_data(
                kpoints=new_kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=self._gradients,
                hessians=self._hessians,
                kgrid=self.kgrid,
            )
            return ebs.sort_by_kpoints(inplace=inplace)

    def reduce_kpoints_to_plane(
        self, k_plane=0, k_plane_tol=0.01, axis=0, inplace=True
    ):
        """
        Reduces the kpoints to a plane
        """

        i_kpoints_near_z_0 = np.where(
            np.logical_and(
                self.kpoints_cartesian[:, axis] < k_plane + k_plane_tol,
                self.kpoints_cartesian[:, axis] > k_plane - k_plane_tol,
            )
        )

        new_properties = {}
        new_gradients = {}
        new_hessians = {}
        new_kpoints = self.kpoints[i_kpoints_near_z_0, ...][0]
        for prop in self.property_names:
            original_value = self.get_property(prop)
            if original_value is not None:
                new_properties[prop] = original_value[i_kpoints_near_z_0, ...][0]
            original_gradient = self.get_gradient(prop, compute=False)
            if original_gradient is not None:
                new_gradients[prop] = original_gradient[i_kpoints_near_z_0, ...][0]
            original_hessian = self.get_hessian(prop, compute=False)
            if original_hessian is not None:
                new_hessians[prop] = original_hessian[i_kpoints_near_z_0, ...][0]

        new_bands = new_properties.pop("bands")

        unique_kx, unique_ky, unique_kz = (
            np.unique(new_kpoints[:, 0]),
            np.unique(new_kpoints[:, 1]),
            np.unique(new_kpoints[:, 2]),
        )

        new_n_kx = len(unique_kx)
        new_n_ky = len(unique_ky)
        new_n_kz = len(unique_kz)

        kgrid = [new_n_kx, new_n_ky, new_n_kz]

        if inplace:
            self._kpoints = new_kpoints
            self._properties = new_properties
            self._gradients = new_gradients
            self._hessians = new_hessians
            self._kgrid = kgrid
            return self
        else:
            return ElectronicBandStructure.from_data(
                kpoints=new_kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kgrid=kgrid,
            )

    def interpolate(self, interpolation_factor=2, inplace=True, order="F"):
        """Interpolates the band structure meshes and properties using FFT interpolation.
        Creates and returns a new ElectronicBandStructure instance with interpolated data.

        Parameters
        ----------
        interpolation_factor : int, optional
            Factor by which to interpolate the mesh, by default 2

        Returns
        -------
        ElectronicBandStructure
            New instance with interpolated data
        """
        logger.info(f"Interpolating band structure by factor {interpolation_factor}")
        # Calculate new mesh dimensions
        kpoints_mesh = self.get_kpoints(as_mesh=True, order=order)
        nkx, nky, nkz = (
            kpoints_mesh.shape[0],
            kpoints_mesh.shape[1],
            kpoints_mesh.shape[2],
        )

        unique_x = kpoints_mesh[:, 0, 0, 0]
        unique_y = kpoints_mesh[0, :, 0, 1]
        unique_z = kpoints_mesh[0, 0, :, 2]

        xmin, xmax = np.min(unique_x), np.max(unique_x)
        ymin, ymax = np.min(unique_y), np.max(unique_y)
        zmin, zmax = np.min(unique_z), np.max(unique_z)

        new_x = np.linspace(xmin, xmax, nkx * interpolation_factor)
        new_y = np.linspace(ymin, ymax, nky * interpolation_factor)
        new_z = np.linspace(zmin, zmax, nkz * interpolation_factor)

        new_kpoints_mesh = np.array(np.meshgrid(new_z, new_y, new_x, indexing="ij"))
        new_kpoints = new_kpoints_mesh.reshape(-1, 3)

        new_properties = {}
        new_gradients = {}
        new_hessians = {}
        for prop in self.property_names:
            original_value = self.get_property(prop)
            if original_value is not None:
                mesh = mathematics.array_to_mesh(
                    original_value, self.n_kx, self.n_ky, self.n_kz, order=order
                )
                interpolated_mesh = mathematics.fft_interpolate_nd_3dmesh(
                    mesh,
                    interpolation_factor,
                )
                interpolated_value = mathematics.mesh_to_array(interpolated_mesh)
                new_properties[prop] = interpolated_value

            original_gradient = self.get_gradient(prop, compute=False)
            if original_gradient is not None:
                mesh = mathematics.array_to_mesh(
                    original_gradient, self.n_kx, self.n_ky, self.n_kz, order=order
                )
                interpolated_mesh = mathematics.fft_interpolate_nd_3dmesh(
                    mesh,
                    interpolation_factor,
                )
                interpolated_gradient = mathematics.mesh_to_array(interpolated_mesh)
                new_gradients[prop] = interpolated_gradient

            original_hessian = self.get_hessian(prop, compute=False)
            if original_hessian is not None:
                mesh = mathematics.array_to_mesh(
                    original_hessian, self.n_kx, self.n_ky, self.n_kz, order=order
                )
                interpolated_mesh = mathematics.fft_interpolate_nd_3dmesh(
                    mesh,
                    interpolation_factor,
                )
                interpolated_hessian = mathematics.mesh_to_array(
                    interpolated_mesh, order=order
                )
                new_hessians[prop] = interpolated_hessian
        new_nkx = len(new_x)
        new_nky = len(new_y)
        new_nkz = len(new_z)
        kgrid = [new_nkx, new_nky, new_nkz]

        if inplace:
            self._kpoints = new_kpoints
            self._properties = new_properties
            self._gradients = new_gradients
            self._hessians = new_hessians
            self._kgrid = kgrid
            return self
        else:
            new_bands = new_properties.pop("bands")
            return ElectronicBandStructure.from_data(
                kpoints=new_kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kgrid=kgrid,
            )

    def expand_kpoints_to_supercell_by_axes(
        self, axes_to_expand=[0, 1, 2], inplace=True
    ):
        # Validate input
        if not set(axes_to_expand).issubset({0, 1, 2}):
            raise ValueError("axes_to_expand must be a subset of [0, 1, 2]")

        # Create supercell directions based on axes to expand
        supercell_directions = list(
            itertools.product([1, 0, -1], repeat=len(axes_to_expand))
        )

        new_properties = {}
        new_gradients = {}
        new_hessians = {}
        # Initialize the new properties
        for prop in self.property_names:
            original_value = self.get_property(prop)
            if original_value is not None:
                new_properties[prop] = original_value

            original_gradient = self.get_gradient(prop, compute=False)
            if original_gradient is not None:
                new_gradients[prop] = original_gradient

            original_hessian = self.get_hessian(prop, compute=False)
            if original_hessian is not None:
                new_hessians[prop] = original_hessian

            new_kpoints = self.kpoints.copy()

        # Iterate over the supercell directions
        for supercell_direction in supercell_directions:
            if supercell_direction != tuple([0] * len(axes_to_expand)):
                for prop in self.property_names:
                    original_value = self.get_property(prop)

                    if prop == "kpoints":
                        for i, axis in enumerate(axes_to_expand):
                            self.kpoints[:, axis] += supercell_direction[i]

                        new_kpoints = np.concatenate(
                            [new_kpoints, original_value], axis=0
                        )
                    else:
                        new_properties[prop] = np.concatenate(
                            [new_properties[prop], original_value], axis=0
                        )

        if inplace:
            self._kpoints = new_kpoints
            self._properties = new_properties
            self._gradients = new_gradients
            self._hessians = new_hessians
            self._kgrid = self.kgrid
            return self
        else:
            new_bands = new_properties.pop("bands")
            return ElectronicBandStructure.from_data(
                kpoints=new_kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kgrid=self.kgrid,
            )

    def pad(self, padding=10, order="F", inplace=True):
        new_properties = {}
        new_gradients = {}
        new_hessians = {}

        for prop in self.property_names:
            original_value = self.get_property(prop, as_mesh=True, order=order)
            if original_value is not None:
                padded_mesh = mathematics.pad_scalar_3d_mesh(
                    original_value, padding, add_1_on_wrap=False
                )

                new_properties[prop] = mathematics.mesh_to_array(
                    padded_mesh, order=order
                )

            original_gradient = self.get_gradient(prop, as_mesh=True, compute=False)
            if original_gradient is not None:
                padded_mesh = mathematics.pad_scalar_3d_mesh(
                    original_gradient, padding, add_1_on_wrap=False
                )
                new_gradients[prop] = mathematics.mesh_to_array(
                    padded_mesh, order=order
                )

            original_hessian = self.get_hessian(prop, as_mesh=True, compute=False)
            if original_hessian is not None:
                padded_mesh = mathematics.pad_scalar_3d_mesh(
                    original_hessian, padding, add_1_on_wrap=False
                )
                new_hessians[prop] = mathematics.mesh_to_array(padded_mesh, order=order)

        kpoints_mesh = self.get_kpoints(as_mesh=True, order=order)
        padded_kpoints_mesh = mathematics.pad_scalar_3d_mesh(
            kpoints_mesh, padding, add_1_on_wrap=True
        )

        new_kpoints = mathematics.mesh_to_array(padded_kpoints_mesh, order=order)
        nn_kx = self.n_kx + 2 * padding
        nn_ky = self.n_ky + 2 * padding
        nn_kz = self.n_kz + 2 * padding
        kgrid = [nn_kx, nn_ky, nn_kz]

        if inplace:
            self._kpoints = new_kpoints
            self._properties = new_properties
            self._gradients = new_gradients
            self._hessians = new_hessians
            self._kgrid = kgrid
            return self
        else:
            new_bands = new_properties.pop("bands")
            return ElectronicBandStructure.from_data(
                kpoints=new_kpoints,
                bands=new_bands,
                efermi=self._efermi,
                orbital_names=self._orbital_names,
                reciprocal_lattice=self._reciprocal_lattice,
                shifted_to_efermi=self._shifted_to_efermi,
                structure=self._structure,
                properties=new_properties,
                gradients=new_gradients,
                hessians=new_hessians,
                kgrid=kgrid,
            )
