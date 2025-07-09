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
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import pyvista as pv
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    RegularGridInterpolator,
    griddata,
)

from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.core.kpath import KPath
from pyprocar.core.property_store import PointSet
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

pv.global_theme.allow_empty_mesh = True

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

NUMERICAL_STABILITY_FACTOR = 0.0001

KPOINTS_DTYPE = np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]]
BANDS_DTYPE = np.ndarray[tuple[int,int,int], np.dtype[np.float_]]
PROJECTED_DTYPE = np.ndarray[tuple[int,int,int,int,int], np.dtype[np.float_]]
PROJECTED_PHASE_DTYPE = np.ndarray[tuple[int,int,int,int,int], np.dtype[np.float_]]
WEIGHTS_DTYPE = np.ndarray[tuple[int,int], np.dtype[np.float_]]
RECIPROCAL_LATTICE_DTYPE = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]

PROPERTIES_DTYPE = npt.NDArray[np.dtype[np.float_]]

def get_ebs_from_data(
        kpoints: KPOINTS_DTYPE,
        bands: BANDS_DTYPE | None = None,
        projected: PROJECTED_DTYPE | None = None,
        projected_phase: PROJECTED_PHASE_DTYPE | None = None,
        weights: WEIGHTS_DTYPE | None = None,
        fermi: float = 0.0,
        reciprocal_lattice: RECIPROCAL_LATTICE_DTYPE | None = None,
        orbital_names: list[str] | None = None,
        structure: Structure | None = None,
        properties: dict[str, PROPERTIES_DTYPE] | None = None,
        gradients: dict[str, PROPERTIES_DTYPE] | None = None,
        hessians: dict[str, PROPERTIES_DTYPE] | None = None,
        kpath: KPath = None,
        kgrid: tuple[int, int, int] | None = None,
        **kwargs):
    
    ebs_args = {
        "kpoints": kpoints,
        "bands": bands,
        "projected": projected,
        "projected_phase": projected_phase,
        "weights": weights,
        "fermi": fermi,
        "reciprocal_lattice": reciprocal_lattice,
        "orbital_names": orbital_names,
        "structure": structure,
        "properties": properties,
        "gradients": gradients,
        "hessians": hessians,
    }
    
    grid_dims = mathematics.get_grid_dims(kpoints)
    is_grid = kpoints.shape[0] == np.prod(grid_dims)
    
    if kpath is not None:
        ebs_args["kpath"] = kpath
        return ElectronicBandStructurePath(**ebs_args)
    elif is_grid:
        return ElectronicBandStructureMesh(**ebs_args)
    else:
        return ElectronicBandStructure(**ebs_args)
    
def get_ebs_from_code(
    code: str,
    dirpath: str,
    use_cache: bool = False,
    ebs_filename: str = "ebs.pkl",
    **kwargs):

    ebs = ElectronicBandStructure.from_code(code, dirpath, use_cache, ebs_filename, **kwargs)
    
    kpath=getattr(ebs, "kpath", None)
    if kpath is not None:
        logger.info("Creating ElectronicBandStructurePath from EBS")
        return ElectronicBandStructurePath.from_ebs(ebs)
    
    if ebs.structure.rotations.shape[0] > 0:
        ebs = ibz2fbz(ebs, rotations=ebs.structure.rotations, decimals=4, inplace=False)
        ebs = sort_by_kpoints(ebs, inplace=False)
        
    grid_dims = mathematics.get_grid_dims(ebs.kpoints)
    is_grid = ebs.kpoints.shape[0] == np.prod(grid_dims)
    
    
    
    logger.debug(f"grid_dims: {grid_dims}")
    
    if is_grid:
        logger.info("Creating ElectronicBandStructureMesh from EBS")
        return ElectronicBandStructureMesh.from_ebs(ebs)
    else:
        return ebs


def reduced_to_cartesian(kpoints: KPOINTS_DTYPE, reciprocal_lattice: RECIPROCAL_LATTICE_DTYPE) -> KPOINTS_DTYPE:
    if reciprocal_lattice is not None:
        return np.dot(kpoints, reciprocal_lattice)
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return


def cartesian_to_reduced(cartesian: KPOINTS_DTYPE, reciprocal_lattice: RECIPROCAL_LATTICE_DTYPE) -> KPOINTS_DTYPE:
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
        kpoints = np.dot(cartesian, np.linalg.inv(reciprocal_lattice))
        return kpoints
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return


def _compare_arrays(array1: npt.NDArray[np.dtype[np.number]], 
                    array2: npt.NDArray[np.dtype[np.number]]) -> bool:
    if array1 is not None and array2 is not None:
        return np.allclose(array1, array2)
    elif array1 is None and array2 is None:
        return True
    else:
        return False

def calculate_avg_inv_effective_mass(hessian: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]):
    # Calculate the trace of each 3x3 matrix along the last two axes
    m_inv = (np.trace(hessian, axis1=-2, axis2=-1) * EV_TO_J / HBAR_J**2) / 3
    # Calculate the harmonic average effective mass for each element
    e_mass = FREE_ELECTRON_MASS * m_inv
    return e_mass


def _deepcopy_dict(d: PROPERTIES_DTYPE | None):
    """Performs a deep copy of a dictionary containing NumPy arrays."""
    if d is None:
        return None
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in d.items()}


# class BaseElectronicBandStructure(ABC):
    
#     @property
#     @abstractmethod
#     def point_store(self) -> PointSet:
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def fermi(self) -> float:
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def structure(self) -> Structure | None:
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def reciprocal_lattice(self) -> RECIPROCAL_LATTICE_DTYPE | None:
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def orbital_names(self) -> list[str] | None:
#         raise NotImplementedError

#     @abstractmethod
#     def compute_property(self, name: str, **kwargs):
#         raise NotImplementedError

#     def get_property(self, 
#             name: str, 
#             calc_type: Literal["value", "gradients", "divergence", "vortex", "laplacian"] = "value", 
#             gradient_order:int=0, 
#             compute:bool=True, 
#             order: Literal["F", "C"] = "F", 
#             **kwargs):
#         if name not in self.point_store:
#             self.compute_property(name, **kwargs)

#         if calc_type == "value":
#             return self.point_store[name].value
#         elif calc_type == "gradients":
#             return self.point_store[name].gradients[gradient_order]
#         elif calc_type == "divergence":
#             return self.point_store[name].divergence
#         elif calc_type == "vortex":
#             return self.point_store[name].vortex
#         elif calc_type == "laplacian":
#             return self.point_store[name].laplacian
#         else:
#             raise ValueError(f"Invalid calc_type: {calc_type}")



class ElectronicBandStructure:
    """This object stores electronic band structure informomration.

    Parameters
    ----------
    kpoints : np.ndarray
        The kpoints array. Will have the shape (n_kpoints, 3)
    bands : np.ndarray
        The bands array. Will have the shape (n_kpoints, n_bands)
    fermi : float
        The fermi energy
    projected : np.ndarray, optional
        The projections array. Will have the shape (n_kpoints, n_bands, n_spins, norbitals,n_atoms), defaults to None
    projected_phase : np.ndarray, optional
        The full projections array that incudes the complex part. Will have the shape (n_kpoints, n_bands, n_spins, norbitals,n_atoms), defaults to None
    weights : np.ndarray, optional
        The weights of the kpoints. Will have the shape (n_kpoints, 1), defaults to None
    orbital_names : list, optional
        The names of the orbitals. Defaults to None
    reciprocal_lattice : np.ndarray, optional
        The reciprocal lattice vector matrix. Will have the shape (3, 3), defaults to None
    shifted_to_fermi : bool, optional
         Boolean to determine if the fermi energy is shifted, defaults to False
    """

    def __init__(
        self,
        kpoints: KPOINTS_DTYPE ,
        fermi: float = 0.0,
        bands: BANDS_DTYPE | None = None,
        projected: PROJECTED_DTYPE | None = None,
        projected_phase: PROJECTED_PHASE_DTYPE | None = None,
        weights: WEIGHTS_DTYPE | None = None,
        orbital_names: list[str] | None = None,
        reciprocal_lattice: RECIPROCAL_LATTICE_DTYPE | None = None,
        shifted_to_fermi: bool = False,
        structure: Structure | None = None,
        point_store: PointSet | None = None,
    ):
        self._kpoints = np.array(kpoints, dtype=float).copy()
        self._point_store = PointSet(self._kpoints)
        
        # We store bands in the properties dict
        # to allow for a unified way of handling derivatives
        
        self._fermi = fermi
        self._orbital_names = orbital_names
        self._reciprocal_lattice = reciprocal_lattice
        self._shifted_to_fermi = shifted_to_fermi

        self._properties = _deepcopy_dict(properties)
        if bands is not None:
            self._properties["bands"] = np.array(bands, dtype=float).copy()
        if projected is not None:
            self._properties["projected"] = np.array(projected, dtype=float).copy()
        if projected_phase is not None:
            self._properties["projected_phase"] = np.array(
                projected_phase, dtype=np.cdouble()
            ).copy()
        if weights is not None:
            self._properties["weights"] = np.array(weights, dtype=float).copy()
            
        self._gradients = _deepcopy_dict(gradients)
        self._hessians = _deepcopy_dict(hessians)
        self._structure = structure

    def __str__(self):
        ret = "\n Electronic Band Structure     \n"
        ret += "============================\n"
        ret += "Total number of kpoints   = {}\n".format(self.n_kpoints)
        if "bands" in self.properties:
            ret += "Total number of bands    = {}\n".format(self.n_bands)
        else:
            ret += "Total number of bands    = None\n"
        if "projected" in self.properties:
            ret += "Total number of atoms    = {}\n".format(self.n_atoms)
        else:
            ret += "Total number of atoms    = None\n"
        if "projected" in self.properties:
            ret += "Total number of orbitals = {}\n".format(self.n_orbitals)
        else:
            ret += "Total number of orbitals = None\n"
        if "bands" in self.properties:
            ret += "Total number of spin channels = {}\n".format(self.n_spin_channels)
        else:
            ret += "Total number of spin channels = None\n"
        if "projected" in self.properties:
            ret += "Total number of spin projections = {}\n".format(self.n_spins)
        else:
            ret += "Total number of spin projections = None\n"

        ret += "\nArray shapes: \n"
        ret += "------------------------     \n"
        k_properties = self.property_names
        for prop in k_properties:
            ret += f"{prop} shape = {self.get_property(prop).shape}\n"

        ret += "\nGradients: \n"
        ret += "------------------------     \n"
        for prop in self._gradients.keys():
            ret += f"{prop} shape = {self.get_property(prop, return_gradient_order=1).shape}\n"

        ret += "\nHessians: \n"
        ret += "------------------------     \n"
        for prop in self._hessians.keys():
            ret += f"{prop} shape = {self.get_property(prop, return_gradient_order=2).shape}\n"

        ret += "\nAdditional information: \n"
        if self.orbital_names is not None:
            ret += "Orbital Names = {}\n".format(self.orbital_names)
            
        if "projected" in self.properties:
            ret += f"Spin Projection Names = {self.spin_projection_names}\n"
            ret += f"Non-colinear = {self.is_non_collinear}\n"
        else:
            ret += "Spin Projection Names = None\n"
            
        
        if self.reciprocal_lattice is not None:
            ret += "Reciprocal Lattice = \n {}\n".format(self.reciprocal_lattice)
        ret += "Fermi Energy = {}\n".format(self.fermi)
        if self.structure is not None:
            ret += "\nStructure: \n"
            ret += "------------------------     \n"
            ret += "Structure = \n {}\n".format(self.structure)

        return ret
    
    def __eq__(self, other):
        kpoints_equal = _compare_arrays(self.kpoints, other.kpoints)
        bands_equal = _compare_arrays(self.bands, other.bands)
        fermi_equal = self.fermi == other.fermi
        projected_equal = _compare_arrays(self.projected, other.projected)
        projected_phase_equal = _compare_arrays(self.projected_phase, other.projected_phase)
        weights_equal = _compare_arrays(self.weights, other.weights)
        
        is_ebs_equal = (kpoints_equal and bands_equal and fermi_equal and 
                        projected_equal and projected_phase_equal and weights_equal)
        return is_ebs_equal

    @property
    def point_store(self) -> PointSet:
        return PointSet(self._kpoints, self._properties)

    @property
    def kpoints(self):
        return self._kpoints

    @property
    def bands(self):
        return self.get_property("bands", compute=False)

    @property
    def projected(self):
        return self.get_property("projected", compute=False)

    @property
    def projected_phase(self):
        return self.get_property("projected_phase", compute=False)

    @property
    def weights(self):
        return self.get_property("weights", compute=False)

    @property
    def orbital_names(self):
        return self._orbital_names

    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice
    
    @property
    def properties(self):
        return self._properties
    
    @property
    def gradients(self):
        return self._gradients
    
    @property
    def hessians(self):
        return self._hessians
    
    @property
    def brillouin_zone(self):
        return BrillouinZone(self.reciprocal_lattice, np.array([1, 1, 1]))

    @property
    def fermi(self):
        return self._fermi

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
    def is_grid(self):
        grid_dims = mathematics.get_grid_dims(self.kpoints)
        return self.n_kpoints == np.prod(grid_dims)

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
    def is_spin_polarized(self):
        return self.n_spin_channels == 2
    
    def has_spin_channels(self, property_value):
        property_value_shape = list(property_value.shape)
        if len(property_value_shape) >= 3:
            nspins = property_value_shape[2]
            return nspins == self.n_spin_channels
        else:
            return False
    
    def is_band_property(self, property_value):
        property_value_shape = list(property_value.shape)
        if len(property_value_shape) >= 3:
            nbands = property_value_shape[1]
            return nbands == self.n_bands
        else:
            return False

    def is_orbital_property(self, property_value):
        property_value_shape = list(property_value.shape)
        if len(property_value_shape) >= 5:
            nkpoints, nbands, nspins, natoms, norbitals = (
                property_value_shape[0],
                property_value_shape[1],
                property_value_shape[2],
                property_value_shape[3],
                property_value_shape[4],
            )
            return (
                nkpoints == self.n_kpoints
                and nbands == self.n_bands
                and nspins == self.n_spin_channels
                and natoms == self.n_atoms
                and norbitals == self.n_orbitals
            )
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
            if self.is_band_property(property_value):
                names.append(property_name)
        return names

    @property
    def property_names(self):
        names = []
        for property_name in self._properties.keys():
            names.append(property_name)
        return names
    
    @property
    def gradient_names(self):
        names = []
        for gradient_name in self._gradients.keys():
            names.append(gradient_name)
        return names
    
    @property
    def hessian_names(self):
        names = []
        for hessian_name in self._hessians.keys():
            names.append(hessian_name)
        return names

    @property
    def bands_velocity(self):
        prop = self.get_property("bands_velocity")
        if prop is not None:
            return prop

        prop = self.compute_band_velocity()
        return prop

    @property
    def bands_speed(self):
        prop = self.get_property("band_speed")
        if prop is not None:
            return prop

        band_speed = self.compute_band_speed()
        return band_speed

    @property
    def avg_inv_effective_mass(self):
        prop = self.get_property("avg_inv_effective_mass")
        if prop is not None:
            return prop

        avg_inv_effective_mass = self.compute_avg_inv_effective_mass()
        return avg_inv_effective_mass

    @property
    def ebs_ipr(self):
        prop = self.get_property("ebs_ipr")
        if prop is not None:
            return prop

        ebs_ipr = self.compute_ebs_ipr()
        return ebs_ipr

    @property
    def ebs_ipr_atom(self):
        """
        It returns the atom-resolved , pIPR:

        pIPR_j =  \\frac{\|c_j\|^4}{(\\sum_i \|c_i^2\|)^2}

        Clearly, \\( \\sum_j pIPR_j = IPR \\).

        Mind: \( c_i \) is the wavefunction \( c(n,k)_i \), in pyprocar we already
        have density projections, \( c_i^2 \).

        *THIS QUANTITY IS NOT READY FOR PLOTTING*, please prefer `self.ebs_ipr()`

        Returns
        -------
        ret : list float
            The IPR projections

        """
        prop = self.get_property("ebs_ipr_atom")
        if prop is not None:
            return prop
        
        ebs_ipr_atom = self.compute_ebs_ipr_atom()
        return ebs_ipr_atom

    @property
    def spin_texture(self):
        prop = self.get_property("spin_texture")
        if prop is not None:
            return prop
        
        spin_texture = self.compute_spin_texture()
        return spin_texture
    
    @property
    def projected_sum(self):
        prop = self.get_property("projected_sum")
        if prop is not None:
            return prop
        
        projected_sum = self.compute_projected_sum()
        return projected_sum
    
    @property
    def projected_sum_spin_texture(self):
        prop = self.get_property("projected_sum_spin_texture")
        if prop is not None:
            return prop
        
        projected_sum_spin_texture = self.compute_projected_sum_spin_texture()
        return projected_sum_spin_texture
    
    def get_property(self, name: str, return_gradient_order:int=0, compute:bool=True, order="F", **kwargs):
        
        if return_gradient_order == 0:
            if name not in self.properties and compute:
                self.compute_property(name, **kwargs)
            
            prop = self.properties.get(name, None)
            return prop
        
        elif return_gradient_order == 1:
            if name not in self.gradients and compute:
                self.compute_gradient(name, order=order)
            
            prop = self.gradients.get(name, None)
            return prop
        elif return_gradient_order == 2:
            if name not in self.hessians and compute:
                self.compute_gradient(name, first_order=False, second_order=True, order=order)
            
            prop = self.hessians.get(name, None)
            return prop
        else:
            raise ValueError(f"Gradient order ({return_gradient_order}) is not valid. Must be 0, 1, or 2.")
    
    def add_property(self, name: str, value: np.ndarray, return_gradient_order:int=0, overwrite: bool = True):
        if return_gradient_order == 0:
            if name in self._properties and not overwrite:
                msg = (f"Property {name} already exists. "
                f"Use overwrite=True to overwrite it.")
                user_logger.warning(msg)
                raise ValueError(msg)
            self._properties[name] = value
        elif return_gradient_order == 1:
            if name in self._gradients and not overwrite:
                msg = (f"Gradient {name} already exists. "
                f"Use overwrite=True to overwrite it.")
                user_logger.warning(msg)
                raise ValueError(msg)
            self._gradients[name] = value
            
        elif return_gradient_order == 2:
            if name in self._hessians and not overwrite:
                msg = (f"Hessian {name} already exists. "
                f"Use overwrite=True to overwrite it.")
                user_logger.warning(msg)
                raise ValueError(msg)
            self._hessians[name] = value
        return self
    
    def set_bands(self, bands: np.ndarray):
        self.add_property("bands", bands, overwrite=True)
        return self
    
    def set_kpoints(self, kpoints: np.ndarray):
        self._kpoints = kpoints
        return self
    
    def compute_property(self, name: str, **kwargs):
        if name == "bands_velocity":
            return self.compute_band_velocity(**kwargs)
        elif name == "band_speed":
            return self.compute_band_speed(**kwargs)
        elif name == "avg_inv_effective_mass":
            return self.compute_avg_inv_effective_mass(**kwargs)
        elif name == "ebs_ipr":
            return self.compute_ebs_ipr(**kwargs)
        elif name == "ebs_ipr_atom":
            return self.compute_ebs_ipr_atom(**kwargs)
        elif name == "spin_texture":
            return self.compute_spin_texture(**kwargs)
        elif name == "projected_sum":
            return self.compute_projected_sum(**kwargs)
        elif name == "projected_sum_spin_texture":
            return self.compute_projected_sum_spin_texture(**kwargs)
        else:
            raise ValueError(f"Property ({name}) does not exist. Use add_property to add it.")
    
    def compute_gradient(
        self,
        name: str,
        first_order: bool = True,
        second_order: bool = False,
        order="F"):
        raise NotImplementedError(
            "Gradient computation is not implemented for this class."
        )
    
    def compute_band_velocity(self, label=None, **kwargs):
        if label is None:
            label="bands_velocity"
        band_gradient, _ = self.compute_gradient("bands", first_order=True, second_order=False, **kwargs)
        band_velocity = band_gradient / HBAR_EV
        self.add_property(label, band_velocity, overwrite = kwargs.pop("overwrite", True))
        return band_velocity
    
    def compute_band_speed(self, label = None, **kwargs):
        if label is None:
            label="band_speed"
        band_velocity = self.compute_band_velocity(**kwargs)
        band_speed = np.linalg.norm(band_velocity, axis=-1)
        self.add_property(label, band_speed, overwrite = kwargs.pop("overwrite", True))
        return band_speed
    
    def compute_avg_inv_effective_mass(self, label=None, **kwargs):
        if label is None:
            label="avg_inv_effective_mass"
        bands_hessian = self.get_property("bands", return_gradient_order=2, **kwargs)
        avg_inv_effective_mass = calculate_avg_inv_effective_mass(bands_hessian)
        self.add_property(label, avg_inv_effective_mass, overwrite = kwargs.pop("overwrite", True))
        return avg_inv_effective_mass
    
    def compute_ebs_ipr(self, label=None, **kwargs):
        if label is None:
            label="ebs_ipr"
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
        self.add_property(label, IPR, overwrite = kwargs.pop("overwrite", True))
        return IPR
    
    def compute_ebs_ipr_atom(self, label=None, **kwargs):
        if label is None:
            label="ebs_ipr_atom"
        orbitals = np.arange(self.n_orbitals, dtype=int)
        # sum over orbitals
        proj = np.sum(self.projected[:, :, :, :, orbitals], axis=-1)
        # keeping only the last principal quantum number
        # selecting all atoms:
        atoms = np.arange(self.n_atoms, dtype=int)

        # the partial pIPR is \frac{|c_j|^4}{(\sum_i |c_i^2|)^2}
        # mind, every c_i is c_{i,n,k} with n,k the band and k-point indexes
        num = np.absolute(proj) ** 2
        den = np.absolute(proj)
        den = np.sum(den[..., atoms], axis=-1) ** 2
        pIPR = num / den[..., np.newaxis]
        self.add_property(label, pIPR, overwrite = kwargs.pop("overwrite", True))
        return pIPR
    
    def compute_projected_sum(self, 
                              atoms: list[int] = None, 
                              orbitals: list[int] = None, 
                              spins: list[int] = None,
                              label: str = None,
                              use_atomic_orbital_label:bool = False,
                              **kwargs):
        
        if spins is None:
            spins = list(np.arange(self.n_spins, dtype=int))
        
        if label is None and use_atomic_orbital_label:
            atomic_orbital_label = ElectronicBandStructure.get_atomic_orbital_label(atoms, orbitals)
            spin_project_label = ElectronicBandStructure.get_spin_projection_label(spins)
            label = f"projected__{atomic_orbital_label}|{spin_project_label}"
        elif label is None:
            label = "projected_sum"
            
        print(f"atoms: {atoms}, orbitals: {orbitals}, spins: {spins}")

        projected_sum = self.ebs_sum(
            atoms=atoms, orbitals=orbitals, spins=spins, sum_noncolinear=True
        )

        self.add_property(label, projected_sum, overwrite = kwargs.pop("overwrite", True))
        return projected_sum
    
    def compute_spin_texture(self, label=None, **kwargs):
        if not self.is_non_collinear:
            raise ValueError(
                "Spin texture is only available for non-collinear calculations"
            )
        if label is None:
            label="spin_texture"
        
        spin_texture = self.projected[:, :, 1:, :, :]
        spin_texture = np.moveaxis(spin_texture, 2, -1)
        self.add_property(label, spin_texture, overwrite = kwargs.pop("overwrite", True))
        return spin_texture

    def compute_projected_sum_spin_texture(self, 
                                           atoms: list[int] = None, 
                                           orbitals: list[int] = None, 
                                           label: str = None, 
                                           use_atomic_orbital_label:bool = False, **kwargs):
        if not self.is_non_collinear:
            raise ValueError(
                "Spin texture is only available for non-collinear calculations"
            )
            
        if atoms is None:
            atoms = np.arange(self.n_atoms, dtype=int)
        if orbitals is None:
            orbitals = np.arange(self.n_orbitals, dtype=int)
            
        if label is None and use_atomic_orbital_label:
            atomic_orbital_label = ElectronicBandStructure.get_atomic_orbital_label(atoms, orbitals)
            label = f"projected_sum_spin_texture__{atomic_orbital_label}"
        elif label is None:
            label = "projected_sum_spin_texture"

        summed_projection = self.ebs_sum(
            atoms=atoms, orbitals=orbitals, sum_noncolinear=False
        )

        projected_spin_texture = summed_projection[..., 1:]
        temp_shape = list(projected_spin_texture.shape)
        temp_shape.insert(2, 1)
        projected_spin_texture = projected_spin_texture.reshape(
            temp_shape, order=kwargs.pop("order", "F")
        )

        self.add_property(label, projected_spin_texture, overwrite = kwargs.pop("overwrite", True))
        return projected_spin_texture

    def ebs_sum(
        self,
        atoms: list[int] = None,
        orbitals: list[int] = None,
        spins: list[int] = None,
        sum_noncolinear: bool = True,
    ):
        """_summary_

        Parameters
        ----------
        atoms : list[int], optional
            list of atoms to be summed over, by default None
        orbitals : list[int], optional
            list of orbitals to be summed over, by default None
        spins : list[int], optional
            list of spins to be summed over, by default None
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
    
    def iter_properties(self, compute:bool=False):
        for gradient_order in [0, 1, 2]:
            for prop_name in self.property_names:
                property_value = self.get_property(prop_name, return_gradient_order=gradient_order, compute=compute)
                if property_value is not None:
                    yield prop_name, property_value, gradient_order
                
    def reduce_bands(self, bands: list[int] = None, near_fermi: bool = False, energy: float = None, 
                     tolerance: float = 0.7, inplace=True):
        if bands is not None:
            return self.reduce_bands_by_index(bands, inplace)
        elif energy is not None:
            return self.reduce_bands_near_energy(energy, tolerance, inplace)
        elif near_fermi:
            return self.reduce_bands_near_fermi(tolerance, inplace)
        else:
            raise ValueError("Either bands or energy or near_fermi must be provided")
        
    def reduce_bands_near_energy(
        self, energy: float, tolerance: float = 0.7, inplace=True
    ):
        """
        Reduces the bands to those near the fermi energy
        """
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
        
        logger.info("____Reducing bands near fermi energy____")
        full_band_index = []
        bands_spin_index = {}

        for ispin in ebs.spin_channels:
            bands_spin_index[ispin] = []
            for iband in range(ebs.n_bands):
                fermi_surface_test = len(
                    np.where(
                        np.logical_and(
                            ebs.bands[:, iband, ispin] >= energy - tolerance,
                            ebs.bands[:, iband, ispin] <= energy + tolerance,
                        )
                    )[0]
                )
                
                if fermi_surface_test != 0:
                    bands_spin_index[ispin].append(iband)

                    if iband not in full_band_index:  # Avoid duplicates
                        full_band_index.append(iband)

        band_property_names = ebs.band_property_names

        for prop_name, prop_value, gradient_order in ebs.iter_properties(compute=False):
            if prop_name in band_property_names:
                ebs.add_property(prop_name, prop_value[:, full_band_index, ...], return_gradient_order=gradient_order)
            else:
                ebs.add_property(prop_name, prop_value, return_gradient_order=gradient_order)

        debug_message = f"Bands near energy {energy}. "
        debug_message += f"Spin-0 {bands_spin_index[0]} |"
        if self.n_spin_channels > 1 and not self.is_non_collinear:
            debug_message += f" Spin-1 {bands_spin_index[1]}"
        logger.debug(debug_message)
        return ebs
        
    def reduce_bands_near_fermi(self, tolerance=0.7, inplace=True):
        """
        Reduces the bands to those near the fermi energy
        """
        return self.reduce_bands_near_energy(self.fermi, tolerance, inplace=inplace)

    def reduce_bands_by_index(self, bands, inplace=True):
        """
        Reduces the bands to those near the fermi energy
        """
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
        band_property_names = ebs.band_property_names
        for prop_name, prop_value, gradient_order in ebs.iter_properties(compute=False):
            if prop_name in band_property_names:
                ebs.add_property(prop_name, prop_value[:, bands, ...], return_gradient_order=gradient_order)
            else:
                ebs.add_property(prop_name, prop_value, return_gradient_order=gradient_order)
        return ebs
        
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
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
        
        if ebs.n_spin_channels != 2:
            raise ValueError("Spin channels must be 2 for this function to work")

        band_property_names = ebs.band_property_names
        for prop_name, prop_value, gradient_order in ebs.iter_properties(compute=False):
            if prop_name in band_property_names and ebs.has_spin_channels(prop_value):
                original_value_shape = list(prop_value.shape)
                band_dim = original_value_shape[1]
                original_value_shape[1] = 2 * band_dim
                original_value_shape[2] = 1
                modified_array = prop_value.reshape(original_value_shape)
                ebs.add_property(prop_name, modified_array, return_gradient_order=gradient_order)
            else:
                ebs.add_property(prop_name, prop_value, return_gradient_order=gradient_order)

        return ebs
        
    def shift_bands(self, shift_value, inplace=False):
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
            
        bands = ebs.get_property("bands")
        bands += shift_value
        ebs.add_property("bands", bands, return_gradient_order=0)
        return ebs
        
    def shift_kpoints_to_fbz(self, inplace=True):
        # Shifting all kpoint to first Brillouin zone
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
            
        bound_ops = -1.0 * (ebs._kpoints > 0.5) + 1.0 * (ebs._kpoints <= -0.5)
        new_kpoints = ebs._kpoints + bound_ops

        ebs.set_kpoints(new_kpoints)
        return  ebs
        
    def unfold(self, transformation_matrix=None, structure=None, inplace=True):
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
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
            
        uf = Unfolder(
            ebs=ebs,
            transformation_matrix=transformation_matrix,
            structure=structure,
        )
        
        ebs.add_property("weights", uf.weights, return_gradient_order=0)
        return ebs
        
    def save(self, path: Path):
        serializer = get_serializer(path)
        serializer.save(self, path)

    @classmethod
    def load(cls, path: Path):
        serializer = get_serializer(path)
        ebs = serializer.load(path)
        return ebs
    
    @classmethod
    def from_code(cls, code: str, dirpath: str, use_cache: bool = False, ebs_filename: str = "ebs.pkl"):
        from pyprocar.io import Parser
        ebs_filepath = Path(dirpath) / ebs_filename
        
        if not use_cache or not ebs_filepath.exists():
            logger.info(f"Parsing EBS calculation directory: {dirpath}")
            parser = Parser(code=code, dirpath=dirpath)
            ebs=parser.ebs
            ebs.save(ebs_filepath)
        else:
            logger.info(f"Loading EBS  from picklefile: {ebs_filepath}")
            ebs = cls.load(ebs_filepath)
        return ebs
    
    @classmethod
    def from_ebs(cls, ebs, **kwargs):
        return cls(
                   kpoints=ebs.kpoints,
                   bands=ebs.bands,
                   projected=ebs.projected,
                   projected_phase=ebs.projected_phase,
                   weights=ebs.weights,
                   fermi=ebs.fermi,
                   reciprocal_lattice=ebs.reciprocal_lattice,
                   orbital_names=ebs.orbital_names,
                   structure=ebs.structure,
                   properties=ebs._properties,
                   gradients=ebs._gradients,
                   hessians=ebs._hessians,
                   **kwargs)

    @staticmethod
    def get_atomic_orbital_label(atoms: list[int], orbitals: list[int]):
        atom_label = ElectronicBandStructure.get_atom_label(atoms)
        orbital_label = ElectronicBandStructure.get_orbital_label(orbitals)
        return f"{atom_label}|{orbital_label}"

    @staticmethod
    def get_orbital_label(orbitals: list[int]):
        orbitals_label = ",".join([str(orbital) for orbital in orbitals])
        return f"orbitals-({orbitals_label})"

    @staticmethod
    def get_atom_label(atoms: list[int]):
        atoms_label = ",".join([str(atom) for atom in atoms])
        return f"atoms-({atoms_label})"

    @staticmethod
    def get_band_label(bands: list[int] | int, spins: list[int] | int):
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
    def get_spin_projection_label(spin_projections: list[int]):
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

    def as_cart(self):
        self.points = self.kpoints_cartesian

    def as_frac(self):
        self.points = self.kpoints


    @property
    def kpath(self):
        return self._kpath

    @property
    def knames(self):
        return self.kpath.knames

    @property
    def n_segments(self):
        return self.kpath.n_segments

    @property
    def tick_positions(self):
        return self.kpath.tick_positions

    @property
    def tick_names(self):
        return self.kpath.tick_names

    @property
    def tick_names_latex(self):
        return self.kpath.tick_names_latex
    
    @property
    def special_kpoint_names(self):
        return self.kpath.special_kpoint_names
    
    def compute_gradient(
        self,
        name: str = None,
        first_order: bool = True,
        second_order: bool = False,
        **kwargs,
    ):
        if name not in self.property_names:
            raise ValueError(f"Property ({name}) does not exist. Use add_property to add it.")
        
        gradients=None
        hessians=None
        if first_order:
            continuous_segments = self.kpath.get_continuous_segments()

            val_array = self.get_property(name)
            gradients = np.zeros(val_array.shape)
            for k_indices in continuous_segments:
                kpath_segment = self.kpoints_cartesian[k_indices]
                delta_k = np.gradient(kpath_segment, axis=0)
                delta_k = np.linalg.norm(delta_k, axis=1)

                gradients[k_indices, ...] = np.gradient(
                    val_array[k_indices, ...],
                    delta_k,
                    axis=0,
                    edge_order=2,
                )
            gradients = gradients * METER_ANGSTROM
            self.add_property(name, gradients, return_gradient_order=1)

        if second_order:
            continuous_segments = self.kpath.get_continuous_segments()

            val_gradients = self.get_property(name, return_gradient_order=1)
            hessians = np.zeros(val_gradients.shape)
            for k_indices in continuous_segments:
                kpath_segment = self.kpoints_cartesian[k_indices]
                delta_k = np.gradient(kpath_segment, axis=0)
                delta_k = np.linalg.norm(delta_k, axis=1)

                hessians[k_indices, ...] = np.gradient(
                    val_gradients[k_indices, ...],
                    delta_k,
                    axis=0,
                    edge_order=2,
                )
            hessians = hessians * METER_ANGSTROM
            self.add_property(name, hessians, return_gradient_order=2)
            
        return gradients, hessians
               
    def as_kdist(self, as_segments=True):
        kdistances = self.kpath.get_distances(as_segments=False, cumlative_across_segments=True)
        n_bands = self.bands.shape[1]
        n_spins = self.bands.shape[2]
        n_kpoints = kdistances.shape[0]
        k_indices = self.kpath.segment_indices
        blocks=pv.MultiBlock()
        
        if as_segments:
            for indices in k_indices:
                n_indices = indices.shape[0]
                for iband in range(n_bands):
                    for ispin in range(n_spins):
                        k_segment_distances=kdistances[indices]
                        bands=self.bands[indices, iband, ispin]

                        band_kpoints=np.zeros(shape=(n_indices, 3))
                        band_kpoints[:, 0] = k_segment_distances
                        band_kpoints[:, 1] = bands
                        blocks.append(pv.PolyData(band_kpoints))
        else:
            for iband in range(n_bands):
                for ispin in range(n_spins):
                    bands=self.bands[:, iband, ispin]
                    k_distances=kdistances.copy()
                    band_kpoints=np.zeros(shape=(n_kpoints, 3))
                    band_kpoints[:, 0] = k_distances
                    band_kpoints[:, 1] = bands
                    band_kpoints[:, 2] = ispin
                    blocks.append(pv.PolyData(band_kpoints))
        return blocks
        
    def plot(
        self,
        add_point_labels_args: dict = None,
        bz_add_mesh_args: dict = None,
        **kwargs,
    ):
        """
        Plots the band structure.

        """
        self.as_cart()
        add_point_labels_args = add_point_labels_args or {}
        bz_add_mesh_args = bz_add_mesh_args or {}

        special_kpoint_names = self.kpath.special_kpoint_names
        special_kpoint_positions = self.kpath.get_special_kpoints(as_segments=False, cartesian=True)

        p = pv.Plotter()
        p.add_mesh(self, **kwargs)
        p.add_point_labels(
            special_kpoint_positions, special_kpoint_names, **add_point_labels_args
        )

        bz_add_mesh_args["style"] = bz_add_mesh_args.get("style", "wireframe")
        bz_add_mesh_args["line_width"] = bz_add_mesh_args.get("line_width", 2.0)
        bz_add_mesh_args["color"] = bz_add_mesh_args.get("color", "black")
        bz_add_mesh_args["opacity"] = bz_add_mesh_args.get("opacity", 1.0)

        p.add_mesh(
            self.brillouin_zone,
            **bz_add_mesh_args,
        )
        p.show()
        
    @classmethod
    def from_ebs(cls, ebs, **kwargs):
        return super().from_ebs(ebs, kpath = ebs.kpath, **kwargs)
        

def is_plane_aligned_with_reciprocal_lattice(normal: np.ndarray, 
                                             reciprocal_lattice: np.ndarray):
    """Check if the plane normal is aligned with reciprocal lattice axes"""
    if reciprocal_lattice is None:
        return False
        
    normal = normal / np.linalg.norm(normal)
    
    # Check alignment with each reciprocal lattice vector
    for i, recip_vec in enumerate(reciprocal_lattice):
        recip_unit = recip_vec / np.linalg.norm(recip_vec)
        dot_product = abs(np.dot(normal, recip_unit))
        if dot_product > 0.99:  # Nearly parallel (within 1 degree)
            return True
    return False

def edge_diff_ramp(vector, pad_width, iaxis, kwargs):
    if pad_width[0] == 0 or pad_width[1] == 0:
        return vector
    
    original_index = pad_width[0] + 1
    original_end_index = vector.shape[0] - pad_width[1] - 1
    
    dx = abs(vector[original_index] - vector[original_index + 1])
    
    # Create left padding using array operations
    left_pad = vector[original_index] - (np.arange(pad_width[0], 0, -1) + 1) * dx
    vector[:pad_width[0]] = left_pad
    
    # Create right padding using array operations  
    right_pad = vector[original_end_index] + (np.arange(1, pad_width[1] + 1)) * dx
    vector[-pad_width[1]:] = right_pad
            
            
class ElectronicBandStructureMesh(ElectronicBandStructure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if self.n_kpoints != np.prod(self.get_kgrid()):
            raise ValueError("n_kpoints must be equal to np.prod(kgrid) (number of kpoints)")
        
        self._property_interpolators = {}
        self._gradient_interpolators = {}
        self._hessian_interpolators = {}

    def __str__(self):
        ret = super().__str__()
        ret += "\nKGrid: \n"
        ret += "------------------------     \n"
        ret += "(nkx, nky, nkz) = \n {}\n".format(self.kgrid)
        return ret

    @property
    def kgrid(self):
        return self.get_kgrid()
    
    @property
    def kbounds(self):
        return self.get_kbounds()

    @property
    def ukx(self):
        return np.linspace(self.kbounds[0, 0], self.kbounds[0, 1], self.n_kx)
    
    @property
    def uky(self):
        return np.linspace(self.kbounds[1, 0], self.kbounds[1, 1], self.n_ky)
    
    @property
    def ukz(self):
        return np.linspace(self.kbounds[2, 0], self.kbounds[2, 1], self.n_kz)

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
        return self.n_kpoints != np.prod(self.kgrid)

    @property
    def is_fbz(self):
        return self.n_kpoints == np.prod(self.kgrid)

    @property
    def kpoints_cartesian_mesh(self):
        return mathematics.array_to_mesh(self.kpoints_cartesian, nkx=self.n_kx, nky=self.n_ky, nkz=self.n_kz)
    
    @property
    def kpoints_mesh(self):
        return mathematics.array_to_mesh(self.kpoints, nkx=self.n_kx, nky=self.n_ky, nkz=self.n_kz)
    
    @property
    def property_interpolators(self):
        return self._property_interpolators
    
    def get_kbounds(self):
        kbounds = np.zeros((3, 2))
        for icoord in range(3):
            coords = self.kpoints[:, icoord]
            kx_min, kx_max = np.min(coords), np.max(coords)
            kbounds[icoord, 0] = kx_min
            kbounds[icoord, 1] = kx_max
        return kbounds
        
    def get_kgrid(self, num_bins:int=1000, height:float=1, coord_tol:float=0.01):
        return mathematics.get_grid_dims(self.kpoints, num_bins=num_bins, height=height, coord_tol=coord_tol)
                
    def get_kpoints_mesh(self, **kwargs):
            return mathematics.array_to_mesh(
                array=self.kpoints,
                nkx=self.n_kx,
                nky=self.n_ky,
                nkz=self.n_kz,
                **kwargs,
            )
        
    def get_property_mesh(self, name: str, return_gradient_order:int=0, compute:bool=True, **kwargs):
        property_array = self.get_property(name, return_gradient_order=return_gradient_order, compute=compute, **kwargs)
        if property_array is None:
            return None
        
        return mathematics.array_to_mesh(
                array=property_array,
                nkx=self.n_kx,
                nky=self.n_ky,
                nkz=self.n_kz,
                **kwargs,
            )
        
    def get_property_interpolator(self, name:str, return_gradient_order:int=0, compute_interpolator:bool=False, **kwargs):
        if return_gradient_order == 0:
            if name not in self.property_interpolators or compute_interpolator:
                interpolator = self.compute_interpolator(name, self.get_property_mesh(name, return_gradient_order=0, **kwargs))
                self._property_interpolators[name] = interpolator
            return self._property_interpolators[name]
        elif return_gradient_order == 1:
            if name not in self._gradient_interpolators or compute_interpolator:
                interpolator = self.compute_interpolator(name, self.get_property_mesh(name, return_gradient_order=1, **kwargs))
                self._gradient_interpolators[name] = interpolator
            return self._gradient_interpolators[name]
        elif return_gradient_order == 2:
            if name not in self._hessian_interpolators or compute_interpolator:
                interpolator = self.compute_interpolator(name, self.get_property_mesh(name, return_gradient_order=2, **kwargs))
                self._hessian_interpolators[name] = interpolator
            return self._hessian_interpolators[name]
        else:
            raise ValueError(f"Invalid return_gradient_order: {return_gradient_order}")
        
    def compute_gradient(
        self,
        name: str = None,
        first_order: bool = True,
        second_order: bool = False,
        **kwargs,
    ):
        if name not in self.property_names:
            raise ValueError(f"Property ({name}) does not exist. Use add_property to add it.")
        
        
        gradients=None
        hessians=None
        
        if first_order:
            val_mesh = self.get_property_mesh(name, return_gradient_order=0, **kwargs)

            gradients_mesh = mathematics.calculate_3d_mesh_scalar_gradients(
                val_mesh, self.reciprocal_lattice
            )
            gradients_mesh *= METER_ANGSTROM

            gradients=mathematics.mesh_to_array(
                mesh=gradients_mesh, **kwargs
            )
            self.add_property(name, gradients, return_gradient_order=1)

        if second_order:
            val_mesh = self.get_property_mesh(name, return_gradient_order=1, **kwargs)

            hessians_mesh = mathematics.calculate_3d_mesh_scalar_gradients(
                val_mesh, self.reciprocal_lattice
            )
            hessians_mesh *= METER_ANGSTROM
            hessians=mathematics.mesh_to_array(
                mesh=hessians_mesh, **kwargs
            )
            self.add_property(name, hessians, return_gradient_order=2)
            
        return gradients, hessians
    
    def compute_interpolator(self, name:str, scalars:np.ndarray, **kwargs):
        if name not in self.property_names:
            raise ValueError(f"Property ({name}) does not exist. Use add_property to add it.")

        interpolator = RegularGridInterpolator((self.ukx,self.uky,self.ukz), scalars, **kwargs)
        return interpolator
        
    def pad(self, padding=10, order="F", inplace=True):
        logger.info(f"Padding kpoints by {padding} in all directions")
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
        
        padding_dims = []
        for i, n in enumerate(ebs.kgrid):
            if n == 1:
                padding_dims.append((0,0))
            else:
                padding_dims.append((padding, padding))
  
        
        kpoints_padding_dims = copy.deepcopy(padding_dims)
        kpoints_padding_dims.append((0, 0))
        padded_kpoints_mesh = np.pad(ebs.kpoints_mesh, kpoints_padding_dims, mode=edge_diff_ramp)
        logger.debug(f"Padded kpoints mesh shape: {padded_kpoints_mesh.shape}")

        for prop_name, prop_value, gradient_order in ebs.iter_properties(compute=False):
            prop_mesh = ebs.get_property_mesh(prop_name, return_gradient_order=gradient_order, compute=False)
            n_scalar_dims = len(prop_mesh.shape[3:])
            scalar_padding_dims = copy.deepcopy(padding_dims)
            for i in range(n_scalar_dims):
                scalar_padding_dims.append((0, 0))
            padded_mesh = np.pad(prop_mesh, scalar_padding_dims, mode='wrap')
            logger.debug(f"Padded {prop_name} mesh shape: {padded_mesh.shape}")
            padded_array = mathematics.mesh_to_array(
                padded_mesh, order=order
            )
            ebs.add_property(prop_name, padded_array, return_gradient_order=gradient_order)

        ebs._kpoints = mathematics.mesh_to_array(padded_kpoints_mesh, order=order)
        return ebs
    
    def expand_kpoints_to_supercell_by_axes(self, axes_to_expand=[0, 1, 2], inplace=True, **kwargs):
        logger.info(f"Expanding kpoints to supercell by axes: {axes_to_expand}")
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
        # Validate input
        if not set(axes_to_expand).issubset({0, 1, 2}):
            raise ValueError("axes_to_expand must be a subset of [0, 1, 2]")

        # Create supercell directions based on axes to expand
        supercell_directions = list(
            itertools.product([1, 0, -1], repeat=len(axes_to_expand))
        )
        
        n_init_points = ebs.n_kpoints
        new_kpoints = ebs.kpoints.copy()
        for supercell_direction in supercell_directions:
            if supercell_direction == tuple([0] * len(axes_to_expand)):
                continue
            
            shifted_kpoints = ebs.kpoints.copy()
            for i, axis in enumerate(axes_to_expand):
                shifted_kpoints[:, axis] += supercell_direction[i]

            new_kpoints = np.concatenate(
                [new_kpoints, shifted_kpoints], axis=0
            )
            for prop_name, prop_value, gradient_order in ebs.iter_properties(compute=False):
                initial_array = prop_value[:n_init_points]
                new_points = np.concatenate([prop_value, initial_array], axis=0)
                
                ebs.add_property(prop_name, new_points, return_gradient_order=gradient_order)
                    
        ebs._kpoints = new_kpoints
        return ebs
        
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
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
        # Calculate new mesh dimensions
        kpoints_mesh = ebs.get_kpoints_mesh(order=order, compute=False)
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
        for prop in ebs.property_names:
            original_value = ebs.get_property(prop)
            if original_value is not None:
                mesh = mathematics.array_to_mesh(
                    original_value, ebs.n_kx, ebs.n_ky, ebs.n_kz, order=order
                )
                interpolated_mesh = mathematics.fft_interpolate_nd_3dmesh(
                    mesh,
                    interpolation_factor,
                )
                interpolated_value = mathematics.mesh_to_array(interpolated_mesh)
                new_properties[prop] = interpolated_value

            original_gradient = ebs.get_gradient(prop, compute=False)
            if original_gradient is not None:
                mesh = mathematics.array_to_mesh(
                    original_gradient, ebs.n_kx, ebs.n_ky, ebs.n_kz, order=order
                )
                interpolated_mesh = mathematics.fft_interpolate_nd_3dmesh(
                    mesh,
                    interpolation_factor,
                )
                interpolated_gradient = mathematics.mesh_to_array(interpolated_mesh)
                new_gradients[prop] = interpolated_gradient

            original_hessian = ebs.get_hessian(prop, compute=False)
            if original_hessian is not None:
                mesh = mathematics.array_to_mesh(
                    original_hessian, ebs.n_kx, ebs.n_ky, ebs.n_kz, order=order
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

        ebs._kpoints = new_kpoints
        ebs._properties = new_properties
        ebs._gradients = new_gradients
        ebs._hessians = new_hessians
        return ebs
    
    def reduce_to_plane(self, normal: np.ndarray, origin: np.ndarray, grid_interpolation: tuple[int, int] = None, **kwargs):
        return ElectronicBandStructurePlane(ebs_mesh=self, 
                                            normal=normal, 
                                            origin=origin, 
                                            grid_interpolation=grid_interpolation,
                                            **kwargs)
    
    @classmethod
    def from_ebs(cls, ebs, **kwargs):
        ebs=ibz2fbz(ebs, rotations=ebs.structure.rotations, decimals=4, inplace=False)
        return cls(
            kpoints=ebs.kpoints,
            bands=ebs.bands,
            fermi=ebs.fermi,
            projected=ebs.projected,
            projected_phase=ebs.projected_phase,
            weights=ebs.weights,
            orbital_names=ebs.orbital_names,
            reciprocal_lattice=ebs.reciprocal_lattice,
            structure=ebs.structure,
            properties=ebs.properties,
            gradients=ebs.gradients,
            hessians=ebs.hessians,
            **kwargs
        )
    
    @classmethod
    def from_code(cls, code: str, dirpath: str, use_cache: bool = False, ebs_filename: str = "ebs.pkl"):
        ebs = super().from_code(code, dirpath, use_cache, ebs_filename)
        ebs=ibz2fbz(ebs, rotations=ebs.structure.rotations, decimals=4, inplace=False)
        return cls.from_ebs(ebs)
    

class ElectronicBandStructurePlane(ElectronicBandStructure):
    
    def __init__(self, 
                 ebs_mesh: ElectronicBandStructureMesh,
                 normal: np.ndarray, 
                 origin: np.ndarray =None, 
                 plane_tol: float = 0.01,
                 grid_interpolation: tuple[int, int] = None,
                 u_limits: tuple[float, float] = None,
                 v_limits: tuple[float, float] = None,
                 **kwargs):
        self._ebs_mesh = ebs_mesh
        self._normal = normal / np.linalg.norm(normal)
        self._origin = origin if origin is not None else np.array([0, 0, 0])
        self._u, self._v = self.get_orthonormal_basis()
        
        if grid_interpolation is None and not self.is_plane_axes_aligned:
            grid_interpolation=(20, 20)
            msg=f"Grid interpolation is required when plane's normal is not aligned with a reciprocal lattice vectors.\n"
            msg+= f"Using default grid interpolation of {grid_interpolation}.\n"
            msg+= f"This can either be set on instantiation or directly with the grid_interpolation property."
            user_logger.warning(msg)
        elif grid_interpolation is None and self.is_plane_axes_aligned:
            grid_interpolation = []
            for i in range(3):
                reciprocal_lattice_vector = self.reciprocal_lattice[i]
                if np.dot(reciprocal_lattice_vector, self.normal) == 0.0:
                    n_dim = self.ebs_mesh.kgrid[i]
                    grid_interpolation.append(n_dim)
            grid_interpolation = tuple(grid_interpolation)
            
        logger.info(f"Grid interpolation: {grid_interpolation}")
        
        self._grid_interpolation = grid_interpolation
        _, i_mesh_points_near_plane = ElectronicBandStructurePlane.find_points_near_plane(ebs_mesh.kpoints_cartesian, normal, origin, 
                                                                                    plane_tol=plane_tol, 
                                                                                    return_indices=True)
        
        plane_points = self.transform_points_to_uv(ebs_mesh.kpoints_cartesian[i_mesh_points_near_plane])
        plane_limits = self._find_plane_limits(plane_points)
        self._u_limits = u_limits if u_limits is not None else plane_limits[0]
        self._v_limits = v_limits if v_limits is not None else plane_limits[1]
            
        super().__init__(kpoints=self.uv_kpoints,
                         fermi=ebs_mesh.fermi,
                         orbital_names=ebs_mesh.orbital_names,
                         reciprocal_lattice=ebs_mesh.reciprocal_lattice,
                         structure=ebs_mesh.structure,
                            **kwargs)
        
        # Transfer existing properties from the 3D mesh to the plane
        for prop_name, prop_value, gradient_order in self.ebs_mesh.iter_properties(compute=False):
            interpolator = self.ebs_mesh.get_property_interpolator(prop_name, return_gradient_order=gradient_order, **kwargs)
            prop_value = interpolator(self.uv_kpoints)
            self.add_property(prop_name, prop_value, return_gradient_order=gradient_order)

    @property
    def ebs_mesh(self):
        return self._ebs_mesh
    
    @property
    def reciprocal_lattice(self):
        return self.ebs_mesh.reciprocal_lattice

    @property
    def normal(self):
        return self._normal
    
    @property
    def origin(self):
        return self._origin
    
    @property
    def grid_interpolation(self):
        return self._grid_interpolation
    @grid_interpolation.setter
    def grid_interpolation(self, grid_interpolation: tuple[int, int]):
        self._grid_interpolation = grid_interpolation

    @property
    def u(self):
        return self._u
    
    @property
    def v(self):
        return self._v
    
    @property
    def u_limits(self):
        return self._u_limits
    
    @property
    def v_limits(self):
        return self._v_limits
    
    @property
    def u_coords(self):
        return self.uv_points[:, 0]
    
    @property
    def v_coords(self):
        return self.uv_points[:, 1]
    
    @property
    def grid_u(self):
        grid_u, grid_v = self.get_uv_grid()
        return grid_u
    
    @property
    def grid_v(self):
        grid_u, grid_v = self.get_uv_grid()
        return grid_v
    
    @property
    def transformation_matrix(self):
        return np.vstack([self.uv_transformation_matrix, self.normal])
    
    @property
    def uv_transformation_matrix(self):
        return np.vstack([self.u, self.v])
    
    @property
    def uv_points(self):
        return np.vstack([self.grid_u.ravel(), self.grid_v.ravel()]).T
    
    @property
    def uv_kpoints(self):
        return self.origin + self.uv_points @ self.uv_transformation_matrix
    
    @property
    def uv_kpoints_cartesian(self):
        return reduced_to_cartesian(self.uv_kpoints, self.reciprocal_lattice)
    
    @property
    def is_plane_axes_aligned(self):
        return is_plane_aligned_with_reciprocal_lattice(self.normal, self.reciprocal_lattice)
    
    def get_orthonormal_basis(self):
        if np.abs(np.dot(self.normal, [0, 0, 1])) < 0.99:
            v_temp = np.array([0, 0, 1])  # Not parallel to normal
        else:
            v_temp = np.array([0, 1, 0])  # Not parallel to normal
            
        u = np.cross(v_temp,self.normal).astype(np.float32)
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u).astype(np.float32)
        v /= np.linalg.norm(v)  # Ensure normalization
        
        return u, v

    def get_uv_grid(self):
        grid_u, grid_v = np.mgrid[
            self.u_limits[0] : self.u_limits[1] : complex(0, self.grid_interpolation[0]),
            self.v_limits[0] : self.v_limits[1] : complex(0, self.grid_interpolation[1]),
        ]
        return grid_u, grid_v
    
    def compute_property(self, name:str, return_gradient_order:int=0, **kwargs):
        interpolator = self.ebs_mesh.get_property_interpolator(name, return_gradient_order=return_gradient_order, **kwargs)
        property_value=interpolator(self.kpoints)
        self.add_property(name, property_value, return_gradient_order=return_gradient_order)
        return self.get_property(name, return_gradient_order=return_gradient_order, **kwargs)
            
    def transform_points_to_uv(self, points:np.ndarray, is_shifted:bool=False):
        if is_shifted:
            points_shifted = points - self.origin
        else:
            points_shifted = points
        return np.column_stack(
            [np.dot(points_shifted, self.u), np.dot(points_shifted, self.v)]
        )
    
    def _find_plane_limits(self, plane_points:np.ndarray):
        u_limits = plane_points[:, 0].min(), plane_points[:, 0].max()
        v_limits = plane_points[:, 1].min(), plane_points[:, 1].max()
        return u_limits, v_limits
    

    @classmethod
    def from_code(cls, code: str, dirpath: str, normal: np.ndarray, 
                  origin: np.ndarray = None, 
                  grid_interpolation: tuple[int, int] = None, 
                  use_cache: bool = False, 
                  ebs_filename: str = "ebs.pkl", 
                  **kwargs):
        ebs_mesh = ElectronicBandStructureMesh.from_code(code, dirpath, use_cache, ebs_filename)
        return cls(ebs_mesh=ebs_mesh, normal=normal, origin=origin, grid_interpolation=grid_interpolation, **kwargs)
    
    @staticmethod
    def find_points_near_plane(points, normal, origin, plane_tol=0.001, return_indices=False):
        if origin is None:
            # Use center of kpoint mesh as origin
            origin = np.array([0,0,0])

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Calculate distance from each kpoint to the plane
        # Distance = |(k - origin) · normal|
        points_shifted = points - origin
        distances = np.abs(np.dot(points_shifted, normal))

        # Find kpoints within tolerance of the plane
        i_points_on_plane = np.where(distances <= plane_tol)[0]
        
        if return_indices:
            return points[i_points_on_plane], i_points_on_plane
        else:
            return points[i_points_on_plane]

def ibz2fbz(ebs, rotations=None, decimals=4, inplace=True,**kwargs):
    """Applys symmetry operations to the kpoints, bands, and projections

    Parameters
    ----------
    rotations : np.ndarray
        The point symmetry operations of the lattice
    decimals : int
        The number of decimals to round the kpoints
        to when checking for uniqueness
    """
    if not inplace:
        ebs = copy.deepcopy(ebs)
        
    rotations = []
    if ebs.is_grid:
        logger.warning("ElectronicBandStructure is already in the grid, skipping ibz2fz")
        return ebs

    if len(rotations) == 0 and ebs.structure is not None:
        rotations = ebs.structure.rotations
    if len(rotations) == 0:
        logger.warning("No rotations provided, skipping ibz2fbz")
        return ebs

    n_kpoints = ebs.n_kpoints

    # Apply rotations and copy properties
    new_kpoints = ebs.kpoints.copy()
    for i, rotation in enumerate(rotations):
        start_idx = i * n_kpoints
        end_idx = start_idx + n_kpoints

        # Rotate kpoints
        new_values = ebs.kpoints.dot(rotation.T)
        new_kpoints = np.concatenate([new_kpoints, new_values], axis=0)
        # Update properties
        for prop_name, prop_value, gradient_order in ebs.iter_properties(compute=False):
            initial_array = prop_value[:n_kpoints]
            new_points = np.concatenate([prop_value, initial_array], axis=0)
            
            ebs.add_property(prop_name, new_points, return_gradient_order=gradient_order)
                

    # Apply boundary conditions to kpoints
    new_kpoints = -np.fmod(new_kpoints + 6.5, 1) + 0.5

    # Floating point error can cause the kpoints to be off by 0.000001 or so
    # causing the unique indices to misidentify the kpoints
    new_kpoints = new_kpoints.round(decimals=decimals)
    _, unique_indices = np.unique(new_kpoints, axis=0, return_index=True)
    
    new_kpoints = new_kpoints[unique_indices, ...]
    for prop_name, prop_value, gradient_order in ebs.iter_properties(compute=False):
        ebs.add_property(prop_name, prop_value[unique_indices, ...], return_gradient_order=gradient_order)
                
    ebs._kpoints = new_kpoints
    return sort_by_kpoints(ebs, inplace=inplace, **kwargs)




def sort_by_kpoints(ebs, inplace=True, order="F"):
    """Sorts the bands and projected arrays by kpoints"""
    logger.info(f"Sorting kpoints by {order}")
    if not inplace:
        ebs = copy.deepcopy(ebs)
            
    if order == "C":
        sorted_indices = np.lexsort(
            (ebs.kpoints[:, 2], ebs.kpoints[:, 1], ebs.kpoints[:, 0])
        )
    elif order == "F":
        sorted_indices = np.lexsort(
            (ebs.kpoints[:, 0], ebs.kpoints[:, 1], ebs.kpoints[:, 2])
        )
    ebs._kpoints = ebs.kpoints[sorted_indices, ...]
    
    for prop_name, prop_value, gradient_order in ebs.iter_properties(compute=False):
        ebs.add_property(prop_name, prop_value[sorted_indices, ...], return_gradient_order=gradient_order)

    return ebs
