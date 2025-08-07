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
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pyvista as pv
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    RegularGridInterpolator,
    griddata,
)
from typing_extensions import override

from pyprocar.core import kpoints
from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.core.property_store import PointSet, Property
from pyprocar.core.serializer import get_serializer
from pyprocar.core.structure import Structure
from pyprocar.utils import math, physics
from pyprocar.utils.info import orbital_index_name_map, orbital_names
from pyprocar.utils.unfolder import Unfolder

pv.global_theme.allow_empty_mesh = True

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

NUMERICAL_STABILITY_FACTOR = 0.0001


BANDS_DTYPE = np.ndarray[tuple[int,int,int], np.dtype[np.float_]]
PROJECTED_DTYPE = np.ndarray[tuple[int,int,int,int,int], np.dtype[np.float_]]
PROJECTED_PHASE_DTYPE = np.ndarray[tuple[int,int,int,int,int], np.dtype[np.float_]]
WEIGHTS_DTYPE = np.ndarray[tuple[int,int], np.dtype[np.float_]]

def get_ebs_from_data(
        kpoints: kpoints.KPOINTS_DTYPE | None = None,
        bands: BANDS_DTYPE | None = None,
        projected: PROJECTED_DTYPE | None = None,
        projected_phase: PROJECTED_PHASE_DTYPE | None = None,
        weights: WEIGHTS_DTYPE | None = None,
        fermi: float = 0.0,
        reciprocal_lattice: kpoints.RECIPROCAL_LATTICE_DTYPE | None = None,
        orbital_names: list[str] | None = None,
        structure: Structure | None = None,
        kpath: kpoints.KPath = None,
        kgrid_info: kpoints.KGridInfo | None = None,
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
    }
    
    # grid_dims = mathematics.get_grid_dims(kpoints)
    
    if kpath is not None:
        logger.debug(f"Creating ElectronicBandStructurePath from EBS")
        ebs_args["kpath"] = kpath
        return ElectronicBandStructurePath(**ebs_args)
    
    if kgrid_info is not None:
        logger.debug(f"Creating ElectronicBandStructureMesh from EBS")
        ebs_args["kgrid_info"] = kgrid_info
        return ElectronicBandStructureMesh(**ebs_args)
    
    
    else:
        logger.debug(f"Creating ElectronicBandStructure from EBS")
        return ElectronicBandStructure(**ebs_args)
    
def get_ebs_from_code(
    code: str,
    dirpath: str,
    use_cache: bool = False,
    ebs_filename: str = "ebs.pkl",
    **kwargs):

    from pyprocar.io import Parser
    ebs_filepath = Path(dirpath) / ebs_filename
    
    if not use_cache or not ebs_filepath.exists():
        logger.info(f"Parsing EBS calculation directory: {dirpath}")
        parser = Parser(code=code, dirpath=dirpath)
        ebs=parser.ebs
        ebs.save(ebs_filepath)
    else:
        logger.info(f"Loading EBS  from picklefile: {ebs_filepath}")
        ebs = ElectronicBandStructure.load(ebs_filepath)
    
    return ebs


class DifferentiablePropertyInterface(ABC):
    
    @abstractmethod
    def gradient_func(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def get_property(self, key: str, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def add_property(self, label: str, value: np.ndarray, **kwargs):
        raise NotImplementedError
    
    def compute_band_velocity(self, **kwargs):
        logger.info("Computing band velocity")
        bands_gradient = self.get_property(("bands","gradients",1))
        return physics.calculate_band_velocity(bands_gradient)
    
    def compute_band_speed(self, **kwargs):
        logger.info("Computing band speed")
        band_velocity = self.compute_band_velocity(**kwargs)
        return physics.calculate_band_speed(band_velocity)
    
    def compute_avg_inv_effective_mass(self, **kwargs):
        logger.info("Computing average inverse effective mass")
        bands_hessian = self.get_property(("bands","gradients",2))
        return physics.calculate_avg_inv_effective_mass(bands_hessian)
        
class PyvistaInterface(ABC):
    _mesh: pv.PolyData | pv.StructuredGrid | pv.PointSet
    
    def mesh(self):
        if self._mesh is None:
            self.to_mesh()
        return self._mesh
    
    def mesh_as_cart(self):
        self._mesh.points = self.kpoints_cartesian
    
    def mesh_as_frac(self):
        self._mesh.points = self.kpoints
    
    @abstractmethod
    def to_mesh(self, 
                active_scalar:tuple[str, np.ndarray] | None = None, 
                active_vector:tuple[str, np.ndarray] | None = None, 
                **kwargs):
        raise NotImplementedError
    
    def set_mesh_scalar(self, name:str, scalar:np.ndarray):
        self._mesh.point_data[name] = scalar
        self._mesh.set_active_scalars(name)
    
    def set_mesh_vector(self, name:str, vector:np.ndarray):
        self._mesh.point_data[name] = vector
        self._mesh.set_active_vectors(name)
    
class ElectronicBandStructure(PointSet, PyvistaInterface):
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
        kpoints: kpoints.KPOINTS_DTYPE | None = None,
        fermi: float = 0.0,
        bands: BANDS_DTYPE | None = None,
        projected: PROJECTED_DTYPE | None = None,
        projected_phase: PROJECTED_PHASE_DTYPE | None = None,
        weights: WEIGHTS_DTYPE | None = None,
        orbital_names: list[str] | None = None,
        reciprocal_lattice: kpoints.RECIPROCAL_LATTICE_DTYPE | None = None,
        shifted_to_fermi: bool = False,
        structure: Structure | None = None,
    ):
        super().__init__(kpoints)

        if bands is not None:
            self.add_property(name="bands", value=bands)
        if projected is not None:
            self.add_property(name="projected", value=projected)
        if projected_phase is not None:
            self.add_property(name="projected_phase", value=projected_phase)
        if weights is not None:
            self.add_property(name="weights", value=weights)
        
        self._fermi = fermi
        self._orbital_names = orbital_names
        self._reciprocal_lattice = reciprocal_lattice
        self._shifted_to_fermi = shifted_to_fermi
        self._structure = structure

    def __str__(self):
        ret = "\n Electronic Band Structure     \n"
        ret += "============================\n"
        ret += "Total number of kpoints   = {}\n".format(self.n_kpoints)
  
        ret += "\nAdditional information: \n"
        if self.orbital_names is not None:
            ret += "Orbital Names = {}\n".format(self.orbital_names)
            
        if "projected" in self.property_store:
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
        kpoints_equal = math.compare_arrays(self.kpoints, other.kpoints)
        bands_equal = math.compare_arrays(self.bands, other.bands)
        fermi_equal = self.fermi == other.fermi
        projected_equal = math.compare_arrays(self.projected, other.projected)
        projected_phase_equal = math.compare_arrays(self.projected_phase, other.projected_phase)
        weights_equal = math.compare_arrays(self.weights, other.weights)
        
        is_ebs_equal = (kpoints_equal and bands_equal and fermi_equal and 
                        projected_equal and projected_phase_equal and weights_equal)
        return is_ebs_equal

    @property
    def kpoints(self):
        return self._points

    @property
    def bands(self):
        prop = self.get_property("bands")
        if prop is None:
            return None
        return prop.value

    @property
    def projected(self):
        prop = self.get_property("projected")
        if prop is None:
            return None
        return prop.value

    @property
    def projected_phase(self):
        prop = self.get_property("projected_phase")
        if prop is None:
            return None
        return prop.value

    @property
    def weights(self):
        prop = self.get_property("weights")
        if prop is None:
            return None
        return prop.value

    @property
    def orbital_names(self):
        return self._orbital_names

    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice
    
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
        return kpoints.reduced_to_cartesian(self.kpoints, self.reciprocal_lattice)

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
        grid_dims = math.get_grid_dims(self.kpoints)
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
    
    def has_spin_channels(self, property: Property):
        property_value_shape = list(property.value.shape)
        if len(property_value_shape) >= 3:
            nspins = property_value_shape[2]
            return nspins == self.n_spin_channels
        else:
            return False
    
    def is_band_property(self, property: Property | np.ndarray):
        if isinstance(property, np.ndarray):
            property_value_shape = list(property.shape)
        else:
            property_value_shape = list(property.value.shape)
        if len(property_value_shape) >= 3:
            nbands = property_value_shape[1]
            return nbands == self.n_bands
        else:
            return False

    def is_orbital_property(self, property: Property):
        property_value_shape = list(property.value.shape)
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
        for property_name in self.property_store.keys():
            property = self.get_property(property_name)
            if self.is_band_property(property):
                names.append(property_name)
        return names

    @property
    def property_names(self):
        names = []
        for property_name in self.property_store.keys():
            names.append(property_name)
        return names
    
    @property
    def ebs_ipr(self):
        prop = self.get_property("ebs_ipr")
        if prop is not None:
            return prop.value
        
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
            return prop.value
        
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
            return prop.value
        
        projected_sum = self.compute_projected_sum()
        return projected_sum
    
    @property
    def projected_sum_spin_texture(self):
        prop = self.get_property("projected_sum_spin_texture")
        if prop is not None:
            return prop
        
        projected_sum_spin_texture = self.compute_projected_sum_spin_texture()
        return projected_sum_spin_texture
    
    def to_mesh(self, as_cartesian:bool = True):
        if as_cartesian:
            point_set = pv.PointSet(self.kpoints_cartesian)
        else:
            point_set = pv.PointSet(self.kpoints)
        return point_set
    
    @override
    def get_property(self, key = None, **kwargs):
        prop_name, (calc_name, gradient_order) = self._extract_key(key)
        if prop_name not in self.property_store:
            prop_value = self.compute_property(prop_name, **kwargs)
            if prop_value is not None:
                self.add_property(name=prop_name, value=prop_value)
        return super().get_property(key)
    
    def to_mesh(self, 
                scalars:tuple[str, np.ndarray] | None = None, 
                vectors:tuple[str, np.ndarray] | None = None,
                as_cartesian:bool = True):
        if as_cartesian:
            mesh_points = self.kpoints_cartesian
        else:
            mesh_points = self.kpoints
        mesh = pv.PointSet(mesh_points)
        if scalars is not None:
            self.set_mesh_scalar(*scalars)
        if vectors is not None:
            self.set_mesh_vector(*vectors)
        self._mesh = mesh
        return mesh
        
    def compute_property(self, name: str, **kwargs):
        if name == "ebs_ipr":
            return self.compute_ebs_ipr(**kwargs)
        elif name == "ebs_ipr_atom":
            return self.compute_ebs_ipr_atom(**kwargs)
        elif name == "spin_texture":
            return self.compute_spin_texture(**kwargs)
        elif name == "projected_sum":
            return self.compute_projected_sum(**kwargs)
        elif name == "projected_sum_spin_texture":
            return self.compute_projected_sum_spin_texture(**kwargs)
    
    def compute_ebs_ipr(self, **kwargs):
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
        return IPR
    
    def compute_ebs_ipr_atom(self, **kwargs):
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
        return pIPR
    
    def compute_projected_sum(self, 
                              atoms: list[int] = None, 
                              orbitals: list[int] = None, 
                              spins: list[int] = None):  
        projected_sum = self.ebs_sum(
            atoms=atoms, orbitals=orbitals, spins=spins, sum_noncolinear=True
        )

        return projected_sum
    
    def compute_spin_texture(self, **kwargs):
        if not self.is_non_collinear:
            raise ValueError(
                "Spin texture is only available for non-collinear calculations"
            )
        
        spin_texture = self.projected[:, :, 1:, :, :]
        spin_texture = np.moveaxis(spin_texture, 2, -1)
        return spin_texture

    def compute_projected_sum_spin_texture(self, 
                                           atoms: list[int] = None, 
                                           orbitals: list[int] = None, 
                                           **kwargs):
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

        projected_spin_texture = summed_projection[..., 1:]
        temp_shape = list(projected_spin_texture.shape)
        temp_shape.insert(2, 1)
        projected_spin_texture = projected_spin_texture.reshape(
            temp_shape, order=kwargs.pop("order", "F")
        )
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
            
        if self.is_spin_polarized:
            # Zero out the spin channel that is not specified
            if np.allclose(np.asarray(spins), np.array([0])):
                ret[..., 1] = 0
            elif np.allclose(np.asarray(spins), np.array([1])):
                ret[..., 0] = 0


        return ret
    
    def iter_properties(self):
        for prop_name, calc_name, gradient_order, value_array in self.iter_property_arrays():
            yield prop_name, calc_name, gradient_order, value_array
   
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
        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            property = ebs.get_property(prop_name)
            if prop_name in band_property_names:
                property[calc_name, gradient_order] = value_array[:, full_band_index, ...]

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
        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            property = ebs.get_property(prop_name)
            if prop_name in band_property_names:
                property[calc_name, gradient_order] = value_array[:, bands, ...]
                
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
        
        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            property = ebs.get_property(prop_name)
            if prop_name in band_property_names and ebs.has_spin_channels(property):
                original_value_shape = list(value_array.shape)
                band_dim = original_value_shape[1]
                original_value_shape[1] = 2 * band_dim
                original_value_shape[2] = 1
                modified_array = value_array.reshape(original_value_shape)
                property[calc_name, gradient_order] = modified_array

        return ebs
        
    def shift_bands(self, shift_value, inplace=False):
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
            
        bands = ebs.get_property("bands").value
        bands += shift_value
        ebs.add_property(name="bands", value=bands)
        return ebs
        
    def shift_kpoints_to_fbz(self, inplace=True):
        # Shifting all kpoint to first Brillouin zone
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
            
        # Old method
        # bound_ops = -1.0 * (ebs.kpoints > 0.5) + 1.0 * (ebs.kpoints <= -0.5)
        # new_kpoints = ebs.kpoints + bound_ops
        
        new_kpoints = -np.fmod(ebs.kpoints + 6.5, 1) + 0.5
        ebs.update_points(new_kpoints)
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
        
        ebs.add_property(name="weights", value=uf.weights)
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
        return get_ebs_from_code(code, dirpath, use_cache, ebs_filename)
    
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

class ElectronicBandStructurePath(ElectronicBandStructure, DifferentiablePropertyInterface, PyvistaInterface):

    def __init__(self, kpath: kpoints.KPath, **kwargs):
        super().__init__(**kwargs)
        self._kpath = kpath
        self.as_cart()
        
    @classmethod
    def from_code(cls, code: str, dirpath: str, use_cache: bool = False, ebs_filename: str = "ebs.pkl"):
        return get_ebs_from_code(code, dirpath, use_cache, ebs_filename)
        
    def __str__(self):
        ret = super().__str__()
        ret += "\nKPath: \n"
        ret += "------------------------     \n"
        ret += "KPath = \n {}\n".format(self.kpath)
        return ret

    def as_cart(self):
        self.transform_points(self.reciprocal_lattice)

    def as_frac(self):
        self.transform_points(np.linalg.inv(self.reciprocal_lattice))

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
    
    def to_mesh(self, 
                scalars:tuple[str, np.ndarray] | None = None, 
                vectors:tuple[str, np.ndarray] | None = None,
                as_cartesian:bool = True,
                **kwargs):
        if as_cartesian:
            mesh_points = self.kpoints_cartesian
        else:
            mesh_points = self.kpoints
        mesh = pv.PointSet(mesh_points)
        if scalars is not None:
            self.set_mesh_scalar(*scalars)
        if vectors is not None:
            self.set_mesh_vector(*vectors)
        self._mesh = mesh
        return mesh
    
    def gradient_func(self, points:npt.NDArray[np.float64], 
                      values:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        continuous_segments = self.kpath.get_continuous_segments()

        gradients = np.zeros(values.shape)
        for k_indices in continuous_segments:
            kpath_segment = points[k_indices]
            delta_k = np.gradient(kpath_segment, axis=0)
            delta_k = np.linalg.norm(delta_k, axis=1)
            
            # Determine distance coordinates for gradient calculation. 
            # cumsum does not include 0 and includes an unnecessary point at the end.
            distance_coordinates = np.cumsum(delta_k, axis=0)[:-1]
            distance_coordinates = np.insert(distance_coordinates, 0, 0, axis=0)
            gradients[k_indices, ...] = np.gradient(
                values[k_indices, ...],
                distance_coordinates,
                axis=0,
                edge_order=2,
            )
        gradients = gradients * physics.METER_ANGSTROM
        return gradients
    
    def compute_property(self, name:str, **kwargs):
        if name == "bands_velocity":
            return self.compute_band_velocity(**kwargs)
        elif name == "bands_speed":
            return self.compute_band_speed(**kwargs)
        elif name == "avg_inv_effective_mass":
            return self.compute_avg_inv_effective_mass(**kwargs)
        else:
            return super().compute_property(name, **kwargs)

    # ------------------------------------------------------------------
    # Overlay weight builders
    # ------------------------------------------------------------------
    def build_overlay_species_weights(
        self,
        spins: list[int] | None = None,
        orbitals: list[int] | None = None,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Build per-species overlay weights for plot overlays.

        For each species present in the structure, this computes the summed
        orbital projections over all atoms of that species and returns a list
        of weight arrays and matching labels.

        Parameters
        ----
        spins : list of int, optional
            Spin channels to include in the sum. Default includes all spins.
        orbitals : list of int, optional
            Orbital indices to include. Defaults to all orbitals.

        Returns
        ----
        (weights, labels) : (list of ndarray, list of str)
            Weights have shape compatible with bands: (n_k, n_bands, n_spins).
        """
        if self.structure is None:
            raise ValueError("Structure is required to build species weights")

        if orbitals is None:
            orbitals = list(np.arange(self.n_orbitals, dtype=int))

        weights_list: list[np.ndarray] = []
        labels_list: list[str] = []

        # Prefer explicit species list if available; otherwise derive from atoms
        species_iterable = getattr(self.structure, "species", None)
        if species_iterable is None:
            species_iterable = np.unique(self.structure.atoms)

        for species_name in species_iterable:
            atom_indices = np.where(self.structure.atoms == species_name)[0]
            if atom_indices.size == 0:
                continue
            summed = self.ebs_sum(atoms=atom_indices.tolist(), orbitals=orbitals, spins=spins)
            weights_list.append(summed)
            labels_list.append(str(species_name))

        return weights_list, labels_list

    def build_overlay_orbitals_weights(
        self,
        atoms: list[int] | None = None,
        spins: list[int] | None = None,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Build per-orbital-group overlay weights for plot overlays.

        Iterates over orbital groups (s, p, d, f when available) and computes
        weights for each group, optionally restricted to given atoms.

        Parameters
        ----
        atoms : list of int, optional
            Atom indices to include. Defaults to all atoms.
        spins : list of int, optional
            Spin channels to include. Default includes all spins.

        Returns
        ----
        (weights, labels) : (list of ndarray, list of str)
            Weights have shape compatible with bands: (n_k, n_bands, n_spins).
        """
        if atoms is not None and len(atoms) == 0:
            atoms = None

        weights_list: list[np.ndarray] = []
        labels_list: list[str] = []

        for orb_name in ["s", "p", "d", "f"]:
            if orb_name == "f" and not (self.n_orbitals > 9):
                continue
            orb_indices = orbital_names[orb_name]
            summed = self.ebs_sum(atoms=atoms, orbitals=orb_indices, spins=spins)
            weights_list.append(summed)
            labels_list.append(orb_name)

        return weights_list, labels_list

    def build_overlay_weights(
        self,
        items: dict | list[dict],
        spins: list[int] | None = None,
        orbitals_as_names: bool = False,
        mathtext: bool = True,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Build overlay weights from a mapping of species to orbital sets.

        The input mirrors the previous script interface where `items` is either
        a dict mapping species strings to lists of orbitals (names like 's','p',
        or integer orbital indices), or a list of such dicts to build multiple
        overlay series.

        Parameters
        ----
        items : dict or list of dict
            Mapping like { 'Fe': ['d'], 'O': ['p'] } or { 'Fe': [4,5,6] }.
        spins : list of int, optional
            Spin channels to include. Default includes all spins.
        orbitals_as_names : bool, optional
            If True, label orbitals by their names using the mapping in
            `pyprocar.utils.info.orbital_index_name_map`. If False, use indices.
            Default is False.
        mathtext : bool, optional
            When `orbitals_as_names` is True, wrap labels with `$...$` so
            matplotlib renders subscripts/superscripts correctly. Default True.

        Returns
        ----
        (weights, labels) : (list of ndarray, list of str)
            Weights have shape compatible with bands: (n_k, n_bands, n_spins).
        """
        if self.structure is None:
            raise ValueError("Structure is required to build overlay weights from items")

        if isinstance(items, dict):
            items_iter = [items]
        else:
            items_iter = items

        weights_list: list[np.ndarray] = []
        labels_list: list[str] = []

        for mapping in items_iter:
            for species_name, orbital_spec in mapping.items():
                atom_indices = np.where(self.structure.atoms == species_name)[0]
                if atom_indices.size == 0:
                    continue

                # Resolve orbital indices from names or pass integers directly
                if len(orbital_spec) > 0 and isinstance(orbital_spec[0], str):
                    resolved: list[int] = []
                    for orb_token in orbital_spec:
                        resolved = np.append(resolved, orbital_names[orb_token]).astype(int).tolist()
                    orbitals = resolved
                    if orbitals_as_names:
                        # Use group name string directly
                        label_suffix = "".join(orbital_spec)
                    else:
                        label_suffix = "".join(orbital_spec)
                else:
                    orbitals = [int(x) for x in orbital_spec]
                    if orbitals_as_names:
                        pretty_parts: list[str] = []
                        for idx in orbitals:
                            name = orbital_index_name_map.get(int(idx), str(idx))
                            if mathtext:
                                name = f"${name}$"
                            pretty_parts.append(name)
                        label_suffix = ",".join(pretty_parts)
                    else:
                        label_suffix = ",".join(str(x) for x in orbital_spec)

                label = f"{species_name}-({label_suffix})"
                summed = self.ebs_sum(atoms=atom_indices.tolist(), orbitals=orbitals, spins=spins)
                weights_list.append(summed)
                labels_list.append(label)

        return weights_list, labels_list
    
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
            
            
class ElectronicBandStructureMesh(ElectronicBandStructure, DifferentiablePropertyInterface, PyvistaInterface):
    def __init__(self, kgrid_info: kpoints.KGridInfo, **kwargs):
        super(ElectronicBandStructureMesh, self).__init__(**kwargs)
        
        self._kgrid_info = kgrid_info
   

        if self.n_kpoints != np.prod(self.kgrid_info.kgrid):
            ibz2fbz(self, rotations=self.structure.rotations, kgrid_info=self.kgrid_info, decimals=4, inplace=True)
        
        sort_by_kpoints(self, inplace=True)
            
        if self.n_kpoints != np.prod(self.kgrid_info.kgrid):
            raise ValueError("n_kpoints must be equal to np.prod(kgrid) (number of kpoints)")
        
    @classmethod
    def from_code(cls, code: str, dirpath: str, use_cache: bool = False, ebs_filename: str = "ebs.pkl"):
        ebs = get_ebs_from_code(code, dirpath, use_cache, ebs_filename)
        return ebs

    def __str__(self):
        ret = super().__str__()
        ret += "\nKGrid: \n"
        ret += "------------------------     \n"
        ret += "(nkx, nky, nkz) = \n {}\n".format(self.kgrid)
        return ret
    
    @property
    def kgrid_info(self):
        return self._kgrid_info
    
    @property
    def kgrid(self):
        return self.get_kgrid()
    
    def get_kgrid(self, num_bins:int=1000, height:float=1, coord_tol:float=0.01):
        return math.get_grid_dims(self.kpoints, num_bins=num_bins, height=height, coord_tol=coord_tol)
    
    @property
    def kshift(self):
        return self.kgrid_info.kshift
    
    @property
    def kgrid_mode(self):
        return self.kgrid_info.kgrid_mode
    
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
    def is2d(self):
        unique_coords = [self.ukx, self.uky, self.ukz]
        
        is_2d = False
        for dim_size in unique_coords:
            if len(dim_size) == 1:
                is_2d = True
                break
        
        return is_2d


    
    def to_mesh(self, 
                scalars:tuple[str, np.ndarray] | None = None, 
                vectors:tuple[str, np.ndarray] | None = None, 
                as_cartesian:bool = True):
        """Explicitly returns a new PyVista StructuredGrid."""
        # This can be the same logic as the `grid` property,
        # but without caching if a fresh object is desired.
        grid = pv.StructuredGrid()
        if as_cartesian:
            grid.points = self.kpoints_cartesian
        else:
            grid.points = self.kpoints
        grid.dimensions = self.kgrid
        self._mesh = grid
        if scalars is not None:
            self.set_mesh_scalar(*scalars)
        if vectors is not None:
            self.set_mesh_vector(*vectors)
        return grid
    
    def get_kbounds(self):
        kbounds = np.zeros((3, 2))
        for icoord in range(3):
            coords = self.kpoints[:, icoord]
            kx_min, kx_max = np.min(coords), np.max(coords)
            kbounds[icoord, 0] = kx_min
            kbounds[icoord, 1] = kx_max
        return kbounds
        

    def get_kpoints_mesh(self, **kwargs):
        return math.array_to_mesh(
            array=self.kpoints,
            nkx=self.n_kx,
            nky=self.n_ky,
            nkz=self.n_kz,
            **kwargs,
        )
        
    def get_property_mesh(self, key , **kwargs):
        property = self.get_property(key, **kwargs)
        if property is None:
            return None
        property_mesh = math.array_to_mesh( 
                array=property.value,
                nkx=self.n_kx,
                nky=self.n_ky,
                nkz=self.n_kz,
                **kwargs,
            )
        
        logger.debug(f"Property mesh shape: {property.value.shape}")
        logger.debug(f"Nkx: {self.n_kx}, Nky: {self.n_ky}, Nkz: {self.n_kz}")
        logger.debug(f"Property mesh shape: {property_mesh.shape}")
        return property_mesh

    def compute_property(self, name:str, **kwargs):
        if name == "bands_velocity":
            return self.compute_band_velocity(**kwargs)
        elif name == "bands_speed":
            return self.compute_band_speed(**kwargs)
        elif name == "avg_inv_effective_mass":
            return self.compute_avg_inv_effective_mass(**kwargs)
        else:
            return super().compute_property(name, **kwargs)
    
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
        kpoints_mesh = ebs.get_kpoints_mesh()
        padded_kpoints_mesh = np.pad(kpoints_mesh, kpoints_padding_dims, mode=edge_diff_ramp)
        logger.debug(f"Padded kpoints mesh shape: {padded_kpoints_mesh.shape}")

        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            property = ebs.get_property(prop_name)
            value_mesh = math.array_to_mesh(array=value_array, nkx=ebs.n_kx, nky=ebs.n_ky, nkz=ebs.n_kz)
            n_scalar_dims = len(value_mesh.shape[3:])
            scalar_padding_dims = copy.deepcopy(padding_dims)
            for i in range(n_scalar_dims):
                scalar_padding_dims.append((0, 0))
            padded_mesh = np.pad(value_mesh, scalar_padding_dims, mode='wrap')
            logger.debug(f"Padded {prop_name} mesh shape: {padded_mesh.shape}")
            padded_array = math.mesh_to_array(
                padded_mesh, order=order
            )
            
            property[calc_name, gradient_order] = padded_array
                
        new_kpoints = math.mesh_to_array(padded_kpoints_mesh, order=order)
        ebs.update_points(new_kpoints)
        ebs._mesh = ebs.to_mesh()
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
            for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
                property = ebs.get_property(prop_name)
                initial_array = value_array[:n_init_points]
                new_points = np.concatenate([value_array, initial_array], axis=0)
                
                ebs.add_property(prop_name, new_points, return_gradient_order=gradient_order)
                    
        ebs.update_points(new_kpoints)
        ebs._mesh = ebs.to_mesh()
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
        kpoints_mesh = ebs.get_kpoints_mesh()
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

        
        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            property = ebs.get_property(prop_name)
            value_mesh = math.array_to_mesh(array=value_array, nkx=ebs.n_kx, nky=ebs.n_ky, nkz=ebs.n_kz)
            interpolated_mesh = math.fft_interpolate_nd_3dmesh(value_mesh, interpolation_factor)
            interpolated_value = math.mesh_to_array(interpolated_mesh)
            property[calc_name, gradient_order] = interpolated_value
             
        ebs.update_points(new_kpoints)
        ebs._mesh = ebs.to_pyvista_grid()
        return ebs
    
    def expand_single_dimension(self, inplace=False, fill_tol=0.1):
        if inplace:
            ebs = self
        else:
          ebs = copy.deepcopy(self)
        
       
        if not ebs.is2d:
            return ebs
        
        unique_coords = [ebs.ukx, ebs.uky, ebs.ukz]
        
        # Expand the kpoints and properties
        new_kpoints = ebs.kpoints.copy()
        for idim, dim_size in enumerate(unique_coords):
            dim_size = len(dim_size)
            if dim_size==1:
                points_plus_fill = ebs.kpoints.copy()
                points_minus_fill = ebs.kpoints.copy()
                points_plus_fill[:, idim] += fill_tol
                points_minus_fill[:, idim] -= fill_tol
                
                new_kpoints = np.concatenate([new_kpoints, points_plus_fill, points_minus_fill], axis=0)
                
                for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
                    property = ebs.get_property(prop_name)
                    initial_array = value_array[:ebs.n_kpoints]
                    new_points = np.concatenate([value_array, initial_array, initial_array], axis=0)
                    
                    property[calc_name, gradient_order] = new_points
                    
        ebs.update_points(new_kpoints)
        sort_by_kpoints(ebs, inplace=True)
        ebs._mesh = ebs.to_mesh()
        return ebs
    
    def reduce_kpoints_to_plane(self, k_z_plane, k_z_plane_tol, inplace=True):
        """
        Reduces the kpoints to a plane
        """
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)
        

        i_kpoints_near_z_0 = np.where(
            np.logical_and(
                ebs.kpoints_cartesian[:, 2] < k_z_plane + k_z_plane_tol,
                ebs.kpoints_cartesian[:, 2] > k_z_plane - k_z_plane_tol,
            )
        )

        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            property = ebs.get_property(prop_name)
            initial_array = value_array[:ebs.n_kpoints]
            new_points = initial_array[i_kpoints_near_z_0, ...][0]
            property[calc_name, gradient_order] = new_points
            
        ebs.update_points(ebs.kpoints[i_kpoints_near_z_0, ...])
        ebs._mesh = ebs.to_mesh()
        return None
    
    def slice(self, normal=(0, 0, 1), origin=(0, 0, 0), 
              scalars:tuple[str, np.ndarray] | None = None,
              vectors:tuple[str, np.ndarray] | None = None,
              as_cartesian:bool = True,
              **kwargs):
        
        mesh = self.to_mesh(scalars=scalars, vectors=vectors, as_cartesian=as_cartesian)
        slice = mesh.slice(normal=normal, origin=origin, **kwargs)
        
        if not as_cartesian:
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = self.reciprocal_lattice.T
            slice = slice.transform(transform_matrix, inplace=False)
            
        n_slice_points = len(slice.points)
        if n_slice_points == 0:
            error_message = "No points found in the slice."
            error_message += "Look at the coordinates in cartesian or fractional space to see if origin is right."
            raise ValueError(error_message)
        if scalars is not None:
            scalar_name, scalar_values = scalars
            slice.set_active_scalars(scalar_name)
        if vectors is not None:
            vector_name, vector_values = vectors
            slice.set_active_vectors(vector_name)
        return slice
    
    def gradient_func(self, points, values, **kwargs):
        val_mesh = math.array_to_mesh(
            array=values,
            nkx=self.n_kx,
            nky=self.n_ky,
            nkz=self.n_kz,
            **kwargs,
        )
        gradients_mesh = math.calculate_3d_mesh_scalar_gradients(
                val_mesh, self.reciprocal_lattice
            )
        gradients_mesh *= physics.METER_ANGSTROM

        gradients=math.mesh_to_array(
            mesh=gradients_mesh, **kwargs
        )
        
        return gradients


def ibz2fbz(ebs, rotations=None, kgrid_info=None, decimals=4, inplace=True,**kwargs):
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
    logger.info(f"Applying symmetry operations to the kpoints")
    logger.info(f"Kgrid info: {kgrid_info}")
    

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
        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            initial_array = value_array[:n_kpoints]
            new_points = np.concatenate([value_array, initial_array], axis=0)
            property = ebs.get_property(prop_name)
            property[calc_name, gradient_order] = new_points
        
    # Apply boundary conditions to kpoints
    new_kpoints = -np.fmod(new_kpoints + 6.5, 1) + 0.5
    
    # 
    if kgrid_info is not None:
        kpoints_grid_points = kpoints.get_kpoints_from_kgrid(
            kgrid=kgrid_info.kgrid, kshift=kgrid_info.kshift, mode=kgrid_info.kgrid_mode)

    
        diff = new_kpoints[:, np.newaxis, :] - kpoints_grid_points[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        # Find the minimum distance for each k-point in new_kpoints to any
        # k-point in kpoints_grid_points
        min_distances = np.min(distances, axis=1)

        # Get the indices in new_kpoints where the minimum distance is within the tolerance
        new_in_original_grid_indices = np.where(min_distances < 0.000001)[0]
        
        new_kpoints = new_kpoints[new_in_original_grid_indices, ...]
   
    # # Floating point error can cause the kpoints to be off by 0.000001 or so
    # # causing the unique indices to misidentify the kpoints
    new_kpoints = new_kpoints.round(decimals=3)
    _, unique_indices = np.unique(new_kpoints, axis=0, return_index=True)
    
    new_kpoints = new_kpoints[unique_indices, ...]

    for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
        property = ebs.get_property(prop_name)
        property[calc_name, gradient_order] = value_array[new_in_original_grid_indices][unique_indices]
                
    ebs.update_points(new_kpoints)
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

    ebs.update_points(ebs.kpoints[sorted_indices, ...])
        
    for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
        property = ebs.get_property(prop_name)
        property[calc_name, gradient_order] = value_array[sorted_indices, ...]

    return ebs





