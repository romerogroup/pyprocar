__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import copy
import logging
import math
import random
import sys
from itertools import product
from typing import Dict, List, Tuple, Union

import numpy as np
import pyvista as pv
import scipy.interpolate as interpolate
from matplotlib import cm
from matplotlib import colors as mpcolors
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree

from pyprocar.core.brillouin_zone import BrillouinZone2D
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.property_store import PointSet, Property

np.set_printoptions(threshold=sys.maxsize)

logger = logging.getLogger(__name__)

# TODO: method to reduce number of points for interpolation need to be modified since the tolerance on the space
# is not lonmg soley reciprocal space, but energy and reciprocal space


def get_orthonormal_basis(normal):
    if np.abs(np.dot(normal, [0, 0, 1])) < 0.99:
        v_temp = np.array([0, 0, 1])  # Not parallel to normal
    else:
        v_temp = np.array([0, 1, 0])  # Not parallel to normal
        
    u = np.cross(v_temp,normal).astype(np.float32)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u).astype(np.float32)
    v /= np.linalg.norm(v)  # Ensure normalization
    
    return u, v

def transform_points_to_uv(
                           points:np.ndarray, 
                           u: np.ndarray, 
                           v: np.ndarray, 
                           origin:np.ndarray = np.array([0, 0, 0]),
                           ):
    points_shifted = points - origin
    return np.column_stack(
        [np.dot(points_shifted, u), np.dot(points_shifted, v)]
    )

def find_plane_limits(plane_points:np.ndarray):
    u_limits = plane_points[:, 0].min(), plane_points[:, 0].max()
    v_limits = plane_points[:, 1].min(), plane_points[:, 1].max()
    return u_limits, v_limits

def get_uv_grid(
    grid_interpolation:tuple[int, int],
    u_limits:tuple[float, float],
    v_limits:tuple[float, float],
    ):
    grid_u, grid_v = np.mgrid[
        u_limits[0] : u_limits[1] : complex(0, grid_interpolation[0]),
        v_limits[0] : v_limits[1] : complex(0, grid_interpolation[1]),
    ]
    return grid_u, grid_v

def get_uv_grid_points(grid_u:np.ndarray, grid_v:np.ndarray):
    return np.vstack([grid_u.ravel(), grid_v.ravel()]).T

def get_uv_grid_kpoints(origin:np.ndarray, uv_points:np.ndarray, uv_transformation_matrix:np.ndarray = None, u:np.ndarray = None, v:np.ndarray = None, normal:np.ndarray = None):
    if uv_transformation_matrix is None:
        uv_transformation_matrix = get_uv_transformation_matrix(u, v)
    return origin + uv_points @ uv_transformation_matrix

def get_uv_transformation_matrix(u:np.ndarray, v:np.ndarray):
    return np.vstack([u, v])

def get_transformation_matrix(u:np.ndarray, v:np.ndarray, normal:np.ndarray):
    uv_transformation_matrix = get_uv_transformation_matrix(u, v)
    transformation_matrix = np.vstack([uv_transformation_matrix, normal])
    return transformation_matrix


class BandStructure2D(pv.PolyData):
    """
    The object is used to store and manapulate a 3d fermi surface.

    Parameters
    ----------
    ebs : ElectronicBandStructure
        The ElectronicBandStructure object
    interpolation_factor : int
        The default is 1. number of kpoints in every direction
        will increase by this factor.
    projection_accuracy : str, optional
        Controls the accuracy of the projects. 2 types ('high', normal)
        The default is ``projection_accuracy=normal``.
    supercell : list int
        This is used to add padding to the array
        to assist in the calculation of the isosurface.
    """

    def __init__(self, ebs: ElectronicBandStructureMesh,
                 normal=(0, 0, 1), 
                 origin=(0, 0, 0),
                 grid_interpolation=(20, 20),
                 as_cartesian=True,
                 padding=15,
                 **kwargs):
        super().__init__()
        self._original_ebs = copy.copy(ebs)
        self._ebs = ebs.pad(padding=padding, inplace=False)
        self._ebs = self._ebs.expand_single_dimension(inplace=False)

        
        self.as_cartesian = as_cartesian

        slice = self.ebs.slice(normal=normal, origin=origin, as_cartesian=self.as_cartesian)
        
        #  # Initialize storage for surface data
        self.band_spin_surface_map = {}
        self.surface_band_spin_map = {}
        self.band_spin_mask = {}
        
        self.transform_to_cart = np.eye(4)
        self.transform_to_frac = np.eye(4)
        self.transform_to_cart[:3, :3] = self.ebs.reciprocal_lattice.T
        self.transform_to_frac[:3, :3] = np.linalg.inv(
            self.ebs.reciprocal_lattice.T
        )
        
        self.normal = normal
        self.origin = origin
        self.grid_interpolation = grid_interpolation
        self.u, self.v = get_orthonormal_basis(normal=normal)
        self.plane_points = transform_points_to_uv(slice.points, self.u, self.v)
        
        
        u_limits, v_limits = find_plane_limits(self.plane_points)

        self.u_grid, self.v_grid = get_uv_grid(grid_interpolation=grid_interpolation,
                                    u_limits=u_limits,
                                    v_limits=v_limits)
        
        self.uv_grid_points = get_uv_grid_points(self.u_grid, self.v_grid)
        
        self.n_grid_points = self.uv_grid_points.shape[0]
        
        logger.debug(f"Normal: {self.normal}")
        logger.debug(f"Origin: {self.origin}")
        logger.debug(f"Grid Interpolation: {self.grid_interpolation}")
        logger.debug(f"U: {self.u}")
        logger.debug(f"V: {self.v}")
        logger.debug(f"U Limits: {u_limits}")
        logger.debug(f"V Limits: {v_limits}")
        logger.debug(f"N_grid_points: {self.n_grid_points}")
        
        self._generate_band_surfaces()
        
        self.point_set = PointSet(self.n_grid_points)
        self.scale_kplane()
        return None

    @property
    def ebs(self):
        return self._ebs
    
    @property
    def original_ebs(self):
        return self._original_ebs
    
    def get_property(self, key, **kwargs):
        prop_name, (calc_name, gradient_order) = self.ebs._extract_key(key)
        
        if prop_name not in self.point_set.property_store:
            property_value = self.compute_property(prop_name, **kwargs)
            property = Property(name=prop_name, value=property_value)
            self.point_set.add_property(property)
            
        self.set_values(prop_name, property_value)
        return property_value
    
    def compute_gradients(self, gradient_order:int, names:list[str] | None = None, **kwargs) -> None:
        if names is None:
            names = list(self.point_set._property_store.keys())
        if gradient_order < 0:
            raise ValueError(f"Gradient order must be greater than 0. Got {gradient_order}.")
        self.ebs.compute_gradients(gradient_order=gradient_order, names=names)
        
        for prop_name, calc_name, gradient_order, value_array in self.ebs.iter_properties():
            property = self.point_set.get_property(prop_name)
            surface_points = self.interpolate_values(value_array)
            property[calc_name, gradient_order] = surface_points
            
        return None

    def compute_property(self, name: str, **kwargs):
        property = self.ebs.get_property(name, **kwargs)
        grid_scalars = self.compute_scalar_grid(property.value)
        
        original_property = self.original_ebs.get_property(name, **kwargs)
        grid_scalars = np.clip(grid_scalars, original_property.value.min(), original_property.value.max())
        return grid_scalars
    
    def interpolate_values(self, values:np.ndarray):
        if values.shape[-1] != 3:
            new_values = np.zeros(self.n_grid_points)
            interpolator = LinearNDInterpolator(self.plane_points, values)
            new_values = interpolator(self.uv_grid_points)
        else:
            new_values = np.zeros((self.n_grid_points, values.shape[-1]))
            for icoord in range(values.shape[-1]):
                interpolator = LinearNDInterpolator(self.plane_points, values[..., icoord])
                new_values[..., icoord] = interpolator(self.uv_grid_points)

        return new_values
    
    def points_to_grid(self, points:np.ndarray, order:str="C"):
        return points.reshape(self.u_grid.shape, order=order)
    
    def project_vector_to_plane(self, vectors:np.ndarray):
        velocity_u = np.dot(vectors, self.u)
        velocity_v = np.dot(vectors, self.v)
        return velocity_u, velocity_v
    
    def get_2d_brillouin_zone(self, e_min:float, e_max:float, supercell:List[int] = None, scale_factor:float = 2 * np.pi):
        return BrillouinZone2D(e_min=e_min,
                               e_max=e_max,
                               axis=2,
                               reciprocal_lattice=self.ebs.reciprocal_lattice*scale_factor)
    
    def set_surface_point_data(self, name:str, values:np.ndarray):
        if values.shape[-1] == 3:
            last_dim = 3
        else:
            last_dim = 1
            
        if self.ebs.is_band_property(values):
            logger.debug(f"Adding band resolved to fermi surface point_data: {name}")
            point_data_array = None
            for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
                values_band_values = values[:, iband, ispin, ...]
                if point_data_array is None:
                    point_data_array = values_band_values
                else:
                    point_data_array = np.insert(point_data_array, 0, values_band_values, axis=0)
            self.point_data[name] = point_data_array
        else:
            logger.debug(f"Adding scalar to fermi surface point_data: {name}")
            self.point_data[name] = values
    
    def set_scalars(self, name:str, value:np.ndarray):
        self.set_surface_point_data(name, value)
        self.set_active_scalars(name, preference="point")
        
    def set_vectors(self, name:str, value:np.ndarray, set_scalar:bool = True):
        if value.shape[-1] != 3:
            raise ValueError(f"Vector data must have 3 dimensions. Got {value.shape[-1]}.")
        
        vector_magnitude = np.linalg.norm(value, axis=-1)
        if set_scalar:
            self.set_surface_point_data(f"{name}-norm", vector_magnitude)
            self.set_active_scalars(f"{name}-norm", preference="point")
        self.set_surface_point_data(name, value)
        self.set_active_vectors(name, preference="point")
        
    def set_values(self, name:str, value:np.ndarray, **kwargs):
        if value.shape[-1] == 3:
            self.set_vectors(name, value, **kwargs)
        else:
            self.set_scalars(name, value, **kwargs)
        
    def compute_scalar_grid(self, scalars:np.ndarray, **kwargs):
        
        scalars_shape = scalars.shape
        slice = self.ebs.slice(normal=self.normal, origin=self.origin, 
                               scalars = ("scalars", scalars.reshape(scalars_shape[0], -1)),
                               as_cartesian=self.as_cartesian)
        
        scalar_grid_points_flat = slice.active_scalars
        new_scalars_grid_points_flat = np.zeros(shape=(self.n_grid_points, scalar_grid_points_flat.shape[-1]))

        for iscalar in range(scalar_grid_points_flat.shape[-1]):
            new_scalars_grid_points_flat[..., iscalar] = self.interpolate_values(scalar_grid_points_flat[..., iscalar])
        
        new_scalars_grid_points = new_scalars_grid_points_flat.reshape((self.n_grid_points, *scalars_shape[1:]))
        return new_scalars_grid_points
    
    def _generate_band_surface(self, scalars):
        logger.info(f"Generating band surface for {scalars.shape} scalars")
        surface_points = np.zeros((self.n_grid_points, 3))
        surface_points[:, 0] = self.u_grid.ravel()
        surface_points[:, 1] = self.v_grid.ravel()
        surface_points[:, 2] = scalars
        surface = pv.StructuredGrid()
        
        surface.points = surface_points
        surface.dimensions = (self.u_grid.shape[0], self.v_grid.shape[0], 1)
        surface = surface.cast_to_unstructured_grid()
        surface = surface.extract_surface()
        return surface
        
    def _generate_band_surfaces(self):
        
        bands = self.ebs.get_property("bands")
        new_bands = self.compute_scalar_grid(bands.value)
        
        band_surfaces = []
        n_grid_points, n_bands, n_spin_channels = new_bands.shape
        for iband in range(n_bands):
            for ispin in range(n_spin_channels):
                band_scalars = new_bands[..., iband, ispin]
                try:
                    surface = self._generate_band_surface(band_scalars)
                    if surface.points.shape[0] > 0:
                        logger.debug(
                            f"Surface for band {iband}, spin {ispin} has {surface.points.shape[0]} points"
                        )
                        band_surfaces.append(surface)
                        surface_idx = len(band_surfaces) - 1

                        surface_idx = len(band_surfaces) - 1
                        self.band_spin_surface_map[(iband, ispin)] = surface_idx
                        self.surface_band_spin_map[surface_idx] = (iband, ispin)
                        
                except Exception as e:
                    logger.exception(
                        f"Failed to generate surface for band {iband}, spin {ispin}: {e}"
                    )
                    continue
        
        if band_surfaces:
            logger.info(f"___Generated {len(band_surfaces)} band surfaces___")
            # Combine all surfaces
            combined_surface = self._merge_surfaces(band_surfaces)
            
            self.points = combined_surface.points
            self.faces = combined_surface.faces

            for key in combined_surface.point_data.keys():
                self.point_data[key] = combined_surface.point_data[key]
            for key in combined_surface.cell_data.keys():
                self.cell_data[key] = combined_surface.cell_data[key]
            
            self.field_data["band_spin_surface_map"] = self.band_spin_surface_map
            self.field_data["surface_band_spin_map"] = self.surface_band_spin_map
            
            self.set_active_scalars("spin_band_index")
            
            self.point_data.pop("vtkOriginalPointIds", None)
            self.cell_data.pop("vtkOriginalCellIds", None)
            
    def _merge_surfaces(self, surfaces: List[pv.PolyData]) -> pv.PolyData:
        """
        Merge multiple Fermi surfaces into a single surface with band and spin information.

        This method combines multiple PyVista PolyData objects representing different
        Fermi surfaces (from different bands and spins) into a single PolyData object.
        It adds point data arrays to track which points belong to which band and spin.

        Parameters
        ----------
        surfaces : List[pv.PolyData]
            List of PyVista PolyData objects representing Fermi surfaces for different
            bands and spins.

        Returns
        -------
        pv.PolyData
            A merged PyVista PolyData object containing all surfaces with added
            point data arrays 'spin_index' and 'spin_band_index' to identify the
            original band and spin of each point.

        """
        if not surfaces:
            return pv.PolyData()

        # Add band and spin indices to each surface before merging
        spin_band_index_array = np.empty(0, dtype=np.int32)
        spin_index_array = np.empty(0, dtype=np.int32)
        for surface_idx, surface in enumerate(surfaces):
            iband, ispin = self.surface_band_spin_map[surface_idx]
            n_points = surface.points.shape[0]

            # spin index identifier
            current_spin_index_array = np.full(n_points, ispin, dtype=np.int32)
            spin_index_array = np.insert(
                spin_index_array, 0, current_spin_index_array, axis=0
            )

            # spin band index identifier
            current_spin_band_index_array = np.full(
                n_points, surface_idx, dtype=np.int32
            )
            spin_band_index_array = np.insert(
                spin_band_index_array, 0, current_spin_band_index_array, axis=0
            )
            
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            mask = spin_band_index_array == surface_idx
            self.band_spin_mask[(iband, ispin)] = mask

        merged = surfaces[0]
        for surface in surfaces[1:]:
            merged = merged.merge(surface, merge_points=False)

        merged.point_data["spin_index"] = spin_index_array
        merged.point_data["spin_band_index"] = spin_band_index_array

        return merged

    def _get_brilloin_zone(self, supercell: List[int], zlim=[-2, 2]):
        """Returns the BrillouinZone of the material

        Returns
        -------
        pyprocar.core.BrillouinZone
            The BrillouinZone of the material
        """

        e_min = zlim[0]
        e_max = zlim[1]

        return BrillouinZone2D(e_min, e_max, self.ebs.reciprocal_lattice, supercell)
    
    def scale_kplane(self, scale_factor: float = 2 * np.pi):
        k_plane_scale_transform = np.eye(4)

        k_plane_scale_transform[0, 0] = scale_factor * k_plane_scale_transform[0, 0]
        k_plane_scale_transform[1, 1] = scale_factor * k_plane_scale_transform[1, 1]
        self.transform(k_plane_scale_transform, inplace=True)
        
    @classmethod
    def from_code(cls, code: str, dirpath: str, normal: np.ndarray, origin: np.ndarray, 
                  reduce_bands_near_energy: float = None,
                  reduce_bands_near_fermi: bool = True,
                  bands: List[int] = None,
                  grid_interpolation: tuple = (120, 120),
                  padding: int = 15,
                  as_cartesian=False,
                  **kwargs):
        ebs = ElectronicBandStructureMesh.from_code(code=code, dirpath=dirpath)
        
        if reduce_bands_near_energy is not None:
            ebs.reduce_bands_near_energy(reduce_bands_near_energy)
        elif reduce_bands_near_fermi:
            ebs.reduce_bands_near_fermi()
        elif bands is not None:
            ebs.reduce_bands_by_index(bands)
        
        
        
        return cls(ebs, normal=normal, origin=origin, 
                   grid_interpolation=grid_interpolation, 
                   as_cartesian=as_cartesian, 
                   padding=padding,
                   **kwargs)

