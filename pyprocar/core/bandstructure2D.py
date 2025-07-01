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
from scipy.spatial import KDTree

from pyprocar.core.brillouin_zone import BrillouinZone2D
from pyprocar.core.ebs import ElectronicBandStructurePlane

np.set_printoptions(threshold=sys.maxsize)

logger = logging.getLogger(__name__)

# TODO: method to reduce number of points for interpolation need to be modified since the tolerance on the space
# is not lonmg soley reciprocal space, but energy and reciprocal space


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

    def __init__(self, ebs: ElectronicBandStructurePlane):
        super().__init__()
        self._ebs = copy.copy(ebs)

        
         # Initialize storage for surface data
        self.band_spin_surface_map = {}
        self.surface_band_spin_map = {}
        self.band_spin_mask = {}
        
        self.transform_to_cart = np.eye(4)
        self.transform_to_frac = np.eye(4)
        self.transform_to_cart[:3, :3] = self.ebs.transformation_matrix
        self.transform_to_frac[:3, :3] = np.linalg.inv(
            self.ebs.reciprocal_lattice.T
        )
        
        # Initialize the Fermi Surface
        self._generate_band_surfaces()
        
        self.transform(
            self.transform_to_cart, transform_all_input_vectors=False, inplace=True
        )
        return None

    @property
    def ebs(self):
        return self._ebs
    
    def _generate_band_surface(self, scalars, order="F"):
        logger.info(f"Generating band surface for {scalars.shape} scalars")
        scalar_grid = self.ebs.scalars_to_mesh(scalars, order=order)
        logger.debug(f"Scalar grid shape: {scalar_grid.shape}")
        surface = pv.StructuredGrid(self.ebs.grid_u, self.ebs.grid_v, scalar_grid)
        surface = surface.cast_to_unstructured_grid()
        surface = surface.extract_surface()
        return surface
        
    def _generate_band_surfaces(self, order="F"):
        
        band_surfaces = []
        n_kpoints, n_bands, n_spin_channels = self.ebs.bands.shape
        for iband in range(n_bands):
            for ispin in range(n_spin_channels):
                band_scalars = self.ebs.bands[:, iband, ispin]
                try:
                    surface = self._generate_band_surface(band_scalars, order)
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
                  reduce_bands_near_fermi: bool = False,
                  bands: List[int] = None,
                  in_cartesian: bool = True, 
                  plane_tol: float = 0.01, 
                  grid_resolution: tuple = (20, 20),
                  **kwargs):
        from pyprocar.io import Parser
        parser = Parser(code=code, dirpath=dirpath, **kwargs)
        ebs = parser.ebs
        
        if reduce_bands_near_energy is not None:
            ebs.reduce_bands_near_energy(reduce_bands_near_energy)
        elif reduce_bands_near_fermi:
            ebs.reduce_bands_near_fermi()
        elif bands is not None:
            ebs.reduce_bands_by_index(bands)
            
        ebs.pad(padding=10, inplace=True)
            
        ebs_plane = ebs.reduce_kpoints_to_plane(normal=normal, 
                                           origin=origin, 
                                           in_cartesian=in_cartesian, 
                                           plane_tol=plane_tol, 
                                           grid_resolution=grid_resolution)

        return cls(ebs_plane)
