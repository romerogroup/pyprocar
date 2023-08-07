__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import random
import math
import sys
import copy
import itertools
from typing import List, Tuple

import numpy as np
import scipy.interpolate as interpolate
from matplotlib import colors as mpcolors
from matplotlib import cm

from . import Surface, BrillouinZone2D

import pyvista as pv
np.set_printoptions(threshold=sys.maxsize)

class BandStructure2D(Surface):
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

    def __init__(
        self,
        ebs,
        ispin,
        interpolation_factor: int=1,
        projection_accuracy: str="Normal",
        supercell: List[int]=[1, 1, 1],
        ):

        self.ebs = copy.copy(ebs)
        self.ispin=ispin
        # Shifts kpoints between [0.5,0.5)
        self.ebs.kpoints = -np.fmod(self.ebs.kpoints + 6.5, 1 ) + 0.5
        self.n_bands=self.ebs.bands.shape[1]
        self.supercell = np.array(supercell)
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy

        self.brillouin_zone = self._get_brilloin_zone(self.supercell)

        grid_cart_x=ebs.kpoints_cartesian_mesh[0,:,:,0]
        grid_cart_y=ebs.kpoints_cartesian_mesh[1,:,:,0]
        
        self.band_surfaces = self._generate_band_structure_2d(grid_cart_x,grid_cart_y)
        self.surface = self._combine_band_surfaces()
        # Initialize the Fermi Surface
        super().__init__(verts=self.surface.points, faces=self.surface.faces)
        self.point_data['band_index'] = self.surface.point_data['band_index']

        return None
    
    def _combine_band_surfaces(self):
        band_surfaces=copy.deepcopy(self.band_surfaces)
        band_indices=[]

        surface=band_surfaces[0]
        band_index_list=[0]*surface.points.shape[0]
        band_indices.extend(band_index_list)
        for i_band,band_surface in enumerate(band_surfaces[1:]):
            # The points are prepended to surface.points, 
            # so at the end we need to reverse this list
            surface.merge(band_surface, merge_points=False, inplace=True)
            band_index_list=[i_band+1]*band_surface.points.shape[0]
            band_indices.extend(band_index_list)


        band_indices.reverse()
        surface.point_data['band_index']=np.array(band_indices)
        return surface
    
    def _generate_band_structure_2d(self,grid_cart_x,grid_cart_y):
        surfaces=[]
        n_bands=self.ebs.bands_mesh.shape[0]
        n_points=0
        for iband in range(n_bands):
            
            grid_z=self.ebs.bands_mesh[iband,self.ispin,:,:,0]
            
            surface = pv.StructuredGrid(grid_cart_x,grid_cart_y,grid_z)
            surface = surface.cast_to_unstructured_grid()
            surface = surface.extract_surface()
            
            surface = Surface(verts=surface.points, faces=surface.faces)
            n_points+=surface.points.shape[0]
            
            band_index_list=[iband]*surface.points.shape[0]
            surface.point_data['bandindex']=np.array(band_index_list)
            surfaces.append(surface)

        return surfaces
     
    def _create_vector_texture(self,
                            vectors_array: np.ndarray, 
                            vectors_name: str="vector" ):
        """
        This method will map a list of vector to the 3d fermi surface mesh

        Parameters
        ----------
        vectors_array : np.ndarray
            The vector array corresponding to the kpoints
        vectors_name : str, optional
            The name of the vectors, by default "vector"
        """
        
        final_vectors_X = []
        final_vectors_Y = []
        final_vectors_Z = []
        for iband, isosurface in enumerate(self.band_surfaces):
            XYZ_extended = copy.copy(self.ebs.kpoints_cartesian)
            XYZ_extended[:,2]=self.ebs.bands[:,iband,self.ispin]

            vectors_extended_X = vectors_array[:,iband,0].copy()
            vectors_extended_Y = vectors_array[:,iband,1].copy()
            vectors_extended_Z = vectors_array[:,iband,2].copy()

            XYZ_transformed=XYZ_extended

            if self.projection_accuracy.lower()[0] == "n":
               
                vectors_X = interpolate.griddata(
                    XYZ_transformed, vectors_extended_X, isosurface.points, method="nearest"
                )
                vectors_Y = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Y, isosurface.points, method="nearest"
                )
                vectors_Z = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Z, isosurface.points, method="nearest"
                )
    
            elif self.projection_accuracy.lower()[0] == "h":
    
                vectors_X = interpolate.griddata(
                    XYZ_transformed, vectors_extended_X, isosurface.points, method="linear"
                )
                vectors_Y = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Y, isosurface.points, method="linear"
                )
                vectors_Z = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Z, isosurface.points, method="linear"
                )

            final_vectors_X.extend( vectors_X)
            final_vectors_Y.extend( vectors_Y)
            final_vectors_Z.extend( vectors_Z)
                
        self.set_vectors(final_vectors_X, final_vectors_Y, final_vectors_Z,vectors_name = vectors_name)
        return None
            
    def _project_color(self, 
                    scalars_array:np.ndarray,
                    scalar_name:str="scalars"):
        """
        Projects the scalars to the 3d fermi surface.

        Parameters
        ----------
        scalars_array : np.array size[len(kpoints),len(self.bands)]   
            the length of the self.bands is the number of bands with a fermi iso surface
        scalar_name :str, optional
            The name of the scalars, by default "scalars"
        
        Returns
        -------
        None.
        """

        points = self.ebs.kpoints_cartesian
        final_scalars = []
        for iband, isosurface in enumerate(self.band_surfaces):
            XYZ_extended = copy.copy(self.ebs.kpoints_cartesian)
            XYZ_extended[:,2]=self.ebs.bands[:,iband,self.ispin]
            scalars_extended =  scalars_array[:,iband].copy()
            XYZ_transformed=XYZ_extended
            if self.projection_accuracy.lower()[0] == "n":
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, isosurface.centers, method="nearest"
                )
            elif self.projection_accuracy.lower()[0] == "h":
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, isosurface.centers, method="linear"
                )
                
            final_scalars.extend(colors)

        self.set_scalars(final_scalars, scalar_name = scalar_name)
        return None
  
    def project_atomic_projections(self,spd):
        """
        Method to calculate the atomic projections of the surface.
        """
        scalars_array = []
        count = 0
        for iband in range(self.n_bands):
            count+=1
            scalars_array.append(spd[:,iband])
        scalars_array = np.vstack(scalars_array).T

        self._project_color(scalars_array = scalars_array, scalar_name = "scalars")

    def project_spin_texture_atomic_projections(self,spd_spin):
        """
        Method to calculate atomic spin texture projections of the surface.
        """
        vectors_array = spd_spin
        self._create_vector_texture(vectors_array = vectors_array, vectors_name = "spin" )

    def project_fermi_velocity(self,fermi_velocity):
        """
        Method to calculate fermi velocity of the surface.
        """
        vectors_array = fermi_velocity.swapaxes(1, 2)
        self._create_vector_texture(vectors_array = vectors_array, vectors_name = "Fermi Velocity Vector" )

    def project_fermi_speed(self,fermi_speed):
        """
        Method to calculate the fermi speed of the surface.
        """
        scalars_array = []
        count = 0
        for iband in range(self.n_bands):
            count+=1
            scalars_array.append(fermi_speed[:,iband])
        scalars_array = np.vstack(scalars_array).T
        self._project_color(scalars_array = scalars_array, scalar_name = "Fermi Speed")

    def project_harmonic_effective_mass(self,harmonic_effective_mass):
        """
        Method to calculate the atomic projections of the surface.
        """
        scalars_array = []
        count = 0
        for iband in range(self.n_bands):
            count+=1
            scalars_array.append(harmonic_effective_mass[:,iband])
        scalars_array = np.vstack(scalars_array).T
        self._project_color(scalars_array = scalars_array, scalar_name = "Harmonic Effective Mass" )

    def extend_surface(self,  extended_zone_directions: List[List[int] or Tuple[int,int,int]]=None,):
        """
        Method to extend the surface in the direction of a reciprocal lattice vecctor

        Parameters
        ----------
        extended_zone_directions : List[List[int] or Tuple[int,int,int]], optional
            List of directions to expand to, by default None
        """
        # The following code  creates exteneded surfaces in a given direction
        extended_surfaces = []
        if extended_zone_directions is not None:
            # new_surface = copy.deepcopy(self)
            initial_surface = copy.deepcopy(self)
            for direction in extended_zone_directions:
                surface = copy.deepcopy(initial_surface)

                self += surface.translate(np.dot(direction, self.ebs.reciprocal_lattice), inplace=True)
            # Clearing unneeded surface from memory
            del surface
    
    def _get_brilloin_zone(self, 
                        supercell: List[int]):

        """Returns the BrillouinZone of the material

        Returns
        -------
        pyprocar.core.BrillouinZone
            The BrillouinZone of the material
        """
        e_min=self.ebs.bands.min()
        e_max=self.ebs.bands.max()
        return BrillouinZone2D(e_min,e_max,self.ebs.reciprocal_lattice, supercell)
    
    