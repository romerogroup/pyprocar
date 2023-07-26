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

from . import Isosurface, Surface, BrillouinZone

import pyvista as pv
np.set_printoptions(threshold=sys.maxsize)


HBAR_EV = 6.582119 *10**(-16) #eV*s
HBAR_J = 1.0545718 *10**(-34) #eV*s
METER_ANGSTROM = 10**(-10) #m /A
EV_TO_J = 1.602*10**(-19)
FREE_ELECTRON_MASS = 9.11*10**-31 #  kg

class FermiSurface3D(Surface):
# class FermiSurface3D(pv.PolyData):
    """
    The object is used to store and manapulate a 3d fermi surface.

    Parameters
    ----------
    ebs : ElectronicBandStructure
        The ElectronicBandStructure object
    fermi : float
        The energy to search for the fermi surface
    fermi_shift : float
        Value to shift fermi energy.
    fermi_tolerance : float = 0.1
        This is used to improve search effiency by doing a prior search selecting band within a tolerance of the fermi energy
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
        fermi:float=0.0,
        fermi_shift: float=0.0,
        interpolation_factor: int=1,
        projection_accuracy: str="Normal",
        supercell: List[int]=[1, 1, 1],
        ):

        self.ebs = copy.copy(ebs)
        if (self.ebs.bands.shape)==3:
            raise "Must reduce ebs.bands into 2d darray"
        
        # Shifts kpoints between [0.5,0.5)
        self.ebs.kpoints = -np.fmod(self.ebs.kpoints + 6.5, 1 ) + 0.5

        self.supercell = np.array(supercell)
        self.fermi = fermi + fermi_shift
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy

        self._input_checks()
        self.brillouin_zone = self._get_brilloin_zone(self.supercell)
        self.isosurfaces = self._generate_isosurfaces()
        self.surface = self._combine_isosurfaces()
        # Initialize the Fermi Surface
        super().__init__(verts=self.surface.points, faces=self.surface.faces)


        self.point_data['band_index'] = self.surface.point_data['band_index'] 
        self.fermi_surface_area = self.area
        return None
    
    def _input_checks(self):
        assert len(self.ebs.bands.shape)==2

    def _generate_isosurfaces(self):
        # code to generate isosurfaces for each band
        isosurfaces=[]
        self.band_index_map = []

        for iband in  range(self.ebs.bands.shape[1]):
            isosurface_band = Isosurface(
                                XYZ=self.ebs.kpoints,
                                V=self.ebs.bands[:,iband],
                                isovalue=self.fermi,
                                algorithm="lewiner",
                                interpolation_factor=self.interpolation_factor,
                                padding=self.supercell,
                                transform_matrix=self.ebs.reciprocal_lattice,
                                boundaries=self.brillouin_zone,
                            )
            

            # Check to see if the generated isosurface has points
            if isosurface_band.points.shape[0] != 0:
                isosurfaces.append(isosurface_band)
                n_isosurface=len(isosurfaces)
                self.band_index_map.append(iband)

        self.ebs.bands=self.ebs.bands[:,self.band_index_map]
        return isosurfaces
    
    def _combine_isosurfaces(self):
        isosurfaces=copy.deepcopy(self.isosurfaces)
        band_indices=[]

        surface=isosurfaces[0]
        band_index_list=[0]*surface.points.shape[0]
        band_indices.extend(band_index_list)
        for i_band,isosurface in enumerate(isosurfaces[1:]):
            # The points are prepended to surface.points, 
            # so at the end we need to reverse this list
            surface.merge(isosurface, merge_points=False, inplace=True)
            band_index_list=[i_band+1]*isosurface.points.shape[0]
            band_indices.extend(band_index_list)
        band_indices.reverse()
        surface.point_data['band_index']=np.array(band_indices)
        return surface
    
    def _get_brilloin_zone(self, 
                        supercell: List[int]):

        """Returns the BrillouinZone of the material

        Returns
        -------
        pyprocar.core.BrillouinZone
            The BrillouinZone of the material
        """

        return BrillouinZone(self.ebs.reciprocal_lattice, supercell)

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
        for iband, isosurface in enumerate(self.isosurfaces):
            XYZ_extended = self.ebs.kpoints.copy()
            
            vectors_extended_X = vectors_array[:,iband,0].copy()
            vectors_extended_Y = vectors_array[:,iband,1].copy()
            vectors_extended_Z = vectors_array[:,iband,2].copy()
    
            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.ebs.kpoints.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, vectors_array[:,iband,0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, vectors_array[:,iband,1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, vectors_array[:,iband,2], axis=0
                    )
                    temp = self.ebs.kpoints.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, vectors_array[:,iband,0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, vectors_array[:,iband,1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, vectors_array[:,iband,2], axis=0
                    )
    
            XYZ_transformed = np.dot(XYZ_extended, self.ebs.reciprocal_lattice)
    
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
        final_scalars = []
        for iband, isosurface in enumerate(self.isosurfaces):
            XYZ_extended = self.ebs.kpoints.copy()
            scalars_extended =  scalars_array[:,iband].copy()
    
    
            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.ebs.kpoints.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
                    temp = self.ebs.kpoints.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
  
            XYZ_transformed = np.dot(XYZ_extended, self.ebs.reciprocal_lattice)
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
        for iband in range(len(self.isosurfaces)):
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
        Method to calculate atomic spin texture projections of the surface.
        """
        vectors_array = fermi_velocity.swapaxes(1, 2)
        self._create_vector_texture(vectors_array = vectors_array, vectors_name = "Fermi Velocity Vector" )

    def project_fermi_speed(self,fermi_speed):
        """
        Method to calculate the fermi speed of the surface.
        """
        scalars_array = []
        count = 0
        for iband in range(len(self.isosurfaces)):
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
        for iband in range(len(self.isosurfaces)):
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
 



