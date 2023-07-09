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
    kpoints : (n,3) float
        A numpy array of kpoints used in the DFT calculation, this list
        has to be (n,3), n being number of kpoints and 3 being the
        3 different cartesian coordinates.
    band : (n,) float
        A numpy array of energies of ith band cooresponding to the
        kpoints.
    fermi : float
        Value of the fermi energy or any energy that one wants to
        find the isosurface with.
    reciprocal_lattice : (3,3) float
        Reciprocal lattice of the structure.
    spd :
        numpy array containing the information about projection of atoms,
        orbitals and spin on each band 
    spd_spin :
        numpy array containing the information about spin projection of atoms
    fermi_shift : float
        Value to shift fermi energy.
    fermi_tolerance : float = 0.1
        This is used to improve search effiency by doing a prior search selecting band within a tolerance of the fermi energy
    interpolation_factor : int
        The default is 1. number of kpoints in every direction
        will increase by this factor.
    colors : list str or list tuples of size 4, optional
        List of colors for each band. If you use tuple, it represents rgba values
        This argument does not work whena 3d file is saved. 
        The colors for when ``save3d`` is used, we
        recomend using qualitative colormaps, as this function will
        automatically choose colors from the colormaps. e.g. 
        
        .. code-block::
            :linenos: 

            colors=['red', 'blue', 'green']
            colors=[(1,0,0,1), (0,1,0,1), (0,0,1,1)]
            
    projection_accuracy : str, optional
        Controls the accuracy of the projects. 2 types ('high', normal) 
        The default is ``projection_accuracy=normal``.
    cmap : str
        The default is 'viridis'. Color map used in projecting the
        colors on the surface
    vmin :  float
        Value to normalize the minimum projection value. The default is 0.
    vmax :  float
        Value to normalize the maximum projection value.. The default is 1.
    supercell : list int
        This is used to add padding to the array 
        to assist in the calculation of the isosurface.
    """

    def __init__(
        self,

        ebs=None,
        bands_to_keep: List[int]=None,

        fermi_shift: float=0.0,
        fermi_tolerance:float=0.1,
        interpolation_factor: int=1,
        colors: List[str] or List[Tuple[float,float,float]]=None,
        surface_color: str or Tuple[float,float,float, float]=None,
        projection_accuracy: str="Normal",
        cmap: str="viridis",
        vmin: float=0,
        vmax: float=1,
        supercell: List[int]=[1, 1, 1],
        # sym: bool=False
        ):

    
        self.kpoints = kpoints
        
        # Shifts kpoints between [0.5,0.5)
        self.XYZ = -np.fmod(self.kpoints + 6.5, 1 ) + 0.5

        self.bands = bands 

        if bands_to_keep is None:
            bands_to_keep = len(self.bands[0,:])
        elif len(bands_to_keep) < len(self.bands[0,:]) :
            self.bands = self.bands[:,bands_to_keep]
        
        self.reciprocal_lattice = reciprocal_lattice
        self.supercell = np.array(supercell)
        self.fermi = fermi + fermi_shift
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy
        self.spd = spd
        self.spd_spin = spd_spin
        self.brillouin_zone = self._get_brilloin_zone(self.supercell)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        


        self.isosurfaces = self._generate_isosurfaces()
        
        self.surface = self._combine_isosurfaces()
        # Initialize the Fermi Surface
        super().__init__(verts=self.surface.points, faces=self.surface.faces)

        self.fermi_surface_area = self.area
        
        # Remapping of the scalar arrays into the combind mesh  
        count = 0
        combined_band_color_array = []

        combined_band_color_array = np.zeros(shape=(len(self.points),4))
        for iband,isosurface_band in enumerate(self.isosurfaces):
            color_array_name = isosurface_band.point_data.keys()[0]
            i_reduced_band = color_array_name.split('_')[1]

            new_color_array = np.zeros(shape=(len(self.points),4))
            for i,point in enumerate(self.points):
                # Check if the point is in the combined surface
                if any(np.equal(isosurface_band.points,point).all(1)):
                    combined_band_color_array[i] = color_band_dict[color_array_name]['color']
                    new_color_array[i] = color_band_dict[color_array_name]['color']
                else:
                    new_color_array[i] = np.array([0,0,0,1])
        
            self.point_data[ "band_"+ str(self.reducedBandIndex_to_fullBandIndex[str(i_reduced_band)])] = np.array(new_color_array)
        self.point_data["bands"] = combined_band_color_array #np.array(combined_band_color_array ) 
        return None
    


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
            XYZ_extended = self.XYZ.copy()
            
            vectors_extended_X = vectors_array[:,iband,0].copy()
            vectors_extended_Y = vectors_array[:,iband,1].copy()
            vectors_extended_Z = vectors_array[:,iband,2].copy()
    
            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
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
                    temp = self.XYZ.copy()
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
    
            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
    
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
            XYZ_extended = self.XYZ.copy()
            scalars_extended =  scalars_array[:,iband].copy()
    
    
            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
                    temp = self.XYZ.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
  
            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
    
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
  
    def _generate_band_colors(self, surface_color, colors):
        # code to generate unique RGBA values for the bands
        nsurface = len(self.bands[0,:])
        if surface_color:
            solid_color_surface = np.arange(nsurface ) / nsurface

            if isinstance(surface_color,str):
                surface_color = mpcolors.to_rgba_array(surface_color, alpha =1 )[0,:]
            band_colors = np.array([surface_color for x in solid_color_surface[:]]).reshape(-1, 4)
        elif colors:
            band_colors =[]
            for color in colors:
                if isinstance(color,str):
                    color = mpcolors.to_rgba_array(color, alpha =1 )[0,:]
                    band_colors.append(color)
        else:
            norm = mpcolors.Normalize(vmin=self.vmin, vmax=self.vmax)
            cmap = cm.get_cmap(self.cmap)
            solid_color_surface = np.arange(nsurface ) / nsurface
            band_colors = np.array([cmap(norm(x)) for x in solid_color_surface[:]]).reshape(-1, 4)

    def _generate_isosurfaces(self):
        # code to generate isosurfaces for each band
        isosurfaces=[]
        for iband in  range(self.bands.shape[1]):
            isosurface_band = Isosurface(
                                XYZ=self.XYZ,
                                V=self.bands[:,iband],
                                isovalue=self.fermi,
                                algorithm="lewiner",
                                interpolation_factor=self.interpolation_factor,
                                padding=self.supercell,
                                transform_matrix=self.reciprocal_lattice,
                                boundaries=self.brillouin_zone,
                            )
            self.isosurfaces.append(isosurface_band)

    def _combine_isosurfaces(self):
        isosurfaces=copy.deepcopy(self.isosurfaces)
        surface=self.isosurfaces[0]
        for isosurface in isosurfaces[1:]:
            surface.merge(isosurface, merge_points=False, inplace=True) 
        return surface
    
    def project_atomic_projections(self):
        """
        Method to calculate the atomic projections of the surface.
        """
        scalars_array = []
        count = 0
        for iband in range(len(self.isosurfaces)):
            count+=1
            scalars_array.append(self.spd[:,iband])
        scalars_array = np.vstack(scalars_array).T
        self._project_color(scalars_array = scalars_array, scalar_name = "scalars")

    def project_spin_texture_atomic_projections(self):
        """
        Method to calculate atomic spin texture projections of the surface.
        """
        vectors_array = self.spd_spin
        self._create_vector_texture(vectors_array = vectors_array, vectors_name = "spin" )

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

                self += surface.translate(np.dot(direction, self.reciprocal_lattice), inplace=True)
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

        return BrillouinZone(self.reciprocal_lattice, supercell)

