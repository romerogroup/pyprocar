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

from .brillouin_zone import BrillouinZone
from ..core import Isosurface, Surface


np.set_printoptions(threshold=sys.maxsize)

# TODO add python typing
# TODO move this module to plotter
# TODO add default settings

HBAR_EV = 6.582119 *10**(-16) #eV*s
HBAR_J = 1.0545718 *10**(-34) #eV*s
METER_ANGSTROM = 10**(-10) #m /A
EV_TO_J = 1.602*10**(-19)
FREE_ELECTRON_MASS = 9.11*10**-31 #  kg

class FermiSurface3D(Surface):

    def __init__(
        self,
        kpoints: np.ndarray,
        bands: np.ndarray,
        fermi: float,
        reciprocal_lattice: np.ndarray,
        bands_to_keep: List[int]=None,
        spd: np.ndarray=None,
        spd_spin:np.ndarray=None,
        calculate_fermi_velocity: bool=False,
        calculate_fermi_speed: bool=False,
        calculate_effective_mass: bool=False,
        fermi_shift: float=0.0,
        fermi_tolerance:float=0.1,
        interpolation_factor: int=1,
        extended_zone_directions: List[List[int] or Tuple[int,int,int]]=None,
        colors: List[str] or List[Tuple[float,float,float]]=None,
        projection_accuracy: str="Normal",
        cmap: str="viridis",
        vmin: float=0,
        vmax: float=1,
        supercell: List[int]=[1, 1, 1],
        # sym: bool=False
        ):

        """
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
        calculate_fermi_velocity : bool, optional (default False)
            Boolean value to calculate fermi velocity vectors on the band surface
            e.g. ``fermi_velocity_vector=True``
        caclulate_fermi_speed : bool, optional (default False)
            Boolean value to calculate magnitude of the fermi velocity on the band surface
            e.g. ``fermi_velocity=True``
        calculate_effective_mass : bool, optional (default False)
            Boolean value to calculate the harmonic mean of the effective mass on the band surface
            e.g. ``effective_mass=True``
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
            automatically choose colors from the colormaps.
            e.g. ``colors=['red', 'blue', 'green']``
                ``colors=[(1,0,0,1), (0,1,0,1), (0,0,1,1)]``
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

        self.kpoints = kpoints
        self.XYZ = np.array(self.kpoints)
        self.bands = bands
        if bands_to_keep is None:
            bands_to_keep = len(self.bands[0,:,0])
        elif len(bands_to_keep) < len(self.bands[0,:,0]) :
            # print("Only considering bands : " , bands_to_keep)
            self.bands = self.bands[:,bands_to_keep,0]

        self.reciprocal_lattice = reciprocal_lattice
        self.supercell = np.array(supercell)
        self.fermi = fermi + fermi_shift
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy
        self.spd_spin = spd_spin
        self.calculate_fermi_velocity = calculate_fermi_velocity
        self.calculate_fermi_speed = calculate_fermi_speed
        self.calculate_effective_mass = calculate_effective_mass
        self.brillouin_zone = self._get_brilloin_zone(self.supercell)
        
        # Finding bands with a fermi iso-surface. This reduces searching
        fullBandIndex = []
        reducedBandIndex = []
        for iband in range(len(self.bands[0,:,0])):
            fermi_surface_test = len(np.where(np.logical_and(self.bands[:,iband,0]>=self.fermi-fermi_tolerance, self.bands[:,iband,0]<=self.fermi+fermi_tolerance))[0])
            if fermi_surface_test != 0:
                fullBandIndex.append(iband)
        if len(fullBandIndex)==0:
            raise Exception("No bands within tolerance. Increase tolerance to increase search space.")
        self.bands = self.bands[:,fullBandIndex,0]

 

        # re-index and creates a mapping to the original bandindex
        reducedBandIndex = np.arange(len(self.bands[0,:]))
        self.fullBandIndex = fullBandIndex
        self.reducedBandIndex = reducedBandIndex
        self.reducedBandIndex_to_fullBandIndex = {f"{key}":value for key,value in zip(reducedBandIndex,fullBandIndex)}
        self.fullBandIndex_to_reducedBandIndex = {f"{key}":value for key,value in zip(fullBandIndex,reducedBandIndex)}
        reduced_bands_to_keep_index = [iband for iband in range(len(bands_to_keep))]

        # Reduces the spd array to the reduces band index scheme
        self.spd = np.array(spd).T
        self.spd_spin = np.array(spd_spin).T
        if self.spd[0] is not None:
            self.spd = self.spd[:,fullBandIndex]
        if self.spd_spin[0] is not None:
            self.spd_spin = self.spd_spin[:,fullBandIndex]

        # Generate unique rgba values for the bands
        band_colors = colors
        if colors is None:
            nsurface = len(self.bands[0,:])
            norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap(cmap)
            solid_color_surface = np.arange(nsurface ) / nsurface
            band_colors = np.array([cmap(norm(x)) for x in solid_color_surface[:]]).reshape(-1, 4)

        # The following loop generates iso surfaces for each band and then stores them in a list
        color_band_dict = {}
        self.isosurfaces = []
        full_isosurface = None
        
        iband_with_surface=0
        for iband, bands in enumerate(self.bands[0,:]):
            # Conditional for shifting values between [-0.5,0.5]
            # Must be done for isosurface algorithm to produce consistent results
            if np.any(self.kpoints > 0.5):
                isosurface_band = Isosurface(
                                        XYZ=self.kpoints,
                                        V=self.bands[:,iband],
                                        isovalue=self.fermi,
                                        algorithm="lewiner",
                                        interpolation_factor=interpolation_factor,
                                        padding=self.supercell * 2,
                                        transform_matrix=self.reciprocal_lattice,
                                        boundaries=self.brillouin_zone,
                                    )
                                    
            else:
                isosurface_band = Isosurface(
                                    XYZ=self.kpoints,
                                    V=self.bands[:,iband],
                                    isovalue=self.fermi,
                                    algorithm="lewiner",
                                    interpolation_factor=interpolation_factor,
                                    padding=self.supercell,
                                    transform_matrix=self.reciprocal_lattice,
                                    boundaries=self.brillouin_zone,
                                )
            isosurface_band_copy = copy.deepcopy(isosurface_band)
            if full_isosurface is None and len(isosurface_band_copy.points)!=0:
                full_isosurface = isosurface_band_copy
            elif len(isosurface_band_copy.points)==0:
                if full_isosurface is None and iband == len(self.bands[0,:])-1:
                    raise Exception("Could not find any fermi surfaces")
                continue
            else:
                full_isosurface += isosurface_band_copy

            color_band_dict.update({f"band_{iband_with_surface}": {"color" : band_colors[iband_with_surface,:]}
                                  })

            band_color = np.array([band_colors[iband_with_surface,:]]*len(isosurface_band.points[:,0]))
            isosurface_band.point_data[f"band_{iband_with_surface}"] = band_color
            self.isosurfaces.append(isosurface_band)

            iband_with_surface +=1
            
            
        # Initialize the Fermi Surface which is the combination of all the 
        # isosurface for each band
        super().__init__(verts=full_isosurface.points, faces=full_isosurface.faces)
        self.fermi_surface_area = self.area
        
        # Remapping of the scalar arrays into the combind mesh
        count = 0
        combined_band_color_array = []
        for iband,isosurface_band in enumerate(self.isosurfaces):
            new_color_array = []
            color_array_name = isosurface_band.point_data.keys()[0]
            for points in self.points:
                if points in isosurface_band.points:
                    new_color_array.append(color_band_dict[color_array_name]['color'])    
                else:
                    new_color_array.append(np.array([0,0,0,1]))
            for ipoints in range(len(isosurface_band.points)):
                combined_band_color_array.append(color_band_dict[color_array_name]['color'])
            self.point_data[ "band_"+ str(self.reducedBandIndex_to_fullBandIndex[str(iband)])] = np.array(new_color_array)
        self.point_data["bands"] = np.array(combined_band_color_array ) 
        
        # Interpolation of scalars to the surface
        if self.spd[0] is not None and self.points is not None:
            scalars_array = []
            count = 0
            for iband in range(len(self.isosurfaces)):
                count+=1
                scalars_array.append(self.spd[:,iband])
            scalars_array = np.vstack(scalars_array).T 
            self.project_color(scalars_array = scalars_array, cmap=cmap, vmin=vmin, vmax=vmax, scalar_name = "scalars")
            
        # Interpolation of spd vectorsto the surface
        if self.spd_spin[0] is not None and self.points is not None:
            
            vectors_array = []
            for iband in range(len(self.bands[0,:])):
                vectors_array.append(self.spd_spin[:,iband])
            vectors_array = np.array(vectors_array).T
            vectors_array = np.swapaxes(vectors_array,axis1 = 1,axis2 = 2)
            
            self.create_vector_texture(vectors_array = vectors_array, vectors_name = "spin" )

        # Interpolation of vectors to the surface
        if self.calculate_fermi_velocity== True or self.calculate_fermi_speed == True or self.calculate_effective_mass == True:
            self.calculate_first_and_second_derivative_energy()
            
            if self.calculate_fermi_velocity == True and self.points is not None:
              
                vectors_array = []
                for band_name in self.first_and_second_derivative_energy_property_dict.keys():
                    vectors_array.append(self.first_and_second_derivative_energy_property_dict[band_name]["group_velocity_vector"])
                    
                vectors_array = np.array(vectors_array).T
                vectors_array = np.swapaxes(vectors_array,axis1 = 1,axis2 = 2)
    
                self.create_vector_texture( vectors_array = vectors_array, vectors_name = "Fermi Velocity Vector"  )

            if self.calculate_fermi_speed == True and self.points is not None:

                scalars_array = []
                for band_name in self.first_and_second_derivative_energy_property_dict.keys():
                    scalars_array.append(self.first_and_second_derivative_energy_property_dict[band_name]["group_velocity_magnitude"])
                    
                scalars_array = np.vstack(scalars_array).T 
                
                self.project_color(scalars_array = scalars_array, cmap=cmap, vmin=vmin, vmax=vmax,  scalar_name = "Fermi Speed")
    
            if self.calculate_effective_mass == True and self.points is not None:

                scalars_array = []
                for band_name in self.first_and_second_derivative_energy_property_dict.keys():
                    scalars_array.append(self.first_and_second_derivative_energy_property_dict[band_name]["effective_mass_list"])
                scalars_array = np.vstack(scalars_array).T 
                
                self.project_color(scalars_array = scalars_array, cmap=cmap, vmin=vmin, vmax=vmax, scalar_name="Geometric Average Effective Mass")
        

        # The following code  creates exteneded surfaces in a given direction
        extended_surfaces = []
        if extended_zone_directions is not None:
            original_surface = copy.deepcopy(self) 
            for direction in extended_zone_directions:
                surface = copy.deepcopy(original_surface)
                self += surface.translate(np.dot(direction, reciprocal_lattice))
            #Clearing unneeded surface from memory
            del original_surface
            del surface

        # if sym == True:
        #     self.ibz2fbz()
            
            
    def create_vector_texture(self,
                            vectors_array: np.ndarray, 
                            vectors_name: str="vector" ):

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
    
            if np.any(self.XYZ >= 0.5): # @logan : I think this should be before the above loop with another else statment
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
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
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
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
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
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
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
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
            
    def create_spin_texture(self):

        if self.spd_spin is not None:
            XYZ_extended = self.XYZ.copy()
            vectors_extended_X = self.spd_spin[0].copy()
            vectors_extended_Y = self.spd_spin[1].copy()
            vectors_extended_Z = self.spd_spin[2].copy()

            for ix in range(3):
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, ix] += 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )
                    temp = self.XYZ.copy()
                    temp[:, ix] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )

            if np.any(self.XYZ >= 0.5): # @logan : I think this should be before the above loop with another else statment
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )
                    temp = self.XYZ.copy()
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )

                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    vectors_extended_X = np.append(
                        vectors_extended_X, self.spd_spin[0], axis=0
                    )
                    vectors_extended_Y = np.append(
                        vectors_extended_Y, self.spd_spin[1], axis=0
                    )
                    vectors_extended_Z = np.append(
                        vectors_extended_Z, self.spd_spin[2], axis=0
                    )

            # XYZ_extended = self.XYZ.copy()
            # scalars_extended = self.spd.copy()

            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
            # XYZ_transformed = XYZ_extended

            if self.projection_accuracy.lower()[0] == "n":

                spin_X = interpolate.griddata(
                    XYZ_transformed, vectors_extended_X, self.points, method="nearest"
                )
                spin_Y = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Y, self.points, method="nearest"
                )
                spin_Z = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Z, self.points, method="nearest"
                )

            elif self.projection_accuracy.lower()[0] == "h":

                spin_X = interpolate.griddata(
                    XYZ_transformed, vectors_extended_X, self.points, method="linear"
                )
                spin_Y = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Y, self.points, method="linear"
                )
                spin_Z = interpolate.griddata(
                    XYZ_transformed, vectors_extended_Z, self.points, method="linear"
                )

            self.set_vectors(spin_X, spin_Y, spin_Z)
   
    def project_color(self, 
                    scalars_array:np.ndarray,
                    cmap: str="viridis", 
                    vmin: float=0.0, 
                    vmax: float=1.0, 
                    scalar_name:str="scalars"):
        """
        Projects the scalars to the surface.
        Parameters
        ----------
        cmap : TYPE string
            DESCRIPTION. Colormaps for the projection.
        vmin : TYPE
            DESCRIPTION.
        vmax : TYPE
            DESCRIPTION.
        scalars_array : np.array size[len(kpoints),len(self.bands)]   
            the length of the self.bands is the number of bands with a fermi iso surface
        
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
            if np.any(self.XYZ >= 0.5): # @logan same here
                for iy in range(self.supercell[ix]):
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
                    temp = self.XYZ.copy()
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
    
                    temp = self.XYZ.copy()
                    temp[:, 0] -= 1 * (iy + 1)
                    temp[:, 1] -= 1 * (iy + 1)
                    temp[:, 2] -= 1 * (iy + 1)
                    XYZ_extended = np.append(XYZ_extended, temp, axis=0)
                    scalars_extended = np.append(scalars_extended,  scalars_array[:,iband], axis=0)
    
            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
    
    
            # XYZ_transformed = XYZ_extended
    
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
        # self.set_color_with_cmap(cmap, vmin, vmax)
              
    def calculate_first_and_second_derivative_energy_band(self, 
                                                        iband: int):
        def get_energy(kp_reduced: np.ndarray):
            return kp_reduced_to_energy[f'({kp_reduced[0]},{kp_reduced[1]},{kp_reduced[2]})']
        def get_cartesian_kp(kp_reduced: np.ndarray):
            return kp_reduced_to_kp_cart[f'({kp_reduced[0]},{kp_reduced[1]},{kp_reduced[2]})']
        def energy_meshgrid_mapping(mesh_list):
            mesh_list_cart = copy.deepcopy(mesh_list)
            F = np.zeros(shape = mesh_list[0].shape)
            for k in range(mesh_list[0].shape[2]):
                for j in range(mesh_list[0].shape[1]):
                    for i in range(mesh_list[0].shape[0]):
                        kx = mesh_list[0][i,j,k]
                        ky = mesh_list[1][i,j,k]
                        kz = mesh_list[2][i,j,k]
        
                        mesh_list_cart[0][i,j,k] = get_cartesian_kp([kx,ky,kz])[0]
                        mesh_list_cart[1][i,j,k] = get_cartesian_kp([kx,ky,kz])[1]
                        mesh_list_cart[2][i,j,k] = get_cartesian_kp([kx,ky,kz])[2]
                        F[i,j,k] = get_energy([kx,ky,kz])
        
            return F
        def fermi_velocity_meshgrid_mapping(mesh_list):
            X = np.zeros(shape = mesh_list[0].shape)
            Y = np.zeros(shape = mesh_list[0].shape)
            Z = np.zeros(shape = mesh_list[0].shape)
            for k in range(mesh_list[0].shape[2]):
                for j in range(mesh_list[0].shape[1]):
                    for i in range(mesh_list[0].shape[0]):
                        kx = mesh_list[0][i,j,k]
                        ky = mesh_list[1][i,j,k]
                        kz = mesh_list[2][i,j,k]
                        
                        X[i,j,k] = get_gradient([kx,ky,kz])[0]
                        Y[i,j,k] = get_gradient([kx,ky,kz])[1]
                        Z[i,j,k] = get_gradient([kx,ky,kz])[2]
        
            return X,Y,Z
        
        kpoints_cart = np.matmul(self.XYZ,  self.reciprocal_lattice)
        
        mesh_list = np.meshgrid(np.unique(self.XYZ[:,0]),
                                np.unique(self.XYZ[:,1]),
                                np.unique(self.XYZ[:,2]), indexing = 'ij')
        
        
        kp_reduced_to_energy = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,self.bands[:,iband])}
        kp_reduced_to_kp_cart = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,kpoints_cart)}
        kp_reduced_to_mesh_index = {f'({kp[0]},{kp[1]},{kp[2]})': np.argwhere((mesh_list[0]==kp[0]) & 
                                                                              (mesh_list[1]==kp[1]) & 
                                                                              (mesh_list[2]==kp[2]) )[0] for kp in self.XYZ}
        
        
        energy_meshgrid = energy_meshgrid_mapping(mesh_list) 

        change_in_energy = np.gradient(energy_meshgrid)

        grad_x_list = [change_in_energy[0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                  kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                  kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                  ]
                  for kp in self.XYZ]
        grad_y_list = [change_in_energy[1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                              kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                              kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                              ]
                  for kp in self.XYZ]
        grad_z_list = [change_in_energy[2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                              kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                              kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                              ]
                  for kp in self.XYZ]
        
        
        gradient_list = np.array([grad_x_list,grad_y_list,grad_z_list]).T

        lattice = np.linalg.inv(self.reciprocal_lattice.T).T

      
        gradient_list = gradient_list/(2*math.pi)
        gradient_list = np.multiply(gradient_list, np.array([np.linalg.norm(lattice[:,0])*METER_ANGSTROM * len(np.unique(self.XYZ[:,0])),
                                                             np.linalg.norm(lattice[:,1])*METER_ANGSTROM * len(np.unique(self.XYZ[:,1])),
                                                             np.linalg.norm(lattice[:,2])*METER_ANGSTROM * len(np.unique(self.XYZ[:,2]))])
                            )
        
        gradient_list_cart = np.array([np.matmul(lattice, gradient) for gradient in gradient_list])
        
 
        
        # kp_reduced_to_grad_x = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,grad_x_list)}
        # kp_reduced_to_grad_y = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,grad_y_list)}
        # kp_reduced_to_grad_z = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,grad_z_list)}
        
        group_velocity_x = gradient_list_cart[:,0]/HBAR_EV
        group_velocity_y = gradient_list_cart[:,1]/HBAR_EV
        group_velocity_z = gradient_list_cart[:,2]/HBAR_EV
        
        group_velocity_vector = [group_velocity_x,group_velocity_y,group_velocity_z]
        
        group_velocity_magnitude = (group_velocity_x**2 + 
                                    group_velocity_y**2 + 
                                    group_velocity_z**2)**0.5

        kp_reduced_to_gradient = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.XYZ,gradient_list_cart)}
        def get_gradient(kp_reduced):
            return kp_reduced_to_gradient[f'({kp_reduced[0]},{kp_reduced[1]},{kp_reduced[2]})']
        
        gradient_mesh = list(fermi_velocity_meshgrid_mapping(mesh_list))
        change_in_energy_gradient = list(map(np.gradient,gradient_mesh))
        
        
        
        grad_xx_list = [change_in_energy_gradient[0][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        grad_xy_list = [change_in_energy_gradient[0][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        grad_xz_list = [change_in_energy_gradient[0][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        
        grad_yx_list = [change_in_energy_gradient[1][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        grad_yy_list = [change_in_energy_gradient[1][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        grad_yz_list = [change_in_energy_gradient[1][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        
        grad_zx_list = [change_in_energy_gradient[2][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        grad_zy_list = [change_in_energy_gradient[2][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        grad_zz_list = [change_in_energy_gradient[2][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
                                                        ]
                        for kp in self.XYZ]
        
        energy_second_derivative_list = np.array([[grad_xx_list,grad_xy_list,grad_xz_list],
                                         [grad_yx_list,grad_yy_list,grad_yz_list],
                                         [grad_zx_list,grad_zy_list,grad_zz_list],
                                         ])
        
        energy_second_derivative_list = np.swapaxes(energy_second_derivative_list,axis1=0,axis2=2)
        energy_second_derivative_list = np.swapaxes(energy_second_derivative_list,axis1=1,axis2=2)
        energy_second_derivative_list = energy_second_derivative_list/(2*math.pi)
        energy_second_derivative_list = np.multiply(energy_second_derivative_list, np.array([np.linalg.norm(lattice[:,0])*METER_ANGSTROM * len(np.unique(self.XYZ[:,0])),
                                                                                                       np.linalg.norm(lattice[:,1])*METER_ANGSTROM * len(np.unique(self.XYZ[:,1])),
                                                                                                       np.linalg.norm(lattice[:,2])*METER_ANGSTROM * len(np.unique(self.XYZ[:,2]))])
                                                    )
        energy_second_derivative_list_cart = energy_second_derivative_list_cart = np.array([ [ np.matmul(lattice,gradient) for gradient in gradient_tensor ] for gradient_tensor in energy_second_derivative_list ])
        
        inverse_effective_mass_tensor_list = energy_second_derivative_list_cart * EV_TO_J/ HBAR_J**2
        effective_mass_tensor_list = np.array([ np.linalg.inv(inverse_effective_mass_tensor) for inverse_effective_mass_tensor in inverse_effective_mass_tensor_list])
        effective_mass_list = np.array([ (3*(1/effective_mass_tensor_list[ikp,0,0] + 
                                                  1/effective_mass_tensor_list[ikp,1,1] +
                                                  1/effective_mass_tensor_list[ikp,2,2])**-1)/FREE_ELECTRON_MASS for ikp in range(len(self.XYZ))])
        return (group_velocity_vector, group_velocity_magnitude, effective_mass_tensor_list,
                effective_mass_list, gradient_list, gradient_list_cart)
    
    def calculate_first_and_second_derivative_energy(self):
        
        self.first_and_second_derivative_energy_property_dict = {}
        for iband in range(len(self.bands[0,:])):
            group_velocity_vector,group_velocity_magnitude,effective_mass_tensor_list,effective_mass_list,gradient_list, gradient_list_cart = self.calculate_first_and_second_derivative_energy_band(iband)
            self.first_and_second_derivative_energy_property_dict.update({f"band_{iband}" : {"group_velocity_vector" : group_velocity_vector,
                                                                                        "group_velocity_magnitude": group_velocity_magnitude,
                                                                                        "effective_mass_tensor_list":effective_mass_tensor_list,
                                                                                        "effective_mass_list":effective_mass_list}
                                                                })
            


    def _get_brilloin_zone(self, 
                        supercell: List[int]):
        return BrillouinZone(self.reciprocal_lattice, supercell)

