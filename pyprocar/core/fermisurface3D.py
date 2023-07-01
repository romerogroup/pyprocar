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
        kpoints: np.ndarray,
        bands: np.ndarray,
        fermi: float,
        reciprocal_lattice: np.ndarray,
        ebs=None,
        bands_to_keep: List[int]=None,
        spd: np.ndarray=None,
        spd_spin:np.ndarray=None,

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
        self.XYZ = np.fmod(self.kpoints + 6.5, 1 ) - 0.5
        print(self.XYZ.shape)
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
        
        # Finding bands with a fermi iso-surface. This reduces searching
        fullBandIndex = []
        reducedBandIndex = []
        for iband in range(len(self.bands[0,:])):
            fermi_surface_test = len(np.where(np.logical_and(self.bands[:,iband]>=self.fermi-fermi_tolerance, self.bands[:,iband]<=self.fermi+fermi_tolerance))[0])
            if fermi_surface_test != 0:
                fullBandIndex.append(iband)
        if len(fullBandIndex)==0:
            raise Exception("No bands within tolerance. Increase tolerance to increase search space.")
        self.bands = self.bands[:,fullBandIndex]
        
        # re-index and creates a mapping to the original bandindex
        reducedBandIndex = np.arange(len(self.bands[0,:]))
        self.fullBandIndex = fullBandIndex
        self.reducedBandIndex = reducedBandIndex
        self.reducedBandIndex_to_fullBandIndex = {f"{key}":value for key,value in zip(reducedBandIndex,fullBandIndex)}
        self.fullBandIndex_to_reducedBandIndex = {f"{key}":value for key,value in zip(fullBandIndex,reducedBandIndex)}
        reduced_bands_to_keep_index = [iband for iband in range(len(bands_to_keep))]

        # Generate unique rgba values for the bands
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
            norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap(cmap)
            solid_color_surface = np.arange(nsurface ) / nsurface
            band_colors = np.array([cmap(norm(x)) for x in solid_color_surface[:]]).reshape(-1, 4)
        
        # The following loop generates iso surfaces for each band and then stores them in a list
        color_band_dict = {}

        self.isosurfaces = []
        full_isosurface = None
        iband_with_surface=0
        for iband in  range(self.bands.shape[1]):
            isosurface_band = Isosurface(
                                XYZ=self.XYZ,
                                V=self.bands[:,iband],
                                isovalue=self.fermi,
                                algorithm="lewiner",
                                interpolation_factor=interpolation_factor,
                                padding=self.supercell,
                                transform_matrix=self.reciprocal_lattice,
                                boundaries=self.brillouin_zone,
                            )

            # Following condition will handle initializing of fermi isosurface
            isosurface_band_copy = copy.deepcopy(isosurface_band)
            if full_isosurface is None and len(isosurface_band_copy.points)!=0:
                full_isosurface = isosurface_band_copy    
            elif len(isosurface_band_copy.points)==0:
                if full_isosurface is None and iband == len(self.bands[0,:])-1:
                    # print(full_isosurface)
                    raise Exception("Could not find any fermi surfaces")
                continue
            else:    
                full_isosurface.merge(isosurface_band_copy, merge_points=False, inplace=True) 

            color_band_dict.update({f"band_{iband_with_surface}": {"color" : band_colors[iband_with_surface,:]} })
            band_color = np.array([band_colors[iband_with_surface,:]]*len(isosurface_band.points[:,0]))
            isosurface_band.point_data[f"band_{iband_with_surface}"] = band_color
            self.isosurfaces.append(isosurface_band)
            iband_with_surface +=1

        # Initialize the Fermi Surface which is the combination of all the 
        # isosurface for each band
        super().__init__(verts=full_isosurface.points, faces=full_isosurface.faces)
        # super().__init__(var_inp=full_isosurface.points, faces=full_isosurface.faces)
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
        

    def create_vector_texture(self,
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
            
    def create_spin_texture(self):
        """
        This method will create the spin textures for the 3d fermi surface in the case of a non-colinear calculation
        """
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



            XYZ_transformed = np.dot(XYZ_extended, self.reciprocal_lattice)
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
            return None
   
    def project_color(self, 
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
        # self.set_color_with_cmap(cmap, vmin, vmax)
  
    def calculate_first_and_second_derivative_energy_band(self, 
                                                        iband: int):
        """Helper method to calculate the first and second derivative of a band

        Parameters
        ----------
        iband : int
            The band index
        """

        
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
        """
        Helper method which calculates the first and secoond derivative of all bands.
        """
        
        self.first_and_second_derivative_energy_property_dict = {}
        for iband in range(len(self.bands[0,:])):
            group_velocity_vector,group_velocity_magnitude,effective_mass_tensor_list,effective_mass_list,gradient_list, gradient_list_cart = self.calculate_first_and_second_derivative_energy_band(iband)
            self.first_and_second_derivative_energy_property_dict.update({f"band_{iband}" : {"group_velocity_vector" : group_velocity_vector,
                                                                                        "group_velocity_magnitude": group_velocity_magnitude,
                                                                                        "effective_mass_tensor_list":effective_mass_tensor_list,
                                                                                        "effective_mass_list":effective_mass_list}
                                                                })

    def calculate_fermi_velocity(self):
        """
        Method to calculate the fermi velocity of the surface.
        """
        self.calculate_first_and_second_derivative_energy()

        vectors_array = []
        for band_name in self.first_and_second_derivative_energy_property_dict.keys():
            vectors_array.append(self.first_and_second_derivative_energy_property_dict[band_name]["group_velocity_vector"])
            
        vectors_array = np.array(vectors_array).T
        vectors_array = np.swapaxes(vectors_array,axis1 = 1,axis2 = 2)

        self.create_vector_texture( vectors_array = vectors_array, vectors_name = "Fermi Velocity Vector"  )  

    def calculate_fermi_speed(self):
        """
        Method to calculate the fermi speed of the surface.
        """
        self.calculate_first_and_second_derivative_energy()

        scalars_array = []
        for band_name in self.first_and_second_derivative_energy_property_dict.keys():
            scalars_array.append(self.first_and_second_derivative_energy_property_dict[band_name]["group_velocity_magnitude"])
            
        scalars_array = np.vstack(scalars_array).T 
        
        self.project_color(scalars_array = scalars_array, scalar_name = "Fermi Speed")

    def calculate_effective_mass(self):
        """
        Method to calculate the effective mass of the surface.
        """
        self.calculate_first_and_second_derivative_energy()

        scalars_array = []
        for band_name in self.first_and_second_derivative_energy_property_dict.keys():
            scalars_array.append(self.first_and_second_derivative_energy_property_dict[band_name]["effective_mass_list"])
        scalars_array = np.vstack(scalars_array).T 
        
        self.project_color(scalars_array = scalars_array, scalar_name="Geometric Average Effective Mass")

    def project_atomic_projections(self):
        """
        Method to calculate the atomic projections of the surface.
        """

        
        self.spd = self.spd[:,self.fullBandIndex]

        scalars_array = []
        count = 0
        for iband in range(len(self.isosurfaces)):
            count+=1
            scalars_array.append(self.spd[:,iband])
        scalars_array = np.vstack(scalars_array).T
        self.project_color(scalars_array = scalars_array, scalar_name = "scalars")

    def project_spin_texture_atomic_projections(self):
        """
        Method to calculate atomic spin texture projections of the surface.
        """
        
        if self.spd_spin[0] is not None:
            self.spd_spin = self.spd_spin[:,self.fullBandIndex,:]

        vectors_array = self.spd_spin

        self.create_vector_texture(vectors_array = vectors_array, vectors_name = "spin" )

    def extend_surface(self,  extended_zone_directions: List[List[int] or Tuple[int,int,int]]=None,):
        """
        Method to extend the surface in the direction of a reciprocal lattice vecctor

        Parameters
        ----------
        extended_zone_directions : List[List[int] or Tuple[int,int,int]], optional
            List of directions to expand to, by default None
        """
        import pyvista as pv
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

