__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import random
import math
import sys
import copy
import itertools
from typing import List, Tuple, Union

import numpy as np
import scipy.interpolate as interpolate
from scipy.spatial import KDTree
from matplotlib import colors as mpcolors
from matplotlib import cm

from . import Isosurface, Surface, BrillouinZone
from pyprocar.utils import LOGGER
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
        max_distance:float=0.2,
        ):
        LOGGER.info(f'___Initializing the FermiSurface3D object___')

        self.ebs = copy.copy(ebs)

        LOGGER.debug(f"ebs.bands shape: {self.ebs.bands.shape}")
        if (self.ebs.bands.shape)==3:
            raise "Must reduce ebs.bands into 2d darray"
        
        # Shifts kpoints between [-0.5,0.5)
        self.ebs.kpoints = -np.fmod(self.ebs.kpoints + 6.5, 1 ) + 0.5

        LOGGER.debug(f"ebs.kpoints shape: {self.ebs.kpoints.shape}")
        LOGGER.debug(f"First 3 ebs.kpoints: {self.ebs.kpoints[:3]}")

        self.supercell = np.array(supercell)
        self.fermi = fermi + fermi_shift
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy
        self.max_distance=max_distance

        LOGGER.info(f'Iso-value used to find isosurfaces: {self.fermi}')
        LOGGER.info(f'Interpolation factor: {self.interpolation_factor}')
        LOGGER.info(f'Projection accuracy: {self.projection_accuracy}')
        LOGGER.info(f'Supercell used to calculate the FermiSurface3D: {self.supercell}')
        LOGGER.info(f'Maximum distance to keep points from isosurface centers: {self.max_distance}')

        # Preocessing steps
        self._input_checks()
        self.brillouin_zone = self._get_brillouin_zone(self.supercell)
        self.isosurfaces = self._generate_isosurfaces()
        self.surface = self._combine_isosurfaces()

        # Initialize the Fermi Surface
        super().__init__(verts=self.surface.points, faces=self.surface.faces)

        # Storing band indices in the initialized surface
        self.point_data['band_index'] = self.surface.point_data['band_index']
        self.cell_data['band_index'] = self.surface.cell_data['band_index'] 
        self.fermi_surface_area = self.area

        LOGGER.info(f'___End of initialization FermiSurface3D object___')
        return None
    
    def _input_checks(self):
        assert len(self.ebs.bands.shape)==2

    def _generate_isosurfaces(self):
        LOGGER.info(f'____Generating isosurfaces for each band___')
        isosurfaces=[]
        self.band_isosurface_index_map={}
        for iband in range(self.ebs.bands.shape[1]):
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
            if isosurface_band.points.shape[0] == 0:
                continue

            LOGGER.debug(f"Found isosurface with {isosurface_band.points.shape[0]} points")

            isosurfaces.append(isosurface_band)
            n_isosurface=len(isosurfaces)-1
            self.band_isosurface_index_map[iband]=n_isosurface

        self.isosurface_band_index_map={value:key for key,value in self.band_isosurface_index_map.items()}
        band_to_surface_indices=list(self.band_isosurface_index_map.keys())
        self.ebs.bands=self.ebs.bands[:,band_to_surface_indices]

        LOGGER.debug(f"self.ebs.bands shape: {self.ebs.bands.shape} after removing bands with no isosurfaces")
        LOGGER.info(f'Band Isosurface index map: {self.band_isosurface_index_map}')
        LOGGER.info(f'____End of generating isosurfaces for each band___')
        return isosurfaces
    
    def _combine_isosurfaces(self):
        LOGGER.info(f'____Combining isosurfaces___')
        isosurfaces=copy.deepcopy(self.isosurfaces)

        surface_band_indices_points=[]
        surface_band_indices_cells=[]
        surface=None
        for i_surface,isosurface in enumerate(isosurfaces[:]):
            LOGGER.info(f"Number of points on isosurface {i_surface} isosurface: {isosurface.points.shape[0]}")
            n_points=isosurface.points.shape[0]
            n_cells=isosurface.n_cells
            
            if i_surface == 0:
                surface=isosurface
            else:
                # The points are prepended to surface.points, 
                # so at the end we need to reverse this list
                surface.merge(isosurface, merge_points=False, inplace=True)
                LOGGER.info(f"Number of points after merging isosurface {i_surface} isosurface to surface: {surface.points.shape[0]}")
            
            band_index_points_list=[i_surface]*n_points
            surface_band_indices_points.extend(band_index_points_list)

            band_indices_cells_list=[i_surface]*n_cells
            surface_band_indices_cells.extend(band_indices_cells_list)

        if surface == None:
            raise ValueError("No Fermi surface found. The structure is probably not metallic, so there will be no Fermi surface.")
        
        # Setting band indices on the points of the combined surfaces
        surface_band_indices_points.reverse()
        surface.point_data['band_index']=np.array(surface_band_indices_points)

        # Setting band indices on the cells of the combined surfaces
        surface_band_indices_cells.reverse()
        surface.cell_data['band_index']=np.array(surface_band_indices_cells)

  
        LOGGER.info(f'___End of combining isosurfaces___')
        return surface
    
    def _get_brillouin_zone(self, 
                        supercell: List[int]):

        """Returns the BrillouinZone of the material
        brillouin_zone

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
        LOGGER.info(f"____Starting Projecting vector texture___")
        
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

            near_isosurface_point=self._keep_points_near_subset(XYZ_transformed,isosurface.points,max_distance=self.max_distance)
            XYZ_transformed=XYZ_transformed[near_isosurface_point]
            vectors_extended_X=vectors_extended_X[near_isosurface_point]
            vectors_extended_Y=vectors_extended_Y[near_isosurface_point]
            vectors_extended_Z=vectors_extended_Z[near_isosurface_point]

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

            # Must flip here because when the values are stored in cell_data, 
            # the values are entered preprended to the cell_data array
            # and are stored in the opposite order of what you would expect
            vectors_X=np.flip(vectors_X,axis=0)
            vectors_Y=np.flip(vectors_Y,axis=0)
            vectors_Z=np.flip(vectors_Z,axis=0)

            final_vectors_X.extend(vectors_X)
            final_vectors_Y.extend(vectors_Y)
            final_vectors_Z.extend(vectors_Z)

        # Again must flip here because when the values are stored in cell_data, 
        # the values are entered preprended to the cell_data array
        # and are stored in the opposite order of what you would expect
        final_vectors_X.reverse()
        final_vectors_Y.reverse()
        final_vectors_Z.reverse()
                
        self.set_vectors(final_vectors_X, final_vectors_Y, final_vectors_Z,vectors_name = vectors_name)
        LOGGER.info(f'___End of projecting vector texture___')
        return None
    
    @staticmethod
    def _keep_points_inside_brillouin_zone(xyz,brillouin):
        """
        This method will keep points inside the brillouin zone
        
        Parameters
        ----------
        xyz : np.ndarray
            These points have to be in cartersian coordinates
        brilloin_zone : BrillouinZone
            The brillouin zone
        
        Returns
        -------
        xyz : np.ndarray
            The xyz array
        """

        face_centers = brillouin.centers
        face_normals = brillouin.face_normals

        LOGGER.debug(f"Brillouin Zone Face Centers: {face_centers}")
        LOGGER.debug(f"Brillouin Zone Face Normals: {face_normals}")

        # Initialize an array to store whether each point is inside the zone
        is_inside = np.ones(len(xyz), dtype=bool)

        # Check each face of the Brillouin zone
        for center, normal in zip(face_centers, face_normals):
            # Calculate the signed distance from each point to the plane
            distance = np.dot(xyz - center, normal)
            
            # Update is_inside: point is inside if distance is negative for all faces
            is_inside &= (distance <= 0)

        # Return only the points that are inside the Brillouin zone
        return is_inside
    
    @staticmethod
    def _keep_points_near_subset(points, subset, max_distance=0.2):
        """
        Keep only the points that are within a specified distance of any point in the subset.

        Parameters
        ----------
        points : np.        Array of shape (n, 3) containing all points to be filtered.
        subset : np.ndarray
            Array of shape (m, 3) containing the subset of points to compare against.
        max_distance : float
            The maximum distance for a point to be considered "
        Returns
        -------
        np.ndarray
            Array of shape (k, 3) containing only the points that are near the subset,
            where k <= n.
        """


        # Create a KDTree for efficient nearest neighbor search
        tree = KDTree(subset)

        # Find the distance to the    
        distances, _ = tree.query(points, k=3)

        # Create a boolean mask for points within the max_distance
        mask = np.ones(distances.shape[0], dtype=bool)
        n_neighbors=distances.shape[1]
        for i_neighbor in range(n_neighbors):
            mask &= (distances[:,i_neighbor] <= max_distance)

        # Return only the points that satisfy the distance criterion
        return mask

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
        LOGGER.info(f"____Starting Projecting atomic projections___")

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
            LOGGER.debug(f"Number of points before projecting inside the Brillouin zone: {len(XYZ_transformed)}")
       

            near_isosurface_point=self._keep_points_near_subset(XYZ_transformed,isosurface.centers,max_distance=self.max_distance)
            XYZ_transformed=XYZ_transformed[near_isosurface_point]
            scalars_extended=scalars_extended[near_isosurface_point]

            LOGGER.debug(f"Number of points after determining which points are near isosurface centers: {len(XYZ_transformed)}")
            LOGGER.debug(f"Number of scalars after determining which points are near isosurface centers: {len(scalars_extended)}")

            if self.projection_accuracy.lower()[0] == "n":
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, isosurface.centers, method="nearest"
                )
            elif self.projection_accuracy.lower()[0] == "h":
                colors = interpolate.griddata(
                    XYZ_transformed, scalars_extended, isosurface.centers, method="linear"
                )
     
            # Again must flip here because when the values are stored in cell_data, 
            # the values are entered preprended to the cell_data array
            # and are stored in the opposite order of what you would expect
            colors=np.flip(colors,axis=0)
            final_scalars.extend(colors)

        # Again, you must flip the values here because the 
        # values are stored in the opposite order of what you would expect
        final_scalars.reverse()

        self.set_scalars(final_scalars, scalar_name = scalar_name)
        LOGGER.info(f'___End of projecting scalars___')
        return None
  
    def project_atomic_projections(self,spd):
        """
        Method to calculate the atomic projections of the surface.
        """
        LOGGER.info(f"____Starting Projecting atomic projections___")
        LOGGER.debug(f"spd shape at this point: {spd.shape}")
        LOGGER.debug(f"First 5 spd values coresponding to the first 5 kpoints: {spd[:5,:]}")

        scalars_array = []
        count = 0
        for iband in range(len(self.isosurfaces)):
            count+=1
            scalars_array.append(spd[:,iband])
        scalars_array = np.vstack(scalars_array).T

        LOGGER.info(f"First 5 scalar array coresponding to the first 5 kpoints: {scalars_array[:5,:]}")
        LOGGER.info(f'scalars_array shape after the creation of the array from the spd: {scalars_array.shape}')

        self._project_color(scalars_array = scalars_array, scalar_name = "scalars")

        LOGGER.info(f"____Ending Projecting atomic projections___")

    def project_spin_texture_atomic_projections(self,spd_spin):
        """
        Method to calculate atomic spin texture projections of the surface.
        """
        LOGGER.info(f"____Starting Projecting spin texture___")
        vectors_array = spd_spin
        self._create_vector_texture(vectors_array = vectors_array, vectors_name = "spin" )
        LOGGER.info(f'___End of projecting spin texture___')

    def project_fermi_velocity(self,fermi_velocity):
        """
        Method to calculate atomic spin texture projections of the surface.
        """
        LOGGER.info(f"____Starting Projecting fermi velocity___")
        vectors_array = fermi_velocity.swapaxes(1, 2)
        self._create_vector_texture(vectors_array = vectors_array, vectors_name = "Fermi Velocity Vector" )
        LOGGER.info(f'___End of projecting fermi velocity___')

    def project_fermi_speed(self,fermi_speed):
        """
        Method to calculate the fermi speed of the surface.
        """
        LOGGER.info(f"____Starting Projecting fermi speed___")
        scalars_array = []
        count = 0
        for iband in range(len(self.isosurfaces)):
            count+=1
            scalars_array.append(fermi_speed[:,iband])
        scalars_array = np.vstack(scalars_array).T
        self._project_color(scalars_array = scalars_array, scalar_name = "Fermi Speed")
        LOGGER.info(f'___End of projecting fermi speed___')

    def project_harmonic_effective_mass(self,harmonic_effective_mass):
        """
        Method to calculate the atomic projections of the surface.
        """
        LOGGER.info(f"____Starting Projecting harmonic effective mass___")
        scalars_array = []
        count = 0
        for iband in range(len(self.isosurfaces)):
            count+=1
            scalars_array.append(harmonic_effective_mass[:,iband])
        scalars_array = np.vstack(scalars_array).T
        self._project_color(scalars_array = scalars_array, scalar_name = "Harmonic Effective Mass" )
        LOGGER.info(f'___End of projecting harmonic effective mass___')

    def extend_surface(self,  extended_zone_directions: List[Union[List[int],Tuple[int,int,int]]]=None,):
        """
        Method to extend the surface in the direction of a reciprocal lattice vecctor

        Parameters
        ----------
        extended_zone_directions : List[List[int] or Tuple[int,int,int]], optional
            List of directions to expand to, by default None
        """
        LOGGER.info(f"____Starting extending surface___")
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
        LOGGER.info(f'___End of extending surface___')
 



