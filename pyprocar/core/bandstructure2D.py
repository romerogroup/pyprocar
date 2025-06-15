__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import copy
import math
import random
import sys
from itertools import product
from typing import List, Tuple, Union

import numpy as np
import pyvista as pv
import scipy.interpolate as interpolate
from matplotlib import cm
from matplotlib import colors as mpcolors
from scipy.spatial import KDTree

from pyprocar.core.brillouin_zone import BrillouinZone2D
from pyprocar.core.surface import Surface

np.set_printoptions(threshold=sys.maxsize)

# TODO: method to reduce number of points for interpolation need to be modified since the tolerance on the space
# is not lonmg soley reciprocal space, but energy and reciprocal space


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
        interpolation_factor: int = 1,
        projection_accuracy: str = "Normal",
        supercell: List[int] = [1, 1, 1],
        zlim=None,
    ):

        self.ebs = copy.copy(ebs)
        self.ispin = ispin

        self.n_bands = self.ebs.bands.shape[1]
        self.supercell = np.array(supercell)
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy

        self.brillouin_zone = self._get_brilloin_zone(self.supercell, zlim=zlim)

        grid_cart_x = self.ebs.kpoints_cartesian_mesh[:, :, 0, 0]
        grid_cart_y = self.ebs.kpoints_cartesian_mesh[:, :, 0, 1]

        self.band_surfaces = self._generate_band_structure_2d(grid_cart_x, grid_cart_y)
        self.surface = self._combine_band_surfaces()

        # Initialize the Fermi Surface
        super().__init__(verts=self.surface.points, faces=self.surface.faces)
        self.point_data["band_index"] = self.surface.point_data["band_index"]
        return None

    def _generate_band_structure_2d(self, grid_cart_x, grid_cart_y):
        surfaces = []
        n_bands = self.ebs.bands_mesh.shape[3]
        n_points = 0
        for iband in range(n_bands):
            grid_z = self.ebs.bands_mesh[:, :, 0, iband, self.ispin]

            surface = pv.StructuredGrid(grid_cart_x, grid_cart_y, grid_z)
            surface = surface.cast_to_unstructured_grid()
            surface = surface.extract_surface()

            surface = Surface(verts=surface.points, faces=surface.faces)
            n_points += surface.points.shape[0]

            band_index_list = [iband] * surface.points.shape[0]
            surface.point_data["band_index"] = np.array(band_index_list)
            surfaces.append(surface)

        return surfaces

    def _combine_band_surfaces(self):
        band_surfaces = copy.deepcopy(self.band_surfaces)

        band_indices = []
        surface = None
        for i_band, band_surface in enumerate(band_surfaces):
            # The points are prepended to surface.points,
            # so at the end we need to reverse this list
            if i_band == 0:
                surface = band_surface
            else:
                surface.merge(band_surface, merge_points=False, inplace=True)

            band_index_list = [i_band] * band_surface.points.shape[0]
            band_indices.extend(band_index_list)

        band_indices.reverse()
        surface.point_data["band_index"] = np.array(band_indices)
        return surface

    @staticmethod
    def _keep_points_near_subset(points, subset, max_distance=0.3):
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
        n_neighbors = distances.shape[1]
        for i_neighbor in range(n_neighbors):
            mask &= distances[:, i_neighbor] <= max_distance

        # Return only the points that satisfy the distance criterion
        return mask

    def _create_vector_texture(
        self, vectors_array: np.ndarray, vectors_name: str = "vector"
    ):
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
            XYZ_extended[:, 2] = self.ebs.bands[:, iband, self.ispin]

            vectors_extended_X = vectors_array[:, iband, 0].copy()
            vectors_extended_Y = vectors_array[:, iband, 1].copy()
            vectors_extended_Z = vectors_array[:, iband, 2].copy()

            XYZ_transformed = XYZ_extended

            near_isosurface_point = self._keep_points_near_subset(
                XYZ_transformed, isosurface.points
            )
            XYZ_transformed = XYZ_transformed[near_isosurface_point]
            vectors_extended_X = vectors_extended_X[near_isosurface_point]
            vectors_extended_Y = vectors_extended_Y[near_isosurface_point]
            vectors_extended_Z = vectors_extended_Z[near_isosurface_point]

            if self.projection_accuracy.lower()[0] == "n":

                vectors_X = interpolate.griddata(
                    XYZ_transformed,
                    vectors_extended_X,
                    isosurface.points,
                    method="nearest",
                )
                vectors_Y = interpolate.griddata(
                    XYZ_transformed,
                    vectors_extended_Y,
                    isosurface.points,
                    method="nearest",
                )
                vectors_Z = interpolate.griddata(
                    XYZ_transformed,
                    vectors_extended_Z,
                    isosurface.points,
                    method="nearest",
                )

            elif self.projection_accuracy.lower()[0] == "h":

                vectors_X = interpolate.griddata(
                    XYZ_transformed,
                    vectors_extended_X,
                    isosurface.points,
                    method="linear",
                )
                vectors_Y = interpolate.griddata(
                    XYZ_transformed,
                    vectors_extended_Y,
                    isosurface.points,
                    method="linear",
                )
                vectors_Z = interpolate.griddata(
                    XYZ_transformed,
                    vectors_extended_Z,
                    isosurface.points,
                    method="linear",
                )

            # Again must flip here because when the values are stored in cell_data,
            # the values are entered preprended to the cell_data array
            # and are stored in the opposite order of what you would expect
            vectors_X = np.flip(vectors_X, axis=0)
            vectors_Y = np.flip(vectors_Y, axis=0)
            vectors_Z = np.flip(vectors_Z, axis=0)
            final_vectors_X.extend(vectors_X)
            final_vectors_Y.extend(vectors_Y)
            final_vectors_Z.extend(vectors_Z)

        final_vectors_X.reverse()
        final_vectors_Y.reverse()
        final_vectors_Z.reverse()

        self.set_vectors(
            final_vectors_X, final_vectors_Y, final_vectors_Z, vectors_name=vectors_name
        )
        return None

    def _project_color(self, scalars_array: np.ndarray, scalar_name: str = "scalars"):
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
            XYZ_extended[:, 2] = self.ebs.bands[:, iband, self.ispin]
            scalars_extended = scalars_array[:, iband].copy()
            XYZ_transformed = XYZ_extended

            near_isosurface_point = self._keep_points_near_subset(
                XYZ_transformed, isosurface.centers
            )
            XYZ_transformed = XYZ_transformed[near_isosurface_point]
            scalars_extended = scalars_extended[near_isosurface_point]

            if self.projection_accuracy.lower()[0] == "n":
                colors = interpolate.griddata(
                    XYZ_transformed,
                    scalars_extended,
                    isosurface.centers,
                    method="nearest",
                )
            elif self.projection_accuracy.lower()[0] == "h":
                colors = interpolate.griddata(
                    XYZ_transformed,
                    scalars_extended,
                    isosurface.centers,
                    method="linear",
                )
            # Again must flip here because when the values are stored in cell_data,
            # the values are entered preprended to the cell_data array
            # and are stored in the opposite order of what you would expect
            colors = np.flip(colors, axis=0)
            final_scalars.extend(colors)
        final_scalars.reverse()

        self.set_scalars(final_scalars, scalar_name=scalar_name)
        return None

    def project_atomic_projections(self, spd):
        """
        Method to calculate the atomic projections of the surface.
        """
        scalars_array = []
        count = 0
        for iband in range(self.n_bands):
            count += 1
            scalars_array.append(spd[:, iband])
        scalars_array = np.vstack(scalars_array).T

        self._project_color(scalars_array=scalars_array, scalar_name="scalars")

    def project_spin_texture_atomic_projections(self, spd_spin):
        """
        Method to calculate atomic spin texture projections of the surface.
        """
        vectors_array = spd_spin
        self._create_vector_texture(vectors_array=vectors_array, vectors_name="spin")

    def project_band_velocity(self, band_velocity):
        """
        Method to calculate band velocity of the surface.
        """
        vectors_array = band_velocity
        self._create_vector_texture(
            vectors_array=vectors_array, vectors_name="Band Velocity Vector"
        )

    def project_band_speed(self, band_speed):
        """
        Method to calculate the fermi speed of the surface.
        """
        scalars_array = []
        count = 0
        for iband in range(self.n_bands):
            count += 1
            scalars_array.append(band_speed[:, iband])
        scalars_array = np.vstack(scalars_array).T
        self._project_color(scalars_array=scalars_array, scalar_name="Band Speed")

    def project_avg_inv_effective_mass(self, avg_inv_effective_mass):
        """
        Method to calculate the atomic projections of the surface.
        """
        scalars_array = []
        count = 0
        for iband in range(self.n_bands):
            count += 1
            scalars_array.append(avg_inv_effective_mass[:, iband])
        scalars_array = np.vstack(scalars_array).T
        self._project_color(
            scalars_array=scalars_array, scalar_name="Avg Inverse Effective Mass"
        )

    def extend_surface(
        self,
        extended_zone_directions: List[Union[List[int], Tuple[int, int, int]]] = None,
    ):
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

                self += surface.translate(
                    np.dot(direction, self.ebs.reciprocal_lattice), inplace=True
                )
            # Clearing unneeded surface from memory
            del surface

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
