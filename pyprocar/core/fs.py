import copy
import logging
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import pyvista as pv
from scipy import interpolate

from pyprocar import io
from pyprocar.cfg.fermi_surface_3d import FermiSurface3DConfig
from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.utils.physics_constants import *

logger = logging.getLogger("pyprocar")


class FermiSurface(pv.PolyData):
    """
    A class to generate and manipulate Fermi surfaces from electronic band structure data.
    Extends PyVista's PolyData class for rich 3D visualization capabilities.

    Parameters
    ----------
    ebs : ElectronicBandStructure
        The electronic band structure object containing band data
    fermi : float, optional
        The Fermi energy level, by default 0.0
    fermi_shift : float, optional
        Energy shift to apply to the Fermi level, by default 0.0
    interpolation_factor : int, optional
        Factor to increase k-point density, by default 1
    projection_accuracy : str, optional
        Accuracy for projections ('high' or 'normal'), by default 'normal'
    max_distance : float, optional
        Maximum distance for point projections, by default 0.2
    """

    def __init__(
        self,
        ebs,
        fermi: float = 0.0,
        fermi_shift: float = 0.0,
        interpolation_factor: int = 1,
        projection_accuracy: str = "normal",
        max_distance: float = 0.2,
        padding: int = 5,
        config: FermiSurface3DConfig = None,
    ):
        super().__init__()
        logger.info("___Initializing FermiSurface object___")

        # Store input parameters
        # self.ebs = ebs

        self.padding = padding
        self._ebs = ebs
        self._padded_ebs = ebs.pad(padding=self.padding, inplace=False)

        self.fermi_shift = fermi_shift
        self.fermi = fermi + fermi_shift
        self.interpolation_factor = interpolation_factor
        self.projection_accuracy = projection_accuracy
        self.max_distance = max_distance

        logger.debug(f"Electronic Band Structure Info: \n")
        logger.debug(f"Bands shape: {self.ebs}")
        logger.debug(f"Fermi Shift: {self.fermi_shift}")
        logger.debug(f"Fermi energy: {self.fermi}")
        logger.debug(f"Interpolation factor: {self.interpolation_factor}")
        logger.debug(f"Projection accuracy: {self.projection_accuracy}")
        logger.debug(f"Max distance: {self.max_distance}")

        # Initialize storage for surface data
        self.band_spin_surface_map = {}
        self.surface_band_spin_map = {}
        self.band_spin_mask = {}

        # Initialize ImageData Grid
        self.nkx, self.nky, self.nkz = self.ebs.n_kx, self.ebs.n_ky, self.ebs.n_kz

        logger.debug(f"nkx: {self.nkx}")
        logger.debug(f"nky: {self.nky}")
        logger.debug(f"nkz: {self.nkz}")

        self.padded_x_unique = np.unique(self.padded_ebs.kpoints[:, 0])
        self.padded_y_unique = np.unique(self.padded_ebs.kpoints[:, 1])
        self.padded_z_unique = np.unique(self.padded_ebs.kpoints[:, 2])

        self.padded_x_min = self.padded_x_unique.min()
        self.padded_y_min = self.padded_y_unique.min()
        self.padded_z_min = self.padded_z_unique.min()

        self.x_spacing = 1 / self.nkx
        self.y_spacing = 1 / self.nky
        self.z_spacing = 1 / self.nkz

        self.padded_grid = pv.ImageData(
            dimensions=(
                self.padded_ebs.n_kx,
                self.padded_ebs.n_ky,
                self.padded_ebs.n_kz,
            ),
            spacing=(self.x_spacing, self.y_spacing, self.z_spacing),
            origin=(self.padded_x_min, self.padded_y_min, self.padded_z_min),
        )

        self.transform_to_cart = np.eye(4)
        self.transform_to_frac = np.eye(4)
        self.transform_to_cart[:3, :3] = self.padded_ebs.reciprocal_lattice.T
        self.transform_to_frac[:3, :3] = np.linalg.inv(
            self.padded_ebs.reciprocal_lattice.T
        )

        self.brillouin_zone = BrillouinZone(
            self.padded_ebs.reciprocal_lattice, np.array([1, 1, 1])
        )

        # Generate the surfaces
        self._generate_all_surfaces()

        logger.debug(f"fermi surface: \n {self}")
        logger.debug(f"fermi surface point data: \n {self.point_data}")
        logger.debug(f"fermi surface cell data: \n {self.cell_data}")
        logger.info("___FermiSurface initialization complete___")

    @property
    def ebs(self):
        return self._ebs

    @property
    def padded_ebs(self):
        return self._padded_ebs

    @property
    def npoints(self):
        return self.points.shape[0]

    def _generate_all_surfaces(self, order="F"):
        """
        Generate isosurfaces for all bands and spins that cross the Fermi level.

        This method iterates through all bands and spins in the electronic band structure,
        generating Fermi surface isosurfaces for each band-spin combination that crosses
        the Fermi level. It then merges all valid surfaces into a single combined surface.

        The method creates mappings between band-spin pairs and their corresponding surface
        indices, which are stored in the class attributes band_spin_surface_map and
        surface_band_spin_map.

        If a surface generation fails for any band-spin combination, the error is logged
        and the method continues with the next combination.

        Returns
        -------
        None
            The generated surfaces are stored in the class instance, updating its points,
            faces, point_data, and cell_data attributes.
        """
        logger.info("___Generating all Fermi surfaces___")

        fermi_surfaces = []

        bands_mesh = self.padded_ebs.get_property("bands", as_mesh=True, order=order)
        # Get dimensions from bands_mesh
        _, _, _, nbands, nspins = bands_mesh.shape

        for ispin in range(nspins):
            for iband in range(nbands):
                scalar_mesh = bands_mesh[..., iband, ispin]

                try:
                    surface = self._generate_single_surface(scalar_mesh, order=order)

                    # Check if the surface is empty
                    if surface.points.shape[0] > 0:
                        logger.debug(
                            f"Surface for band {iband}, spin {ispin} has {surface.points.shape[0]} points"
                        )
                        fermi_surfaces.append(surface)
                        # Map the surface index to band and spin
                        surface_idx = len(fermi_surfaces) - 1
                        self.band_spin_surface_map[(iband, ispin)] = surface_idx
                        self.surface_band_spin_map[surface_idx] = (iband, ispin)

                except Exception as e:
                    logger.exception(
                        f"Failed to generate surface for band {iband}, spin {ispin}: {e}"
                    )
                    continue

        if fermi_surfaces:
            self.band_surfaces = fermi_surfaces
            logger.info(f"___Generated {len(fermi_surfaces)} Fermi surfaces___")
            # Combine all surfaces
            combined_surface = self._merge_surfaces(fermi_surfaces)

            # Update self with the combined surface data
            self.points = combined_surface.points
            self.faces = combined_surface.faces

            for x in combined_surface.point_data:
                if "Contour Data" not in x:
                    self.point_data[x] = combined_surface.point_data[x]
            for x in combined_surface.cell_data:
                self.cell_data[x] = combined_surface.cell_data[x]

    def _generate_single_surface(
        self,
        scalar_mesh: np.ndarray,
        method: str = "marching_cubes",
        order: str = "F",
    ) -> pv.PolyData:
        """
        Generate a single isosurface for given band and spin.

        Parameters
        ----------
        scalar_mesh : np.ndarray
            3D mesh of energy values
        method : str, optional
            Method for isosurface generation, by default "marching_cubes"

        Returns
        -------
        pv.PolyData
            Surface for the given band and spin
        """

        grid = copy.deepcopy(self.padded_grid)
        padded_scalar_mesh = scalar_mesh
        # Generate isosurface
        surface = grid.contour(
            [self.fermi],
            scalars=padded_scalar_mesh.reshape(-1, order=order),
            method=method,
        )

        if surface.points.shape[0] > 0:
            # Transform to cartesian coordinates
            surface = surface.transform(
                self.transform_to_cart, transform_all_input_vectors=False, inplace=False
            )

            # Clip surface with each face of the Brillouin zone
            for normal, center in zip(
                self.brillouin_zone.face_normals, self.brillouin_zone.centers
            ):
                surface = surface.clip(origin=center, normal=normal, inplace=False)
                if surface.points.shape[0] == 0:
                    break

        return surface

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

        # Create boolean masks for each band and spin combination
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            mask = spin_band_index_array == surface_idx
            self.band_spin_mask[(iband, ispin)] = mask

        merged = surfaces[0]
        for surface in surfaces[1:]:
            merged = merged.merge(surface, merge_points=False)

        merged.point_data["spin_index"] = spin_index_array
        merged.point_data["spin_band_index"] = spin_band_index_array

        return merged

    def get_brillouin_zone(self, supercell: List[int]):
        """Returns the BrillouinZone of the material

        Parameters
        ----------
        supercell : List[int]
            The supercell to use for the BrillouinZone

        Returns
        -------
        pyprocar.core.BrillouinZone
            The BrillouinZone of the material
        """

        return BrillouinZone(self.ebs.reciprocal_lattice, supercell)

    def _interpolate_to_surface(
        self,
        grid: pv.ImageData = None,
        meshgrids: Dict[str, np.ndarray] = None,
        interpolate_by_band: bool = False,
        save_band_data: bool = False,
    ):
        """
        Interpolate data from a grid or meshgrids onto the Fermi surface.

        This method takes either a PyVista ImageData object or a dictionary of
        numpy meshgrids and interpolates the data onto the Fermi surface points.
        The interpolated data is stored in the point_data dictionary of the
        Fermi surface.

        Parameters
        ----------
        grid : pv.ImageData, optional
            PyVista ImageData object containing the data to interpolate.
            Must have point_data attributes to interpolate.
        meshgrids : Dict[str, np.ndarray], optional
            Dictionary mapping field names to numpy meshgrids. Each meshgrid
            should have the same shape as the grid used to define the Fermi surface.

        Returns
        -------
        pyvista.UnstructuredGrid
            The interpolated grid containing the data as point attributes.

        Raises
        ------
        ValueError
            If neither grid nor meshgrids is provided.

        Notes
        -----
        The interpolated data is also stored in the point_data dictionary
        of the Fermi surface object.
        """
        logger.info("___Interpolating to surface___")
        # if image_data is None and meshgrids is None and unstructured_grid is None:
        #     raise ValueError("image_data, meshgrids, or unstructured_grid must be provided")

        if grid is None and meshgrids is None:
            raise ValueError(
                "image_data, meshgrids, or unstructured_grid must be provided"
            )

        if meshgrids is not None:
            logger.info("___Interpolating to surface from meshgrids___")
            grid = copy.deepcopy(self.padded_grid)
            for name, meshgrid in meshgrids.items():
                grid.point_data[name] = meshgrid.reshape(-1, order="F")
            unstructured_grid = grid.cast_to_unstructured_grid()

        if grid is not None:
            logger.info("___Interpolating to surface from grid___")
            unstructured_grid = grid.cast_to_unstructured_grid()

        unstructured_grid_cart = unstructured_grid.transform(
            self.transform_to_cart, transform_all_input_vectors=False, inplace=False
        )

        keys_to_interpolate = unstructured_grid_cart.point_data.keys()
        logger.debug(f"point_data to be interpolated: {keys_to_interpolate}")

        interpolated_surface = self.interpolate(
            unstructured_grid_cart,
            n_points=40,
            sharpness=20,
            strategy="null_value",
        )

        if interpolate_by_band:
            total_property_value = None
            for incoming_key in keys_to_interpolate:
                property_name = self.padded_ebs.extract_property_label(incoming_key)

                logger.debug(f"property_name: {property_name}")
                logger.debug(f"incoming_key: {incoming_key}")
                band_index, spin_index = self.padded_ebs.extract_band_index(
                    incoming_key
                )

                mask = self.band_spin_mask[(band_index, spin_index)]

                point_data_array = interpolated_surface.point_data[incoming_key]

                if point_data_array.ndim == 1:
                    # For scalar data
                    point_data_array[~mask] = 0
                else:
                    # For vector data
                    point_data_array[~mask, :] = 0

                interpolated_surface.point_data[incoming_key] = point_data_array

                if property_name not in interpolated_surface.point_data:
                    interpolated_surface.point_data[property_name] = point_data_array
                else:
                    interpolated_surface.point_data[property_name] += point_data_array

                if not save_band_data:
                    interpolated_surface.point_data.pop(incoming_key)

        self.point_data.update(interpolated_surface.point_data)

        return interpolated_surface

    def _compute_property(self, name, property_value, save_band_data=False):

        logger.debug(f"property_value: {property_value.shape}")

        padded_grid = copy.deepcopy(self.padded_grid)
        if self.ebs.is_band_property(property_value):
            nbands, nspins = property_value.shape[1:3]
            for iband in range(nbands):
                for ispin in range(nspins):
                    band_property_label = self.padded_ebs.get_band_property_label(
                        name, iband, ispin
                    )

                    padded_grid.point_data[band_property_label] = property_value[
                        :, iband, ispin, ...
                    ]

            self._interpolate_to_surface(
                grid=padded_grid,
                interpolate_by_band=True,
                save_band_data=save_band_data,
            )

        else:
            padded_grid.point_data[name] = property_value
            self._interpolate_to_surface(grid=padded_grid)

        is_point_data_vector = self.point_data[name].shape[-1] == 3
        logger.debug(f"is_point_data_vector|{name}: {is_point_data_vector}")
        if is_point_data_vector:
            scalar_name = f"{name}-norm"
            self.point_data[scalar_name] = np.linalg.norm(
                self.point_data[name], axis=-1
            )
            self.set_active_scalars(scalar_name, preference="point")
            self.set_active_vectors(name, preference="point")
        else:
            self.set_active_scalars(name, preference="point")

    def compute_property(
        self,
        property_name: str,
        property=True,
        gradient=False,
        hessian=False,
        save_band_data=False,
    ):

        if property:
            property_value = self.padded_ebs.get_property(
                property_name, as_mesh=False, order="F"
            )
            self._compute_property(property_name, property_value, save_band_data)
            return self.point_data
        if gradient:
            gradient_value = self.padded_ebs.get_gradient(
                property_name, as_mesh=False, order="F"
            )
            gradient_label = self.padded_ebs.get_property_gradient_label(property_name)
            self._compute_property(gradient_label, gradient_value, save_band_data)
        if hessian:
            hessian_value = self.padded_ebs.get_hessian(
                property_name, as_mesh=False, order="F"
            )
            hessian_label = self.padded_ebs.get_property_hessian_label(property_name)
            self._compute_property(hessian_label, hessian_value, save_band_data)
        return self.point_data

    def compute_atomic_projection(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        spins: List[int] = None,
        save_band_data: bool = False,
    ):

        property_value = self.padded_ebs.get_projected_sum(
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
        )

        property_value = self.padded_ebs.get_property(
            "projected_sum", as_mesh=False, order="F"
        )
        self._compute_property(
            "projected_sum", property_value, save_band_data=save_band_data
        )
        return self.point_data

    def compute_spin_texture(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        save_band_data: bool = False,
    ):
        property_name = "spin_texture"
        if not self.padded_ebs.is_non_collinear:
            raise ValueError(
                "Spin texture is only available for non-collinear calculations"
            )

        padded_grid = copy.deepcopy(self.padded_grid)
        spin_texture = self.padded_ebs.get_projected_spin_texture(
            atoms=atoms, orbitals=orbitals
        )

        self._compute_property(
            "spin_texture", spin_texture, save_band_data=save_band_data
        )

    def extend_surface(
        self, zone_directions: List[Union[List[int], Tuple[int, int, int]]]
    ):
        """
        Method to extend the surface in the direction of a reciprocal lattice vecctor

        Parameters
        ----------
        zone_directions : List[List[int] or Tuple[int,int,int]], optional
            List of directions to expand to, by default None
        """
        # The following code  creates exteneded surfaces in a given direction
        print(
            f"Warning. Extending the surface will disable the further calculation of properties.\n"
            "Existing properties will still be available."
        )
        print(
            f"To enable the calculation of properties, please create a new FermiSurface object.\n"
        )

        extended_surfaces = []

        new_surface = copy.deepcopy(self)
        initial_surface = copy.deepcopy(self)
        for direction in zone_directions:
            surface = copy.deepcopy(initial_surface)

            new_surface += surface.translate(
                np.dot(direction, self.ebs.reciprocal_lattice), inplace=True
            )

        return new_surface

    @classmethod
    def from_code(cls, code, dirpath, fermi: float = None, **kwargs):
        parser = io.Parser(code=code, dirpath=dirpath)
        ebs = parser.ebs
        ebs.reduce_bands_near_fermi()

        if fermi is None:
            fermi = ebs.efermi

        return cls(ebs, fermi=fermi, **kwargs)
