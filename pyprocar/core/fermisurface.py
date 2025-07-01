import copy
import logging
import re
from functools import cached_property
from typing import Dict, List, Tuple, Union

import numpy as np
import pyvista as pv
from scipy import interpolate

from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.core.ebs import ElectronicBandStructureMesh
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
        padding: int = 5,
    ):
        super().__init__()
        logger.info("___Initializing FermiSurface object___")

        self._ebs = ebs
        self._fermi_shift = fermi_shift
        self._fermi = fermi + fermi_shift
        self._padding = padding
        
        self._padded_ebs = ebs.pad(padding=self._padding, inplace=False)

        logger.debug(f"Padded Electronic Band Structure Info: \n {self.padded_ebs}")

        # Initialize storage for surface data
        self._band_spin_surface_map = {}
        self._surface_band_spin_map = {}
        self._band_spin_mask = {}

        # Generate the surfaces
        self._generate_all_surfaces()

        logger.debug(f"Fermi Surface: \n {self}")
        logger.debug(f"Fermi Surface Point Data: \n {self.point_data}")
        logger.debug(f"Fermi Surface Cell Data: \n {self.cell_data}")
        logger.info("___FermiSurface initialization complete___")
        

    @property
    def ebs(self):
        return self._ebs
    
    @property
    def padded_ebs(self):
        return self._padded_ebs
    
    @property
    def padding(self):
        return self._padding
    
    @property
    def padded_grid(self):
        x_coords = self.padded_ebs.kpoints[:, 0]
        y_coords = self.padded_ebs.kpoints[:, 1]
        z_coords = self.padded_ebs.kpoints[:, 2]
        
        
        nx = self.padded_ebs.n_kx
        ny = self.padded_ebs.n_ky
        nz = self.padded_ebs.n_kz
        
        padded_x_min = x_coords.min()
        padded_y_min = y_coords.min()
        padded_z_min = z_coords.min()
        
        # Must be the points on non-padded grid
        x_spacing = 1 / self.ebs.n_kx
        y_spacing = 1 / self.ebs.n_kx
        z_spacing = 1 / self.ebs.n_kx
        
        
        padded_grid = pv.ImageData(
            dimensions=(nx,ny,nz),
            spacing=(x_spacing, y_spacing, z_spacing),
            origin=(padded_x_min, padded_y_min, padded_z_min),
        )
        return padded_grid
    
    @property
    def transform_matrix_to_cart(self):
        transform_to_cart = np.eye(4)
        transform_to_cart[:3, :3] = self.reciprocal_lattice.T
        return transform_to_cart
    
    @property
    def transform_matrix_to_frac(self):
        transform_to_frac = np.eye(4)
        transform_to_frac[:3, :3] = np.linalg.inv(self.reciprocal_lattice.T)
        return transform_to_frac

    @property
    def fermi_shift(self):
        return self._fermi_shift
    
    @property
    def fermi(self):
        return self._fermi
    
    @property
    def reciprocal_lattice(self):
        return self.ebs.reciprocal_lattice

    @property
    def n_points(self):
        return self.points.shape[0]
    
    @property
    def band_indices(self):
        return self._band_spin_surface_map.keys()
    
    @property
    def band_spin_surface_map(self):
        return self._band_spin_surface_map
    
    @property
    def surface_band_spin_map(self):
        return self._surface_band_spin_map
    
    @cached_property
    def brillouin_zone(self):
        return self.get_brillouin_zone(np.array([1, 1, 1]))

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

        return BrillouinZone(self.reciprocal_lattice, supercell)
    
    def get_property(self, name: str, compute=True):
        if name not in self.point_data and compute:
            self.compute_property(name)
        return self.point_data.get(name, None)
    
    def get_gradient(self, name: str, compute=True):
        if name not in self.point_data and compute:
            self.compute_gradient(name, first_order=True, second_order=False)
        return self.point_data.get(name, None)
    
    def get_hessian(self, name: str, compute=True):
        if name not in self.point_data and compute:
            self.compute_gradient(name, first_order=False, second_order=True)
        return self.point_data.get(name, None)

    def compute_property(self, name: str, include_band_resolved_data=False, **kwargs):
        
        if name == "fermi_velocity":
            return self.compute_fermi_velocity(include_band_resolved_data=include_band_resolved_data, **kwargs)
        elif name == "fermi_speed":
            return self.compute_fermi_speed(include_band_resolved_data=include_band_resolved_data, **kwargs)
        else:
            property_value = self.padded_ebs.get_property(name, **kwargs)
            
        self.project_property(name, property_value, include_band_resolved_data, spin_channels=kwargs.pop("spins", None))
        return self.point_data[name]
    
    def compute_gradient(self, name: str, first_order: bool = True, second_order: bool = False, order: str = "F"):
        if first_order:
            gradient_value = self.padded_ebs.get_gradient(name, order=order)
            gradient_label = self.padded_ebs.get_property_gradient_label(name)
            self.project_property(gradient_label, gradient_value)
            
        if second_order:
            hessian_value = self.padded_ebs.get_hessian(name, order=order)
            hessian_label = self.padded_ebs.get_property_hessian_label(name)
            self.project_property(hessian_label, hessian_value)
            
        return self.point_data[name]

    def compute_fermi_speed(self, label=None, include_band_resolved_data=False, **kwargs):
        if label is None:
            label="fermi_speed"
        self.padded_ebs.compute_band_speed(label=label, **kwargs)
        self.project_property(label, self.padded_ebs.get_property(label, **kwargs), include_band_resolved_data=include_band_resolved_data)
        return self.point_data[label]

    def compute_fermi_velocity(self, label=None, include_band_resolved_data=False, **kwargs):
        if label is None:
            label="fermi_velocity"
        self.padded_ebs.compute_band_velocity(label=label, **kwargs)
        self.project_property(label, self.padded_ebs.get_property(label, **kwargs), include_band_resolved_data=include_band_resolved_data)
        return self.point_data[label]

    def compute_avg_inv_effective_mass(self, label=None, include_band_resolved_data=False, **kwargs):
        if label is None:
            label="avg_inv_effective_mass"
        self.padded_ebs.compute_avg_inv_effective_mass(label=label, **kwargs)
        self.project_property(label, self.padded_ebs.get_property(label, **kwargs), include_band_resolved_data=include_band_resolved_data)
        return self.point_data[label]

    def compute_ebs_ipr(self, label=None, include_band_resolved_data=False, **kwargs):
        if label is None:
            label="ebs_ipr"
        self.padded_ebs.compute_ebs_ipr(label=label, **kwargs)
        self.project_property(label, self.padded_ebs.get_property(label, **kwargs), include_band_resolved_data=include_band_resolved_data)
        return self.point_data[label]
    
    def compute_ebs_ipr_atom(self, label=None, include_band_resolved_data=False, **kwargs):
        if label is None:
            label="ebs_ipr_atom"
        self.padded_ebs.compute_ebs_ipr_atom(label=label, **kwargs)
        self.project_property(label, self.padded_ebs.get_property(label, **kwargs), include_band_resolved_data=include_band_resolved_data)
        return self.point_data[label]

    def compute_atomic_projection(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        spins: List[int] = None,
        include_band_resolved_data: bool = False,
        label=None,
        **kwargs
    ):

        if label is None:
            label="projected_sum"

        property_values = self.padded_ebs.compute_projected_sum(
                                                    atoms=atoms, orbitals=orbitals, spins=spins,
                                                    label=label,
                                                    **kwargs
                                                    )

        self.project_property(
            label, property_values, include_band_resolved_data=include_band_resolved_data, spin_channels=spins
        )
        return self.point_data[label]
    
    
    def compute_spin_texture(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        include_band_resolved_data: bool = False,
        label=None,
        **kwargs
    ):

        if label is None:
            label="spin_texture"

        property_value = self.padded_ebs.compute_projected_sum_spin_texture(
                                                      atoms=atoms, orbitals=orbitals,
                                                      label=label,
                                                      **kwargs
                                                      )
        self.project_property(
            label, property_value, include_band_resolved_data=include_band_resolved_data
        )

        return self.point_data[label]
    
    def compute_projected_sum_spin_texture(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        include_band_resolved_data: bool = False,
        label=None,
        **kwargs
    ):

        if label is None:
            label="projected_sum_spin_texture"

        property_value = self.padded_ebs.get_property(label, 
                                                      atoms=atoms, orbitals=orbitals,
                                                      label=label,
                                                      **kwargs
                                                      )
        
        self.project_property(
            label, property_value, include_band_resolved_data=include_band_resolved_data
        )

        return self.point_data[label]
        
    def project_property(self, name, property_value, include_band_resolved_data=False, spin_channels=None):
        property_value = self.format_property_data(property_value, spin_channels=spin_channels)
        
        if isinstance(property_value, dict):
            self._interpolate_to_surface_by_band(name, property_value, include_band_resolved_data)
        else:
            self._interpolate_to_surface(name, property_value)

            
    def update_point_data(self, name: str, 
                          point_data: Union[Dict[str, np.ndarray], np.ndarray]= None, 
                          surface:pv.UnstructuredGrid = None):
        if isinstance(point_data, dict):
            self.point_data.update(point_data)
        elif isinstance(point_data, np.ndarray):
            self.point_data[name] = point_data
        elif surface is not None:
            self.point_data.update(surface.point_data)
        else:
            raise ValueError("Either point_data or a surface with point_data attribute must be provided")
        
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
            
        active_scalars_name = self.active_scalars_name
        active_vectors_name = self.active_vectors_name
        if active_vectors_name is not None and active_scalars_name is not None:
            new_surface.set_active_scalars(active_scalars_name, preference="point")
            new_surface.set_active_vectors(active_vectors_name, preference="point")
        elif active_scalars_name is not None:
            self.set_active_scalars(active_scalars_name, preference="point")

        return new_surface

    @classmethod
    def from_code(cls, code, dirpath, 
                  reduce_bands_near_fermi: bool = False, 
                  reduce_bands_near_energy: float = None,
                  reduce_bands_by_index: List[int] = None,
                  padding: int = 10,
                  fermi: float = None,
                  fermi_shift: float = 0.0,
                  **kwargs):
        ebs = ElectronicBandStructureMesh.from_code(code, dirpath)
        if fermi is None:
            fermi = ebs.efermi
            
        if reduce_bands_near_fermi:
            ebs.reduce_bands_near_fermi()
        elif reduce_bands_near_energy is not None:
            ebs.reduce_bands_near_energy(reduce_bands_near_energy)
        elif reduce_bands_by_index is not None:
            ebs.reduce_bands_by_index(reduce_bands_by_index)

        return cls(ebs, padding=padding, fermi=fermi, fermi_shift=fermi_shift, **kwargs)
    
    def format_property_data(self, property_value: np.ndarray, spin_channels: List[int] = None):
        if self.padded_ebs.is_band_property(property_value):
            if spin_channels is None:
                spin_channels = self.ebs.spin_channels
            band_property_values={}
            if self.ebs.n_spins==2:
                n_bands = self.padded_ebs.n_bands
                for iband in range(n_bands):
                    for ispin,spin_channel in enumerate(spin_channels):
                        band_property_values[(iband, spin_channel)] = property_value[:, iband, ispin, ...]
            else:
                for iband in range(self.padded_ebs.n_bands):
                    band_property_values[(iband, 0)] = property_value[:, iband, 0, ...]
            return band_property_values
        else:
            return property_value
    
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

        bands_mesh = self.padded_ebs.get_property_mesh("bands", order=order)
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
                        self._band_spin_surface_map[(iband, ispin)] = surface_idx
                        self._surface_band_spin_map[surface_idx] = (iband, ispin)

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
        order : str, optional
            Order of the scalar mesh, by default "F"

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
                self.transform_matrix_to_cart, transform_all_input_vectors=False, inplace=False
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
            iband, ispin = self._surface_band_spin_map[surface_idx]
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
        for (iband, ispin), surface_idx in self._band_spin_surface_map.items():
            mask = spin_band_index_array == surface_idx
            self._band_spin_mask[(iband, ispin)] = mask

        merged = surfaces[0]
        for surface in surfaces[1:]:
            merged = merged.merge(surface, merge_points=False)

        merged.point_data["spin_index"] = spin_index_array
        merged.point_data["spin_band_index"] = spin_band_index_array

        return merged
    
    def _create_interpolated_surface(self,
                                     grid: pv.ImageData = None,
                                     meshgrids: Dict[str, np.ndarray] = None,
                                     n_points: int = 40,
                                     sharpness: int = 20,
                                     strategy: str = "null_value",
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
        n_points : int, optional
            Number of points to sample the Fermi surface.
        sharpness : int, optional
            Sharpness of the interpolation.
        strategy : str, optional
            Strategy to use for interpolation.

        Returns
        -------
        pyvista.UnstructuredGrid
            The interpolated grid containing the data as point attributes.

        Notes
        -----
        The interpolated data is also stored in the point_data dictionary
        of the Fermi surface object.
        """
        logger.info("___Interpolating to surface___")

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
            self.transform_matrix_to_cart, transform_all_input_vectors=False, inplace=False
        )

        keys_to_interpolate = unstructured_grid_cart.point_data.keys()
        logger.debug(f"point_data to be interpolated: {keys_to_interpolate}")

        interpolated_surface = self.interpolate(
            unstructured_grid_cart,
            n_points=n_points,
            sharpness=sharpness,
            strategy=strategy,
        )

        return interpolated_surface
    
    def _interpolate_to_surface(self, name: str, property_value: np.ndarray):
        """Interpolate a property value to the Fermi surface.

        This method takes a property value array and interpolates it to the Fermi
        surface using the padded grid. The interpolated data is then stored in
        the point_data dictionary.

        Parameters
        ----------
        name : str
            The name of the property to interpolate.
        property_value : np.ndarray
            The property values to interpolate. Should be a 1D array that can be
            reshaped to match the padded grid dimensions.

        Returns
        -------
        None
            The interpolated data is stored in self.point_data.
        """
        padded_grid = copy.deepcopy(self.padded_grid)
        padded_grid.point_data[name] = property_value.reshape(-1, order="F")
        interpolated_surface = self._create_interpolated_surface(grid=padded_grid)
        self.update_point_data(name, surface=interpolated_surface)
        return None
        
    
    def _interpolate_to_surface_by_band(self, name: str, band_property_values:  Dict[Tuple[int, int], np.ndarray], include_band_resolved_data: bool = False):
        """Interpolate band-resolved property values to the Fermi surface.

        This method handles properties that are resolved by band and spin indices.
        It creates separate interpolated surfaces for each band-spin combination
        and optionally merges them into a single property.

        Parameters
        ----------
        name : str
            The base name of the property to interpolate.
        band_property_values : Dict[Tuple[int, int], np.ndarray]
            The property values to interpolate. The keys of the dictionary are 
            tuples of (band_index, spin_index). Each value should be a numpy 
            array with shape (npoints, ...) where the last dimensions depend 
            on the property type (scalar or vector).
        include_band_resolved_data : bool, optional
            If True, keeps individual band-spin resolved data in point_data.
            If False, removes individual band-spin data after merging, by default False.

        Returns
        -------
        None
            The interpolated data is stored in self.point_data.
        """
        padded_grid = copy.deepcopy(self.padded_grid)
        
        for (iband, spin_channel), property_value in band_property_values.items():
            band_property_label = self.padded_ebs.get_band_property_label(
                name, iband, spin_channel
            )
            padded_grid.point_data[band_property_label] = property_value

        interpolated_surface = self._create_interpolated_surface(grid=padded_grid)
        
        incoming_keys = padded_grid.point_data.keys()
        interpolated_surface = self._merge_band_resolved_data(interpolated_surface, incoming_keys)
        if not include_band_resolved_data:
            for incoming_key in incoming_keys:
                interpolated_surface.point_data.pop(incoming_key)
        
        if name in interpolated_surface.point_data:
            self.update_point_data(name, surface=interpolated_surface)
        else:
            raise ValueError(f"The property {name} is not present on the fermi surface")
    
    def _merge_band_resolved_data(self, interpolated_surface, incoming_keys:List[str]):
        """Merge band-resolved data into a single property on the Fermi surface.

        This method takes individual band-spin resolved data and merges them into
        a single property by applying band-spin masks and summing contributions.

        Parameters
        ----------
        interpolated_surface : pyvista.UnstructuredGrid
            The interpolated surface containing band-resolved point data.
        incoming_keys : List[str]
            List of keys corresponding to band-spin resolved properties to merge.

        Returns
        -------
        pyvista.UnstructuredGrid
            The interpolated surface with merged property data added to point_data.
        """
        for incoming_key in incoming_keys:
            property_name = self.padded_ebs.extract_property_label(incoming_key)
            band_index, spin_index = self.padded_ebs.extract_band_index(
                incoming_key
            )

            if (band_index, spin_index) in self._band_spin_mask:
                mask = self._band_spin_mask[(band_index, spin_index)]
            else:
                mask = np.zeros(interpolated_surface.points.shape[0], dtype=bool)

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
                
        return interpolated_surface


    
