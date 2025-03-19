import copy
import logging
from typing import Dict, List

import numpy as np
import pyvista as pv
from scipy import interpolate

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
        config: FermiSurface3DConfig = None,
    ):
        super().__init__()
        logger.info("___Initializing FermiSurface object___")

        # Store input parameters
        self.ebs = copy.deepcopy(ebs)
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

        self._has_band_gradients = False
        self._has_band_hessian = False
        self.hessian_matrix_name_map = {
            (0, 0): "dxdx",
            (0, 1): "dxdy",
            (0, 2): "dxdz",
            (1, 0): "dydx",
            (1, 1): "dydy",
            (1, 2): "dydz",
            (2, 0): "dzdx",
            (2, 1): "dzdy",
            (2, 2): "dzdz",
        }

        # Initialize storage for surface data
        self.band_spin_surface_map = {}
        self.surface_band_spin_map = {}
        self.band_spin_mask = {}

        # Initialize ImageData Grid
        self.nkx, self.nky, self.nkz = self.ebs.n_kx, self.ebs.n_ky, self.ebs.n_kz

        logger.debug(f"nkx: {self.nkx}")
        logger.debug(f"nky: {self.nky}")
        logger.debug(f"nkz: {self.nkz}")

        self.x_unique = np.unique(self.ebs.kpoints[:, 0])
        self.y_unique = np.unique(self.ebs.kpoints[:, 1])
        self.z_unique = np.unique(self.ebs.kpoints[:, 2])

        self.x_min = self.x_unique.min()
        self.y_min = self.y_unique.min()
        self.z_min = self.z_unique.min()

        self.x_spacing = 1 / self.nkx
        self.y_spacing = 1 / self.nky
        self.z_spacing = 1 / self.nkz

        self.grid = pv.ImageData(
            dimensions=(
                self.nkx,
                self.nky,
                self.nkz,
            ),
            spacing=(self.x_spacing, self.y_spacing, self.z_spacing),
            origin=(self.x_min, self.y_min, self.z_min),
        )

        self.transform_to_cart = np.eye(4)
        self.transform_to_frac = np.eye(4)
        self.transform_to_cart[:3, :3] = self.ebs.reciprocal_lattice.T
        self.transform_to_frac[:3, :3] = np.linalg.inv(self.ebs.reciprocal_lattice.T)

        self.brillouin_zone = BrillouinZone(
            self.ebs.reciprocal_lattice, np.array([1, 1, 1])
        )

        # Generate the surfaces
        self._generate_all_surfaces()

        logger.debug(f"fermi surface: \n {self}")
        logger.debug(f"fermi surface point data: \n {self.point_data}")
        logger.debug(f"fermi surface cell data: \n {self.cell_data}")
        logger.info("___FermiSurface initialization complete___")

    @property
    def has_band_gradients(self):
        return self._has_band_gradients

    @property
    def has_band_hessian(self):
        return self._has_band_hessian

    @property
    def npoints(self):
        return self.points.shape[0]

    def _generate_all_surfaces(self):
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

        # Get dimensions from bands_mesh
        _, _, _, nbands, nspins = self.ebs.bands_mesh.shape

        for ispin in range(nspins):
            for iband in range(nbands):
                scalar_mesh = self.ebs.bands_mesh[..., iband, ispin]

                try:
                    surface = self._generate_single_surface(
                        scalar_mesh, self.ebs.reciprocal_lattice
                    )

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

        logger.info(f"___Generated {len(fermi_surfaces)} Fermi surfaces___")

    def _generate_single_surface(
        self,
        scalar_mesh: np.ndarray,
        reciprocal_lattice: np.ndarray,
        method: str = "marching_cubes",
    ) -> pv.PolyData:
        """
        Generate a single isosurface for given band and spin.

        Parameters
        ----------
        scalar_mesh : np.ndarray
            3D mesh of energy values
        reciprocal_lattice : np.ndarray
            Reciprocal lattice vectors
        method : str, optional
            Method for isosurface generation, by default "marching_cubes"

        Returns
        -------
        pv.PolyData
            Surface for the given band and spin
        """
        padding = 1

        padded_nkx = self.nkx + 2 * padding
        padded_nky = self.nky + 2 * padding
        padded_nkz = self.nkz + 2 * padding

        # Adjust origin to account for padding
        padded_origin = (
            self.x_min - padding * self.x_spacing,
            self.y_min - padding * self.y_spacing,
            self.z_min - padding * self.z_spacing,
        )

        grid = pv.ImageData(
            dimensions=(padded_nkx, padded_nky, padded_nkz),
            spacing=(self.x_spacing, self.y_spacing, self.z_spacing),
            origin=padded_origin,
        )

        padded_scalar_mesh = np.pad(
            scalar_mesh,
            pad_width=((padding, padding), (padding, padding), (padding, padding)),
            mode="wrap",  # Use edge values for padding
        )

        # Generate isosurface
        surface = grid.contour(
            [self.fermi],
            scalars=padded_scalar_mesh.reshape(-1, order="F"),
            method=method,
        )

        if surface.points.shape[0] > 0:
            # Transform to cartesian coordinates
            surface = surface.transform(
                self.transform_to_cart, transform_all_input_vectors=False
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
        # unstructured_grid: pv.UnstructuredGrid = None,
        grid: pv.ImageData = None,
        meshgrids: Dict[str, np.ndarray] = None,
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
            grid = copy.deepcopy(self.grid)
            for name, meshgrid in meshgrids.items():
                grid.point_data[name] = meshgrid.reshape(-1, order="F")
            unstructured_grid = grid.cast_to_unstructured_grid()

        if grid is not None:
            logger.info("___Interpolating to surface from grid___")
            unstructured_grid = grid.cast_to_unstructured_grid()

        unstructured_grid_cart = unstructured_grid.transform(
            self.transform_to_cart, transform_all_input_vectors=True
        )

        logger.debug(
            f"point_data to be interpolated: {unstructured_grid_cart.point_data.keys()}"
        )
        interpolated_grid = self.interpolate(
            unstructured_grid_cart,
            n_points=40,
            # radius=0.5,
            sharpness=20,
            strategy="null_value",
        )

        logger.debug(
            f"self.point_data after interpolation: {interpolated_grid.point_data.keys()}"
        )
        if "vtkValidPointMask" in interpolated_grid.point_data:
            interpolated_grid.point_data.pop("vtkValidPointMask")

        self.point_data.update(interpolated_grid.point_data)

        return interpolated_grid

    def compute_mesh_derivatives(
        self,
        scalar_mesh_dict: Dict[str, np.ndarray] = None,
        # vector_mesh_dict: Dict[str, np.ndarray] = None,
        compute_hessian: bool = False,
    ):
        """
        Compute derivatives of scalar fields on the mesh grid.

        This method calculates the gradients and optionally the Hessian matrices
        of scalar fields defined on the mesh grid. The results are interpolated
        to the Fermi surface points and stored in the point_data dictionary.

        Parameters
        ----------
        scalar_mesh_dict : Dict[str, np.ndarray], optional
            Dictionary mapping field names to scalar meshes. Each scalar mesh should
            have the same shape as the grid used to define the Fermi surface.
        compute_hessian : bool, default=False
            Whether to compute the Hessian matrices (second derivatives) in addition
            to gradients. Hessians are useful for effective mass calculations.

        Returns
        -------
        pyvista.UnstructuredGrid
            The grid containing the computed derivatives as point data.

        Notes
        -----
        The computed derivatives are stored in the point_data dictionary with keys:
        - '{name}-gradient': Gradient vectors for each scalar field
        - '{name}-hessian-dxdx', '{name}-hessian-dxdy', etc.: Components of the
          Hessian matrix for each scalar field (if compute_hessian=True)
        """
        logger.info("___Computing mesh derivatives___")

        def gradients_to_dict(gradient, prefix=""):
            """A helper method to label the gradients into a dictionary."""
            keys = np.array(
                [
                    f"{prefix}dxdx",
                    f"{prefix}dxdy",
                    f"{prefix}dxdz",
                    f"{prefix}dydx",
                    f"{prefix}dydy",
                    f"{prefix}dydz",
                    f"{prefix}dzdx",
                    f"{prefix}dzdy",
                    f"{prefix}dzdz",
                ],
            )
            keys = keys.reshape((3, 3))[:, : gradient.shape[1]].ravel()
            return dict(zip(keys, gradient.T))

        grid = copy.deepcopy(self.grid)

        # if scalar_mesh_dict is not None:
        #     for name, scalar_mesh in scalar_mesh_dict.items():
        #         grid.point_data[name] = scalar_mesh.reshape(-1, order="F")

        # if vector_mesh_dict is not None:

        for name, scalar_mesh in scalar_mesh_dict.items():
            if name not in grid.point_data:
                grid.point_data[name] = scalar_mesh.reshape(-1, order="F")

            gradients_grid = grid.compute_derivative(scalars=name)
            gradients_grid[f"{name}_gradient"] = gradients_grid.point_data.pop(
                "gradient"
            )

            if compute_hessian:
                gradients_grid = gradients_grid.compute_derivative(
                    scalars=f"{name}_gradient"
                )
                hessian_point_data = gradients_grid.point_data.pop("gradient")

                hessian_dict = gradients_to_dict(
                    hessian_point_data, prefix=f"{name}_hessian-"
                )

                gradients_grid.point_data.update(hessian_dict)

            grid.point_data.update(gradients_grid.point_data)

        logger.debug(f"gradients_point_data \n {gradients_grid.point_data.keys()}")

        self._interpolate_to_surface(grid=grid)

        return gradients_grid

    def compute_band_derivatives(self, compute_hessian: bool = False):
        """
        Compute the derivatives of the bands mesh for the Fermi surface.

        This method calculates the gradients and optionally the Hessian matrices
        of the band energies at each point on the Fermi surface. The results are
        stored in the point_data dictionary with appropriate keys.

        For each band and spin combination that contributes to the Fermi surface,
        this method:
        1. Extracts the corresponding scalar mesh from the electronic band structure
        2. Computes the derivatives using compute_mesh_derivatives
        3. Aggregates the results into combined gradient and Hessian fields

        Parameters
        ----------
        compute_hessian : bool, default=False
            Whether to compute the Hessian matrices in addition to gradients.
            The Hessian contains second derivatives that are useful for
            effective mass calculations.

        Returns
        -------
        None
            Results are stored in the point_data dictionary with keys:
            - 'band-gradient': Combined gradient vectors
            - 'band-hessian-xx', 'band-hessian-xy', etc.: Components of the
              Hessian matrix (if compute_hessian=True)
        """
        scalar_mesh_dict = {}
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            scalar_mesh = self.ebs.bands_mesh[..., iband, ispin]
            scalar_mesh_dict[f"band-{iband}|spin-{ispin}"] = scalar_mesh

        self.compute_mesh_derivatives(
            scalar_mesh_dict=scalar_mesh_dict, compute_hessian=compute_hessian
        )

        n_points = self.points.shape[0]
        band_gradients = np.zeros((n_points, 3))
        band_hessian = np.zeros((9, n_points))
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            band_mask = self.band_spin_mask[(iband, ispin)]
            band_gradients[band_mask, :] += self.point_data[
                f"band-{iband}|spin-{ispin}_gradient"
            ][band_mask, :]

            if compute_hessian:
                self._has_band_hessian = True
                for i, name in enumerate(self.hessian_matrix_name_map.values()):
                    band_hessian[i, band_mask] += self.point_data[
                        f"band-{iband}|spin-{ispin}_hessian-{name}"
                    ][band_mask]

        self.point_data["band-gradient"] = band_gradients

        for i, name in enumerate(self.hessian_matrix_name_map.values()):
            self.point_data[f"band-hessian-{name}"] = band_hessian[i, :]

        self._has_band_gradients = True

    def compute_fermi_velocity(self):
        """
        Compute the Fermi velocity for each point on the Fermi surface.

        This method calculates the Fermi velocity by taking the gradient of the band
        energy at the Fermi level and converting it to SI velocity units. The calculation
        follows the relation:

        v_F = ∇E / ħ

        where:
        - ∇E is the gradient of the band energy
        - ħ is the reduced Planck constant

        The method stores the result in the point_data dictionary with the key
        'fermi-velocity'.

        Returns
        -------
        None
            Results are stored in the point_data dictionary.
        """
        if not self.has_band_gradients:
            self.compute_band_derivatives()

        self.point_data["fermi-velocity"] = (
            -self.point_data["band-gradient"] * METER_ANGSTROM / HBAR_EV
        )
        # self.point_data["fermi-velocity"] = (
        #     np.dot(
        #         self.point_data["band-gradient"],
        #         np.linalg.inv(self.reciprocal_lattice).T,
        #     )
        #     * METER_ANGSTROM
        #     / HBAR_EV
        # )

    def compute_fermi_speed(self):
        """
        Compute the Fermi speed (magnitude of Fermi velocity) for each point on the Fermi surface.

        This method calculates the magnitude of the Fermi velocity vector for each point
        on the Fermi surface. It first ensures that the Fermi velocity has been computed,
        then calculates the Euclidean norm of each velocity vector. In SI units.

        The method stores the result in the point_data dictionary with the key
        'fermi-speed'.

        Returns
        -------
        None
            Results are stored in the point_data dictionary.
        """
        if "fermi-velocity" not in self.point_data:
            self.compute_fermi_velocity()

        self.point_data["fermi-speed"] = np.linalg.norm(
            self.point_data["fermi-velocity"], axis=1
        )

        # self.point_data["band-gradient"] = band_gradients

    def compute_average_inverse_effective_mass(self):
        """
        Compute the average inverse effective mass for each point on the Fermi surface.

        This method calculates the average inverse effective mass by taking the trace
        of the Hessian matrix and computing the inverse. The inverseeffective mass is expressed
        in units of the free electron mass (m_e).

        The calculation follows:
        m* -1 = m_e * Tr(∇²E) / 3

        where:
        - Tr(∇²E) is the trace of the Hessian matrix of the band energy
        - m_e is the free electron mass

        The method stores the result in the point_data dictionary with the key
        'avg-inv-effective-mass' for the overall surface and individual keys for
        each band and spin combination.

        Returns
        -------
        None
            Results are stored in the point_data dictionary.
        """
        if not self.has_band_hessian:
            self.compute_bands_mesh_derivatives(compute_hessian=True)

        hessian_matrix = np.zeros((3, 3, self.points.shape[0]))
        effective_masses_per_band_spin = {}
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            band_spin_name = f"band-{iband}|spin-{ispin}"
            band_spin_prefix = f"{band_spin_name}_hessian"
            for (i, j), name in self.hessian_matrix_name_map.items():
                hessian_comp_name = f"{band_spin_prefix}-{name}"
                hessian_matrix[i, j, :] = self.point_data[hessian_comp_name]

            scale_factor = EV_TO_J * METER_ANGSTROM**2 / HBAR_J**2

            hessian_matrix = hessian_matrix * scale_factor
            hessian_xx = self.point_data[f"{band_spin_prefix}-dxdx"] * scale_factor
            hessian_yy = self.point_data[f"{band_spin_prefix}-dydy"] * scale_factor
            hessian_zz = self.point_data[f"{band_spin_prefix}-dzdz"] * scale_factor
            m_inv = (hessian_xx + hessian_yy + hessian_zz) / 3

            e_mass = 1 / m_inv

            effective_masses_per_band_spin[(iband, ispin)] = e_mass

            self.point_data[f"band-{iband}|spin-{ispin}_avg-inv-effective-mass"] = (
                FREE_ELECTRON_MASS / e_mass
            )

        n_points = self.points.shape[0]
        total_effective_masses = np.zeros(n_points)
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            band_mask = self.band_spin_mask[(iband, ispin)]
            band_spin_name = f"band-{iband}|spin-{ispin}_avg-inv-effective-mass"
            total_effective_masses[band_mask] += self.point_data[band_spin_name][
                band_mask
            ]

        self.point_data["avg-inv-effective-mass"] = total_effective_masses

    def compute_atomic_projections(
        self,
        atoms: List[int] = None,
        orbitals: List[int] = None,
        spins: List[int] = None,
        compute_derivative: bool = False,
        compute_hessian: bool = False,
    ):

        if orbitals is None and self.ebs.projected is not None:
            orbitals = np.arange(self.ebs.norbitals, dtype=int)
        if atoms is None and self.ebs.projected is not None:
            atoms = np.arange(self.ebs.natoms, dtype=int)

        if self.ebs.is_non_collinear:
            projected = self.ebs.ebs_sum(
                spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=True
            )
        else:
            projected = self.ebs.ebs_sum(
                spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False
            )
        grid = copy.deepcopy(self.grid)
        atom_names = "".join(str(atom) for atom in atoms)
        orbital_names = "".join(str(orbital) for orbital in orbitals)
        projection_name = f"atoms-{atom_names}|orbitals-{orbital_names}-projection"
        scalar_mesh_dict = {}
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            band_spin_projection = projected[:, iband, ispin]

            scalar_name = f"band-{iband}|spin-{ispin}|{projection_name}"
            grid.point_data[scalar_name] = band_spin_projection

            scalar_mesh_dict[scalar_name] = band_spin_projection.reshape(
                (self.nkx, self.nky, self.nkz), order="F"
            )

        if compute_derivative:
            self.compute_mesh_derivatives(
                scalar_mesh_dict=scalar_mesh_dict, compute_hessian=compute_hessian
            )
            projection_gradient = np.zeros((self.npoints, 3))
            for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
                scalar_name = f"band-{iband}|spin-{ispin}|{projection_name}"
                projection_gradient += self.point_data[f"{scalar_name}-gradient"]

            self.point_data[f"{projection_name}-gradient"] = projection_gradient

        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            band_mask = self.band_spin_mask[(iband, ispin)]
            scalar_name = f"band-{iband}|spin-{ispin}|{projection_name}"
            self.point_data[f"{projection_name}"][band_mask] += self.point_data[
                f"{scalar_name}"
            ][band_mask]

        self._interpolate_to_surface(grid=grid)

        return grid

    # def project_vectors(self, vector_data: np.ndarray, name: str = "vectors"):

    # def project_vectors(self, vector_data: np.ndarray, name: str = "vectors"):
    #     """
    #     Project vector data onto the Fermi surface.

    #     Parameters
    #     ----------
    #     vector_data : np.ndarray
    #         Vector data with shape matching ebs.bands_mesh
    #     name : str, optional
    #         Name for the vector field, by default "vectors"
    #     """
    #     if vector_data.shape[:-1] != self.ebs.bands_mesh.shape[:-1]:
    #         raise ValueError("Vector data shape must match bands_mesh shape")

    #     # Project vectors for each surface
    #     for surface_idx, (iband, ispin) in self.surface_band_spin_map.items():
    #         mask = (self.point_data["band_index"] == iband) & (
    #             self.point_data["spin_index"] == ispin
    #         )
    #         points = self.points[mask]

    #         # Get vectors for this band/spin
    #         vectors = vector_data[..., iband, ispin, :]

    #         # Interpolate vectors to surface points
    #         interpolated_vectors = self._interpolate_to_surface(points, vectors)

    #         # Store in point_data
    #         if name not in self.point_data:
    #             self.point_data[name] = np.zeros((self.n_points, 3))
    #         self.point_data[name][mask] = interpolated_vectors

    # def project_scalars(self, scalar_data: np.ndarray, name: str = "scalars"):
    #     """
    #     Project scalar data onto the Fermi surface.

    #     Parameters
    #     ----------
    #     scalar_data : np.ndarray
    #         Scalar data with shape matching ebs.bands_mesh
    #     name : str
    #         Name for the scalar field
    #     """
    #     if scalar_data.shape[:-1] != self.ebs.bands_mesh.shape[:-1]:
    #         raise ValueError("Scalar data shape must match bands_mesh shape")

    #     # Project scalars for each surface
    #     for surface_idx, (iband, ispin) in self.surface_band_spin_map.items():
    #         mask = (self.point_data["band_index"] == iband) & (
    #             self.point_data["spin_index"] == ispin
    #         )
    #         points = self.points[mask]

    #         # Get scalars for this band/spin
    #         scalars = scalar_data[..., iband, ispin]

    #         # Interpolate scalars to surface points
    #         interpolated_scalars = self._interpolate_to_surface(
    #             points, scalars, vector=False
    #         )

    #         # Store in point_data
    #         if name not in self.point_data:
    #             self.point_data[name] = np.zeros(self.n_points)
    #         self.point_data[name][mask] = interpolated_scalars

    # def _interpolate_grid_to_surface(
    #     self, meshgrid: np.ndarray, vector: bool = True
    # ) -> np.ndarray:
    #     """Helper method to interpolate data to surface points"""
    #     grid = copy.deepcopy(self.grid)

    #     grid =
    #     unstructured_grid_cart = grid.cast_to_unstructured_grid()
    #     self = self.interpolate(
    #         unstructured_grid_cart, radius=1, sharpness=10, strategy="mask_points"
    #     )

    #     return self

    # def _interpolate_to_surface(
    #     self, points: np.ndarray, data: np.ndarray, vector: bool = True
    # ) -> np.ndarray:
    #     """Helper method to interpolate data to surface points"""
    # Create regular grid points
    #     nkx, nky, nkz = self.ebs.bands_mesh.shape[:3]
    #     x = np.linspace(0, 1, nkx)
    #     y = np.linspace(0, 1, nky)
    #     z = np.linspace(0, 1, nkz)

    #     # Transform surface points to fractional coordinates
    #     inv_lattice = np.linalg.inv(self.ebs.reciprocal_lattice)
    #     frac_points = np.dot(points, inv_lattice.T)

    #     if vector:
    #         # Interpolate each component
    #         interpolated = np.zeros((len(points), 3))
    #         for i in range(3):
    #             interpolated[:, i] = interpolate.interpn(
    #                 (x, y, z),
    #                 data[..., i],
    #                 frac_points,
    #                 method="linear",
    #                 bounds_error=False,
    #                 fill_value=None,
    #             )
    #     else:
    #         # Interpolate scalar
    #         interpolated = interpolate.interpn(
    #             (x, y, z),
    #             data,
    #             frac_points,
    #             method="linear",
    #             bounds_error=False,
    #             fill_value=None,
    #         )

    #     return interpolated
