from __future__ import annotations

__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import copy
import logging
import sys
from dataclasses import dataclass
from typing import Any, Literal, cast

from typing_extensions import override

import numpy as np
import numpy.typing as npt
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator

from pyprocar.core.brillouin_zone import BrillouinZone2D
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.property_store import PointSet, Property  # noqa: TC002 - used for isinstance checks

np.set_printoptions(threshold=sys.maxsize)

logger = logging.getLogger(__name__)

# TODO: method to reduce number of points for interpolation need to be modified since the tolerance on the space
# is not lonmg soley reciprocal space, but energy and reciprocal space


@dataclass
class PlaneInfo:
    """Container for 2D plane cut parameters.

    This dataclass holds all the computed parameters needed to define
    a 2D cutting plane through k-space for band structure visualization.
    """

    normal: np.ndarray
    origin: np.ndarray
    u: np.ndarray  # First orthonormal basis vector
    v: np.ndarray  # Second orthonormal basis vector
    u_limits: tuple[float, float]
    v_limits: tuple[float, float]
    grid_interpolation: tuple[int, int]
    u_grid: np.ndarray
    v_grid: np.ndarray
    uv_grid_points: np.ndarray
    as_cartesian: bool


def get_orthonormal_basis(
    normal: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if np.abs(np.dot(normal, [0, 0, 1])) < 0.99:
        v_temp = np.array([0, 0, 1])  # Not parallel to normal
    else:
        v_temp = np.array([0, 1, 0])  # Not parallel to normal

    u: npt.NDArray[np.float32] = np.cross(v_temp, normal).astype(np.float32)
    u /= np.linalg.norm(u)
    v: npt.NDArray[np.float32] = np.cross(normal, u).astype(np.float32)
    v /= np.linalg.norm(v)  # Ensure normalization

    return u, v


def transform_points_to_uv(
    points: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    origin: np.ndarray | None = None,
):
    if origin is None:
        origin = np.array([0, 0, 0])
    points_shifted = points - origin
    return np.column_stack([np.dot(points_shifted, u), np.dot(points_shifted, v)])


def find_plane_limits(plane_points: np.ndarray):
    u_limits = plane_points[:, 0].min(), plane_points[:, 0].max()
    v_limits = plane_points[:, 1].min(), plane_points[:, 1].max()
    return u_limits, v_limits


def get_uv_grid(
    grid_interpolation: tuple[int, int],
    u_limits: tuple[float, float],
    v_limits: tuple[float, float],
):
    grid_u, grid_v = np.mgrid[
        u_limits[0] : u_limits[1] : complex(0, grid_interpolation[0]),
        v_limits[0] : v_limits[1] : complex(0, grid_interpolation[1]),
    ]
    return grid_u, grid_v


def get_uv_grid_points(grid_u: np.ndarray, grid_v: np.ndarray):
    return np.vstack([grid_u.ravel(), grid_v.ravel()]).T


def get_uv_grid_kpoints(
    origin: npt.NDArray[np.floating[Any]],
    uv_points: npt.NDArray[np.floating[Any]],
    uv_transformation_matrix: npt.NDArray[np.floating[Any]] | None = None,
    u: npt.NDArray[np.floating[Any]] | None = None,
    v: npt.NDArray[np.floating[Any]] | None = None,
    _normal: npt.NDArray[np.floating[Any]] | None = None,
) -> npt.NDArray[np.floating[Any]]:
    if uv_transformation_matrix is None:
        if u is None or v is None:
            raise ValueError("Either uv_transformation_matrix or both u and v must be provided")
        uv_transformation_matrix = get_uv_transformation_matrix(u, v)
    return origin + uv_points @ uv_transformation_matrix


def get_uv_transformation_matrix(u: np.ndarray, v: np.ndarray):
    return np.vstack([u, v])


def get_transformation_matrix(u: np.ndarray, v: np.ndarray, normal: np.ndarray):
    uv_transformation_matrix = get_uv_transformation_matrix(u, v)
    transformation_matrix = np.vstack([uv_transformation_matrix, normal])
    return transformation_matrix


def compute_plane_info(
    ebs: ElectronicBandStructureMesh,
    normal: tuple[float, float, float],
    origin: tuple[float, float, float],
    grid_interpolation: tuple[int, int],
    as_cartesian: bool,
) -> PlaneInfo:
    """Compute plane parameters from EBS and plane specification.

    Parameters
    ----------
    ebs : ElectronicBandStructureMesh
        The padded/expanded electronic band structure
    normal : tuple[float, float, float]
        Normal vector defining the cutting plane
    origin : tuple[float, float, float]
        Origin point of the cutting plane
    grid_interpolation : tuple[int, int]
        Number of grid points in (u, v) directions
    as_cartesian : bool
        Whether to interpret coordinates in Cartesian space

    Returns
    -------
    PlaneInfo
        Computed plane parameters
    """
    normal_arr = np.array(normal)
    origin_arr = np.array(origin)

    # Convert to tuples for pyvista slice method
    normal_tuple: tuple[float, float, float] = (float(normal_arr[0]), float(normal_arr[1]), float(normal_arr[2]))
    origin_tuple: tuple[float, float, float] = (float(origin_arr[0]), float(origin_arr[1]), float(origin_arr[2]))
    slice_mesh = ebs.slice(normal=normal_tuple, origin=origin_tuple, as_cartesian=as_cartesian)
    u, v = get_orthonormal_basis(normal=normal_arr)
    plane_points = transform_points_to_uv(slice_mesh.points, u, v)
    u_limits, v_limits = find_plane_limits(plane_points)
    u_grid, v_grid = get_uv_grid(
        grid_interpolation=grid_interpolation,
        u_limits=u_limits,
        v_limits=v_limits,
    )
    uv_grid_points = get_uv_grid_points(u_grid, v_grid)

    return PlaneInfo(
        normal=normal_arr,
        origin=origin_arr,
        u=np.asarray(u),
        v=np.asarray(v),
        u_limits=u_limits,
        v_limits=v_limits,
        grid_interpolation=grid_interpolation,
        u_grid=u_grid,
        v_grid=v_grid,
        uv_grid_points=uv_grid_points,
        as_cartesian=as_cartesian,
    )


def _generate_single_band_surface(
    scalars: npt.NDArray[np.floating[Any]],
    u_grid: npt.NDArray[np.floating[Any]],
    v_grid: npt.NDArray[np.floating[Any]],
) -> pv.PolyData:
    """Generate surface mesh for a single band's scalar values.

    Parameters
    ----------
    scalars : np.ndarray
        1D array of energy values at grid points
    u_grid : np.ndarray
        2D array of u-coordinates
    v_grid : np.ndarray
        2D array of v-coordinates

    Returns
    -------
    pv.PolyData
        Surface mesh with z-values set to energy
    """
    n_grid_points = u_grid.size
    surface_points = np.zeros((n_grid_points, 3))
    surface_points[:, 0] = u_grid.ravel()
    surface_points[:, 1] = v_grid.ravel()
    surface_points[:, 2] = scalars

    grid: Any = pv.StructuredGrid()
    grid.points = surface_points
    grid.dimensions = (u_grid.shape[0], v_grid.shape[0], 1)
    unstructured: Any = grid.cast_to_unstructured_grid()
    surface: Any = unstructured.extract_surface()
    if not isinstance(surface, pv.PolyData):
        raise TypeError(f"Expected PolyData, got {type(surface)}")
    return surface


def _merge_band_surfaces(
    surfaces: list[pv.PolyData],
    surface_band_spin_map: dict[int, tuple[int, int]],
) -> tuple[pv.PolyData, npt.NDArray[np.int32], npt.NDArray[np.int32], dict[tuple[int, int], npt.NDArray[np.bool_]]]:
    """Merge individual band surfaces with tracking arrays.

    Parameters
    ----------
    surfaces : list[pv.PolyData]
        List of individual band surfaces
    surface_band_spin_map : dict[int, tuple[int, int]]
        Mapping from surface index to (band_index, spin_index)

    Returns
    -------
    merged : pv.PolyData
        Merged surface
    spin_index_array : np.ndarray
        Array tracking spin index for each point
    spin_band_index_array : np.ndarray
        Array tracking surface index for each point
    band_spin_mask : dict[tuple[int, int], np.ndarray]
        Mask arrays for each (band, spin) pair
    """
    if not surfaces:
        return pv.PolyData(), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), {}

    # Build tracking arrays
    spin_band_index_array = np.empty(0, dtype=np.int32)
    spin_index_array = np.empty(0, dtype=np.int32)

    for surface_idx, surface in enumerate(surfaces):
        iband, ispin = surface_band_spin_map[surface_idx]
        n_points = surface.points.shape[0]

        # spin index identifier
        current_spin_index_array = np.full(n_points, ispin, dtype=np.int32)
        spin_index_array = np.insert(spin_index_array, 0, current_spin_index_array, axis=0)

        # spin band index identifier
        current_spin_band_index_array = np.full(n_points, surface_idx, dtype=np.int32)
        spin_band_index_array = np.insert(
            spin_band_index_array, 0, current_spin_band_index_array, axis=0
        )

    # Build band_spin_mask
    band_spin_mask: dict[tuple[int, int], npt.NDArray[np.bool_]] = {}
    band_spin_surface_map: dict[tuple[int, int], int] = {v: k for k, v in surface_band_spin_map.items()}
    for (iband, ispin), surface_idx in band_spin_surface_map.items():
        mask: npt.NDArray[np.bool_] = spin_band_index_array == surface_idx
        band_spin_mask[(iband, ispin)] = mask

    # Merge surfaces
    merged: pv.PolyData = surfaces[0]
    for surface in surfaces[1:]:
        merge_result = merged.merge(surface, merge_points=False)
        if not isinstance(merge_result, pv.PolyData):
            raise TypeError(f"Expected PolyData from merge, got {type(merge_result)}")
        merged = merge_result

    merged.point_data["spin_index"] = spin_index_array
    merged.point_data["spin_band_index"] = spin_band_index_array

    return merged, spin_index_array, spin_band_index_array, band_spin_mask


def _compute_scalar_grid(
    ebs: ElectronicBandStructureMesh,
    scalars: npt.NDArray[np.floating[Any]],
    plane_info: PlaneInfo,
) -> npt.NDArray[np.floating[Any]]:
    """Interpolate scalar values onto the 2D grid.

    Parameters
    ----------
    ebs : ElectronicBandStructureMesh
        The electronic band structure
    scalars : np.ndarray
        Scalar values to interpolate
    plane_info : PlaneInfo
        Plane parameters

    Returns
    -------
    np.ndarray
        Interpolated scalar values on the grid
    """
    scalars_shape = scalars.shape
    # Convert to tuples for pyvista slice method
    normal_tuple: tuple[float, float, float] = (
        float(plane_info.normal[0]),
        float(plane_info.normal[1]),
        float(plane_info.normal[2]),
    )
    origin_tuple: tuple[float, float, float] = (
        float(plane_info.origin[0]),
        float(plane_info.origin[1]),
        float(plane_info.origin[2]),
    )
    slice_mesh = ebs.slice(
        normal=normal_tuple,
        origin=origin_tuple,
        scalars=("scalars", scalars.reshape(scalars_shape[0], -1)),
        as_cartesian=plane_info.as_cartesian,
    )

    # Get slice points in UV coordinates for interpolation
    plane_points = transform_points_to_uv(slice_mesh.points, plane_info.u, plane_info.v)

    scalar_grid_points_flat = slice_mesh.active_scalars
    if scalar_grid_points_flat is None:
        raise ValueError("slice_mesh.active_scalars is None")
    n_grid_points = plane_info.uv_grid_points.shape[0]
    new_scalars_grid_points_flat = np.zeros(
        shape=(n_grid_points, scalar_grid_points_flat.shape[-1])
    )

    for iscalar in range(scalar_grid_points_flat.shape[-1]):
        interpolator = LinearNDInterpolator(
            plane_points, scalar_grid_points_flat[..., iscalar]
        )
        new_scalars_grid_points_flat[..., iscalar] = interpolator(plane_info.uv_grid_points)

    new_scalars_grid_points = new_scalars_grid_points_flat.reshape(
        (n_grid_points, *scalars_shape[1:])
    )
    return new_scalars_grid_points


def generate_band_2d_surfaces(
    ebs: ElectronicBandStructureMesh,
    plane_info: PlaneInfo,
    original_ebs: ElectronicBandStructureMesh | None = None,
    scale_factor: float = 2 * np.pi,
) -> tuple[pv.PolyData, dict[tuple[int, int], pv.PolyData], PointSet]:
    """Generate 2D band structure surfaces for all bands and spins.

    Parameters
    ----------
    ebs : ElectronicBandStructureMesh
        The padded/expanded electronic band structure
    plane_info : PlaneInfo
        Computed plane parameters
    original_ebs : ElectronicBandStructureMesh | None
        Original EBS for value clipping (optional)
    scale_factor : float
        K-plane scale factor (default: 2π)

    Returns
    -------
    combined_surface : pv.PolyData
        Merged surface containing all bands
    band_surfaces : dict[tuple[int, int], pv.PolyData]
        Individual surfaces keyed by (band_index, spin_index)
    point_set : PointSet
        PointSet with spin_index and spin_band_index properties
    """
    bands = ebs.get_property("bands")
    if bands is None or not isinstance(bands, Property):
        raise ValueError("bands property not found in EBS")
    bands_value: npt.NDArray[np.float64] = bands.value
    new_bands = _compute_scalar_grid(ebs, bands_value, plane_info)

    # Clip to original range if provided
    if original_ebs is not None:
        original_bands = original_ebs.get_property("bands")
        if original_bands is not None and isinstance(original_bands, Property):
            original_bands_value: npt.NDArray[np.float64] = original_bands.value
            new_bands = np.clip(
                new_bands, float(original_bands_value.min()), float(original_bands_value.max())
            )

    band_surfaces_list: list[pv.PolyData] = []
    band_surfaces_dict: dict[tuple[int, int], pv.PolyData] = {}
    band_spin_surface_map: dict[tuple[int, int], int] = {}
    surface_band_spin_map: dict[int, tuple[int, int]] = {}

    _, n_bands, n_spin_channels = new_bands.shape

    for iband in range(n_bands):
        for ispin in range(n_spin_channels):
            band_scalars = new_bands[..., iband, ispin]
            try:
                surface = _generate_single_band_surface(
                    band_scalars, plane_info.u_grid, plane_info.v_grid
                )
                if surface.points.shape[0] > 0:
                    logger.debug(
                        f"Surface for band {iband}, spin {ispin} has {surface.points.shape[0]} points"
                    )
                    band_surfaces_list.append(surface)
                    surface_idx = len(band_surfaces_list) - 1

                    band_surfaces_dict[(iband, ispin)] = surface
                    band_spin_surface_map[(iband, ispin)] = surface_idx
                    surface_band_spin_map[surface_idx] = (iband, ispin)

            except Exception as e:
                logger.exception(
                    f"Failed to generate surface for band {iband}, spin {ispin}: {e}"
                )
                continue

    if not band_surfaces_list:
        logger.warning("No band surfaces generated")
        empty_surface = pv.PolyData()
        empty_point_set = PointSet(np.zeros((0, 3)))
        empty_point_set.add_property(
            Property(name="spin_index", value=np.empty(0, dtype=np.float64))
        )
        empty_point_set.add_property(
            Property(name="spin_band_index", value=np.empty(0, dtype=np.float64))
        )
        return empty_surface, {}, empty_point_set

    logger.info(f"___Generated {len(band_surfaces_list)} band surfaces___")

    # Merge surfaces
    combined_surface, spin_index_array, spin_band_index_array, _ = _merge_band_surfaces(
        band_surfaces_list, surface_band_spin_map
    )

    # Apply k-plane scaling
    k_plane_scale_transform: npt.NDArray[np.float64] = np.eye(4)
    k_plane_scale_transform[0, 0] = scale_factor
    k_plane_scale_transform[1, 1] = scale_factor
    combined_surface.transform(k_plane_scale_transform, inplace=True)

    # Also scale individual surfaces
    for surface in band_surfaces_dict.values():
        surface.transform(k_plane_scale_transform, inplace=True)

    # Clean up PyVista internal arrays
    combined_surface.point_data.pop("vtkOriginalPointIds", None)
    combined_surface.cell_data.pop("vtkOriginalCellIds", None)

    # Create PointSet with required properties
    point_set = PointSet(combined_surface.points)
    point_set.add_property(
        Property(name="spin_index", value=spin_index_array.astype(np.float64))
    )
    point_set.add_property(
        Property(name="spin_band_index", value=spin_band_index_array.astype(np.float64))
    )

    # Store band_spin_mask in field_data for later access
    # Store as flattened int array that PyVista can serialize
    combined_surface.field_data["band_spin_surface_map"] = np.array(
        [item for key in band_spin_surface_map.keys() for item in key], dtype=np.int32
    )
    combined_surface.field_data["surface_band_spin_map"] = np.array(
        list(surface_band_spin_map.keys()), dtype=np.int32
    )

    combined_surface.set_active_scalars("spin_band_index")

    return combined_surface, band_surfaces_dict, point_set


class BandStructure2D(pv.PolyData):
    """
    2D band structure visualization from a plane cut through k-space.

    This class represents a 2D slice of the electronic band structure,
    visualized as a 3D surface where the z-axis represents energy.

    Use the factory methods `from_code()` or `from_ebs()` to create instances.
    """

    points: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int_]
    _band_surfaces: dict[tuple[int, int], pv.PolyData]
    _point_set: PointSet
    _original_ebs: ElectronicBandStructureMesh | None
    _ebs: ElectronicBandStructureMesh | None
    _plane_info: PlaneInfo

    def __init__(
        self,
        points: np.ndarray,
        faces: np.ndarray,
        band_surfaces: dict[tuple[int, int], pv.PolyData],
        point_set: PointSet,
        original_ebs: ElectronicBandStructureMesh,
        ebs: ElectronicBandStructureMesh,
        plane_info: PlaneInfo,
        point_data: dict[str, np.ndarray] | None = None,
        cell_data: dict[str, np.ndarray] | None = None,
        field_data: dict[str, Any] | None = None,
    ):
        """
        Initialize BandStructure2D with pre-computed surface data.

        Use from_code() or from_ebs() factory methods to create instances.
        """
        super().__init__()

        self.points = points
        self.faces = faces
        self._band_surfaces = band_surfaces
        self._point_set = point_set
        self._original_ebs = original_ebs
        self._ebs = ebs
        self._plane_info = plane_info

        # Cache invalidation tracking
        self._ebs_cache_version: int = 0
        self._cached_properties: dict[str, int] = {}

        # Validate required properties
        if "spin_band_index" not in point_set.property_store.keys():
            raise ValueError("spin_band_index not found in point_set.property_store")
        if "spin_index" not in point_set.property_store.keys():
            raise ValueError("spin_index not found in point_set.property_store")

        # Apply point/cell/field data
        point_data = point_data if point_data is not None else {}
        cell_data = cell_data if cell_data is not None else {}
        field_data = field_data if field_data is not None else {}

        self.point_data.update(point_data)
        self.cell_data.update(cell_data)
        self.field_data.update(field_data)

        # Set default active scalars
        if "spin_band_index" in self.point_data:
            self.set_active_scalars("spin_band_index", preference="point")

        # Compute transformation matrices (for reciprocal space conversions)
        self.transform_to_cart: npt.NDArray[np.float64] = np.eye(4)
        self.transform_to_frac: npt.NDArray[np.float64] = np.eye(4)
        if self._ebs.reciprocal_lattice is not None:
            self.transform_to_cart[:3, :3] = self._ebs.reciprocal_lattice.T
            self.transform_to_frac[:3, :3] = np.linalg.inv(self._ebs.reciprocal_lattice.T)

        logger.info("___BandStructure2D initialization complete___")

    # -------------------------------------------------------------------------
    # Cache invalidation methods
    # -------------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        """Invalidate all cached interpolated properties."""
        self._ebs_cache_version += 1
        logger.debug(f"Cache invalidated, new version: {self._ebs_cache_version}")

    def _is_cache_valid(self, property_name: str) -> bool:
        """Check if cached property is still valid."""
        if property_name not in self._cached_properties:
            return False
        return self._cached_properties[property_name] == self._ebs_cache_version

    def _mark_cached(self, property_name: str) -> None:
        """Mark property as cached at current version."""
        self._cached_properties[property_name] = self._ebs_cache_version

    @classmethod
    def from_ebs(
        cls,
        ebs: ElectronicBandStructureMesh,
        normal: tuple[float, float, float] = (0, 0, 1),
        origin: tuple[float, float, float] = (0, 0, 0),
        grid_interpolation: tuple[int, int] = (20, 20),
        as_cartesian: bool = True,
        padding: int = 15,
        scale_factor: float = 2 * np.pi,
        **_kwargs: Any,
    ) -> BandStructure2D:
        """
        Create BandStructure2D from an ElectronicBandStructureMesh.

        This factory method performs all heavy computation:
        - Pads the EBS
        - Computes plane parameters
        - Generates band surfaces

        Parameters
        ----------
        ebs : ElectronicBandStructureMesh
            Electronic band structure on a uniform k-grid
        normal : tuple
            Normal vector defining the cutting plane
        origin : tuple
            Origin point of the cutting plane
        grid_interpolation : tuple
            Number of grid points in (u, v) directions
        as_cartesian : bool
            Whether to interpret coordinates in Cartesian space
        padding : int
            Number of k-points to pad in each direction
        scale_factor : float
            Scale factor for k-plane (default: 2π)

        Returns
        -------
        BandStructure2D
            The constructed 2D band structure surface
        """
        original_ebs = copy.copy(ebs)
        padded_ebs = ebs.pad(padding=padding, inplace=False)
        padded_ebs = padded_ebs.expand_single_dimension(inplace=False)

        plane_info = compute_plane_info(
            ebs=padded_ebs,
            normal=normal,
            origin=origin,
            grid_interpolation=grid_interpolation,
            as_cartesian=as_cartesian,
        )

        combined_surface, band_surfaces, point_set = generate_band_2d_surfaces(
            ebs=padded_ebs,
            plane_info=plane_info,
            original_ebs=original_ebs,
            scale_factor=scale_factor,
        )

        return cls(
            points=combined_surface.points,
            faces=combined_surface.faces,
            band_surfaces=band_surfaces,
            point_set=point_set,
            original_ebs=original_ebs,
            ebs=padded_ebs,
            plane_info=plane_info,
            point_data=dict(combined_surface.point_data),
            cell_data=dict(combined_surface.cell_data),
            field_data=dict(combined_surface.field_data),
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def ebs(self) -> ElectronicBandStructureMesh:
        """The padded electronic band structure mesh."""
        if self._ebs is None:
            raise ValueError(
                "EBS not available. Pass ebs parameter to load() "
                "or create BandStructure2D from calculation."
            )
        return self._ebs

    @property
    def original_ebs(self) -> ElectronicBandStructureMesh:
        """The original (unpadded) electronic band structure mesh."""
        if self._original_ebs is None:
            raise ValueError(
                "Original EBS not available. Pass ebs parameter to load() "
                "or create BandStructure2D from calculation."
            )
        return self._original_ebs

    @property
    def point_set(self) -> PointSet:
        """PointSet containing surface properties."""
        return self._point_set

    @property
    def plane_info(self) -> PlaneInfo:
        """Plane parameters for the 2D cut."""
        return self._plane_info

    @property
    def normal(self) -> np.ndarray:
        """Normal vector of the cutting plane."""
        return self._plane_info.normal

    @property
    def origin(self) -> np.ndarray:
        """Origin point of the cutting plane."""
        return self._plane_info.origin

    @property
    def u(self) -> np.ndarray:
        """First orthonormal basis vector in the plane."""
        return self._plane_info.u

    @property
    def v(self) -> np.ndarray:
        """Second orthonormal basis vector in the plane."""
        return self._plane_info.v

    @property
    def n_grid_points(self) -> int:
        """Number of grid points in the 2D plane."""
        return self._plane_info.uv_grid_points.shape[0]

    @property
    def band_surfaces(self) -> dict[tuple[int, int], pv.PolyData]:
        """Individual band surfaces keyed by (band_index, spin_index)."""
        return self._band_surfaces

    @property
    def band_spin_surface_map(self) -> dict[tuple[int, int], int]:
        """Mapping from (band, spin) to surface index."""
        return {key: idx for idx, key in enumerate(self._band_surfaces.keys())}

    @property
    def surface_band_spin_map(self) -> dict[int, tuple[int, int]]:
        """Mapping from surface index to (band, spin)."""
        return {idx: key for idx, key in enumerate(self._band_surfaces.keys())}

    @property
    def band_spin_mask(self) -> dict[tuple[int, int], npt.NDArray[np.bool_]]:
        """Boolean masks for each (band, spin) pair."""
        spin_band_index_prop = self._point_set.get_property("spin_band_index")
        if spin_band_index_prop is None or not isinstance(spin_band_index_prop, Property):
            raise ValueError("spin_band_index property not found")
        spin_band_index: npt.NDArray[np.float64] = spin_band_index_prop.value
        return {key: spin_band_index == idx for key, idx in self.band_spin_surface_map.items()}

    # -------------------------------------------------------------------------
    # Backwards-compatible attribute accessors
    # -------------------------------------------------------------------------

    @property
    def u_grid(self) -> np.ndarray:
        """U-coordinate grid (for backwards compatibility)."""
        return self._plane_info.u_grid

    @property
    def v_grid(self) -> np.ndarray:
        """V-coordinate grid (for backwards compatibility)."""
        return self._plane_info.v_grid

    @property
    def uv_grid_points(self) -> np.ndarray:
        """UV grid points (for backwards compatibility)."""
        return self._plane_info.uv_grid_points

    @property
    def as_cartesian(self) -> bool:
        """Whether coordinates are in Cartesian space."""
        return self._plane_info.as_cartesian

    @property
    def grid_interpolation(self) -> tuple[int, int]:
        """Grid interpolation parameters."""
        return self._plane_info.grid_interpolation

    @property
    def plane_points(self) -> npt.NDArray[np.floating[Any]]:
        """Points on the slice plane in UV coordinates."""
        # Convert to tuples for pyvista slice method
        normal_tuple: tuple[float, float, float] = (float(self.normal[0]), float(self.normal[1]), float(self.normal[2]))
        origin_tuple: tuple[float, float, float] = (float(self.origin[0]), float(self.origin[1]), float(self.origin[2]))
        slice_mesh = self.ebs.slice(
            normal=normal_tuple, origin=origin_tuple, as_cartesian=self.as_cartesian
        )
        return transform_points_to_uv(slice_mesh.points, self.u, self.v)

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    def get_property(self, key: str, **kwargs: Any) -> npt.NDArray[np.float64]:
        """Get property values, computing if necessary.

        Uses cache invalidation to track property staleness.
        """
        prop_name, _ = self.ebs.extract_key(key)

        # Check if property exists and cache is valid
        property_value: npt.NDArray[np.float64]
        if prop_name in self.point_set.property_store and self._is_cache_valid(prop_name):
            prop = self.point_set.get_property(prop_name)
            if prop is not None and isinstance(prop, Property):
                property_value = prop.value
            else:
                property_value = self.compute_property(prop_name, **kwargs)
        else:
            # Compute and cache the property
            property_value = self.compute_property(prop_name, **kwargs)
            prop = Property(name=prop_name, value=property_value)
            self.point_set.add_property(prop)
            self._mark_cached(prop_name)

        self.set_values(prop_name, property_value)
        return property_value

    def compute_gradients(
        self, gradient_order: int, names: list[str] | None = None
    ) -> None:
        """Compute property gradients on the surface."""
        if names is None:
            names = list(self.point_set.property_store.keys())
        if gradient_order < 0:
            raise ValueError(f"Gradient order must be greater than 0. Got {gradient_order}.")
        self.ebs.compute_gradients(gradient_order=gradient_order, names=names)

        for prop_name, calc_name, grad_order, value_array in self.ebs.iter_properties():
            prop = self.point_set.get_property(prop_name)
            if prop is not None and isinstance(prop, Property):
                surface_points = self.interpolate_values(value_array)
                # Cast from floating[Any] to float64 for Property setter
                surface_points_f64: npt.NDArray[np.float64] = surface_points.astype(np.float64)
                prop[calc_name, grad_order] = surface_points_f64

    def compute_property(self, name: str, **kwargs: Any) -> npt.NDArray[np.float64]:
        ebs_property = self.ebs.get_property(name, **kwargs)
        if ebs_property is None or not isinstance(ebs_property, Property):
            raise ValueError(f"Property '{name}' not found in EBS")
        grid_scalars: npt.NDArray[np.float64] = self.compute_scalar_grid(ebs_property.value)

        original_property = self.original_ebs.get_property(name, **kwargs)
        if original_property is None or not isinstance(original_property, Property):
            raise ValueError(f"Property '{name}' not found in original EBS")
        original_value: npt.NDArray[np.float64] = original_property.value
        clipped_scalars: npt.NDArray[np.float64] = np.clip(
            grid_scalars, float(original_value.min()), float(original_value.max())
        )
        return clipped_scalars

    def interpolate_values(self, values: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        new_values: npt.NDArray[np.floating[Any]]
        if values.shape[-1] != 3:
            new_values = np.zeros(self.n_grid_points)
            interpolator = LinearNDInterpolator(self.plane_points, values)
            new_values = interpolator(self.uv_grid_points)
        else:
            new_values = np.zeros((self.n_grid_points, values.shape[-1]))
            for icoord in range(values.shape[-1]):
                interpolator = LinearNDInterpolator(self.plane_points, values[..., icoord])
                new_values[..., icoord] = interpolator(self.uv_grid_points)

        return new_values

    def points_to_grid(self, points: npt.NDArray[np.floating[Any]], order: Literal["C", "F", "A"] = "C") -> npt.NDArray[np.floating[Any]]:
        return points.reshape(self.u_grid.shape, order=order)

    def project_vector_to_plane(self, vectors: npt.NDArray[np.floating[Any]]) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        velocity_u: npt.NDArray[np.floating[Any]] = np.dot(vectors, self.u)
        velocity_v: npt.NDArray[np.floating[Any]] = np.dot(vectors, self.v)
        return velocity_u, velocity_v

    def get_2d_brillouin_zone(
        self,
        e_min: float,
        e_max: float,
        _supercell: list[int] | None = None,
        scale_factor: float = 2 * np.pi,
    ) -> BrillouinZone2D:
        if self.ebs.reciprocal_lattice is None:
            raise ValueError("reciprocal_lattice is None")
        reciprocal_lattice: npt.NDArray[np.float64] = self.ebs.reciprocal_lattice * scale_factor
        return BrillouinZone2D(
            e_min=e_min,
            e_max=e_max,
            axis=2,
            reciprocal_lattice=reciprocal_lattice,
        )

    def set_surface_point_data(self, name: str, values: npt.NDArray[np.floating[Any]]) -> None:
        """Set point data on the surface, handling band-resolved data."""
        if self.ebs.is_band_property(values):
            logger.debug(f"Adding band resolved to surface point_data: {name}")
            point_data_array: npt.NDArray[np.floating[Any]] | None = None
            for (iband, ispin), _ in self.band_spin_surface_map.items():
                values_band_values = values[:, iband, ispin, ...]
                if point_data_array is None:
                    point_data_array = values_band_values
                else:
                    point_data_array = np.insert(point_data_array, 0, values_band_values, axis=0)
            if point_data_array is not None:
                self.point_data[name] = point_data_array
        else:
            logger.debug(f"Adding scalar to surface point_data: {name}")
            self.point_data[name] = values

    def set_scalars(self, name: str, value: npt.NDArray[np.floating[Any]]) -> None:
        self.set_surface_point_data(name, value)
        self.set_active_scalars(name, preference="point")

    def set_vectors(self, name: str, value: npt.NDArray[np.floating[Any]], set_scalar: bool = True) -> None:
        if value.shape[-1] != 3:
            raise ValueError(f"Vector data must have 3 dimensions. Got {value.shape[-1]}.")

        vector_magnitude: npt.NDArray[np.floating[Any]] = np.linalg.norm(value, axis=-1)
        if set_scalar:
            self.set_surface_point_data(f"{name}-norm", vector_magnitude)
            self.set_active_scalars(f"{name}-norm", preference="point")
        self.set_surface_point_data(name, value)
        self.set_active_vectors(name, preference="point")

    def set_values(self, name: str, value: npt.NDArray[np.floating[Any]], **kwargs: Any) -> None:
        if value.shape[-1] == 3:
            self.set_vectors(name, value, **kwargs)
        else:
            self.set_scalars(name, value, **kwargs)

    def compute_scalar_grid(self, scalars: npt.NDArray[np.floating[Any]], **_kwargs: Any) -> npt.NDArray[np.float64]:
        scalars_shape = scalars.shape
        # Convert to tuples for pyvista slice method
        normal_tuple: tuple[float, float, float] = (float(self.normal[0]), float(self.normal[1]), float(self.normal[2]))
        origin_tuple: tuple[float, float, float] = (float(self.origin[0]), float(self.origin[1]), float(self.origin[2]))
        slice_mesh = self.ebs.slice(
            normal=normal_tuple,
            origin=origin_tuple,
            scalars=("scalars", scalars.reshape(scalars_shape[0], -1)),
            as_cartesian=self.as_cartesian,
        )

        scalar_grid_points_flat = slice_mesh.active_scalars
        if scalar_grid_points_flat is None:
            raise ValueError("slice_mesh.active_scalars is None")
        new_scalars_grid_points_flat = np.zeros(
            shape=(self.n_grid_points, scalar_grid_points_flat.shape[-1])
        )

        for iscalar in range(scalar_grid_points_flat.shape[-1]):
            new_scalars_grid_points_flat[..., iscalar] = self.interpolate_values(
                scalar_grid_points_flat[..., iscalar]
            )

        new_scalars_grid_points = new_scalars_grid_points_flat.reshape(
            (self.n_grid_points, *scalars_shape[1:])
        )
        return new_scalars_grid_points

    @classmethod
    def from_code(
        cls,
        code: str,
        dirpath: str,
        normal: tuple[float, float, float] = (0, 0, 1),
        origin: tuple[float, float, float] = (0, 0, 0),
        reduce_bands_near_energy: float | None = None,
        reduce_bands_near_fermi: bool = True,
        bands: list[int] | None = None,
        grid_interpolation: tuple[int, int] = (120, 120),
        padding: int = 15,
        as_cartesian: bool = False,
        scale_factor: float = 2 * np.pi,
        **kwargs: Any,
    ) -> BandStructure2D:
        """
        Create BandStructure2D from DFT output files.

        Parameters
        ----------
        code : str
            DFT code name ('vasp', 'qe', etc.)
        dirpath : str
            Path to calculation directory
        normal : tuple
            Normal vector defining the cutting plane
        origin : tuple
            Origin point of the cutting plane
        reduce_bands_near_energy : float | None
            Energy value to reduce bands around
        reduce_bands_near_fermi : bool
            Whether to reduce bands near Fermi level (default True)
        bands : list[int] | None
            Specific band indices to include
        grid_interpolation : tuple
            Number of grid points in (u, v) directions
        padding : int
            Number of k-points to pad in each direction
        as_cartesian : bool
            Whether to interpret coordinates in Cartesian space
        scale_factor : float
            Scale factor for k-plane (default: 2π)

        Returns
        -------
        BandStructure2D
            The constructed 2D band structure surface
        """
        ebs_result = ElectronicBandStructureMesh.from_code(code=code, dirpath=dirpath)
        # from_code returns ElectronicBandStructure, cast to Mesh type
        ebs = cast(ElectronicBandStructureMesh, ebs_result)

        if reduce_bands_near_energy is not None:
            ebs.reduce_bands_near_energy(reduce_bands_near_energy)
        elif reduce_bands_near_fermi:
            ebs.reduce_bands_near_fermi()
        elif bands is not None:
            ebs.reduce_bands_by_index(bands)

        return cls.from_ebs(
            ebs,
            normal=normal,
            origin=origin,
            grid_interpolation=grid_interpolation,
            as_cartesian=as_cartesian,
            padding=padding,
            scale_factor=scale_factor,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Serialization methods
    # -------------------------------------------------------------------------

    @override
    def save(self, path: str) -> None:  # pyright: ignore[reportIncompatibleMethodOverride] - different signature from pv.PolyData.save
        """
        Save BandStructure2D to file.

        Parameters
        ----------
        path : str
            Output file path. Extension determines format (.pkl).

        Note
        ----
        EBS objects are not saved due to weak reference complexity.
        Users should keep a reference to EBS or reload from calculation.
        """
        import pickle
        from pathlib import Path

        path_obj = Path(path)

        # Serialize point_set properties
        point_set_data: dict[str, Any] = {
            "points": self._point_set.points.copy(),
            "properties": {},
        }
        for prop_name, prop in self._point_set.property_store.items():
            point_set_data["properties"][prop_name] = {
                "value": prop.value.copy(),
                "gradients": {
                    k: v.copy() for k, v in prop.gradients.items() if len(v) > 0
                },
                "units": prop.units,
                "label": prop.label,
                "metadata": copy.deepcopy(prop.metadata) if prop.metadata else {},
                "data_lim": prop.data_lim,
            }

        # Serialize band surfaces (geometry only)
        band_surfaces_data: dict[tuple[int, int], dict[str, Any]] = {}
        for key, surface in self._band_surfaces.items():
            band_surfaces_data[key] = {
                "points": np.asarray(surface.points),
                "faces": np.asarray(surface.faces),
            }

        # Serialize plane_info
        plane_info_data = {
            "normal": self._plane_info.normal.copy(),
            "origin": self._plane_info.origin.copy(),
            "u": self._plane_info.u.copy(),
            "v": self._plane_info.v.copy(),
            "u_limits": self._plane_info.u_limits,
            "v_limits": self._plane_info.v_limits,
            "grid_interpolation": self._plane_info.grid_interpolation,
            "u_grid": self._plane_info.u_grid.copy(),
            "v_grid": self._plane_info.v_grid.copy(),
            "uv_grid_points": self._plane_info.uv_grid_points.copy(),
            "as_cartesian": self._plane_info.as_cartesian,
        }

        save_data: dict[str, Any] = {
            "points": np.asarray(self.points).copy(),
            "faces": np.asarray(self.faces).copy(),
            "point_data": {k: np.asarray(v) for k, v in self.point_data.items()},
            "cell_data": {k: np.asarray(v) for k, v in self.cell_data.items()},
            "field_data": dict(self.field_data),
            "point_set_data": point_set_data,
            "band_surfaces_data": band_surfaces_data,
            "plane_info_data": plane_info_data,
        }

        with open(path_obj, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"BandStructure2D saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        ebs: ElectronicBandStructureMesh | None = None,
    ) -> "BandStructure2D":
        """
        Load BandStructure2D from file.

        Parameters
        ----------
        path : str
            Input file path
        ebs : ElectronicBandStructureMesh | None
            Optional EBS to associate. If not provided, property computation
            methods will not be available.

        Returns
        -------
        BandStructure2D
            Loaded instance
        """
        import pickle
        from pathlib import Path

        path_obj = Path(path)

        with open(path_obj, "rb") as f:
            save_data = pickle.load(f)

        # Reconstruct point_set
        point_set_data = save_data["point_set_data"]
        point_set = PointSet(points=point_set_data["points"])

        for prop_name, prop_data in point_set_data["properties"].items():
            prop = Property(
                name=prop_name,
                value=prop_data["value"],
                gradients=prop_data.get("gradients"),
                units=prop_data.get("units"),
                label=prop_data.get("label"),
                point_set=point_set,
                metadata=prop_data.get("metadata"),
                data_lim=prop_data.get("data_lim"),
            )
            point_set.add_property(prop)

        # Reconstruct band_surfaces
        band_surfaces: dict[tuple[int, int], pv.PolyData] = {}
        for key, surface_data in save_data["band_surfaces_data"].items():
            surface = pv.PolyData(surface_data["points"], surface_data["faces"])
            band_surfaces[key] = surface

        # Reconstruct plane_info
        pi_data = save_data["plane_info_data"]
        plane_info = PlaneInfo(
            normal=pi_data["normal"],
            origin=pi_data["origin"],
            u=pi_data["u"],
            v=pi_data["v"],
            u_limits=pi_data["u_limits"],
            v_limits=pi_data["v_limits"],
            grid_interpolation=pi_data["grid_interpolation"],
            u_grid=pi_data["u_grid"],
            v_grid=pi_data["v_grid"],
            uv_grid_points=pi_data["uv_grid_points"],
            as_cartesian=pi_data["as_cartesian"],
        )

        # Reconstruct via shallow copy pattern (like FermiSurface)
        temp_polydata = pv.PolyData(save_data["points"], save_data["faces"])
        for k, v in save_data["point_data"].items():
            temp_polydata.point_data[k] = v
        for k, v in save_data["cell_data"].items():
            temp_polydata.cell_data[k] = v
        temp_polydata.field_data.update(save_data["field_data"])

        bs2d = pv.PolyData.__new__(cls)
        bs2d.shallow_copy(temp_polydata)

        # Copy PyVista internal state
        for attr in ["_association_bitarray_names", "_association_complex_names"]:
            if hasattr(temp_polydata, attr):
                setattr(bs2d, attr, getattr(temp_polydata, attr).copy())

        bs2d._band_surfaces = band_surfaces
        bs2d._point_set = point_set
        bs2d._original_ebs = ebs
        bs2d._ebs = ebs
        bs2d._plane_info = plane_info
        bs2d._ebs_cache_version = 0
        bs2d._cached_properties = {}

        # Initialize transformation matrices if EBS is provided
        if ebs is not None and ebs.reciprocal_lattice is not None:
            bs2d.transform_to_cart = np.eye(4)
            bs2d.transform_to_frac = np.eye(4)
            bs2d.transform_to_cart[:3, :3] = ebs.reciprocal_lattice.T
            bs2d.transform_to_frac[:3, :3] = np.linalg.inv(ebs.reciprocal_lattice.T)
        else:
            bs2d.transform_to_cart = np.eye(4)
            bs2d.transform_to_frac = np.eye(4)

        logger.info(f"BandStructure2D loaded from {path}")
        return bs2d
