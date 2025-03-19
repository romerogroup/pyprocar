# -*- coding: utf-8 -*-
import logging

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def get_angle(v, w, radians=False):
    """
    Calculates angle between two vectors

    Parameters
    ----------
    v : float
        vector 1.
    w : float
        vector 1.
    radians : bool, optional
        To return the result in radians or degrees. The default is False.

    Returns
    -------
    float
        Angle between v and w.

    """

    if np.linalg.norm(v) == 0 or np.linalg.norm(w) == 0 or np.all(v == w):
        return 0
    cosine = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))

    if radians:
        return np.arccos(cosine)
    else:
        return np.rad2deg(np.arccos(cosine))


def fft_interpolate(function, interpolation_factor=2, axis=None):
    """
    This method will interpolate using a Fast-Fourier Transform

    if I = interpolation_factor
    This function withh recieve f(x,y,z) with dimensions of (nx,ny,nz)
    and returns f(x,y,z) with dimensions of (nx*I,ny*I,nz*I)

    Parameters
    ----------
    function : np.ndarray
        The values array to do the interpolation on.
    interpolation_factor : int, optional
        Interpolation Factor, by default 2

    Returns
    -------
    np.ndarray
        The interpolated points
    """

    if axis is None:
        axis = np.arange(function.ndim)
    if type(axis) is int:
        axis = [axis]
    function = np.array(function)
    eigen_fft = np.fft.fftn(function)
    shifted_fft = np.fft.fftshift(eigen_fft)
    pad_width = []
    factor = 0
    for idim in range(function.ndim):
        if idim in axis:
            n = shifted_fft.shape[idim]
            pad = n * (interpolation_factor - 1) // 2
            factor += 1
        else:
            pad = 0
        pad_width.append([pad, pad])
    new_matrix = np.pad(shifted_fft, pad_width, "constant", constant_values=0)
    new_matrix = np.fft.ifftshift(new_matrix)
    if "complex" in function.dtype.name:
        interpolated = np.fft.ifftn(new_matrix) * (interpolation_factor * factor)
    else:
        interpolated = np.real(np.fft.ifftn(new_matrix)) * (
            interpolation_factor * factor
        )
    return interpolated


def change_of_basis(tensor, A, B):
    """changes the basis of a tensor given the column vectors of A and B

    This changes the basis from B to A. The tensor has to be in the A basis.

    Parameters
    ----------
    tensor : np.ndarray
        Rank 1 or rank 2 tensor
    A : np.ndarray
        column vectors of the A basis
    B : np.ndarray
        column vectors of the B basis
    """
    transform = np.linalg.inv(B).dot(A)
    n_dim = len(tensor.shape)
    if n_dim == 1:
        tensor_b = transform.dot(tensor)
    else:
        transform_inv = np.linalg.inv(transform)
        tensor_b = transform_inv.dot(tensor).dot(transform)
        # tensor_b = transform.dot(tensor).dot(transform_inv)
    return tensor_b


def interpolate_nd_3dmesh(
    x_values, y_values, z_values, mesh, interpolation_factor, **kwargs
):
    """Interpolate a Nd 3D mesh using FFT while preserving coordinate ranges and C-ordering.

    Parameters
    ----------
    mesh : np.ndarray
        The mesh to interpolate with shape (nx, ny, nz, ...)
    interpolation_factor : int
        Factor by which to interpolate the mesh

    Returns
    -------
    np.ndarray
        The interpolated mesh with shape (new_nx, new_ny, new_nz, ...)
    """
    scalar_dims = mesh.shape[3:]

    new_mesh_shape = (
        mesh.shape[0] * interpolation_factor,
        mesh.shape[1] * interpolation_factor,
        mesh.shape[2] * interpolation_factor,
        *scalar_dims,
    )

    new_mesh = np.zeros(new_mesh_shape)

    # If this is just a 3D array, use fft_interpolate directly
    if len(scalar_dims) == 0:
        return interpolate_3d_mesh(
            x_values, y_values, z_values, mesh, interpolation_factor, **kwargs
        )

    # For higher dimensional arrays, iterate through the scalar dimensions

    # Iterate through all combinations of scalar indices
    for idx in np.ndindex(*scalar_dims):
        # Extract the 3D grid at the current scalar indices
        # Convert idx to a tuple of slices for proper indexing
        idx_slices = (slice(None), slice(None), slice(None)) + idx

        grid_3d = mesh[idx_slices]  # Take just the first three dimensions

        # Interpolate the 3D grid
        interpolated_grid = interpolate_3d_mesh(
            x_values,
            y_values,
            z_values,
            grid_3d,
            interpolation_factor,
            wrap_axes=[0, 1, 2],
            **kwargs,
        )

        # Store the result in the corresponding location in the new mesh
        new_mesh[idx_slices] = interpolated_grid

    return new_mesh


def interpolate_3d_mesh(
    x_values,
    y_values,
    z_values,
    mesh,
    interpolation_factor,
    wrap_axes=None,
    **kwargs,
):
    """
    Interpolate a 3D scalar mesh using FFT while ensuring coordinates stay
    within the range [-0.5, 0.5] and preventing duplicate points.

    Parameters
    ----------
    x_values : np.ndarray
        The x values of the mesh
    y_values : np.ndarray
        The y values of the mesh
    z_values : np.ndarray
        The z values of the mesh
    mesh : np.ndarray
        The 3D scalar mesh to interpolate, with shape (nx, ny, nz)
    wrap_axes : list, optional
        Dimensions to wrap, by default None
    interpolation_factor : int
        Factor by which to interpolate the mesh

    Returns
    -------
    np.ndarray
        The interpolated mesh with shape (new_nx, new_ny, new_nz)
    """
    # If no wrap dimensions specified, create an empty list
    if wrap_axes is None:
        wrap_axes = []

    # Get original dimensions
    nkx, nky, nkz = mesh.shape

    xmin, xmax = np.min(x_values), np.max(x_values)
    ymin, ymax = np.min(y_values), np.max(y_values)
    zmin, zmax = np.min(z_values), np.max(z_values)

    new_x = np.linspace(xmin, xmax, nkx * interpolation_factor)
    new_y = np.linspace(ymin, ymax, nky * interpolation_factor)
    new_z = np.linspace(zmin, zmax, nkz * interpolation_factor)

    # new_x = np.linspace(0, 1, nkx * interpolation_factor)
    # new_y = np.linspace(0, 1, nky * interpolation_factor)
    # new_z = np.linspace(0, 1, nkz * interpolation_factor)

    padding_x = nkx * 2 // 2
    padding_y = nky * 2 // 2
    padding_z = nkz * 2 // 2

    padded_mesh = np.pad(
        mesh,
        ((padding_x, padding_x), (padding_y, padding_y), (padding_z, padding_z)),
        "wrap",
    )

    padded_x = np.pad(x_values, (padding_x, padding_x), "reflect", reflect_type="odd")
    padded_y = np.pad(y_values, (padding_y, padding_y), "reflect", reflect_type="odd")
    padded_z = np.pad(z_values, (padding_z, padding_z), "reflect", reflect_type="odd")

    interfunc = RegularGridInterpolator(
        (padded_x, padded_y, padded_z), padded_mesh, **kwargs
    )
    new_x_mesh, new_y_mesh, new_z_mesh = np.meshgrid(new_z, new_y, new_x, indexing="ij")
    interpolated_values = interfunc((new_x_mesh, new_y_mesh, new_z_mesh))
    return interpolated_values


def fft_interpolate_nd_3dmesh(mesh, interpolation_factor):
    """Interpolate a Nd 3D mesh using FFT while preserving coordinate ranges and C-ordering.

    Parameters
    ----------
    mesh : np.ndarray
        The mesh to interpolate with shape (nx, ny, nz, ...)
    interpolation_factor : int
        Factor by which to interpolate the mesh

    Returns
    -------
    np.ndarray
        The interpolated mesh with shape (new_nx, new_ny, new_nz, ...)
    """
    scalar_dims = mesh.shape[3:]

    new_mesh_shape = (
        mesh.shape[0] * interpolation_factor,
        mesh.shape[1] * interpolation_factor,
        mesh.shape[2] * interpolation_factor,
        *scalar_dims,
    )

    new_mesh = np.zeros(new_mesh_shape)

    # If this is just a 3D array, use fft_interpolate directly
    if len(scalar_dims) == 0:
        return fft_interpolate_mesh(grid_3d, interpolation_factor)

    # For higher dimensional arrays, iterate through the scalar dimensions

    # Iterate through all combinations of scalar indices
    for idx in np.ndindex(*scalar_dims):
        # Extract the 3D grid at the current scalar indices
        # Convert idx to a tuple of slices for proper indexing
        idx_slices = (slice(None), slice(None), slice(None)) + idx

        grid_3d = mesh[idx_slices]  # Take just the first three dimensions

        # Interpolate the 3D grid
        interpolated_grid = fft_interpolate_mesh(grid_3d, interpolation_factor)

        # Store the result in the corresponding location in the new mesh
        new_mesh[idx_slices] = interpolated_grid

    return new_mesh


def fft_interpolate_mesh(function, interpolation_factor=2):
    """
    This method will interpolate using a Fast-Fourier Transform

    if I = interpolation_factor
    This function will receive f(x,y,z) with dimensions of (nx,ny,nz)
    and returns f(x,y,z) with dimensions of (nx*I,ny*I,nz*I)

    Parameters
    ----------
    function : np.ndarray
        The values array to do the interpolation on.
    interpolation_factor : int, optional
        Interpolation Factor, by default 2

    Returns
    -------
    np.ndarray
        The interpolated points
    """
    # Handle NaN values if present
    has_nan = np.isnan(function).any()
    if has_nan:
        # Replace NaN with zeros for FFT
        function_copy = np.nan_to_num(function, nan=0.0)
    else:
        function_copy = function.copy()

    # Perform FFT

    # Get dimensions of the input array
    nx, ny, nz = function_copy.shape

    # Create larger output array filled with zeros
    new_shape = (
        nx * interpolation_factor,
        ny * interpolation_factor,
        nz * interpolation_factor,
    )
    new_fft = np.zeros(new_shape, dtype=complex)
    # Calculate half-dimensions for proper frequency component placement
    nx_half = nx // 2
    ny_half = ny // 2
    nz_half = nz // 2

    # Copy low frequency components to the new array
    eigen_fft = np.fft.fftn(function_copy)
    new_fft[:nx_half, :ny_half, :nz_half] = eigen_fft[:nx_half, :ny_half, :nz_half]

    new_fft[-nx_half:, :ny_half, :nz_half] = eigen_fft[-nx_half:, :ny_half, :nz_half]
    new_fft[:nx_half, -ny_half:, :nz_half] = eigen_fft[:nx_half, -ny_half:, :nz_half]
    new_fft[:nx_half, :ny_half, -nz_half:] = eigen_fft[:nx_half, :ny_half, -nz_half:]

    new_fft[:nx_half, -ny_half:, -nz_half:] = eigen_fft[:nx_half, -ny_half:, -nz_half:]
    new_fft[-nx_half:, -ny_half:, :nz_half] = eigen_fft[-nx_half:, -ny_half:, :nz_half]
    new_fft[-nx_half:, :ny_half, -nz_half:] = eigen_fft[-nx_half:, :ny_half, -nz_half:]

    new_fft[-nx_half:, -ny_half:, -nz_half:] = eigen_fft[
        -nx_half:, -ny_half:, -nz_half:
    ]

    # Perform inverse FFT to get the interpolated result
    interpolated = np.real(np.fft.ifftn(new_fft)) * interpolation_factor**3
    return interpolated
