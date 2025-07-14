# -*- coding: utf-8 -*-
import logging
from typing import Dict

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.signal import find_peaks

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


def calculate_central_differences_on_meshgrid_axis(scalar_mesh, axis):
    """Calculates the scalar differences over the
    k mesh grid using central differences

    Parameters
    ----------
    scalar_mesh : np.ndarray
        The scalar mesh. shape = [n_kx,n_ky,n_kz]

    Returns
    -------
    np.ndarray
        scalar_gradient_mesh shape = [n_kx,n_ky,n_kz]
    """
    n = scalar_mesh.shape[axis]
    # Calculate indices with periodic boundary conditions
    plus_one_indices = np.arange(n) + 1
    minus_one_indices = np.arange(n) - 1
    plus_one_indices[-1] = 0
    minus_one_indices[0] = n - 1

    if axis == 0:
        return (
            scalar_mesh[plus_one_indices, ...] - scalar_mesh[minus_one_indices, ...]
        ) / 2
    elif axis == 1:
        return (
            scalar_mesh[:, plus_one_indices, :, ...]
            - scalar_mesh[:, minus_one_indices, :, ...]
        ) / 2
    elif axis == 2:
        return (
            scalar_mesh[:, :, plus_one_indices, ...]
            - scalar_mesh[:, :, minus_one_indices, ...]
        ) / 2


def calculate_forward_averages_on_meshgrid_axis(scalar_mesh, axis):
    """Calculates the scalar differences over the
    k mesh grid using central differences

    Parameters
    ----------
    scalar_mesh : np.ndarray
        The scalar mesh. shape = [n_kx,n_ky,n_kz]

    Returns
    -------
    np.ndarray
        scalar_gradient_mesh shape = [n_kx,n_ky,n_kz]
    """
    n = scalar_mesh.shape[axis]

    # Calculate indices with periodic boundary conditions
    plus_one_indices = np.arange(n) + 1
    zero_one_indices = np.arange(n)
    plus_one_indices[-1] = 0
    if axis == 0:
        return (
            scalar_mesh[zero_one_indices, ...] + scalar_mesh[plus_one_indices, ...]
        ) / 2
    elif axis == 1:
        return (
            scalar_mesh[:, zero_one_indices, :, ...]
            + scalar_mesh[:, plus_one_indices, :, ...]
        ) / 2
    elif axis == 2:
        return (
            scalar_mesh[:, :, zero_one_indices, ...]
            + scalar_mesh[:, :, plus_one_indices, ...]
        ) / 2


def calculate_scalar_volume_averages(scalar_mesh):
    """Calculates the scalar averages over the k mesh grid in cartesian coordinates"""
    scalar_sums_i = calculate_forward_averages_on_meshgrid_axis(scalar_mesh, axis=0)
    scalar_sums_j = calculate_forward_averages_on_meshgrid_axis(scalar_mesh, axis=1)
    scalar_sums_k = calculate_forward_averages_on_meshgrid_axis(scalar_mesh, axis=2)
    scalar_sums = (scalar_sums_i + scalar_sums_j + scalar_sums_k) / 3
    return scalar_sums


def calculate_scalar_differences(scalar_mesh):
    """Calculates the scalar gradient over the k mesh grid in cartesian coordinates

    Uses gradient trnasformation matrix to calculate the gradient
    scalar_differens are calculated by central differences


    Parameters
    ----------
    scalar_mesh : np.ndarray
        The scalar mesh. shape = [n_kx,n_ky,n_kz,...,3]
    """
    scalar_diffs_i = calculate_central_differences_on_meshgrid_axis(scalar_mesh, axis=0)
    scalar_diffs_j = calculate_central_differences_on_meshgrid_axis(scalar_mesh, axis=1)
    scalar_diffs_k = calculate_central_differences_on_meshgrid_axis(scalar_mesh, axis=2)
    scalar_diffs = np.array([scalar_diffs_i, scalar_diffs_j, scalar_diffs_k])
    scalar_diffs = np.moveaxis(scalar_diffs, 0, -1)
    return scalar_diffs


def calculate_scalar_differences_2(scalar_mesh, transform_matrix):
    """Calculates the scalar gradient over the k mesh grid in cartesian coordinates

    Uses gradient trnasformation matrix to calculate the gradient
    scalar_differens are calculated by central differences


    Parameters
    ----------
    scalar_mesh : np.ndarray
        The scalar mesh. shape = [n_kx,n_ky,n_kz,...,3]
    """
    scalar_diffs_i = calculate_central_differences_on_meshgrid_axis(scalar_mesh, axis=0)
    scalar_diffs_j = calculate_central_differences_on_meshgrid_axis(scalar_mesh, axis=1)
    scalar_diffs_k = calculate_central_differences_on_meshgrid_axis(scalar_mesh, axis=2)
    scalar_diffs = np.array([scalar_diffs_i, scalar_diffs_j, scalar_diffs_k])
    scalar_diffs = np.moveaxis(scalar_diffs, 0, -1)

    scalar_diffs_2 = np.einsum("ij,uvwj->uvwi", transform_matrix, scalar_diffs)
    return scalar_diffs_2


def calculate_3d_mesh_scalar_gradients(
    scalar_array,
    reciprocal_lattice,
):
    """Transforms the derivatives to cartesian coordinates
        (n,j,k,...)->(n,j,k,...,3)

    Parameters
    ----------
    derivatives : np.ndarray
        The derivatives to transform
    reciprocal_lattice : np.ndarray
        The reciprocal lattice

    Returns
    -------
    np.ndarray
        The transformed derivatives
    """
    # expanded_freq_mesh = []

    # for i in range(ndim):
    #     # Start with the original frequency mesh component
    #     component = freq_mesh[i]

    #     # For each extra dimension in scalar_grid, expand the frequency mesh
    #     for dim_size in extra_dims:
    #         component = np.expand_dims(component, axis=-1)
    #         # Repeat the values along the new axis
    #         component = np.repeat(component, dim_size, axis=-1)

    #     expanded_freq_mesh.append(component)
    # return fourier_reciprocal_gradient(scalar_array, reciprocal_lattice)

    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    scalar_diffs = calculate_scalar_differences(scalar_array)

    del_k1 = 1 / scalar_diffs.shape[0]
    del_k2 = 1 / scalar_diffs.shape[1]
    del_k3 = 1 / scalar_diffs.shape[2]

    scalar_diffs[..., 0] = scalar_diffs[..., 0] / del_k1
    scalar_diffs[..., 1] = scalar_diffs[..., 1] / del_k2
    scalar_diffs[..., 2] = scalar_diffs[..., 2] / del_k3

    n_dim = len(scalar_diffs.shape[3:]) - 1
    transform_matrix_einsum_string = "ij"
    dim_letters = "".join(letters[0:n_dim])
    scalar_array_einsum_string = "uvw" + dim_letters + "j"
    transformed_scalar_string = "uvw" + dim_letters + "i"
    ein_sum_string = (
        transform_matrix_einsum_string
        + ","
        + scalar_array_einsum_string
        + "->"
        + transformed_scalar_string
    )
    logger.debug(f"ein_sum_string: {ein_sum_string}")

    scalar_gradients = np.einsum(
        ein_sum_string,
        np.linalg.inv(reciprocal_lattice.T).T,
        scalar_diffs,
    )
    # scalar_gradients = np.einsum(ein_sum_string, reciprocal_lattice.T, scalar_diffs)

    return scalar_gradients


def calculate_3d_mesh_scalar_integral(scalar_mesh, reciprocal_lattice):
    """Calculate the scalar integral"""
    n1, n2, n3 = scalar_mesh.shape[:3]
    volume_reduced_vector = np.array([1, 1, 1])
    volume_cartesian_vector = np.dot(reciprocal_lattice, volume_reduced_vector)
    volume = np.prod(volume_cartesian_vector)
    dv = volume / (n1 * n2 * n3)

    scalar_volume_avg = calculate_scalar_volume_averages(scalar_mesh)
    # Compute the integral by summing up the product of scalar values and the volume of each grid cell.
    integral = np.sum(scalar_volume_avg * dv, axis=(0, 1, 2))

    return integral


def q_multi(q1, q2):
    """
    Multiplication of quaternions, it doesn't fit in any other place
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array((w, x, y, z))


def fourier_reciprocal_gradient(scalar_grid, reciprocal_lattice):
    """
    Calculate the reciprocal space gradient of a scalar field using Fourier methods.
    It first finds the gradient in the fractional basis,
    and then transforms to cartesian coordinates through the reciprocal lattice vectors.
    Units of angstoms and eV are assumed.

    Parameters:
    -----------
    scalar_grid : ndarray
        N-dimensional array of scalar values on a mesh grid
    dk_values : tuple or list
        Grid spacing in each dimension
    reciprocal_lattice : ndarray, optional
        Reciprocal lattice vectors for non-orthogonal grids

    Returns:
    --------
    gradient : list of ndarrays
        List of gradient components, one for each dimension
    """
    # Get dimensions of the grid
    scalar_grid_shape = scalar_grid.shape
    ndim = reciprocal_lattice.shape[0]

    # Create frequency meshgrid

    nx = scalar_grid_shape[0]
    ny = scalar_grid_shape[1]
    nz = scalar_grid_shape[2]

    dk_values = np.array([1 / nx, 1 / ny, 1 / nz])

    wavenumbers = []
    for i in range(ndim):
        wavenumbers_1d_full = (
            np.fft.fftfreq(scalar_grid_shape[i], d=dk_values[i]) * 2 * np.pi
        )
        wavenumbers.append(wavenumbers_1d_full)

    freq_mesh = np.stack(np.meshgrid(*wavenumbers, indexing="ij"))
    # Get the shape of scalar_grid beyond the first 3 dimensions (if any)
    extra_dims = scalar_grid_shape[3:] if len(scalar_grid_shape) > 3 else ()

    # Expand freq_mesh to match the expected scalar_gradient_grid_shape
    # First, create a list to hold the expanded dimensions
    # expanded_freq_mesh = []

    # for i in range(ndim):
    #     # Start with the original frequency mesh component
    #     component = freq_mesh[i]

    #     # For each extra dimension in scalar_grid, expand the frequency mesh
    #     for dim_size in extra_dims:
    #         component = np.expand_dims(component, axis=-1)
    #         # Repeat the values along the new axis
    #         component = np.repeat(component, dim_size, axis=-1)

    #     expanded_freq_mesh.append(component)

    # # Replace the original freq_mesh with the expanded version
    # freq_mesh = np.stack(expanded_freq_mesh)
    # print(freq_mesh.shape)
    scalar_gradient_grid_shape = (3, *scalar_grid_shape)
    print(scalar_gradient_grid_shape)
    # Standard orthogonal case
    derivative_operator = freq_mesh * 1j
    spectral_derivative = np.fft.ifftn(
        derivative_operator * np.fft.fftn(scalar_grid),
        s=(3, nx, ny, nz),
    )

    derivatives = np.real(spectral_derivative)

    print(derivatives.shape)
    derivatives = np.moveaxis(derivatives, 0, -1)

    print(f"Derivatives shape: {derivatives.shape}")
    # cart_derivatives = np.dot(derivatives, np.linalg.inv(reciprocal_lattice.T))

    transform_matrix_einsum_string = "ij"
    ndim = len(derivatives.shape[3:]) - 1
    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    dim_letters = "".join(letters[0:ndim])
    scalar_array_einsum_string = "uvw" + dim_letters + "j"
    transformed_scalar_string = "uvw" + dim_letters + "i"
    ein_sum_string = (
        transform_matrix_einsum_string
        + ","
        + scalar_array_einsum_string
        + "->"
        + transformed_scalar_string
    )
    logger.debug(f"ein_sum_string: {ein_sum_string}")

    cart_derivatives = np.einsum(
        ein_sum_string,
        np.linalg.inv(reciprocal_lattice.T).T,
        derivatives,
    )

    return cart_derivatives


def ravel_array(mesh_grid):
    shape = mesh_grid.shape
    mesh_grid = mesh_grid.reshape(shape[:-3] + (-1,))
    mesh_grid = np.moveaxis(mesh_grid, -1, 0)
    return mesh_grid


def array_to_mesh(array, nkx, nky, nkz, order="F"):
    """
    Converts a list to a mesh that corresponds to ebs.kpoints
    [n_kx*n_ky*n_kz,...]->[n_kx,n_ky,n_kz,...]. Make sure array is sorted by lexisort

    Parameters
    ----------
    array : np.ndarray
        The array to convert to a mesh
    nkx : int
        The number of kx points
    nky : int
        The number of ky points
    nkz : int
        The number of kz points
    order : str, optional
        The order of the array. Defaults to "C"

    Returns
    -------
    np.ndarray
        mesh
    """
    prop_shape = (
        nkx,
        nky,
        nkz,
    ) + array.shape[1:]

    try:
        scalar_grid = array.reshape(prop_shape, order=order)
    except ValueError:
        error_msg = "This array can not be converted to a 3d mesh.\n"
        error_msg += f"Array shape: {array.shape}\n"
        error_msg += f"Prop shape: {prop_shape}\n"
        error_msg += f"Order: {order}"
        raise ValueError(error_msg)

    return scalar_grid


def mesh_to_array(mesh, order="F"):
    """
    Converts a mesh to a list that corresponds to ebs.kpoints
    [n_kx,n_ky,n_kz,...]->[n_kx*n_ky*n_kz,...]
    Parameters
    ----------
    mesh : np.ndarray
        The mesh to convert to a list
    order : str, optional
        The order of the array. Defaults to "C"

    Returns
    -------
    np.ndarray
        lsit
    """
    if mesh is None:
        return None
    nkx, nky, nkz = mesh.shape[:3]
    prop_shape = (nkx * nky * nkz,) + mesh.shape[3:]
    array = mesh.reshape(prop_shape, order=order)
    return array


def get_padding_dims(n_coords, padding):
    if n_coords == 1:
        return 1
    else:
        return n_coords + 2 * padding
    
def get_coord_diffs(coords):
    if len(coords) == 1:
        return 0
    else:
        return np.diff(coords)
    
    
def get_grid_dims(points, num_bins=1000, height=1, coord_tol=0.01):
    grid = np.zeros(3, dtype=int)
    
    for icoord in range(3):
        coords = points[:, icoord]
        coord_min, coord_max = np.min(coords), np.max(coords)
        hist, bin_edges = np.histogram(coords, bins=num_bins, range=(coord_min-coord_tol, coord_max+coord_tol))

        peaks, _ = find_peaks(hist, height=height)
        grid[icoord] = len(peaks)
    return grid


def compare_arrays(array1: np.ndarray, array2: np.ndarray) -> bool:
    if array1 is not None and array2 is not None:
        return np.allclose(array1, array2)
    elif array1 is None and array2 is None:
        return True
    else:
        return False