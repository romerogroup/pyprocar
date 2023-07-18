# -*- coding: utf-8 -*-

import numpy as np


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

def change_of_basis(tensor,A,B):
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
