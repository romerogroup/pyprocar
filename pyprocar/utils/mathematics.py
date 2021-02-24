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


def fft_interpolate(function, interpolation_factor=2):
    """


    Parameters
    ----------
    function : TYPE
        DESCRIPTION.
    interpolation_factor : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    interpolated : TYPE
        DESCRIPTION.

    """
    function = np.array(function)
    eigen_fft = np.fft.fftn(function)
    shifted_fft = np.fft.fftshift(eigen_fft)
    pad_width = []
    for idim in range(function.ndim):
        n = shifted_fft.shape[idim]
        pad = n * (interpolation_factor - 1) // 2
        pad_width.append([pad, pad])
    new_matrix = np.pad(shifted_fft, pad_width, "constant", constant_values=0)
    new_matrix = np.fft.ifftshift(new_matrix)
    if "complex" in function.dtype.name:
        interpolated = np.fft.ifftn(new_matrix) * (interpolation_factor * function.ndim)
    else:
        interpolated = np.real(np.fft.ifftn(new_matrix)) * (
            interpolation_factor * function.ndim
        )
    return interpolated
