from __future__ import annotations

import numpy as np
import numpy.typing as npt

HBAR_EV = 6.582119 * 10 ** (-16)  # eV*s
HBAR_J = 1.0545718 * 10 ** (-34)  # eV*s
METER_ANGSTROM = 10 ** (-10)  # m /A
EV_TO_J = 1.602 * 10 ** (-19)
FREE_ELECTRON_MASS = 9.11 * 10**-31  #  kg


def calculate_avg_inv_effective_mass(
    hessian: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate the average inverse effective mass from band structure Hessian.

    Parameters
    ----------
    hessian : npt.NDArray[np.float64]
        The Hessian matrix of the band structure with shape (..., 3, 3).

    Returns
    -------
    npt.NDArray[np.float64]
        The average inverse effective mass.
    """
    # Calculate the trace of each 3x3 matrix along the last two axes
    m_inv = (np.trace(hessian, axis1=-2, axis2=-1) * EV_TO_J / HBAR_J**2) / 3
    # Calculate the harmonic average effective mass for each element
    e_mass: npt.NDArray[np.float64] = FREE_ELECTRON_MASS * m_inv
    return e_mass


def calculate_band_velocity(
    bands_gradient: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate band velocity from band gradient.

    Parameters
    ----------
    bands_gradient : npt.NDArray[np.float64]
        The gradient of the band structure.

    Returns
    -------
    npt.NDArray[np.float64]
        The band velocity.
    """
    result: npt.NDArray[np.float64] = bands_gradient / HBAR_EV
    return result


def calculate_band_speed(
    band_velocity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate band speed from band velocity.

    Parameters
    ----------
    band_velocity : npt.NDArray[np.float64]
        The band velocity.

    Returns
    -------
    npt.NDArray[np.float64]
        The band speed (magnitude of velocity).
    """
    result: npt.NDArray[np.float64] = np.linalg.norm(band_velocity, axis=-1)
    return result
