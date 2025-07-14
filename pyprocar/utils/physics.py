from typing import Literal

import numpy as np

HBAR_EV = 6.582119 * 10 ** (-16)  # eV*s
HBAR_J = 1.0545718 * 10 ** (-34)  # eV*s
METER_ANGSTROM = 10 ** (-10)  # m /A
EV_TO_J = 1.602 * 10 ** (-19)
FREE_ELECTRON_MASS = 9.11 * 10**-31  #  kg




def calculate_avg_inv_effective_mass(hessian: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]):
    # Calculate the trace of each 3x3 matrix along the last two axes
    m_inv = (np.trace(hessian, axis1=-2, axis2=-1) * EV_TO_J / HBAR_J**2) / 3
    # Calculate the harmonic average effective mass for each element
    e_mass = FREE_ELECTRON_MASS * m_inv
    return e_mass


def calculate_band_velocity(bands_gradient: np.ndarray):
    return bands_gradient / HBAR_EV

def calculate_band_speed(band_velocity: np.ndarray):
    return np.linalg.norm(band_velocity, axis=-1)