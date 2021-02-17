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
