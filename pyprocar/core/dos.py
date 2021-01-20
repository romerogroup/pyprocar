# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:49:03 2020

@author: Pedram Tavadze, Logan Lang
"""
from scipy.interpolate import CubicSpline
import numpy as np


class DensityOfStates:
    def __init__(self,
                 energies=None,
                 total=None,
                 projected=None,
                 interpolation_factor=None):
        """
        A class that contains density of states calcuated by the a density
        functional theory calculation.

        Parameters
        ----------
        energies : list float, optional
            List of points on energy spectrum . The default is None.
        total : list float, optional
            List of densities at each point. The default is None.
        projected : list float, optional
            dictionary by the following order
            projected[iatom][iprincipal][iorbital][ispin].
            ``iprincipal`` works like the principal quantum number n. The last
            index should be the total. (iprincipal = -1)
            n = iprincipal => 0, 1, 2, 3, -1 => s, p, d, total
            ``iorbital`` works similar to angular quantum number l, but not the
            same. iorbital follows this order
            (0,1,2,3,4,5,6,7,8) => s,py,pz,px,dxy,dyz,dz2,dxz,dx2-y2.
            ``ispin`` works as magnetic quantum number.
            m = 0,1, for spin up and down
            The default is None.


        interpolation_factor : int, optional
            The number of density of states points will increase by this factor
            in the interpolation.
        Returns
        -------
        None.

        """

        self.energies = energies
        self.total = total
        self.projected = projected

        if interpolation_factor is not None:
            interpolated = []
            for ispin in range(len(self.total)):
                new_energy, new_total = interpolate(
                    self.energies,
                    self.total[ispin],
                    factor=interpolation_factor)
                interpolated.append(new_total)

            self.total = interpolated

            for iatom in range(len(projected)):
                for iprincipal in range(len(projected[iatom])):
                    for iorbital in range(len(projected[iatom][iprincipal])):
                        for ispin in range(
                                len(projected[iatom][iprincipal][iorbital])):
                            x = energies
                            y = projected[iatom][iprincipal][iorbital][ispin]
                            xs, ys = interpolate(x,
                                                 y,
                                                 factor=interpolation_factor)

                            self.projected[iatom][iprincipal][iorbital][
                                ispin] = ys

            self.energies = xs

        self.ndos = len(self.energies)
        self.total = np.array(self.total)

    def dos_sum(self,
                atoms=None,
                principal_q_numbers=[-1],
                orbitals=None,
                spins=None):
        """
        +-------+-----+------+------+------+------+------+------+------+------+
        |n-lm   |  0  |   1  |  2   |   3  |   4  |   5  |   6  |   7  |   8  |
        +=======+=====+======+======+======+======+======+======+======+======+
        |-1(tot)|  s  |  py  |  pz  |  px  | dxy  | dyz  | dz2  | dxz  |x2-y2 |
        +=======+=====+======+======+======+======+======+======+======+======+
        |   0   |  s  |      |      |      |      |      |      |      |      |
        +=======+=====+======+======+======+======+======+======+======+======+
        |   1   |  s  |  py  |  pz  |  px  |      |      |      |      |      |
        +=======+=====+======+======+======+======+======+======+======+======+
        |   2   |  s  |  py  |  pz  |  px  | dxy  | dyz  | dz2  | dxz  |x2-y2 |
        +=======+=====+======+======+======+======+======+======+======+======+
        |  ...  | ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |
        +=======+=====+======+======+======+======+======+======+======+======+


        Parameters
        ----------
        atoms : list int, optional
             list of atom index needed to be sumed over.
             count from zero with the same
             order as input. The default is None.
        principal_q_numbers : list int, optional
            list of . The default is [-1].
        orbitals : TYPE, optional
            DESCRIPTION. The default is None.
        spins : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        ret : list float
            .

        """

        projected = self.projected
        principal_q_numbers = np.array(principal_q_numbers)
        if atoms is None:
            atoms = np.arange(len(projected), dtype=int)
        if spins is None:
            spins = np.arange(len(projected[0][0][0]), dtype=int)

        if orbitals is None:
            orbitals = np.arange(len(projected[0][0]), dtype=int)
        orbitals = np.array(orbitals)

        ret = np.zeros(shape=(2, self.ndos))
        for iatom in atoms:
            for iprinc in principal_q_numbers:
                for ispin in spins:
                    temp = np.array(projected[iatom][iprinc])

                    ret[ispin, :] += temp[orbitals, ispin].sum(axis=0)

        return ret


def interpolate(x, y, factor=2):
    """
    Interplates the function y=f(x) by increasing the x points by the factor.

    Parameters
    ----------
    x : list float
        points of x.
    y : list float
        points of y=f(x).
    factor : int, optional
        Multiplies the number of x points by this factor. The default is 2.

    Returns
    -------
    xs : list float
        points in which y was interpolated.
    ys : list float
        interpolated points.

    """

    cs = CubicSpline(x, y)
    xs = np.linspace(min(x), max(x), len(x) * factor)
    ys = cs(xs)

    return xs, ys


