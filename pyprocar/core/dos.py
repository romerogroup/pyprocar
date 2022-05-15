"""This module defines an object to handle the density of states in a DFT 
calculations.
"""

__author__ = "Pedram Tavadze, Logan Lang"
__copyright__ = "Copyright (C) 2007 Free Software Foundation,"
__credits__ = ["Uthpala Herath"]
__license__ = "GNU GENERAL PUBLIC LICENSE"
__version__ = "2.0"
__maintainer__ = "Logan Lang, Pedram Tavadze"
__email__ = "petavazohi@spot.colorado.edu"
__status__ = "Production"

from scipy.interpolate import CubicSpline
import numpy as np
import numpy.typing as npt

# TODO When PEP 646 is introduced in numpy. need to update the python typing.

class DensityOfStates:
    def __init__(
        self, 
        energies: npt.NDArray[np.float64],
        total: npt.NDArray[np.float64], 
        projected: npt.NDArray[np.float64] = None, 
        interpolation_factor: int = 1,
        interpolation_kind: str = 'cubic',
    ):
        """A class that contains density of states calculated by the a density
        functional theory calculation.

        Parameters
        ----------
        energies : npt.NDArray[np.float64]
            Points on energy spectrum. shape = (n_dos, )
        total : npt.NDArray[np.float64]
            Densities at each point. shape = (n_dos, )
        projected : npt.NDArray[np.float64], optional
            Projection of elements, orbitals, spin, etc. shape = (n_atoms, n_principals, n_orbitals, n_spins, n_dos)
            ``i_principal`` works like the principal quantum number n. The last
            index should be the total. (i_principal = -1)
            n = i_principal => 0, 1, 2, 3, -1 => s, p, d, total
            ``i_orbital`` works similar to angular quantum number l, but not the
            same. i_orbital follows this order
            (0, 1, 2, 3, 4, 5, 6, 7, 8) => s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2.
            ``i_spin`` works as magnetic quantum number.
            m = 0, 1, for spin up and down, by default None.
        interpolation_factor : int, optional
            The number of density of states points will increase by this factor
            in the interpolation, by default 1.
            
        """        


        self.energies = energies
        self.total = total
        self.projected = projected

        if interpolation_factor not in [1, 0]:
            interpolated = []
            for i_spin in range(len(self.total)):
                new_energy, new_total = interpolate(
                    self.energies, self.total[i_spin], factor=interpolation_factor
                )
                interpolated.append(new_total)

            self.total = interpolated

            for i_atom in range(len(projected)):
                for i_principal in range(len(projected[i_atom])):
                    for i_orbital in range(len(projected[i_atom][i_principal])):
                        for i_spin in range(len(projected[i_atom][i_principal][i_orbital])):
                            x = energies
                            y = projected[i_atom][iprincipal][i_orbital][i_spin]
                            xs, ys = interpolate(x, y, factor=interpolation_factor)

                            self.projected[i_atom][i_principal][i_orbital][i_spin] = ys

            self.energies = xs

        self.total = np.array(self.total)
        self.projected = np.array(self.projected)

    @property
    def n_dos(self):
        return len(self.energies)
    
    @property
    def n_energies(self):
        return self.n_dos

    @property
    def n_spins(self):
        return len(self.total)
    
    @property
    def is_non_collinear(self):
        if self.n_spins == 3:
            return True
        else:
            return False

    def dos_sum(self, atoms=None, principal_q_numbers=[-1], orbitals=None, spins=None):
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

        ret = np.zeros(shape=(2, self.n_dos))
        for iatom in atoms:
            for iprinc in principal_q_numbers:
                for ispin in spins:
                    temp = np.array(projected[iatom][iprinc])

                    ret[ispin, :] += temp[orbitals, ispin].sum(axis=0)

        return ret


def interpolate(x, y, factor=2):
    """
    Interplates the function y=f(x) by increasing the x points by the factor.
    # TODO need to add ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’

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
