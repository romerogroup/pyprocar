"""This module defines an object to handle the density of states in a DFT
calculations.
"""

__author__ = "Pedram Tavadze, Logan Lang"
__copyright__ = "Copyright (C) 2007 Free Software Foundation,"
__credits__ = ["Uthpala Herath"]
__license__ = "GNU GENERAL PUBLIC LICENSE"
__maintainer__ = "Logan Lang, Pedram Tavadze"
__email__ = "petavazohi@mix.wvu.edu"
__status__ = "Production"

import logging
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from sympy.physics.quantum.cg import CG

from pyprocar.core.serializer import get_serializer

logger = logging.getLogger(__name__)

class DensityOfStates:
    """A class that contains density of states calculated by the a density
    functional theory calculation.

    Parameters
    ----------
    energies : np.ndarray,
        Points on energy spectrum. shape = (n_dos, )
    total : np.ndarray
        Densities at each point. shape = (n_dos, )
    efermi : float
        Fermi energy of the system.
    projected : np.ndarray, optional
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

    def __init__(
        self,
        energies: npt.NDArray[np.float64],
        total: npt.NDArray[np.float64],
        efermi: float,
        projected: npt.NDArray[np.float64] = None,
        interpolation_factor: int = 1,
        # interpolation_kind: str = 'cubic',
    ):

        self.energies = energies
        self.total = total
        self.efermi = efermi
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
                        for i_spin in range(
                            len(projected[i_atom][i_principal][i_orbital])
                        ):
                            x = energies
                            y = projected[i_atom][i_principal][i_orbital][i_spin]
                            xs, ys = interpolate(x, y, factor=interpolation_factor)

                            self.projected[i_atom][i_principal][i_orbital][i_spin] = ys

            self.energies = xs

        self.total = np.array(self.total)
        self.projected = np.array(self.projected)
        
    def __repr__(self):
        repr_str = "DensityOfStates(\n"
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                repr_str += f"    {key}: {value.shape}\n"
            else:
                repr_str += f"    {key}: {value}\n"
        repr_str += ")"
        return repr_str

    def __eq__(self, other):
        energies_equal = np.allclose(self.energies, other.energies)
        total_equal = np.allclose(self.total, other.total)
        efermi_equal = self.efermi == other.efermi
        projected_equal = np.allclose(self.projected, other.projected)

        n_spins_equal = self.n_spins == other.n_spins

        dos_equal = (
            energies_equal
            and total_equal
            and efermi_equal
            and projected_equal
            and n_spins_equal
        )

        return dos_equal
    
    

    @property
    def n_dos(self):
        """The number of dos values

        Returns
        -------
        int
            The number of dos values
        """
        return len(self.energies)

    @property
    def n_energies(self):
        """The number of energy values

        Returns
        -------
        int
            The number of energy values
        """
        return self.n_dos

    @property
    def n_spins(self):
        """The number of spin channels

        Returns
        -------
        int
            The number of spin channels
        """
        return len(self.total)

    @property
    def is_non_collinear(self):
        """Boolean for if this is non-colinear calc

        Returns
        -------
        bool
            Boolean for if this is non-colinear calc
        """
        # last condition is for quantum espresso total angular momentum basis
        if self.n_spins == 3 or self.n_spins == 4 or len(self.projected[0][0]) == 2 + 2 + 4 + 4 + 6 + 6 + 8:
            return True
        else:
            return False

    def dos_sum(
        self,
        atoms: List[int] = None,
        principal_q_numbers: List[int] = [-1],
        orbitals: List[int] = None,
        spins: List[int] = None,
        sum_noncolinear: bool = True,
    ):
        """

        .. code-block::
            :linenos:

            Orbital table

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
        atoms : List[int], optional
             list of atom index needed to be sumed over.
             count from zero with the same
             order as input. The default is None.
        principal_q_numbers :List[int], optional
            List of n quantum numbers to be summed over. The default is [-1].
        orbitals : List[int], optional
            List of orbitals to be summed over. The default is None.
        spins : List[int], optional
            List of spins to be summed over. The default is None.

        Returns
        -------
        ret : list float
            The summed density of states.

        """

        projected = self.projected

        principal_q_numbers = np.array(principal_q_numbers)
        if atoms is None:
            atoms = np.arange(len(projected), dtype=int)
            
        if spins is None and self.n_spins == 4:
            raise ValueError("Spins must be provided for non-colinear calculations. This is because the spin projections cannot be summed over")
        elif spins is None and self.n_spins != 4:
            spins = np.arange(len(projected[0][0][0]), dtype=int)
        
        if orbitals is None:
            orbitals = np.arange(len(projected[0][0]), dtype=int)
            
        logger.debug(f"Summing over atoms: {atoms}")
        logger.debug(f"Summing over principal_q_numbers: {principal_q_numbers}")
        logger.debug(f"Summing over orbitals: {orbitals}")
        
        # print(orbitals)
        
        # Adjusting for spin type calculation
        if self.n_spins == 2:
            ret = np.zeros(shape=(2, self.n_dos))
            for iatom in atoms:
                for iprinc in principal_q_numbers:
                    for ispin in spins:
                        temp = np.array(projected[iatom][iprinc])
                        ret[ispin, :] += temp[orbitals, ispin].sum(axis=0)

        else:
            ret = np.zeros(shape=(1, self.n_dos))
            for iatom in atoms:
                for iprinc in principal_q_numbers:
                    for ispin in spins:
                        temp = np.array(projected[iatom][iprinc])
                        ret[0, :] += temp[orbitals, ispin].sum(axis=0)

        return ret


    def get_current_basis(self):
        """Returns a string of current orbital basis

        Returns
        -------
        str
            Returns a string of current orbital basis
        """

        n_orbitals = self.projected.shape[2]

        if n_orbitals == 18:
            basis = "jm basis"
        elif n_orbitals == 9:
            basis = "spd basis"
        elif n_orbitals == 9:
            basis = "spdf basis"
        else:
            basis = "I do not know"
        return basis

    def coupled_to_uncoupled_basis(self):
        """
        Converts coupled projections to uncoupled projections. This assumes the orbitals are order by

        .. code-block::
            :linenos:

            coupled_orbitals = [
                            {"l": 's', "j": 0.5, "m": -0.5},
                            {"l": 's', "j": 0.5, "m": 0.5},

                            {"l": 'p', "j": 0.5, "m": -0.5},
                            {"l": 'p', "j": 0.5, "m": 0.5},

                            {"l": 'p', "j": 1.5, "m": -1.5},
                            {"l": 'p', "j": 1.5, "m": -0.5},
                            {"l": 'p', "j": 1.5, "m": -0.5},
                            {"l": 'p', "j": 1.5, "m": 1.5},

                            {"l": 'd', "j": 1.5, "m": -1.5},
                            {"l": 'd', "j": 1.5, "m": -0.5},
                            {"l": 'd', "j": 1.5, "m": -0.5},
                            {"l": 'd', "j": 1.5, "m": 1.5},

                            {"l": 'd', "j": 2.5, "m": -2.5},
                            {"l": 'd', "j": 2.5, "m": -1.5},
                            {"l": 'd', "j": 2.5, "m": -0.5},
                            {"l": 'd', "j": 2.5, "m": 0.5},
                            {"l": 'd', "j": 2.5, "m": 1.5},
                            {"l": 'd', "j": 2.5, "m": 2.5},
                        ]
            uncoupled_orbitals = [
                        {"l": 0, "m": 1},
                        {"l": 1, "m": 3},
                        {"l": 1, "m": 1},
                        {"l": 1, "m": 2},
                        {"l": 2, "m": 5},
                        {"l": 2, "m": 3},
                        {"l": 2, "m": 1},
                        {"l": 2, "m": 2},
                        {"l": 2, "m": 4},
                    ]

        Returns
        -------
        ret :  None
            None

        """
        n_atoms = self.projected.shape[0]
        n_principle = self.projected.shape[1]
        n_uncoupled_orbitals = self.projected.shape[2]
        n_spins = self.projected.shape[3]
        n_energy = self.projected.shape[4]

        uncoupled_projected = np.zeros(
            shape=(n_atoms, n_principle, n_uncoupled_orbitals, 2, n_energy)
        )

        def x(proj_1, proj_2, clebsh_1, clebsch_2, clebsch_3, clebsch_4):
            a = proj_2
            b = (
                clebsch_4
                * (proj_1 - (clebsh_1 / clebsch_3) * proj_2)
                / (clebsch_2 - (clebsh_1 / clebsch_3))
            )
            return (a - b) / clebsch_3

        def y(proj_1, proj_2, clebsh_1, clebsch_2, clebsch_3, clebsch_4):
            a = proj_1
            b = (clebsh_1 / clebsch_3) * proj_2
            c = clebsch_2 - (clebsh_1 / clebsch_3)
            return (a - b) / c

        paired_uncoupled_obritals = [
            [0, 1],
            [2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ]
        print(float(CG(j1=1 / 2, m1=-1 / 2, j2=1 / 2, m2=+1 / 2, j3=1, m3=0).doit()))

        uncoupled_projected[:, :, 0, :, :] = (
            self.projected[:, :, 0, :, :] + self.projected[:, :, 1, :, :]
        )

        return None

    def normalize_dos(self, mode="max"):
        """
        Normalizes the density of states.

        Returns
        -------
        None
            None
            The density of states is normalized.
        """
        possible_modes = ["max", "integral"]
        if mode not in possible_modes:
            raise ValueError(f"The mode must be {possible_modes}")
        if mode == "max":
            for i in range(len(self.total)):
                self.total[i] = self.total[i] / np.max(self.total[i])
        elif mode == "integral":
            for i in range(len(self.total)):
                y = self.total[i]
                x = self.energies
                integral = trapezoid(y, x=self.energies)
                self.total[i] = self.total[i] / integral
        return None

    def save(self, path: Path):
        serializer = get_serializer(path)
        serializer.save(self, path)

    @classmethod
    def load(cls, path: Path):
        serializer = get_serializer(path)
        return serializer.load(path)


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
