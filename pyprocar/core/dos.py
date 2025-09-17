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

from pyprocar.core.property_store import PointSet, Property
from pyprocar.core.serializer import get_serializer

logger = logging.getLogger(__name__)


def get_dos_from_code(
    code: str,
    dirpath: str,
    use_cache: bool = False,
    filename: str = "dos.pkl"
    ):

    from pyprocar.io import Parser
    dos_filepath = Path(dirpath) / filename
    
    if not use_cache or not dos_filepath.exists():
        logger.info(f"Parsing EBS calculation directory: {dirpath}")
        parser = Parser(code=code, dirpath=dirpath)
        dos=parser.dos
        dos.save(dos_filepath)
    else:
        logger.info(f"Loading EBS  from picklefile: {dos_filepath}")
        dos = DensityOfStates.load(dos_filepath)
        
    return dos
    



class DensityOfStates(PointSet):
    """A class that contains density of states calculated by the a density
    functional theory calculation.

    Parameters
    ----------
    energies : np.ndarray,
        Points on energy spectrum. shape = (n_dos, )
    total : np.ndarray
        Densities at each point. shape = (n_dos, )
    fermi : float
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
        fermi: float = 0.0,
        projected: npt.NDArray[np.float64] = None,
        orbital_names: list[str] | None = None,
    ):
        logger.info(f"Initializing DensityOfStates")
        logger.debug(f"energies: {energies.shape}")
        logger.debug(f"total: {total.shape}")
        logger.debug(f"projected: {projected.shape}")
        super().__init__(energies)
        
        
        projected = np.array(projected)
        self.add_property(name="total", value=total)
        
        if projected is not None:
            self.add_property(name="projected", value=projected)
            
        self._orbital_names = orbital_names      
        
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
        fermi_equal = self.fermi == other.fermi
        projected_equal = np.allclose(self.projected, other.projected)

        n_spins_equal = self.n_spins == other.n_spins

        dos_equal = (
            energies_equal
            and total_equal
            and fermi_equal
            and projected_equal
            and n_spins_equal
        )

        return dos_equal
    
    @classmethod
    def from_code(cls, code: str, dirpath: str, use_cache: bool = False, filename: str = "dos.pkl"):
        return get_dos_from_code(code, dirpath, use_cache, filename)
    
    @classmethod
    def load(cls, path: Path):
        serializer = get_serializer(path)
        dos = serializer.load(path)
        return dos
    
    def save(self, path: Path):
        serializer = get_serializer(path)
        serializer.save(self, path)
    
    @property
    def energies(self) -> npt.NDArray[np.float64]:
        return self._points
    
    @property
    def total(self) -> npt.NDArray[np.float64]:
        return self.get_property("total").value
    
    @property
    def projected(self) -> npt.NDArray[np.float64]:
        return self.get_property("projected").value
    
    @property
    def orbital_names(self):
        return self._orbital_names

    @property
    def n_energies(self) -> int:
        return self._points.shape[0]
    
    @property
    def n_dos(self):
        """The number of dos values

        Returns
        -------
        int
            The number of dos values
        """
        return self.n_energies
    
    @property
    def n_spin_channels(self):
        """The number of spin channels

        Returns
        -------
        int
            The number of spin channels
        """
        return self.total.shape[1]
    
    @property
    def n_spins(self):
        """The number of spin projections

        Returns
        -------
        int
            The number of spin projections
        """
        return self.projected.shape[1]

    @property
    def n_atoms(self):
        """The number of atoms

        Returns
        -------
        int
            The number of atoms
        """
        return self.projected.shape[2]

    @property
    def n_orbitals(self):
        """The number of orbitals

        Returns
        -------
        int
            The number of orbitals
        """
        return self.projected.shape[3]

    

    @property
    def spin_channels(self):
        """The number of spin channels

        Returns
        -------
        int
            The number of spin channels
        """

        return np.arange(self.n_spin_channels)
    
    @property
    def is_spin_polarized(self):
        """Boolean for if this is spin polarized
        """
        return self.n_spins == 2
    
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

    @property
    def spin_projection_names(self):
        spin_projection_names = ["Spin-up", "Spin-down"]
        if self.is_non_collinear:
            return ["total", "x", "y", "z"]
        elif self.n_spins == 2:
            return spin_projection_names
        else:
            return spin_projection_names[:1]

    def dos_sum(
        self,
        atoms: List[int] = None,
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


        if spins is None and self.n_spins == 4:
            raise ValueError("Spins must be provided for non-colinear calculations. This is because the spin projections cannot be summed over")
        elif spins is None and self.n_spins != 4:
            spins = np.arange(self.n_spins, dtype=int)
        if atoms is None:
            atoms = np.arange(self.n_atoms, dtype=int)
        if orbitals is None:
            orbitals = np.arange(self.n_orbitals, dtype=int)

        logger.debug(f"Summing over atoms: {atoms}")
        logger.debug(f"Summing over orbitals: {orbitals}")
        
        projected_sum = np.sum(self.projected[..., orbitals], axis=-1)
        # sum over atoms
        projected_sum = np.sum(projected_sum[... , atoms], axis=-1)
        # sum over spins only in non collinear and reshaping for consistency (nkpoints, nbands, nspins)
        if self.is_non_collinear and sum_noncolinear:
            projected_sum = np.sum(projected_sum[..., spins], axis=-1)
            projected_sum = projected_sum[..., np.newaxis]
        elif self.is_non_collinear:
            projected_sum = projected_sum[..., spins]
            
        if self.is_spin_polarized:
            # Zero out the spin channel that is not specified
            if np.allclose(np.asarray(spins), np.array([0])):
                projected_sum[..., 1] = 0
            elif np.allclose(np.asarray(spins), np.array([1])):
                projected_sum[..., 0] = 0
        
        return projected_sum


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
        total_normalized = np.zeros_like(self.total)
        if mode not in possible_modes:
            raise ValueError(f"The mode must be {possible_modes}")
        if mode == "max":
            for i in range(self.n_spin_channels):
                total_normalized[:,i] = self.total[:,i] / np.max(self.total[:,i])
        elif mode == "integral":
            for i in range(self.n_spin_channels):
                y = self.total[:,i]
                x = self.energies
                integral = trapezoid(y, x=self.energies)
                total_normalized[:,i] = self.total[:,i] / integral
        return DensityOfStates(energies=self.energies, total=total_normalized, projected=self.projected, orbital_names=self.orbital_names)
    
    def compute_projected_sum(self, atoms, orbitals, spin_projections):
        dos_total = np.array(self.total)
        if self.n_spins == 4:
            dos_total_projected = self.dos_sum(spins=spin_projections)
        else:
            dos_total_projected = self.dos_sum()
        dos_projected = self.dos_sum(
            atoms=atoms,
            orbitals=orbitals,
            spins=spin_projections,
        )
            
        return dos_total_projected, dos_projected
    
    def interpolate(self, factor=2):
        if factor in [1, 0]:
            return
        
        # Interpolate total DOS (shape: (n_energies, n_spin))
        energies, total = interpolate(self.energies, self.total, factor=factor)

        # Interpolate projected DOS 
        # shape: (n_energies, n_spin, n_atoms, n_orbitals)
        _, projected = interpolate(self.energies, self.projected, factor=factor)
        
        return DensityOfStates(energies=energies, total=total, projected=projected, orbital_names=self.orbital_names)


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
