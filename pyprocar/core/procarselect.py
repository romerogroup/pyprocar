"""Procar data selection utilities."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class ProcarDataProtocol(Protocol):
    """Protocol for objects that provide PROCAR data."""

    @property
    def spd(self) -> npt.NDArray[np.float64]:
        """Projected density of states data."""
        ...

    @property
    def bands(self) -> npt.NDArray[np.float64]:
        """Band energies."""
        ...

    @property
    def kpoints(self) -> npt.NDArray[np.float64]:
        """K-points coordinates."""
        ...


class ProcarSelect:
    """
    Reduces the dimensionality of the data making it uselful to
    plot bands.

    The main data to manipulate is the projected electronic structure.
    Its shape original is:

    spd[kpoint][band][ispin][atom][orbital].

    The selection of components should be done in order, says, first
    "ispin", then "atom", and at last "orbital".

    Note: once any selection has been performed, the data itself
    changes. Say, if you want compare atom [0] and [1,2], you need two
    instances of this class.


    Example to compare the bandstructure of two set of atoms
    >>>

    """

    spd: npt.NDArray[np.float64] | None
    bands: npt.NDArray[np.float64] | None
    kpoints: npt.NDArray[np.float64] | None
    mode: str | None
    numspin: int
    log: logging.Logger
    ch: logging.StreamHandler[Any]

    def __init__(
        self,
        ProcarData: ProcarDataProtocol | None = None,
        deepCopy: bool = True,
        loglevel: int = logging.WARNING,
        mode: str | None = None,
    ) -> None:
        self.spd = None
        self.bands = None
        self.kpoints = None
        # self.cspd=None
        self.mode = mode
        self.numspin = 1

        # We want a logging to tell us what is happening
        self.log = logging.getLogger("ProcarSelect")
        self.log.setLevel(loglevel)
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(logging.Formatter("%(name)s::%(levelname)s: %(message)s"))
        self.ch.setLevel(logging.DEBUG)
        self.log.addHandler(self.ch)
        # At last, one message to the logger.
        self.log.debug("ProcarSelect: instanciated")

        if ProcarData is not None:
            self.setData(ProcarData, deepCopy)
        return

    def setData(self, ProcarData: ProcarDataProtocol, deepCopy: bool = True) -> None:
        """
        The data from ProcarData is deepCopy-ed by default (ie: their
        elements are not modified by this class.

        Args:

        -ProcarData: is a ProcarParser instance (or anything with similar
         functionality, duck typing)

        -deepCopy=True: If false a shallow copy will be made (saves memory).
        """
        self.log.debug("setData: ...")
        if deepCopy is True:
            self.spd = ProcarData.spd.copy()
            self.bands = ProcarData.bands.copy()
            self.kpoints = ProcarData.kpoints.copy()
            # self.cspd = ProcarData.cspd.copy()
        else:
            self.spd = ProcarData.spd
            self.bands = ProcarData.bands
            self.kpoints = ProcarData.kpoints
            # self.cspd = ProcarData.cspd

        self.numspin = self.spd.shape[2]  # Number of spins
        self.log.debug("setData: ... Done")
        print(
            "spd shape      :  %s [kpoints, bands, spins, atoms+1, orbitals+2]"
            % str(self.spd.shape)
        )

        return

    def selectIspin(self, value: list[int] | None = None, separate: bool = False) -> None:
        """
        value is a list with the values of Ispin to select.

        UPDATE:
            if separate == true, then spin = 0 corresponds to spin up
            and spin = 1 corresponds to spin down. If not, they give
            spin density and spin magnetization, respectively.

        Example:
        >>> foo = ProcarParser()
        >>> foo.readFile("PROCAR")
        >>> bar = ProcarSelect(foo)
        >>> bar.selectIspin([0]) #just the density
        """
        if self.spd is None:
            raise ValueError("spd data not set. Call setData first.")
        if value is None:
            raise ValueError("value must be provided")

        # all kpoint, all bands, VALUE spin, all the rest
        self.log.debug("selectIspin: ...")
        self.log.debug("old spd shape =" + str(self.spd.shape))
        # first, testing if the domensionaluty is rigth:
        dimen = len(self.spd.shape)
        if dimen != 5:
            self.log.error(
                "The array is " + str(dimen) + " dimensional, expecting a 5 dimensional array."
            )
            self.log.error("You should call selectIspin->selecAtom->selectOrbitals, in this order.")
            raise RuntimeError("Wrong dimensionality of the array")
        self.log.debug("ispin value = " + str(value))

        numofbands = int(self.spd.shape[1] / 2)

        if separate == False:
            # spin density or magnetization
            spd_selected: npt.NDArray[np.float64] = self.spd[:, :, value]
            spd_summed: npt.NDArray[np.float64] = spd_selected.sum(axis=2)
            self.spd = spd_summed
            self.log.info("new spd shape =" + str(spd_summed.shape))
            self.log.debug("selectIspin: ...Done")
            if self.mode == "parametric":
                if self.numspin > 1:
                    if value == [0]:
                        print("Plotting spin density...")
                    elif value == [1]:
                        print("Plotting spin magnetization...")

        else:
            # spin up (spin = 0) and spin down (spin = 1) separately.
            if value == [0]:
                # select spin up block
                self.spd = self.spd[:, :numofbands, 0]

            elif value == [1]:
                # select spin down block
                self.spd = self.spd[:, numofbands:, 0]

        return

    def selectAtoms(self, value: list[int] | None = None, fortran: bool = False) -> None:
        """
        value is a list with the values of Atoms to select. The optional
        `fortran` argument indicates whether a c-like 0-based indexing
        (`=False`, default) or a fortran-like 1-based (`=True`) is
        provided in `value`.

        Example:
        >>> foo = ProcarParser()
        >>> foo.readFile("PROCAR")
        >>> bar = ProcarSelect(foo)
        >>> bar.selectIspin([...])
        >>> bar.selectAtoms([0,1,2]) #atom0+atom1+atom2

        Note: this method should be called after select.Ispin
        """
        if self.spd is None:
            raise ValueError("spd data not set. Call setData first.")
        if value is None:
            raise ValueError("value must be provided")

        self.log.debug("selectAtoms: ...")

        # taking care about stupid fortran indexing
        if fortran is True:
            value = [x - 1 for x in value]

        # all kpoint, all bands, VALUE atoms, all the rest
        self.log.debug("old shape =" + str(self.spd.shape))

        # testing if the dimensionaluty is rigth:
        dimen = len(self.spd.shape)
        if dimen != 4:
            self.log.error(
                "The array is " + str(dimen) + " dimensional, expecting a 4 dimensional array."
            )
            self.log.error("You should call selectIspin->selecAtom->selectOrbitals, in this order.")
            raise RuntimeError("Wrong dimensionality of the array")
        spd_selected: npt.NDArray[np.float64] = self.spd[:, :, value]
        spd_summed: npt.NDArray[np.float64] = spd_selected.sum(axis=2)
        self.spd = spd_summed
        # self.cspd = self.cspd[:,:,value]
        # self.cspd = self.cspd.sum(axis=2)
        self.log.info("new shape =" + str(spd_summed.shape))
        self.log.debug("selectAtoms: ...Done")
        return

    def selectOrbital(self, value: list[int]) -> None:
        """
        value is a list with the values of orbital to select.

        Example:
        >>> foo = ProcarParser()
        >>> foo.readFile("PROCAR")
        >>> bar = ProcarSelect(foo)
        >>> bar.selectIspin([...])
        >>> bar.selectAtoms([...])
        >>> bar.selectOrbital([-1]) #the last (`tot`) field

        to select "p" orbitals just change the argument in the last line
        to [2,3,4] or as needed

        Note: this method should be called after `select.Ispin` and
        `select.Atoms`
        """
        if self.spd is None:
            raise ValueError("spd data not set. Call setData first.")

        self.log.debug("selectOrbital: ...")
        self.log.debug("Changing the orbital `values` to have a 0-based indexes")
        # Mind: the first orbital field is the atoms number, which is not
        # an orbital, therefore the orbital index is an affective 1-based
        # therefore all `value` indexes += 1 (well, negative values do not
        # change )
        for i in range(len(value)):
            if value[i] >= 0:
                value[i] += 1

        self.log.debug("New values (indexes to select) :" + str(value))

        # all kpoint, all bands, VALUE orbitals, nothing else?
        spd_selected: npt.NDArray[np.float64] = self.spd[:, :, value]
        # self.cspd = self.cspd[:,:,value]
        self.log.debug("old shape =" + str(spd_selected.shape))

        # testing if the dimensionaluty is rigth:
        dimen = len(spd_selected.shape)
        if dimen != 3:
            self.log.error(
                "The array is " + str(dimen) + " dimensional, expecting a 3 dimensional array."
            )
            self.log.error("You should call selectIspin->selecAtom->selectOrbitals, in this order.")
            raise RuntimeError("Wrong dimensionality of the array")

        spd_summed: npt.NDArray[np.float64] = spd_selected.sum(axis=2)
        self.spd = spd_summed
        # self.cspd=self.cspd.sum(axis=2)
        self.log.info("new shape =" + str(spd_summed.shape))
        self.log.debug("selectOrbital: ...Done")
        return
