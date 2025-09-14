import collections
import gzip
import logging
import re
from pathlib import Path
from typing import Union

import numpy as np

from pyprocar.utils import np_utils

logger = logging.getLogger(__name__)

class Procar(collections.abc.Mapping):
    """
    A class to parse the PROCAR file

    Parameters
    ----------
    filename : str, optional
        The PROCAR filename, by default "PROCAR"
    structure : pyprocar.core.Structure, optional
        The structure of the calculation, by default None
    reciprocal_lattice : np.ndarray, optional
        The reciprocal lattice matrix, by default None
    kpath :  pyprocar.core.KPath, optional
        The pyprocar.core.KPath object, by default None
    kpoints : np.ndarray, optional
        The kpoints, by default None
    fermi : float, optional
        The fermi energy, by default None
    interpolation_factor : int, optional
        The interpolation factor, by default 1
    """

    orbital_names = [
        "s",
        "py",
        "pz",
        "px",
        "dxy",
        "dyz",
        "dz2",
        "dxz",
        "x2-y2",
        "fy3x2",
        "fxyz",
        "fyz2",
        "fz3",
        "fxz2",
        "fzx2",
        "fx3",
        "tot",
    ]

    orbital_names_old = [
        "s",
        "py",
        "pz",
        "px",
        "dxy",
        "dyz",
        "dz2",
        "dxz",
        "dx2",
        "tot",
    ]

    orbital_names_short = ["s", "p", "d", "f", "tot"]

    labels = orbital_names_old[:-1]

    def __init__(self, filepath: Union[str, Path]):

        self.filepath = self._validate_file(filepath)
        self.filename = self.filepath.name

        self.file_str = None
        self.meta_lines = []

        self.has_phase = None
        self.kpoints = None
        self.bands = None
        self.occupancies = None
        self.spd = None
        self.spd_phase = None
        self.kpointsCount = None
        self.bandsCount = None
        self.ionsCount = None
        self.ispin = None
        self._read()

        if self.has_phase:
            self.carray = self.spd_phase[:, :, :, :-1, 1:-1]

    def _read(self):
        """
        Helper method to parse the procar file
        """

        file_stream = self._open_file(self.filepath)

        # Line 1: PROCAR lm decomposed
        self.meta_lines.append(file_stream.readline())
        if "phase" in self.meta_lines[-1]:
            self.has_phase = True
        else:
            self.has_phase = False
        # Line 2: # of k-points:  816   # of bands:  52   # of ions:   8
        self.meta_lines.append(file_stream.readline())
        self.kpointsCount, self.bandsCount, self.ionsCount = map(
            int, re.findall(r"#[^:]+:([^#]+)", self.meta_lines[-1])
        )

        self.file_str = file_stream.read()
        self._repair()
        self._read_kpoints()
        self._read_bands()
        self._read_orbitals()
        if self.has_phase:
            self._read_phases()

        file_stream.close()

        return

    def _repair(self):
        """
        It Tries to repair some stupid problems due the stupid fixed
        format of the stupid fortran.

        Up to now it only separes k-points as the following:
        k-point    61 :    0.00000000-0.50000000 0.00000000 ...
        to
        k-point    61 :    0.00000000 -0.50000000 0.00000000 ...

        But as I found new stupid errors they should be fixed here.
        """
        if (
            len(re.findall(r"(band\s)(\*\*\*)", self.file_str)) != 0
            or len(re.findall(r"(\.\d{8})(\d{2}\.)", self.file_str)) != 0
            or len(re.findall(r"(\d)-(\d)", self.file_str)) != 0
            or len(re.findall(r"\*+", self.file_str)) != 0
        ):
            logger.info("PROCAR needs repairing")
            # Fixing bands issues (when there are more than 999 bands)
            # band *** # energy    6.49554019 # occ.  0.00000000
            self.file_str = re.sub(r"(band\s)(\*\*\*)", r"\1 1000", self.file_str)
            # Fixing k-point issues

            self.file_str = re.sub(r"(\.\d{8})(\d{2}\.)", r"\1 \2", self.file_str)
            self.file_str = re.sub(r"(\d)-(\d)", r"\1 -\2", self.file_str)

            self.file_str = re.sub(r"\*+", r" -10.0000 ", self.file_str)

            outfile = open(self.filename + "-repaired", "w")
            for iline in self.meta_lines:
                outfile.write(iline)
            outfile.write(self.file_str)
            outfile.close()
            print("Repaired PROCAR is written at {}-repaired".format(self.filename))
            print(
                "Please use {}-repaired next time for better efficiency".format(
                    self.filename
                )
            )
        return

    def _validate_file(self, filepath: Union[str, Path]):
        """
        Tries to open a File, it has suitable values for PROCAR and can
        handle gzipped files

        Example:

        """
        filepath = Path(filepath)
        if filepath.is_dir():
            filepath / "PROCAR"
        elif filepath.is_file():
            return filepath
        elif filepath.with_suffix(".gz").is_file():
            return filepath.with_suffix(".gz")
        else:
            raise IOError("File not found")

        return filepath

    def _open_file(self, filepath: Union[str, Path]):
        """
        Tries to open a File, it has suitable values for PROCAR and can
        handle gzipped files

        Example:

        """

        file_stream = None
        if filepath.with_suffix(".gz").is_file():
            file_stream = gzip.open(filepath.with_suffix(".gz"), mode="rt")
        elif filepath.is_file():
            file_stream = open(filepath, "r")
        else:
            raise IOError("File not found")

        return file_stream

    def _read_kpoints(self):
        """
        Reads the k-point headers. A typical k-point line is:
        k-point    1 :    0.00000000 0.00000000 0.00000000  weight = 0.00003704\n
        fills self.kpoint[kpointsCount][3]
        The weights are discarded (are they useful?)
        """
        if not self.file_str:
            print("You should invoke `procar.readFile()` instead. Returning")
            return

        # finding all the K-points headers
        self.kpoints = re.findall(r"k-point\s+\d+\s*:\s+([-.\d\s]+)", self.file_str)

        # trying to build an array
        self.kpoints = [x.split() for x in self.kpoints]
        try:
            self.kpoints = np.array(self.kpoints, dtype=float)
        except ValueError:
            print("\n".join([str(x) for x in self.kpoints]))
            if self.permissive:
                # Discarding the kpoints list, however I need to set
                # self.ispin beforehand.
                if len(self.kpoints) == self.kpointsCount:
                    self.ispin = 1
                elif len(self.kpoints) == 2 * self.kpointsCount:
                    self.ispin = 2
                else:
                    raise ValueError("Kpoints do not match with ispin=1 or 2.")
                self.kpoints = None
                return
            else:

                raise ValueError("Badly formated Kpoints headers, try `--permissive`")
        # if successful, go on

        # trying to identify an non-polarized or non-collinear case, a
        # polarized case or a defective file

        if len(self.kpoints) != self.kpointsCount:
            # if they do not match, may means two things a spin polarized
            # case or a bad file, lets check
            # lets start testing if it is spin polarized, if so, there
            # should be 2 identical blocks of kpoints.
            up, down = np.vsplit(self.kpoints, 2)
            if (up == down).all():
                self.ispin = 2
                # just keeping one set of kpoints (the other will be
                # discarded)
                self.kpoints = up
            else:
                raise RuntimeError("Bad Kpoints list.")
        # if ISPIN != 2 setting ISPIN=1 (later for the non-collinear case 1->4)
        # It is unknown until parsing the projected data
        else:
            self.ispin = 1

        # checking again, for compatibility,
        if len(self.kpoints) != self.kpointsCount:
            raise RuntimeError(
                "Kpoints number do not match with metadata (header of PROCAR)"
            )
        return

    @property
    def kpoints_reduced(self):
        """The kpoints in reduced coordinates

        Returns
        -------
        np.ndarray
            The kpoints in reduced coordinates
        """
        return self.kpoints

    def _read_bands(self):
        """
        Reads the bands header. A typical bands is:
        band   1 # energy   -7.11986315 # occ.  1.00000000

        fills self.bands[kpointsCount][bandsCount]

        The occupation numbers are discarded (are they useful?)
        """
        if not self.file_str:
            print("You should invoke `procar.read()` instead. Returning")
            return

        # finding all bands
        self.bands = re.findall(
            r"band\s*(\d+)\s*#\s*energy\s*([-.\d\s]+)", self.file_str
        )

        # checking if the number of bands match

        if len(self.bands) != self.bandsCount * self.kpointsCount * self.ispin:
            raise RuntimeError("Number of bands don't match")

        # casting to array to manipulate the bands
        self.bands = np.array(self.bands, dtype=float)[:, 1]

        # Now I will deal with the spin polarized case. The goal is join
        # them like for a non-magnetic case
        # new version of pyprocar will have an (nkpoints, nbands, 2) dimensions
        if self.ispin == 2:
            # up and down are along the first axis

            up, down = np.split(self.bands, 2)

            # reshapping (the 2  means both band index and energy)
            up = up.reshape(self.kpointsCount, self.bandsCount)
            down = down.reshape(self.kpointsCount, self.bandsCount)

            # setting the correct number of bands (up+down)
            # self.bandsCount *= 2

            # and joining along the second axis (axis=1), ie: bands-like
            # self.bands = np.concatenate((up, down), axis=1)
            self.bands = np.stack((up, down), axis=-1)
        else:
            self.bands = self.bands.reshape(self.kpointsCount, self.bandsCount, 1)

        # otherwise just reshaping is needed
        # else:
        #     self.bands.shape = (self.kpointsCount, self.bandsCount, 2)

        # Making a test if the broadcast is rigth, otherwise just print
        # test = [x.max() - x.min() for x in self.bands[:, :, 0].transpose()]
        # if np.array(test).any():
        #     print(
        #         "The indexes of bands do not match. CHECK IT. "
        #         "Likely the data was wrongly broadcasted"
        #     )
        #     print(str(self.bands[:, :, 0]))
        # # Now safely removing the band index
        # self.bands = self.bands[:, :, 1]
        return

    def _read_orbitals(self):
        """
        Reads all the spd-projected data. A typical/expected block is:

            ion      s     py     pz     px    dxy    dyz    dz2    dxz    dx2    tot
            1  0.079  0.000  0.001  0.000  0.000  0.000  0.000  0.000  0.000  0.079
            2  0.152  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.152
            3  0.079  0.000  0.001  0.000  0.000  0.000  0.000  0.000  0.000  0.079
            4  0.188  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.188
            5  0.188  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.188
            tot  0.686  0.000  0.002  0.000  0.000  0.000  0.000  0.000  0.000  0.688
            (x2 for spin-polarized -akwardkly formatted-, x4 non-collinear -nicely
             formatted-).

        The data is stored in an array self.spd[kpoint][band][ispin][atom][orbital]

        Undefined behavior in case of phase factors (LORBIT = 12).
        """
        # if not self.file_str:
        #     print("You should invoke `procar.readFile()` instead. Returning")
        #     return

        # finding all orbital headers
        self.spd = re.findall(r"ion(.+)", self.file_str)

        # testing if the orbital names are known (the standard ones)
        FoundOrbs = self.spd[0].split()
        size = len(FoundOrbs)
        # only the first 'size' orbital
        StdOrbs = self.orbital_names[: size - 1] + self.orbital_names[-1:]
        StdOrbs_short = (
            self.orbital_names_short[: size - 1] + self.orbital_names_short[-1:]
        )
        StdOrbs_old = self.orbital_names_old[: size - 1] + self.orbital_names_old[-1:]
        if (
            FoundOrbs != (StdOrbs)
            and FoundOrbs != (StdOrbs_short)
            and FoundOrbs != (StdOrbs_old)
        ):
            print(
                str(size) + " orbitals. (Some of) They are unknow (if "
                "you did 'filter' them it is OK)."
            )
        self.orbitalCount = size
        self.orbitalNames = self.spd[0].split()

        # Now reading the bulk of data
        # The case of just one atom is handled differently since the VASP
        # output is a little different
        if self.ionsCount == 1:
            self.spd = re.findall(r"^(\s*1\s+.+)$", self.file_str, re.MULTILINE)

            raw_spd_natom_axis = self.ionsCount
        else:
            # Added by Francisco to speed up filtering on June 4th, 2019
            # get rid of phase factors
            self.spd = re.findall(r"ion.+tot\n([-.\d\seto]+)", self.file_str)
            self.spd = "".join(self.spd)
            self.spd = re.findall(r"([-.\d\se]+tot.+)\n", self.spd)
            raw_spd_natom_axis = self.ionsCount + 1
        # free the memory (could be a lot)
        if not self.has_phase:
            self.file_str = None

        # Now the method will try to find the value of self.ispin,
        # previously it was set to either 1 or 2. If "1", it could be 1 or
        # 4, but previously it was impossible to find the rigth value. If
        # "2" it has to macth with the number of entries of spd data.

        expected = self.bandsCount * self.kpointsCount * self.ispin

        if expected == len(self.spd):
            pass
        # catching a non-collinear calc.
        elif expected * 4 == len(self.spd):
            # testing if previous ispin value is ok
            if self.ispin != 1:
                print(
                    "Incompatible data: self.ispin= " + str(self.ispin) + ". Now is 4"
                )
            self.ispin = 4
        else:
            raise RuntimeError("Incompatible file.")

        # checking for consistency
        for line in self.spd:
            if self.ionsCount != 1:
                if len(line.split()) != (self.ionsCount + 1) * (self.orbitalCount + 1):
                    raise RuntimeError("Flats happens")
        # replacing the "tot" string by a number, to allows a conversion
        # to numpy
        self.spd = [x.replace("tot", "0").split() for x in self.spd]
        # self.spd = [x.split() for x in self.spd]
        self.spd = np.array(self.spd, dtype=float)

        # handling collinear polarized case
        if self.ispin == 2:
            # splitting both spin components, now they are along k-points
            # axis (1st axis) but, then should be concatenated along the
            # bands.
            up, down = np.vsplit(self.spd, 2)
            # ispin = 1 for a while, we will made the distinction
            up = up.reshape(
                self.kpointsCount,
                self.bandsCount,
                1,
                raw_spd_natom_axis,
                self.orbitalCount + 1,
            )
            down = down.reshape(
                self.kpointsCount,
                self.bandsCount,
                1,
                raw_spd_natom_axis,
                self.orbitalCount + 1,
            )
            # concatenating bandwise. Density and magntization, their
            # meaning is obvious, and do uses 2 times more memory than
            # required, but I *WANT* to keep it as close as possible to the
            # non-collinear or non-polarized case
            density = np.concatenate((up, down), axis=1)
            magnet = np.concatenate((up, -down), axis=1)
            # concatenated along 'ispin axis'
            self.spd = np.concatenate((density, magnet), axis=2)

        # otherwise, just a reshaping suffices
        else:
            self.spd = self.spd.reshape(
                self.kpointsCount,
                self.bandsCount,
                self.ispin,
                raw_spd_natom_axis,
                self.orbitalCount + 1,
            )

        if self.ionsCount == 1:
            self.spd = np.pad(
                self.spd,
                ((0, 0), (0, 0), (0, 0), (0, 1), (0, 0)),
                "constant",
                constant_values=(0),
            )
            self.spd[:, :, :, 1, :] = self.spd[:, :, :, 0, :]

        return

    def _read_phases(self):
        """
        Helped method to parse the projection phases
        """
        if self.ionsCount == 1:
            self.spd_phase = re.findall(r"^(\s*1\s+.+)$", self.file_str, re.MULTILINE)
            raw_spd_natom_axis = self.ionsCount
        else:
            # Added by Francisco to speed up filtering on June 4th, 2019
            # get rid of phase factors
            self.spd_phase = re.findall(
                r"ion.+\n([-.\d\se]+charge[-.\d\se]+)", self.file_str
            )
            self.spd_phase = "".join(self.spd_phase)
            self.spd_phase = re.findall(r"([-.\d\se]+charge.+)\n", self.spd_phase)
            raw_spd_natom_axis = self.ionsCount + 1
        # free the memory (could be a lot)
        self.file_str = None

        # replacing the "charge" string by a number, to allows a conversion
        # to numpy, adding columns of zeros next to charge row to be able to
        # convert to imaginary

        self.spd_phase = [
            x.replace(
                re.findall("charge.*", x)[0],
                (
                    " 0.000 ".join(re.findall("charge.*", x)[0].split()).replace(
                        "charge", ""
                    )
                ),
            )
            for x in self.spd_phase
        ]
        self.spd_phase = [x.split() for x in self.spd_phase]
        self.spd_phase = np.array(self.spd_phase, dtype=float)

        # handling collinear polarized case
        if self.ispin == 2:
            # splitting both spin components, now they are along k-points
            # axis (1st axis) but, then should be concatenated along the
            # bands.
            up, down = np.vsplit(self.spd_phase, 2)
            # ispin = 1 for a while, we will made the distinction
            up = up.reshape(
                self.kpointsCount,
                self.bandsCount,
                1,
                self.ionsCount + 1,
                self.orbitalCount * 2,
            )
            down = down.reshape(
                self.kpointsCount,
                self.bandsCount,
                1,
                self.ionsCount + 1,
                self.orbitalCount * 2,
            )
            # concatenating bandwise. Density and magntization, their
            # meaning is obvious, and do uses 2 times more memory than
            # required, but I *WANT* to keep it as close as possible to the
            # non-collinear or non-polarized case
            density = np.concatenate((up, down), axis=1)
            magnet = np.concatenate((up, -down), axis=1)
            # concatenated along 'ispin axis'
            self.spd_phase = np.concatenate((density, magnet), axis=2)

        # otherwise, just a reshaping suffices
        elif self.ispin == 4:
            self.spd_phase = self.spd_phase.reshape(
                self.kpointsCount,
                self.bandsCount,
                1,
                self.ionsCount + 1,
                self.orbitalCount * 2,
            )
        else:
            self.spd_phase = self.spd_phase.reshape(
                self.kpointsCount,
                self.bandsCount,
                self.ispin,
                self.ionsCount + 1,
                self.orbitalCount * 2,
            )
        temp = np.zeros(
            shape=(
                self.spd_phase.shape[0],
                self.spd_phase.shape[1],
                self.spd_phase.shape[2],
                self.spd_phase.shape[3],
                int(self.spd_phase.shape[4] / 2) + 1,
            ),
            dtype=np_utils.COMPLEX_DTYPE,
        )

        for i in range(1, (self.orbitalCount) * 2 - 2, 2):
            temp[:, :, :, :, (i + 1) // 2].real = self.spd_phase[:, :, :, :, i]
            temp[:, :, :, :, (i + 1) // 2].imag = self.spd_phase[:, :, :, :, i + 1]
        temp[:, :, :, :, 0].real = self.spd_phase[:, :, :, :, 0]
        temp[:, :, :, :, -1].real = self.spd_phase[:, :, :, :, -1]
        self.spd_phase = temp

        if self.ionsCount == 1:
            self.spd_phase = np.pad(
                self.spd_phase,
                ((0, 0), (0, 0), (0, 0), (0, 1), (0, 0)),
                "constant",
                constant_values=(0),
            )
            self.spd_phase[:, :, :, 1, :] = self.spd_phase[:, :, :, 0, :]

        return

    def _spd2projected(self, spd):
        """
        Helpermethod to project the spd array to the projected array
        which will be fed into pyprocar.coreElectronicBandStructure object

        Parameters
        ----------
        spd : np.ndarray
            The spd array from the earlier parse. This has a structure simlar to the PROCAR output in vasp
            Has the shape [n_kpoints,n_band,n_spins,n_orbital,n_atoms]
        nprinciples : int, optional
            The prinicipal quantum numbers, by default 1

        Returns
        -------
        np.ndarray
            The projected array. Has the shape [n_kpoints,n_band,n_spin, n_atom,n_principal,n_orbital]
        """
        # This function is for VASP
        # non-pol and colinear
        # spd is formed as (nkpoints,nbands, nspin, natom+1, norbital+2)
        # natom+1 > last column is total
        # norbital+2 > 1st column is the number of atom last is total
        # non-colinear
        # spd is formed as (nkpoints,nbands, nspin +1 , natom+1, norbital+2)
        # natom+1 > last column is total
        # norbital+2 > 1st column is the number of atom last is total
        # nspin +1 > last column is total
        if spd is None:
            return None
        natoms = spd.shape[3] - 1

        nkpoints = spd.shape[0]

        nbands = spd.shape[1]
        norbitals = spd.shape[4] - 2
        if spd.shape[2] == 4:
            nspins = 4
        else:
            nspins = spd.shape[2]
        if nspins == 2:
            nbands = int(spd.shape[1] / 2)
        else:
            nbands = spd.shape[1]

        projected = np.zeros(
            shape=(nkpoints, nbands, nspins, natoms, norbitals),
            dtype=spd.dtype,
        )

        if nspins == 2:
            projected[:, :, 0, :, :] = spd[:, :nbands, 0, :-1, 1:-1]
            projected[:, :, 1, :, :] = spd[:, nbands:, 1, :-1, 1:-1]
        else:
            projected[:, :, :, :, :] = spd[:, :, :, :-1, 1:-1]

        return projected

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)
