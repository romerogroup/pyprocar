# -*- coding: utf-8 -*-

from ..core import Structure, DensityOfStates, ElectronicBandStructure, KPath
import numpy as np
from numpy import array
import os
import re
import xml.etree.ElementTree as ET
import collections
import gzip


class Outcar(collections.abc.Mapping):
    def __init__(self, filename="OUTCAR"):
        self.variables = {}
        self.filename = filename

        rf = open(self.filename, "r")
        self.file_str = rf.read()
        rf.close()

    @property
    def efermi(self):
        """Just finds all E-fermi fields in the outcar file and keeps the
        last one (if more than one found).

        Args:
            -filename: the file name of the outcar to be read

        """
        return float(re.findall(r"E-fermi\s*:\s*(-?\d+.\d+)", self.file_str)[-1])

    @property
    def reciprocal_lattice(self):
        """Finds and return the reciprocal lattice vectors, if more than
        one set present, it return just the last one.

        Args:
            -filename: the name of the outcar file  to be read

        """
        reciprocal_lattice = re.findall(
            r"reciprocal\s*lattice\s*vectors\s*([-.\s\d]*)", self.file_str
        )[-1]
        reciprocal_lattice = reciprocal_lattice.split()
        reciprocal_lattice = np.array(reciprocal_lattice, dtype=float)
        # up to now I have, both direct and rec. lattices (3+3=6 columns)
        reciprocal_lattice = np.reshape(reciprocal_lattice, (3, 6))
        reciprocal_lattice = reciprocal_lattice[:, 3:]
        return reciprocal_lattice

    @property
    def rotations(self):
        """Finds the point symmetry operations included in the OUTCAR file
        and returns them in matrix form.

        Args:
            -filename: the file name of the outcar file to be read
            -reciprocal_lattice: reciprocal lattice of the structure

        """

        with open(self.filename) as f:
            txt = f.readlines()

        for i, line in enumerate(txt):
            if 'irot' in line:
                begin_table = i+1
            if 'Subroutine' in line:
                end_table = i-1

        operators = np.zeros((end_table-begin_table, 9))
        for i, line in enumerate(txt[begin_table:end_table]):
            str_list = line.split()
            num_list = [float(s) for s in str_list]
            operator = np.array(num_list)
            operators[i, :] = operator

        rotations = []

        for operator in operators:
            det_A = operator[1]
            # convert alpha to radians
            alpha = np.pi * operator[2] / 180.0
            # get rotation axis
            x = operator[3]
            y = operator[4]
            z = operator[5]

            R = (
                np.array(
                    [
                        [
                            np.cos(alpha) + x ** 2 * (1 - np.cos(alpha)),
                            x * y * (1 - np.cos(alpha)) - z * np.sin(alpha),
                            x * z * (1 - np.cos(alpha)) + y * np.sin(alpha),
                        ],
                        [
                            y * x * (1 - np.cos(alpha)) + z * np.sin(alpha),
                            np.cos(alpha) + y ** 2 * (1 - np.cos(alpha)),
                            y * z * (1 - np.cos(alpha)) - x * np.sin(alpha),
                        ],
                        [
                            z * x * (1 - np.cos(alpha)) - y * np.sin(alpha),
                            z * y * (1 - np.cos(alpha)) + x * np.sin(alpha),
                            np.cos(alpha) + z ** 2 * (1 - np.cos(alpha)),
                        ],
                    ]
                )
                * det_A
            )

            reciprocal_lattice = np.transpose(self.reciprocal_lattice)
            R = np.dot(
                np.dot(np.linalg.inv(reciprocal_lattice), R),
                reciprocal_lattice
            )
            R = np.round_(R, decimals=3)
            rotations.append(R)

        return np.array(rotations)

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class Poscar(collections.abc.Mapping):
    def __init__(self, filename="POSCAR"):
        self.variables = {}
        self.filename = filename
        atoms, coordinates, lattice = self._parse_poscar()
        self.structure = Structure(
            atoms=atoms, fractional_coordinates=coordinates, lattice=lattice
        )

    def _parse_poscar(self):
        """
        Reads VASP POSCAR file-type and returns the pyprocar structure

        Parameters
        ----------
        filename : str, optional
            Path to POSCAR file. The default is 'CONTCAR'.

        Returns
        -------
        None.

        """
        rf = open(self.filename, "r")
        lines = rf.readlines()
        rf.close()
        comment = lines[0]
        self.comment = comment
        scale = float(lines[1])
        lattice = np.zeros(shape=(3, 3))
        for i in range(3):
            lattice[i, :] = [float(x) for x in lines[i + 2].split()[:3]]
        lattice *= scale
        if any([char.isalpha() for char in lines[5]]):
            species = [x for x in lines[5].split()]
            shift = 1
        else:
            shift = 0
            if os.path.exists("POTCAR"):
                base_dir = self.filename.replace(
                    self.filename.split(os.sep)[-1], "")
                if base_dir == "":
                    base_dir = "."
                rf = open(base_dir + os.sep + "POTCAR", "r")
                potcar = rf.read()
                rf.close()
                species = re.findall(
                    "\s*PAW[PBE_\s]*([A-Z][a-z]*)[_a-z]*[0-9]*[a-zA-Z]*[0-9]*.*\s[0-9.]*",
                    potcar,
                )[::2]
        composition = [int(x) for x in lines[5 + shift].split()]
        atoms = []
        for i in range(len(composition)):
            for x in composition[i] * [species[i]]:
                atoms.append(x)
        natom = sum(composition)
        if lines[6 + shift][0].lower() == "s":
            shift = 2
        if lines[6 + shift][0].lower() == "d":
            direct = True
        elif lines[6 + shift][0].lower() == "c":
            print("havn't implemented conversion to cartesian yet")
            direct = False
        coordinates = np.zeros(shape=(natom, 3))
        for i in range(natom):
            coordinates[i, :] = [float(x)
                                 for x in lines[i + 7 + shift].split()[:3]]

        if direct:
            return atoms, coordinates, lattice

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class Kpoints(collections.abc.Mapping):
    def __init__(self, filename="KPOINTS", has_time_reversal=True):
        self.variables = {}
        self.filename = filename
        self.file_str = None
        self.metadata = None
        self.mode = None
        self.kgrid = None
        self.kshift = None
        self.ngrids = None
        self.special_kpoints = None
        self.knames = None
        self.cartesian = False
        self.automatic = False
        self._parse_kpoints()
        self.kpath = KPath(
            knames=self.knames,
            special_kpoints=self.special_kpoints,
            ngrids=self.ngrids,
            has_time_reversal=has_time_reversal,
        )

    def _parse_kpoints(self):
        rf = open(self.filename, "r")
        self.comment = rf.readline()
        grids = rf.readline()
        grids = grids[: grids.find("!")]
        self.ngrids = [int(x) for x in grids.split()]
        if self.ngrids[0] == 0:
            self.automatic = True
        mode = rf.readline()
        if mode[0].lower() == "m":
            self.mode = "monkhorst-pack"
        elif mode[0].lower() == "g":
            self.mode = "gamma"
        elif mode[0].lower() == "l":
            self.mode = "line"
        if self.mode == "gamma" or self.mode == "monkhorst-pack":
            kgrid = rf.readline()
            kgrid = kgrid[: kgrid.find("!")]
            self.kgrid = [int(x) for x in kgrid.split()]
            shift = rf.readline()
            shift = shift[: shift.find("!")]
            self.kshift = [int(x) for x in shift.split()]
            rf.close()
        elif self.mode == "line":
            if rf.readline()[0].lower() == "c":
                self.cartesian = True
            else:
                self.cartesian = False
            self.file_str = rf.read()
            rf.close()
            temp = np.array(
                re.findall(
                    "([0-9.-]+)\s*([0-9.-]+)\s*([0-9.-]+)(.*)", self.file_str)
            )
            temp_special_kp = temp[:, :3].astype(float)
            temp_knames = temp[:, -1]
            nsegments = temp_special_kp.shape[0] // 2
            if len(self.ngrids) == 1:
                self.ngrids = [self.ngrids[0]] * nsegments
            self.knames = np.reshape(
                [x.replace("!", "").strip()
                 for x in temp_knames], (nsegments, 2)
            )
            self.special_kpoints = temp_special_kp.reshape(nsegments, 2, 3)

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()

    # @property
    # def mode(self):

    #     KPmatrix = re.findall("reciprocal[\s\S]*", KPread)
    #     tick_labels = np.array(re.findall("!\s(.*)", KPmatrix[0]))
    #     knames = []
    #     knames = [tick_labels[0]]

    #     ################## Checking for discontinuities ########################
    #     discont_indx = []
    #     icounter = 1
    #     while icounter < len(tick_labels) - 1:
    #         if tick_labels[icounter] == tick_labels[icounter + 1]:
    #             knames.append(tick_labels[icounter])
    #             icounter = icounter + 2
    #         else:
    #             discont_indx.append(icounter)
    #             knames.append(tick_labels[icounter] + "|" + tick_labels[icounter + 1])
    #             icounter = icounter + 2
    #     knames.append(tick_labels[-1])
    #     discont_indx = list(dict.fromkeys(discont_indx))

    #     ################# End of discontinuity check ##########################

    #     # Added by Nicholas Pike to modify the output of seekpath to allow for
    #     # latex rendering.
    #     for i in range(len(knames)):
    #         if knames[i] == "GAMMA":
    #             knames[i] = "\Gamma"
    #         else:
    #             pass

    #     knames = [str("$" + latx + "$") for latx in knames]

    #     # getting the number of grid points from the KPOINTS file
    #     f2 = open(kpointsfile)
    #     KPreadlines = f2.readlines()
    #     f2.close()
    #     numgridpoints = int(KPreadlines[1].split()[0])

    #     kticks = [0]
    #     gridpoint = 0
    #     for kt in range(len(knames) - 1):
    #         gridpoint = gridpoint + numgridpoints
    #         kticks.append(gridpoint - 1)

    #     print("knames         : ", knames)
    #     print("kticks         : ", kticks)

    #     # creating an array for discontunuity k-points. These are the indexes
    #     # of the discontinuity k-points.
    #     for k in discont_indx:
    #         discontinuities.append(kticks[int(k / 2) + 1])
    #     if discontinuities:
    #         print("discont. list  : ", discontinuities)

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class Procar(collections.abc.Mapping):
    def __init__(
        self,
        filename="PROCAR",
        structure=None,
        reciprocal_lattice=None,
        kpath=None,
        efermi=None,
        interpolation_factor=1,
    ):
        self.variables = {}
        self.filename = filename
        self.meta_lines = []

        self.reciprocal_lattice = reciprocal_lattice
        self.file_str = None
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
        self.structure = structure

        self.orbitalName = [
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
        self.orbitalName_old = [
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
        self.orbitalName_short = ["s", "p", "d", "f", "tot"]
        self.labels = self.orbitalName_old[:-1]

        self._read()
        if self.has_phase:
            self.carray = self.spd_phase[:, :, :, :-1, 1:-1]
        self.ebs = ElectronicBandStructure(
            kpoints=self.kpoints,
            bands=self.bands,
            projected=self._spd2projected(self.spd),
            efermi=efermi,
            kpath=kpath,
            projected_phase=self._spd2projected(self.spd_phase),
            labels=self.orbitalNames[:-1],
            reciprocal_lattice=reciprocal_lattice,
            interpolation_factor=interpolation_factor,
            shifted_to_efermi=False,
        )

    def repair(self):
        """It Tries to repair some stupid problems due the stupid fixed
        format of the stupid fortran.

        Up to now it only separes k-points as the following:
        k-point    61 :    0.00000000-0.50000000 0.00000000 ...
        to
        k-point    61 :    0.00000000 -0.50000000 0.00000000 ...

        But as I found new stupid errors they should be fixed here.
        """

        print("PROCAR needs repairing")
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

    def _open_file(self):
        """
        Tries to open a File, it has suitable values for PROCAR and can
        handle gzipped files

        Example:

            >>> foo =  UtilsProcar.Openfile()
            Tries to open "PROCAR", then "PROCAR.gz"

            >>> foo = UtilsProcar.Openfile("../bar")
            Tries to open "../bar". If it is a directory, it will try to open
            "../bar/PROCAR" and if fails again "../bar/PROCAR.gz"

            >>> foo = UtilsProcar.Openfile("PROCAR-spd.gz")
            Tries to open a gzipped file "PROCAR-spd.gz"

            If unable to open a file, it raises a "IOError" exception.
        """

        # checking if fileName is just a path and needs a "PROCAR to " be
        # appended
        if os.path.isdir(self.filename):
            if self.filename[-1] != r"/":
                self.filename += "/"
            self.filename += "PROCAR"

        # checking that the file exist
        if os.path.isfile(self.filename):
            # Checking if compressed
            if self.filename[-2:] == "gz":
                in_file = gzip.open(self.filename, mode="rt")
            else:
                in_file = open(self.filename, "r")
            return in_file

        # otherwise a gzipped version may exist
        elif os.path.isfile(self.filename + ".gz"):
            in_file = gzip.open(self.filename + ".gz", mode="rt")

        else:
            raise IOError("File not found")

        return in_file

    def _read(self):

        rf = self._open_file()
        # Line 1: PROCAR lm decomposed
        self.meta_lines.append(rf.readline())
        if "phase" in self.meta_lines[-1]:
            self.has_phase = True
        else:
            self.has_phase = False
        # Line 2: # of k-points:  816   # of bands:  52   # of ions:   8
        self.meta_lines.append(rf.readline())
        self.kpointsCount, self.bandsCount, self.ionsCount = map(
            int, re.findall(r"#[^:]+:([^#]+)", self.meta_lines[-1])
        )
        if self.ionsCount == 1:
            print(
                "Special case: only one atom found. The program may not work as expected"
            )
        else:
            self.ionsCount = self.ionsCount + 1

        # reading all the rest of the file to be parsed below

        self.file_str = rf.read()
        if (
            len(re.findall(r"(band\s)(\*\*\*)", self.file_str)) != 0
            or len(re.findall(r"(\.\d{8})(\d{2}\.)", self.file_str)) != 0
            or len(re.findall(r"(\d)-(\d)", self.file_str)) != 0
            or len(re.findall(r"\*+", self.file_str)) != 0
        ):
            self.repair()

        self._read_kpoints()
        self._read_bands()
        self._read_orbitals()
        if self.has_phase:
            self._read_phases()
        rf.close()
        return

    def _read_kpoints(self):
        """Reads the k-point headers. A typical k-point line is:
        k-point    1 :    0.00000000 0.00000000 0.00000000  weight = 0.00003704\n
        fills self.kpoint[kpointsCount][3]
        The weights are discarded (are they useful?)
        """
        if not self.file_str:
            print("You should invoke `procar.readFile()` instead. Returning")
            return

        # finding all the K-points headers
        self.kpoints = re.findall(
            r"k-point\s+\d+\s*:\s+([-.\d\s]+)", self.file_str)

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

                raise ValueError(
                    "Badly formated Kpoints headers, try `--permissive`")
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
    def kpoints_cartesian(self):
        if self.reciprocal_lattice is not None:
            return np.dot(self.kpoints, self.reciprocal_lattice)
        else:
            print(
                "Please provide a reciprocal lattice when initiating the Procar class"
            )
            return

    @property
    def kpoints_reduced(self):
        return self.kpoints

    def _read_bands(self):
        """Reads the bands header. A typical bands is:
        band   1 # energy   -7.11986315 # occ.  1.00000000

        fills self.bands[kpointsCount][bandsCount]

        The occupation numbers are discarded (are they useful?)"""
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
        self.bands = np.array(self.bands, dtype=float)[:,1]

        # Now I will deal with the spin polarized case. The goal is join
        # them like for a non-magnetic case
        # new version of pyprocar will have an (nkpoints, nbands, 2) dimensions
        if self.ispin == 2:
            # up and down are along the first axis

            up, down = np.split(self.bands,2)

            # reshapping (the 2  means both band index and energy)
            up = up.reshape(self.kpointsCount, self.bandsCount)
            down = down.reshape(self.kpointsCount, self.bandsCount)

            # setting the correct number of bands (up+down)
            # self.bandsCount *= 2

            # and joining along the second axis (axis=1), ie: bands-like
            # self.bands = np.concatenate((up, down), axis=1)
            self.bands = np.stack(
                (up, down), axis=-1)
        else :
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
        """Reads all the spd-projected data. A typical/expected block is:
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
        StdOrbs = self.orbitalName[: size - 1] + self.orbitalName[-1:]
        StdOrbs_short = self.orbitalName_short[: size -
                                               1] + self.orbitalName_short[-1:]
        StdOrbs_old = self.orbitalName_old[: size -
                                           1] + self.orbitalName_old[-1:]
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
            self.spd = re.findall(
                r"^(\s*1\s+.+)$", self.file_str, re.MULTILINE)
        else:
            # Added by Francisco to speed up filtering on June 4th, 2019
            # get rid of phase factors
            self.spd = re.findall(r"ion.+tot\n([-.\d\seto]+)", self.file_str)
            self.spd = "".join(self.spd)
            self.spd = re.findall(r"([-.\d\se]+tot.+)\n", self.spd)
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
                    "Incompatible data: self.ispin= " +
                    str(self.ispin) + ". Now is 4"
                )
            self.ispin = 4
        else:
            raise RuntimeError("Incompatible file.")

        # checking for consistency
        for line in self.spd:
            if len(line.split()) != (self.ionsCount) * (self.orbitalCount + 1):
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
                self.bandsCount ,
                1,
                self.ionsCount,
                self.orbitalCount + 1,
            )
            down = down.reshape(
                self.kpointsCount,
                self.bandsCount ,
                1,
                self.ionsCount,
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
                self.ionsCount,
                self.orbitalCount + 1,
            )
        return

    def _read_phases(self):

        if self.ionsCount == 1:
            self.spd_phase = re.findall(
                r"^(\s*1\s+.+)$", self.file_str, re.MULTILINE)
        else:
            # Added by Francisco to speed up filtering on June 4th, 2019
            # get rid of phase factors
            self.spd_phase = re.findall(
                r"ion.+\n([-.\d\se]+charge[-.\d\se]+)", self.file_str
            )
            self.spd_phase = "".join(self.spd_phase)
            self.spd_phase = re.findall(
                r"([-.\d\se]+charge.+)\n", self.spd_phase)
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
                int(self.bandsCount / 2),
                1,
                self.ionsCount,
                self.orbitalCount * 2,
            )
            down = down.reshape(
                self.kpointsCount,
                int(self.bandsCount / 2),
                1,
                self.ionsCount,
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
                self.ionsCount,
                self.orbitalCount * 2,
            )
        else:
            self.spd_phase = self.spd_phase.reshape(
                self.kpointsCount,
                self.bandsCount,
                self.ispin,
                self.ionsCount,
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
            dtype=np.complex_,
        )

        for i in range(1, (self.orbitalCount) * 2 - 2, 2):
            temp[:, :, :, :, (i + 1) // 2].real = self.spd_phase[:, :, :, :, i]
            temp[:, :, :, :, (i + 1) //
                 2].imag = self.spd_phase[:, :, :, :, i + 1]
        temp[:, :, :, :, 0].real = self.spd_phase[:, :, :, :, 0]
        temp[:, :, :, :, -1].real = self.spd_phase[:, :, :, :, -1]
        self.spd_phase = temp
        return

    def _spd2projected(self, spd, nprinciples=1):
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
            nspins = 3
        else:
            nspins = spd.shape[2]
        if nspins == 2:
            nbands = int(spd.shape[1] / 2)
        else:
            nbands = spd.shape[1]
        projected = np.zeros(
            shape=(nkpoints, nbands, natoms, nprinciples, norbitals, nspins),
            dtype=spd.dtype,
        )
        temp_spd = spd.copy()
        # (nkpoints,nbands, nspin, natom, norbital)
        temp_spd = np.swapaxes(temp_spd, 2, 4)
        # (nkpoints,nbands, norbital , natom , nspin)
        temp_spd = np.swapaxes(temp_spd, 2, 3)
        # (nkpoints,nbands, natom, norbital, nspin)
        # projected[ikpoint][iband][iatom][iprincipal][iorbital][ispin]
        if nspins == 3:
            projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :-1]
        elif nspins == 2:
            projected[:, :, :, 0, :, 0] = temp_spd[:, :nbands, :-1, 1:-1, 0]
            projected[:, :, :, 0, :, 1] = temp_spd[:, nbands:, :-1, 1:-1, 0]
        else:
            projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]
        return projected

    def symmetrize(self, symprec=1e-5, outcar=None, structure=None, spglib=True):
        if outcar is not None:
            with open(outcar) as f:
                txt = f.readlines()
            for i, line in enumerate(txt):
                if "irot" in line:
                    begin_table = i + 1
                if "Subroutine" in line:
                    end_table = i - 1

            operators = np.zeros((end_table - begin_table, 9))
            for i, line in enumerate(txt[begin_table:end_table]):
                str_list = line.split()
                num_list = [float(s) for s in str_list]
                operator = np.array(num_list)
                operators[i, :] = operator
            rotations = []

            for operator in operators:
                det_A = operator[1]
                # convert alpha to radians
                alpha = np.pi * operator[2] / 180.0
                # get rotation axis
                x = operator[3]
                y = operator[4]
                z = operator[5]

                R = (
                    np.array(
                        [
                            [
                                np.cos(alpha) + x ** 2 * (1 - np.cos(alpha)),
                                x * y * (1 - np.cos(alpha)) -
                                z * np.sin(alpha),
                                x * z * (1 - np.cos(alpha)) +
                                y * np.sin(alpha),
                            ],
                            [
                                y * x * (1 - np.cos(alpha)) +
                                z * np.sin(alpha),
                                np.cos(alpha) + y ** 2 * (1 - np.cos(alpha)),
                                y * z * (1 - np.cos(alpha)) -
                                x * np.sin(alpha),
                            ],
                            [
                                z * x * (1 - np.cos(alpha)) -
                                y * np.sin(alpha),
                                z * y * (1 - np.cos(alpha)) +
                                x * np.sin(alpha),
                                np.cos(alpha) + z ** 2 * (1 - np.cos(alpha)),
                            ],
                        ]
                    )
                    * det_A
                )

                R = np.dot(
                    np.dot(np.linalg.inv(structure.reciprocal_lattice), R),
                    structure.reciprocal_lattice,
                )
                R = np.round_(R, decimals=3)
                rotations.append(R)
        elif structure is not None:
            rotations = structure.get_spglib_symmetry_dataset(symprec)

        klist = []
        bandlist = []
        spdlist = []
        # for each symmetry operation

        for i, _ in enumerate(rotations):
            # for each point
            for j, _ in enumerate(self.kpoints):
                # apply symmetry operation to kpoint
                sympoint_vector = np.dot(rotations[i], self.kpoints[j])
                sympoint = sympoint_vector.tolist()

                if sympoint not in klist:
                    klist.append(sympoint)

                    band = self.bands[j].tolist()
                    bandlist.append(band)
                    spd = self.spd[j].tolist()
                    spdlist.append(spd)

        self.kpoints = np.array(klist)
        self.bands = np.array(bandlist)
        self.spd = np.array(spdlist)
        self.spd = self._spd2projected(spd)

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class VaspXML(collections.abc.Mapping):
    """contains."""

    def __init__(self, filename="vasprun.xml", dos_interpolation_factor=None):

        self.variables = {}
        self.dos_interpolation_factor = dos_interpolation_factor

        if not os.path.isfile(filename):
            raise ValueError("File not found " + filename)
        else:
            self.filename = filename

        self.spins_dict = {"spin 1": "Spin-up", "spin 2": "Spin-down"}
        # self.positions = None
        # self.stress = None
        # self.array_sizes = {}
        self.data = self.read()

    def read(self):
        """
        Read and parse vasprun.xml.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.parse_vasprun(self.filename)

    @property
    def bands(self):
        spins = list(self.data["general"]["eigenvalues"]
                     ["array"]["data"].keys())
        kpoints_list = list(
            self.data["general"]["eigenvalues"]["array"]["data"]["spin 1"].keys()
        )
        eigen_values = {}
        nbands = len(
            self.data["general"]["eigenvalues"]["array"]["data"][spins[0]][
                kpoints_list[0]
            ][kpoints_list[0]]
        )
        nkpoints = len(kpoints_list)
        for ispin in spins:
            eigen_values[ispin] = {}
            eigen_values[ispin]["eigen_values"] = np.zeros(
                shape=(nbands, nkpoints))
            eigen_values[ispin]["occupancies"] = np.zeros(
                shape=(nbands, nkpoints))
            for ikpoint, kpt in enumerate(kpoints_list):
                temp = np.array(
                    self.data["general"]["eigenvalues"]["array"]["data"][ispin][kpt][
                        kpt
                    ]
                )
                eigen_values[ispin]["eigen_values"][:, ikpoint] = (
                    temp[:, 0] - self.fermi
                )
                eigen_values[ispin]["occupancies"][:, ikpoint] = temp[:, 1]
        return eigen_values

    @property
    def bands_projected(self):
        # projected[iatom][ikpoint][iband][iprincipal][iorbital][ispin]
        labels = self.data["general"]["projected"]["array"]["info"]
        spins = list(self.data["general"]["projected"]["array"]["data"].keys())
        kpoints_list = list(
            self.data["general"]["projected"]["array"]["data"][spins[0]].keys()
        )
        bands_list = list(
            self.data["general"]["projected"]["array"]["data"][spins[0]][
                kpoints_list[0]
            ][kpoints_list[0]].keys()
        )
        bands_projected = {"labels": labels}

        nspins = len(spins)
        nkpoints = len(kpoints_list)
        nbands = len(bands_list)
        norbitals = len(labels)
        natoms = self.initial_structure.natoms
        bands_projected["projection"] = np.zeros(
            shape=(nspins, nkpoints, nbands, natoms, norbitals)
        )
        for ispin, spn in enumerate(spins):
            for ikpoint, kpt in enumerate(kpoints_list):
                for iband, bnd in enumerate(bands_list):
                    bands_projected["projection"][
                        ispin, ikpoint, iband, :, :
                    ] = np.array(
                        self.data["general"]["projected"]["array"]["data"][spn][kpt][
                            kpt
                        ][bnd][bnd]
                    )
        # ispin, ikpoint, iband, iatom, iorbital
        bands_projected["projection"] = np.swapaxes(
            bands_projected["projection"], 0, 3)
        # iatom, ikpoint, iband, ispin, iorbital
        bands_projected["projection"] = np.swapaxes(
            bands_projected["projection"], 3, 4)
        # iatom, ikpoint, iband, iorbital, ispin
        bands_projected["projection"] = bands_projected["projection"].reshape(
            natoms, nkpoints, nbands, 1, norbitals, nspins
        )

        return bands_projected

    def _get_dos_total(self):

        spins = list(self.data["general"]["dos"]
                     ["total"]["array"]["data"].keys())
        energies = np.array(
            self.data["general"]["dos"]["total"]["array"]["data"][spins[0]]
        )[:, 0]
        dos_total = {"energies": energies}
        for ispin in spins:
            dos_total[self.spins_dict[ispin]] = np.array(
                self.data["general"]["dos"]["total"]["array"]["data"][ispin]
            )[:, 1]

        return dos_total, list(dos_total.keys())

    def _get_dos_projected(self, atoms=[]):

        if len(atoms) == 0:
            atoms = np.arange(self.initial_structure.natoms)

        if "partial" in self.data["general"]["dos"]:
            dos_projected = {}
            ion_list = [
                "ion %s" % str(x + 1) for x in atoms
            ]  # using this name as vasrun.xml uses ion #
            for i in range(len(ion_list)):
                iatom = ion_list[i]
                name = self.initial_structure.atoms[atoms[i]] + str(atoms[i])
                spins = list(
                    self.data["general"]["dos"]["partial"]["array"]["data"][
                        iatom
                    ].keys()
                )
                energies = np.array(
                    self.data["general"]["dos"]["partial"]["array"]["data"][iatom][
                        spins[0]
                    ][spins[0]]
                )[:, 0]
                dos_projected[name] = {"energies": energies}
                for ispin in spins:
                    dos_projected[name][self.spins_dict[ispin]] = np.array(
                        self.data["general"]["dos"]["partial"]["array"]["data"][iatom][
                            ispin
                        ][ispin]
                    )[:, 1:]
            return (
                dos_projected,
                self.data["general"]["dos"]["partial"]["array"]["info"],
            )
        else:
            print("This calculation does not include partial density of states")
            return None, None

    @property
    def dos(self):
        energies = self.dos_total["energies"]
        total = []
        for ispin in self.dos_total:
            if ispin == "energies":
                continue
            total.append(self.dos_total[ispin])
        # total = np.array(total).T
        return DensityOfStates(
            energies=energies,
            total=total,
            projected=self.dos_projected,
            interpolation_factor=self.dos_interpolation_factor,
        )

    @property
    def dos_to_dict(self):
        """
        the complete density (total,projected) of states as a python dictionary
        """
        return {"total": self._get_dos_total(), "projected": self._get_dos_projected()}

    @property
    def dos_total(self):
        """
        Returns the total density of states as a pychemia.visual.DensityOfSates object
        """
        dos_total, labels = self._get_dos_total()
        dos_total["energies"] -= self.fermi

        return dos_total

    @property
    def dos_projected(self):
        """
        Returns the projected DOS as a multi-dimentional array, to be used in the
        pyprocar.core.dos object
        """
        ret = []
        dos_projected, info = self._get_dos_projected()
        if dos_projected is None:
            return None
        norbitals = len(info) - 1
        info[0] = info[0].capitalize()
        labels = []
        labels.append(info[0])
        ret = []
        for iatom in dos_projected:
            temp_atom = []
            for iorbital in range(norbitals):
                temp_spin = []
                for key in dos_projected[iatom]:
                    if key == "energies":
                        continue
                    temp_spin.append(dos_projected[iatom][key][:, iorbital])
                temp_atom.append(temp_spin)
            ret.append([temp_atom])
        return ret

    @property
    def kpoints(self):
        """
        Returns the kpoints used in the calculation in form of a pychemia.core.KPoints object
        """

        if self.data["kpoints_info"]["mode"] == "listgenerated":
            kpoints = dict(
                mode="path", kvertices=self.data["kpoints_info"]["kpoint_vertices"]
            )
        else:
            kpoints = dict(
                mode=self.data["kpoints_info"]["mode"].lower(),
                grid=self.data["kpoints_info"]["kgrid"],
                shifts=self.data["kpoints_info"]["user_shift"],
            )
        return kpoints

    @property
    def kpoints_list(self):
        """
        Returns the list of kpoints and weights used in the calculation in form of a pychemia.core.KPoints object
        """
        return dict(
            mode="reduced",
            kpoints_list=self.data["kpoints"]["kpoints_list"],
            weights=self.data["kpoints"]["k_weights"],
        )

    @property
    def incar(self):
        """
        Returns the incar parameters used in the calculation as pychemia.code.vasp.VaspIncar object
        """
        return self.data["incar"]

    @property
    def vasp_parameters(self):
        """
        Returns all of the parameters vasp has used in this calculation
        """
        return self.data["vasp_params"]

    @property
    def potcar_info(self):
        """
        Returns the information about pseudopotentials(POTCAR) used in this calculation
        """
        return self.data["atom_info"]["atom_types"]

    @property
    def fermi(self):
        """
        Returns the fermi energy
        """
        return self.data["general"]["dos"]["efermi"]

    @property
    def species(self):
        """
        Returns the species in POSCAR
        """
        return self.initial_structure.species

    @property
    def structures(self):
        """
        Returns a list of pychemia.core.Structure representing all the ionic step structures
        """
        symbols = [x.strip() for x in self.data["atom_info"]["symbols"]]
        structures = []
        for ist in self.data["structures"]:

            st = Structure(
                atoms=symbols,
                fractional_coordinates=ist["reduced"],
                lattice=ist["cell"],
            )
            structures.append(st)
        return structures

    @property
    def structure(self):
        """
        crystal structure of the last step
        """
        return self.structures[-1]

    @property
    def forces(self):
        """
        Returns all the forces in ionic steps
        """
        return self.data["forces"]

    @property
    def initial_structure(self):
        """
        Returns the initial Structure as a pychemia structure
        """
        return self.structures[0]

    @property
    def final_structure(self):
        """
        Returns the final Structure as a pychemia structure
        """

        return self.structures[-1]

    @property
    def iteration_data(self):
        """
        Returns a list of information in each electronic and ionic step of calculation
        """
        return self.data["calculation"]

    @property
    def energies(self):
        """
        Returns a list of energies in each electronic and ionic step [ionic step,electronic step, energy]
        """
        scf_step = 0
        ion_step = 0
        double_counter = 1
        energies = []
        for calc in self.data["calculation"]:
            if "ewald" in calc["energy"]:
                if double_counter == 0:
                    double_counter += 1
                    scf_step += 1
                elif double_counter == 1:
                    double_counter = 0
                    ion_step += 1
                    scf_step = 1
            else:
                scf_step += 1
            energies.append([ion_step, scf_step, calc["energy"]["e_0_energy"]])
        return energies

    @property
    def last_energy(self):
        """
        Returns the last calculated energy of the system
        """
        return self.energies[-1][-1]

    @property
    def energy(self):
        """
        Returns the last calculated energy of the system
        """
        return self.last_energy

    @property
    def convergence_electronic(self):
        """
        Returns a boolian representing if the last electronic self-consistent
        calculation converged
        """
        ediff = self.vasp_parameters["electronic"]["EDIFF"]
        last_dE = abs(self.energies[-1][-1] - self.energies[-2][-1])
        if last_dE < ediff:
            return True
        else:
            return False

    @property
    def convergence_ionic(self):
        """
        Returns a boolian representing if the ionic part of the
        calculation converged
        """
        energies = np.array(self.energies)
        nsteps = len(np.unique(np.array(self.energies)[:, 0]))
        if nsteps == 1:
            print("This calculation does not have ionic steps")
            return True
        else:
            ediffg = self.vasp_parameters["ionic"]["EDIFFG"]
            if ediffg < 0:
                last_forces_abs = np.abs(np.array(self.forces[-1]))
                return not (np.any(last_forces_abs > abs(ediffg)))
            else:
                last_ionic_energy = energies[(
                    energies[:, 0] == nsteps)][-1][-1]
                penultimate_ionic_energy = energies[(energies[:, 0] == (nsteps - 1))][
                    -1
                ][-1]
                last_dE = abs(last_ionic_energy - penultimate_ionic_energy)
                if last_dE < ediffg:
                    return True
        return False

    @property
    def convergence(self):
        """
        Returns a boolian representing if the the electronic self-consistent
        and ionic calculation converged
        """
        return self.convergence_electronic and self.convergence_ionic

    @property
    def is_finished(self):
        """
        Always returns True, need to fix this according to reading the xml as if the calc is
        not finished we will have errors in xml parser
        """
        # if vasprun.xml is read the calculation is finished
        return True

    def text_to_bool(self, text):
        """boolians in vaspxml are stores as T or F in str format, this function coverts them to python boolians """
        text = text.strip(" ")
        if text == "T" or text == ".True." or text == ".TRUE.":
            return True
        else:
            return False

    def conv(self, ele, _type):
        """This function converts the xml text to the type specified in the attrib of xml tree """

        if _type == "string":
            return ele.strip()
        elif _type == "int":
            return int(ele)
        elif _type == "logical":
            return self.text_to_bool(ele)
        elif _type == "float":
            if "*" in ele:
                return np.nan
            else:
                return float(ele)

    def get_varray(self, xml_tree):
        """Returns an array for each varray tag in vaspxml """
        ret = []
        for ielement in xml_tree:
            ret.append([float(x) for x in ielement.text.split()])
        return ret

    def get_params(self, xml_tree, dest):
        """dest should be a dictionary
        This function is recurcive #check spelling"""
        for ielement in xml_tree:
            if ielement.tag == "separator":
                dest[ielement.attrib["name"].strip()] = {}
                dest[ielement.attrib["name"].strip()] = self.get_params(
                    ielement, dest[ielement.attrib["name"]]
                )
            else:
                if "type" in ielement.attrib:
                    _type = ielement.attrib["type"]
                else:
                    _type = "float"
                if ielement.text is None:
                    dest[ielement.attrib["name"].strip()] = None

                elif len(ielement.text.split()) > 1:
                    dest[ielement.attrib["name"].strip()] = [
                        self.conv(x, _type) for x in ielement.text.split()
                    ]
                else:
                    dest[ielement.attrib["name"].strip()] = self.conv(
                        ielement.text, _type
                    )

        return dest

    def get_structure(self, xml_tree):
        """Returns a dictionary of the structure """
        ret = {}
        for ielement in xml_tree:
            if ielement.tag == "crystal":
                for isub in ielement:
                    if isub.attrib["name"] == "basis":
                        ret["cell"] = self.get_varray(isub)
                    elif isub.attrib["name"] == "volume":
                        ret["volume"] = float(isub.text)
                    elif isub.attrib["name"] == "rec_basis":
                        ret["rec_cell"] = self.get_varray(isub)
            elif ielement.tag == "varray":
                if ielement.attrib["name"] == "positions":
                    ret["reduced"] = self.get_varray(ielement)
        return ret

    def get_scstep(self, xml_tree):
        """This function extracts the self-consistent step information """
        scstep = {"time": {}, "energy": {}}
        for isub in xml_tree:
            if isub.tag == "time":
                scstep["time"][isub.attrib["name"]] = [
                    float(x) for x in isub.text.split()
                ]
            elif isub.tag == "energy":
                for ienergy in isub:
                    scstep["energy"][ienergy.attrib["name"]
                                     ] = float(ienergy.text)
        return scstep

    def get_set(self, xml_tree, ret):
        """ This function will extract any element taged set recurcively"""
        if xml_tree[0].tag == "r":
            ret[xml_tree.attrib["comment"]] = self.get_varray(xml_tree)
            return ret
        else:
            ret[xml_tree.attrib["comment"]] = {}
            for ielement in xml_tree:

                if ielement.tag == "set":
                    ret[xml_tree.attrib["comment"]
                        ][ielement.attrib["comment"]] = {}
                    ret[xml_tree.attrib["comment"]][
                        ielement.attrib["comment"]
                    ] = self.get_set(
                        ielement,
                        ret[xml_tree.attrib["comment"]
                            ][ielement.attrib["comment"]],
                    )
            return ret

    def get_general(self, xml_tree, ret):
        """ This function will parse any element in calculatio other than the structures, scsteps"""
        if "dimension" in [x.tag for x in xml_tree]:
            ret["info"] = []
            ret["data"] = {}
            for ielement in xml_tree:
                if ielement.tag == "field":
                    ret["info"].append(ielement.text.strip(" "))
                elif ielement.tag == "set":
                    for iset in ielement:
                        ret["data"] = self.get_set(iset, ret["data"])
            return ret
        else:
            for ielement in xml_tree:
                if ielement.tag == "i":
                    if "name" in ielement.attrib:
                        if ielement.attrib["name"] == "efermi":
                            ret["efermi"] = float(ielement.text)
                    continue
                ret[ielement.tag] = {}
                ret[ielement.tag] = self.get_general(
                    ielement, ret[ielement.tag])
            return ret

    def parse_vasprun(self, vasprun):
        tree = ET.parse(vasprun)
        root = tree.getroot()

        calculation = []
        structures = []
        forces = []
        stresses = []
        orbital_magnetization = {}
        run_info = {}
        incar = {}
        general = {}
        kpoints_info = {}
        vasp_params = {}
        kpoints_list = []
        k_weights = []
        atom_info = {}
        for ichild in root:

            if ichild.tag == "generator":
                for ielement in ichild:
                    run_info[ielement.attrib["name"]] = ielement.text

            elif ichild.tag == "incar":
                incar = self.get_params(ichild, incar)

            # Skipping 1st structure which is primitive cell
            elif ichild.tag == "kpoints":

                for ielement in ichild:
                    if ielement.items()[0][0] == "param":
                        kpoints_info["mode"] = ielement.items()[0][1]
                        if kpoints_info["mode"] == "listgenerated":
                            kpoints_info["kpoint_vertices"] = []
                            for isub in ielement:

                                if isub.attrib == "divisions":
                                    kpoints_info["ndivision"] = int(isub.text)
                                else:
                                    if len(isub.text.split()) != 3:
                                        continue
                                    kpoints_info["kpoint_vertices"].append(
                                        [float(x) for x in isub.text.split()]
                                    )
                        else:
                            for isub in ielement:
                                if isub.attrib["name"] == "divisions":
                                    kpoints_info["kgrid"] = [
                                        int(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "usershift":
                                    kpoints_info["user_shift"] = [
                                        float(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "genvec1":
                                    kpoints_info["genvec1"] = [
                                        float(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "genvec2":
                                    kpoints_info["genvec2"] = [
                                        float(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "genvec3":
                                    kpoints_info["genvec3"] = [
                                        float(x) for x in isub.text.split()
                                    ]
                                elif isub.attrib["name"] == "shift":
                                    kpoints_info["shift"] = [
                                        float(x) for x in isub.text.split()
                                    ]

                    elif ielement.items()[0][1] == "kpointlist":
                        for ik in ielement:
                            kpoints_list.append([float(x)
                                                 for x in ik.text.split()])
                        kpoints_list = array(kpoints_list)
                    elif ielement.items()[0][1] == "weights":
                        for ik in ielement:
                            k_weights.append(float(ik.text))
                        k_weights = array(k_weights)

            # Vasp Parameters
            elif ichild.tag == "parameters":
                vasp_params = self.get_params(ichild, vasp_params)

            # Atom info
            elif ichild.tag == "atominfo":

                for ielement in ichild:
                    if ielement.tag == "atoms":
                        atom_info["natom"] = int(ielement.text)
                    elif ielement.tag == "types":
                        atom_info["nspecies"] = int(ielement.text)
                    elif ielement.tag == "array":
                        if ielement.attrib["name"] == "atoms":
                            for isub in ielement:
                                if isub.tag == "set":
                                    atom_info["symbols"] = []
                                    for isym in isub:
                                        atom_info["symbols"].append(
                                            isym[0].text)
                        elif ielement.attrib["name"] == "atomtypes":
                            atom_info["atom_types"] = {}
                            for isub in ielement:
                                if isub.tag == "set":
                                    for iatom in isub:
                                        atom_info["atom_types"][iatom[1].text] = {}
                                        atom_info["atom_types"][iatom[1].text][
                                            "natom_per_specie"
                                        ] = int(iatom[0].text)
                                        atom_info["atom_types"][iatom[1].text][
                                            "mass"
                                        ] = float(iatom[2].text)
                                        atom_info["atom_types"][iatom[1].text][
                                            "valance"
                                        ] = float(iatom[3].text)
                                        atom_info["atom_types"][iatom[1].text][
                                            "pseudopotential"
                                        ] = iatom[4].text.strip()

            elif ichild.tag == "structure":
                if ichild.attrib["name"] == "initialpos":
                    initial_pos = self.get_structure(ichild)
                elif ichild.attrib["name"] == "finalpos":
                    final_pos = self.get_structure(ichild)

            elif ichild.tag == "calculation":
                for ielement in ichild:
                    if ielement.tag == "scstep":
                        calculation.append(self.get_scstep(ielement))
                    elif ielement.tag == "structure":
                        structures.append(self.get_structure(ielement))
                    elif ielement.tag == "varray":
                        if ielement.attrib["name"] == "forces":
                            forces.append(self.get_varray(ielement))
                        elif ielement.attrib["name"] == "stress":
                            stresses.append(self.get_varray(ielement))

                    # elif ielement.tag == 'eigenvalues':
                    #     for isub in ielement[0] :
                    #         if isub.tag == 'set':
                    #             for iset in isub :
                    #                 eigen_values[iset.attrib['comment']] = {}
                    #                 for ikpt in iset :
                    #                     eigen_values[iset.attrib['comment']][ikpt.attrib['comment']] = get_varray(ikpt)

                    elif ielement.tag == "separator":
                        if ielement.attrib["name"] == "orbital magnetization":
                            for isub in ielement:
                                orbital_magnetization[isub.attrib["name"]] = [
                                    float(x) for x in isub.text.split()
                                ]

                    # elif ielement.tag == 'dos':
                    #     for isub in ielement :
                    #         if 'name' in isub.attrib:
                    #             if isub.attrib['name'] == 'efermi' :
                    #                 dos['efermi'] = float(isub.text)
                    #             else :
                    #                 dos[isub.tag] = {}
                    #                 dos[isub.tag]['info'] = []
                    #               for iset in isub[0]  :
                    #                   if iset.tag == 'set' :
                    #                       for isub_set in iset:
                    #                           dos[isub.tag] = get_set(isub_set,dos[isub.tag])
                    #                   elif iset.tag == 'field' :
                    #                       dos[isub.tag]['info'].append(iset.text.strip(' '))
                    else:
                        general[ielement.tag] = {}
                        general[ielement.tag] = self.get_general(
                            ielement, general[ielement.tag]
                        )
            # NEED TO ADD ORBITAL MAGNETIZATION

        return {
            "calculation": calculation,
            "structures": structures,
            "forces": forces,
            "run_info": run_info,
            "incar": incar,
            "general": general,
            "kpoints_info": kpoints_info,
            "vasp_params": vasp_params,
            "kpoints": {"kpoints_list": kpoints_list, "k_weights": k_weights},
            "atom_info": atom_info,
        }

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()
