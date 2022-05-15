import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import sys
from ..utilsprocar import UtilsProcar


class ProcarParser:
    """Parses a PROCAR file and store it in memory. It only deals with
    PROCAR files, that means no Fermi energy (UtilsProcar.FermiOutcar
    can help), and the reciprocal vectors should be supplied (if used,
    see UtilsProcar class).

    Members:

    __init__(self, loglevel): The setup the variables internally, `loglevel`
      sets the verbosity level ie: `loglevel=logging.DEBUG` for debugging. Its
      default is `logging.WARNING`

    readFile(self, procar=None, permissive=False, recLattice=None):
      The only method of the API it load the file completely.

      Arguments:
    `procar=None`: name of the PROCAR file, can be a gzipped file (the
                    extension is no required). The default covers a wide range
                    of obvious alternatives.
      `permissive=False`: Set to `True` if the PROCAR file has problems reading
                        the Kpoints (stupid Fortran), but in that case the
                          Kpoints mesh will be discarded. Future updates could
                          allow it to handle other formating/corruption issues.
      `recLattice`=None: Reciprical Vectors, you want to provide them since not
                        all the paths on the BZ are the same.

    Don't use the other methods beggining with underscores "_"

    Example:
    To read a PROCAR or PROCAR.gz file:
    >>> foo = ProcarParser()
    >>> foo.readFile()

    To include the reciprocal vectors, and file name MyFirstPROCAR
    >>> outcarparser = UtilsProcar()
    >>> recLat = outcarparser.RecLatOutcar(args.outcar)
    >>> foo = ProcarParser()
    >>> foo.readFile("MyFirstPROCAR", recLat=recLat)

    """

    def __init__(self, loglevel=logging.WARNING):
        # array with k-points, they have the following values
        # -None: if not parsed (yet) or parsed with a `permissive` flag on
        # -direct coordinates: if a recLattice was not supplied to the parser
        # -cartesian coords: if a recLattice was supplied to the parser.
        # In the later cases, self.kpoints.shape=(self.kpointsCount, 3)
        self.kpoints = None
        # Number of kpoints, as given by the KPOINTS header (PROCAR file)
        self.kpointsCount = None

        # bands headers present in PROCAR file.
        # self.bands.shape=(self.kpointsCount,self.bandsCount)
        self.bands = None
        # Number of bands. For a spin polarized calculation the number of
        # bands is double (spin ip + spin down). On this array there is no
        # distinction between spin up and down
        self.bandsCount = None

        # Number of ions+1 the +1 is the 'tot' field, ie: the sum over all atoms
        self.ionsCount = None

        self.fileStr = None  # the actual file, stored in memory
        self.spd = None  # the atom/orbital projected data
        self.cspd = None  # spd data with the phase
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
        self.orbitalCount = None  # number of orbitals

        # number of spin components (blocks of data), 1: non-magnetic non
        # polarized, 2: spin polarized collinear, 4: non-collinear
        # spin.
        # NOTE: before calling to `self._readOrbital` the case '4'
        # is marked as '1'
        self.ispin = None
        self.recLattice = None  # reciprocal lattice vectors
        self.utils = UtilsProcar()

        self.log = logging.getLogger("ProcarParser")
        self.log.setLevel(loglevel)
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(
            logging.Formatter("%(name)s::%(levelname)s:" " %(message)s")
        )
        self.ch.setLevel(logging.DEBUG)
        self.log.addHandler(self.ch)
        # At last, one message to the logger.
        self.log.debug("Procar instanciated")
        return

    @property
    def nspin(self):
        """
        number of spin, default is 1.
        """
        nspindict = {1: 1, 2: 2, 4: 2, None: 1}
        return nspindict[self.ispin]

    @property
    def spd_orb(self):
        # indices: ikpt, iband, ispin, iion, iorb
        # remove indices and total from iorb.
        return self.spd[:, :, :, 1:-1]

    def _readKpoints(self, permissive=False):
        """Reads the k-point headers. A typical k-point line is:
        k-point    1 :    0.00000000 0.00000000 0.00000000  weight = 0.00003704\n
        fills self.kpoint[kpointsCount][3]
        The weights are discarded (are they useful?)
        """
        self.log.debug("readKpoints")
        if not self.fileStr:
            log.warning("You should invoke `procar.readFile()` instead. Returning")
            return

        # finding all the K-points headers
        self.kpoints = re.findall(r"k-point\s+\d+\s*:\s+([-.\d\s]+)", self.fileStr)
        self.log.debug(str(len(self.kpoints)) + " K-point headers found")
        self.log.debug("The first match found is: " + str(self.kpoints[0]))

        # trying to build an array
        self.kpoints = [x.split() for x in self.kpoints]
        try:
            self.kpoints = np.array(self.kpoints, dtype=float)
        except ValueError:
            self.log.error("Ill-formatted data:")
            print("\n".join([str(x) for x in self.kpoints]))
            if permissive is True:
                # Discarding the kpoints list, however I need to set
                # self.ispin beforehand.
                if len(self.kpoints) == self.kpointsCount:
                    self.ispin = 1
                elif len(self.kpoints) == 2 * self.kpointsCount:
                    self.ispin = 2
                else:
                    raise ValueError("Kpoints do not match with ispin=1 or 2.")
                self.kpoints = None
                self.log.warning("K-points list is useless, setting it to `None`")
                return
            else:
                raise ValueError("Badly formated Kpoints headers, try `--permissive`")
        # if successful, go on

        # trying to identify an non-polarized or non-collinear case, a
        # polarized case or a defective file

        if len(self.kpoints) != self.kpointsCount:
            # if they do not match, may means two things a spin polarized
            # case or a bad file, lets check
            self.log.debug(
                "Number of kpoints do not match, looking for a " "spin-polarized case"
            )
            # lets start testing if it is spin polarized, if so, there
            # should be 2 identical blocks of kpoints.
            up, down = np.vsplit(self.kpoints, 2)
            if (up == down).all():
                self.log.info("Spin-polarized calculation found")
                self.ispin = 2
                # just keeping one set of kpoints (the other will be
                # discarded)
                self.kpoints = up
            else:
                self.log.error("Number of K-points do not match! check them.")
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

        self.log.debug(str(self.kpoints))
        self.log.info("The kpoints shape is " + str(self.kpoints.shape))

        if self.recLattice is not None:
            self.log.info("Changing to cartesians coordinates")
            self.kpoints = np.dot(self.kpoints, self.recLattice)
            self.log.debug("New kpoints: \n" + str(self.kpoints))
        return

    def _readBands(self):
        """Reads the bands header. A typical bands is:
        band   1 # energy   -7.11986315 # occ.  1.00000000

        fills self.bands[kpointsCount][bandsCount]

        The occupation numbers are discarded (are they useful?)"""
        self.log.debug("readBands")
        if not self.fileStr:
            log.warning("You should invoke `procar.read()` instead. Returning")
            return

        # finding all bands
        self.bands = re.findall(
            r"band\s*(\d+)\s*#\s*energy\s*([-.\d\s]+)", self.fileStr
        )
        self.log.debug(
            str(len(self.bands))
            + " bands headers found, bands*Kpoints = "
            + str(self.bandsCount * self.kpointsCount)
        )
        self.log.debug("The first match found is: " + str(self.bands[0]))

        # checking if the number of bands match

        if len(self.bands) != self.bandsCount * self.kpointsCount * self.ispin:
            self.log.error("Number of bands headers do not match")
            raise RuntimeError("Number of bands don't match")

        # casting to array to manipulate the bands
        self.bands = np.array(self.bands, dtype=float)
        self.log.debug(str(self.bands))

        # Now I will deal with the spin polarized case. The goal is join
        # them like for a non-magnetic case
        if self.ispin == 2:
            # up and down are along the first axis
            up, down = np.vsplit(self.bands, 2)
            self.log.debug("up   , " + str(up.shape))
            self.log.debug("down , " + str(down.shape))

            # reshapping (the 2  means both band index and energy)
            up.shape = (self.kpointsCount, self.bandsCount, 2)
            down.shape = (self.kpointsCount, self.bandsCount, 2)

            # setting the correct number of bands (up+down)
            self.bandsCount *= 2
            self.log.debug("New number of bands : " + str(self.bandsCount))

            # and joining along the second axis (axis=1), ie: bands-like
            self.bands = np.concatenate((up, down), axis=1)

        # otherwise just reshaping is needed
        else:
            self.bands.shape = (self.kpointsCount, self.bandsCount, 2)

        # Making a test if the broadcast is rigth, otherwise just print
        test = [x.max() - x.min() for x in self.bands[:, :, 0].transpose()]
        if np.array(test).any():
            self.log.warning(
                "The indexes of bands do not match. CHECK IT. "
                "Likely the data was wrongly broadcasted"
            )
            self.log.warning(str(self.bands[:, :, 0]))
        # Now safely removing the band index
        self.bands = self.bands[:, :, 1]
        self.log.info("The bands shape is " + str(self.bands.shape))
        return

    def _readOrbital(self):
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
        self.log.debug("readOrbital")
        if not self.fileStr:
            log.warning("You should invoke `procar.readFile()` instead. Returning")
            return

        # finding all orbital headers
        self.spd = re.findall(r"ion(.+)", self.fileStr)
        self.log.info("the first orbital match reads: " + self.spd[0])
        self.log.debug("And I found " + str(len(self.spd)) + " orbitals headers")

        # testing if the orbital names are known (the standard ones)
        FoundOrbs = self.spd[0].split()
        size = len(FoundOrbs)
        # only the first 'size' orbital
        StdOrbs = self.orbitalName[: size - 1] + self.orbitalName[-1:]
        StdOrbs_short = self.orbitalName_short[: size - 1] + self.orbitalName_short[-1:]
        StdOrbs_old = self.orbitalName_old[: size - 1] + self.orbitalName_old[-1:]
        if (
            FoundOrbs != (StdOrbs)
            and FoundOrbs != (StdOrbs_short)
            and FoundOrbs != (StdOrbs_old)
        ):
            self.log.warning(
                str(size) + " orbitals. (Some of) They are unknow (if "
                "you did 'filter' them it is OK)."
            )
        self.orbitalCount = size
        self.orbitalNames = self.spd[0].split()
        self.log.debug(
            "Anyway, I will use the following set of orbitals: "
            + str(self.orbitalNames)
        )

        # Now reading the bulk of data
        self.log.debug("Now searching the values")
        # The case of just one atom is handled differently since the VASP
        # output is a little different
        if self.ionsCount == 1:
            self.spd = re.findall(r"^(\s*1\s+.+)$", self.fileStr, re.MULTILINE)
        else:
            # Added by Francisco to speed up filtering on June 4th, 2019
            # get rid of phase factors
            self.spd = re.findall(r"ion.+tot\n([-.\d\seto]+)", self.fileStr)
            self.spd = "".join(self.spd)
            self.spd = re.findall(r"([-.\d\se]+tot.+)\n", self.spd)
        # free the memory (could be a lot)
        self.fileStr = None
        self.log.debug("the first entry is \n" + self.spd[0])

        # Now the method will try to find the value of self.ispin,
        # previously it was set to either 1 or 2. If "1", it could be 1 or
        # 4, but previously it was impossible to find the rigth value. If
        # "2" it has to macth with the number of entries of spd data.

        self.log.debug("Number of entries found: " + str(len(self.spd)))
        expected = self.bandsCount * self.kpointsCount
        self.log.debug(
            "The number of entries for a non magnetic calc. is: " + str(expected)
        )
        if expected == len(self.spd):
            self.log.info("Both numbers match, ok, going ahead")
        # catching a non-collinear calc.
        elif expected * 4 == len(self.spd):
            self.log.info("non-collinear calculation found")
            # testing if previous ispin value is ok
            if self.ispin != 1:
                self.log.warning(
                    "Incompatible data: self.ispin= " + str(self.ispin) + ". Now is 4"
                )
            self.ispin = 4
        else:
            self.log.error("The parser or data is wrong!!!")
            self.log.info("bandsCount: " + str(self.bandsCount))
            self.log.info("KpointsCount: " + str(self.kpointsCount))
            raise RuntimeError("Shit happens")

        # checking for consistency
        for line in self.spd:
            if len(line.split()) != (self.ionsCount) * (self.orbitalCount + 1):
                self.log.error(
                    "Expected: "
                    + str(self.ionsCount)
                    + "*"
                    + str(self.orbitalCount + 1)
                    + " = "
                    + str((self.ionsCount) * (self.orbitalCount + 1))
                    + " Fields. Present block: "
                    + str(len(line.split()))
                )
                raise RuntimeError("Flats happens")

        # replacing the "tot" string by a number, to allows a conversion
        # to numpy
        self.spd = [x.replace("tot", "0") for x in self.spd]
        self.spd = [x.split() for x in self.spd]
        self.spd = np.array(self.spd, dtype=float)
        self.log.debug("The spd (old) array shape is:" + str(self.spd.shape))

        # handling collinear polarized case
        if self.ispin == 2:
            self.log.debug("Handling spin-polarized collinear case...")
            # splitting both spin components, now they are along k-points
            # axis (1st axis) but, then should be concatenated along the
            # bands.
            up, down = np.vsplit(self.spd, 2)
            # ispin = 1 for a while, we will made the distinction
            up.shape = (
                self.kpointsCount,
                int(self.bandsCount / 2),
                1,
                self.ionsCount,
                self.orbitalCount + 1,
            )
            down.shape = (
                self.kpointsCount,
                int(self.bandsCount / 2),
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
            self.log.debug("polarized collinear spd.shape= " + str(self.spd.shape))

        # otherwise, just a reshaping suffices
        else:
            self.spd.shape = (
                self.kpointsCount,
                self.bandsCount,
                self.ispin,
                self.ionsCount,
                self.orbitalCount + 1,
            )

        self.log.info("spd array ready. Its shape is:" + str(self.spd.shape))
        return

    def readFile(self, procar=None, phase=False, permissive=False, recLattice=None):
        """Reads and parses the whole PROCAR file. This method is a sort
    of metamethod: it opens the file, reads the meta data and call the
    respective functions for parsing kpoints, bands, and projected
    data.

    Args:

    -procar: The file name, if `None` or a directory, a suitable set
     of defaults will be used. Default=None

    -permissive: turn on (or off) some features to deal with badly
     written PROCAR files (stupid fortran), up to now just ignores the
     kpoints coordinates, which -as side effect- prevent he rigth
     space between kpoints. Default=False (off)


    -recLattice: a 3x3 array containing the reciprocal vectors, to
     change the Kpoints from rec. coordinates to cartesians. Rarely
     given by hand, see `UtilsProcar.RecLatProcar`. If given, the
     kpoints will be converted from direct coordinates to cartesian
     ones. Default=None

    """
        self.log.debug("readFile...")

        self.recLattice = recLattice

        self.log.debug("Opening file: '" + str(procar) + "'")
        f = self.utils.OpenFile(procar)
        # Line 1: PROCAR lm decomposed
        f.readline()  # throwaway
        # Line 2: # of k-points:  816   # of bands:  52   # of ions:   8
        metaLine = f.readline()  # metadata
        self.log.debug("The metadata line is: " + metaLine)
        re.findall(r"#[^:]+:([^#]+)", metaLine)
        self.kpointsCount, self.bandsCount, self.ionsCount = map(
            int, re.findall(r"#[^:]+:([^#]+)", metaLine)
        )
        self.log.info("kpointsCount = " + str(self.kpointsCount))
        self.log.info("bandsCount = " + str(self.bandsCount))
        self.log.info("ionsCount = " + str(self.ionsCount))
        if self.ionsCount == 1:
            self.log.warning(
                "Special case: only one atom found. The program may not work as expected"
            )
        else:
            self.log.debug("An extra ion representing the  total value will be added")
            self.ionsCount = self.ionsCount + 1

        # reading all the rest of the file to be parsed below
        self.fileStr = f.read()
        self._readKpoints(permissive)
        self._readBands()
        self._readOrbital()
        self.log.debug("readfile...done")
        return

    # Slower way to parse the phase included PROCAR. He Xu implemented a faster way. See readFile2()
    # if phase == False:
    #     self.log.debug("readFile...")

    #     self.recLattice = recLattice

    #     self.log.debug("Opening file: '" + str(procar) + "'")
    #     f = self.utils.OpenFile(procar)
    #     # Line 1: PROCAR lm decomposed
    #     f.readline()  # throwaway
    #     # Line 2: # of k-points:  816   # of bands:  52   # of ions:   8
    #     metaLine = f.readline()  # metadata
    #     self.log.debug("The metadata line is: " + metaLine)
    #     re.findall(r"#[^:]+:([^#]+)", metaLine)
    #     self.kpointsCount, self.bandsCount, self.ionsCount = \
    #         list(map(int, re.findall(r"#[^:]+:([^#]+)", metaLine)))
    #     self.log.info("kpointsCount = " + str(self.kpointsCount))
    #     self.log.info("bandsCount = " + str(self.bandsCount))
    #     self.log.info("ionsCount = " + str(self.ionsCount))
    #     if self.ionsCount is 1:
    #         self.log.warning(
    #             "Special case: only one atom found. The program may not work as expected"
    #         )
    #     else:
    #         self.log.debug(
    #             "An extra ion representing the  total value will be added")
    #         self.ionsCount = self.ionsCount + 1

    #     #reading all the rest of the file to be parsed below
    #     self.fileStr = f.read()
    #     self._readKpoints(permissive)
    #     self._readBands()
    #     self._readOrbital()
    #     self.log.debug("readfile...done")

    # elif phase == True:  #for LORBIT=12
    #     self.recLattice = recLattice
    #     f = self.utils.OpenFile(procar)
    #     f.readline()  #throw away first line
    #     metaLine = f.readline()  # header

    #     #parsing header information
    #     self.kpointsCount, self.bandsCount, self.ionsCount = list(
    #         map(int, re.findall(r"#[^:]+:([^#]+)", metaLine)))

    #     if self.ionsCount is 1:
    #         self.log.warning(
    #             "Special case: only one atom found. The program may not work as expected"
    #         )
    #     else:
    #         self.log.debug(
    #             "An extra ion representing the  total value will be added")
    #         self.ionsCount = self.ionsCount + 1

    #     #reading all the rest of the file to be parsed below saved as spd
    #     self.fileStr = f.read()
    #     self._readKpoints(permissive)
    #     self._readBands()
    #     self._readOrbital()
    #     self.log.debug("readfile...done")

    #     #reading the complex phase data. will be saved as cspd
    #     f2 = self.utils.OpenFile(procar)
    #     f2.readline()  #throw away first line
    #     f2.readline()  #throw away header line
    #     data2 = f2.read()

    #     #parsing
    #     spd0 = re.findall(r"ion(.+)", data2)  #headers of the blocks
    #     FoundOrbs = spd0[1].split()
    #     size = len(FoundOrbs)
    #     corbitalCount = size
    #     spd_phase = re.findall(
    #         r"(?<=dx2-y2)([charge0-9.\s-]*)(?=band|k-point|\Z)",
    #         data2)  #for LORBIT=12

    #     spd_new = []
    #     for i in range(len(spd_phase)):
    #         #get last list of original block and replace spd last line and append all to get new spd.
    #         # for charge line use spd instead of spd_real
    #         spd_last = spd_phase[i].split()[-(corbitalCount + 2):]
    #         result = []
    #         for counter, value in enumerate(spd_last):
    #             result.append(value)
    #             result.append('0')
    #         del result[1]
    #         del result[-1]

    #         #replace last line of each spd block
    #         spd_block = spd_phase[i].split()
    #         spd_block[-(corbitalCount + 2):] = result
    #         spd_block = [x.replace('charge', '0') for x in spd_block]
    #         spd_new.append(spd_block)

    #     # conversion to numpy
    #     spd_phase = np.array(spd_new, dtype=float)

    #     #reshaping
    #     spd_phase.shape = (self.kpointsCount, self.bandsCount, 1,
    #                        self.ionsCount, 2 * corbitalCount + 2)

    #     #matrix to hold complex values
    #     self.cspd = np.zeros([
    #         self.kpointsCount, self.bandsCount, 1, self.ionsCount,
    #         corbitalCount + 2
    #     ],
    #                          dtype='complex')

    #     for ikpointsCount in range(self.kpointsCount):
    #         for ibandsCount in range(self.bandsCount):
    #             for iionsCount in range(self.ionsCount):
    #                 orbs_real = spd_phase[ikpointsCount][ibandsCount][0][
    #                     iionsCount][1:-1:2]
    #                 orbs_imag = spd_phase[ikpointsCount][ibandsCount][0][
    #                     iionsCount][2::2]
    #                 orbs_all = orbs_real + (1j * orbs_imag)
    #                 self.cspd[ikpointsCount, ibandsCount, 0,
    #                           iionsCount, :] = np.concatenate(
    #                               [[
    #                                   spd_phase[ikpointsCount][ibandsCount]
    #                                   [0][iionsCount][0]
    #                               ], orbs_all,
    #                                [
    #                                    spd_phase[ikpointsCount]
    #                                    [ibandsCount][0][iionsCount][-1]
    #                                ]])

    # return

    def readFile2(
        self,
        procar=None,
        phase=False,
        permissive=False,
        recLattice=None,
        ispin=None,  # the only spin channle to read
    ):
        """
        Read file in a line by line manner.
        Only used when the phase factor is in procar. (for vasp, lorbit=12)
        """
        # Fall back to readFile function if no phase
        self.bands = None
        if not phase:
            self.readFile(
                procar=procar, phase=False, permissive=permissive, recLattice=recLattice
            )
        else:
            if ispin is None:
                nspin = 1
            else:
                nspin = 2
            iispin = 0
            self.projections = None
            ikpt = 0
            iband = 0
            nkread = 0
            # with open(self.fname) as myfile:
            f = self.utils.OpenFile(procar)
            lines = iter(f.readlines())
            last_iband = -1
            for line in lines:
                if line.startswith("# of k-points"):
                    a = re.findall(":\s*([0-9]*)", line)
                    self.kpointsCount, self.bandsCount, self.ionsCount = map(int, a)
                    self.kpoints = np.zeros([self.kpointsCount, 3])
                    self.kweights = np.zeros(self.kpointsCount)
                    if self.bands is None:
                        self.bands = np.zeros(
                            [nspin, self.kpointsCount, self.bandsCount]
                        )
                if line.strip().startswith("k-point"):
                    ss = line.strip().split()
                    ikpt = int(ss[1]) - 1
                    k0 = float(ss[3])
                    k1 = float(ss[4])
                    k2 = float(ss[5])
                    w = float(ss[-1])
                    self.kpoints[ikpt, :] = [k0, k1, k2]
                    self.kweights[ikpt] = w
                    nkread += 1
                    if nkread <= self.kpointsCount:
                        iispin = 0
                    else:
                        iispin = 1
                if line.strip().startswith("band"):
                    ss = line.strip().split()
                    try:
                        iband = int(ss[1]) - 1
                    except ValueError:
                        iband = last_iband + 1
                    last_iband = iband
                    e = float(ss[4])
                    occ = float(ss[-1])
                    self.bands[iispin, ikpt, iband] = e
                if line.strip().startswith("ion"):
                    if line.strip().endswith("tot"):
                        self.orbitalName = line.strip().split()[1:-1]
                        self.orbitalCount = len(self.orbitalName)
                    if self.projections is None:
                        self.projections = np.zeros(
                            [
                                self.kpointsCount,
                                self.bandsCount,
                                self.ionsCount,
                                self.orbitalCount,
                            ]
                        )
                        self.carray = np.zeros(
                            [
                                self.kpointsCount,
                                self.bandsCount,
                                nspin,
                                self.ionsCount,
                                self.orbitalCount,
                            ],
                            dtype="complex",
                        )
                    for i in range(self.ionsCount):
                        line = next(lines)
                        t = line.strip().split()
                        if len(t) == self.orbitalCount + 2:
                            self.projections[ikpt, iband, iispin, :] = [
                                float(x) for x in t[1:-1]
                            ]
                        elif len(t) == self.orbitalCount * 2 + 2:
                            self.carray[ikpt, iband, iispin, i, :] += np.array(
                                [float(x) for x in t[1:-1:2]]
                            )
                            self.carray[ikpt, iband, iispin, i, :] += 1j * np.array(
                                [float(x) for x in t[2::2]]
                            )

                        # Added by Francisco to parse older version of PROCAR format on Jun 11, 2019
                        elif len(t) == self.orbitalCount * 1 + 1:
                            self.carray[ikpt, iband, iispin, i, :] += np.array(
                                [float(x) for x in t[1:]]
                            )
                            line = next(lines)
                            t = line.strip().split()
                            self.carray[ikpt, iband, iispin, i, :] += 1j * np.array(
                                [float(x) for x in t[1:]]
                            )
                        else:
                            raise Exception(
                                "Cannot parse line to projection: %s" % line
                            )
