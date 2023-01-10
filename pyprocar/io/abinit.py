import re
from numpy import array
from ..core import Structure, DensityOfStates, ElectronicBandStructure, KPath
from ..utils import elements
import numpy as np
import glob
import collections
import os
import sys


class Output(collections.abc.Mapping):
    """This class contains methods to parse the fermi energy and reciprocal
    lattice vectors from the Abinit output file.

    Since abinit v9.x creates the PROCAR file,
    the methods in vasp.py will be used to parse it.
    """

    def __init__(self, abinit_output=None):

        # variables
        self.abinit_output = abinit_output
        self.fermi = None
        self.reclat = None  # reciprocal lattice vectors
        self.nspin = None  # spin
        self.coordinates = None  # reduced atomic coordinates
        self.variables = {}

        # call parsing functions
        self._readFermi()
        self._readRecLattice()
        self._readCoordinates()
        self._readLattice()
        self._readAtoms()

        # creating structure object
        self.structure = Structure(
            atoms=self.atoms,
            fractional_coordinates=self.coordinates,
            lattice=self.lattice,
        )

        return

    def _readFermi(self):
        """Reads the Fermi energy from the Abinit output file."""

        with open(self.abinit_output, "r") as rf:
            data = rf.read()
            self.fermi = float(
                re.findall(
                    r"Fermi\w*.\(\w*.HOMO\)\s*\w*\s*\(\w*\)\s*\=\s*([0-9.+-]*)", data
                )[0]
            )

            # Converting from Hartree to eV
            self.fermi = 27.211396641308 * self.fermi

            # read spin (nsppol)
            self.nspin = re.findall(r"nsppol\s*=\s*([1-9]*)", data)[0]

            return

    def _readRecLattice(self):
        """Reads the reciprocal lattice vectors
        from the Abinit output file. This is used to calculate
        the k-path in cartesian coordinates if required."""

        with open(self.abinit_output, "r") as rf:
            data = rf.read()
            lattice_block = re.findall(r"G\([1,2,3]\)=\s*([0-9.\s-]*)", data)
            lattice_block = lattice_block[3:]
            self.reclat = array(
                [lattice_block[0:3][i].split() for i in range(len(lattice_block))],
                dtype=float,
            )

            return

    def _readCoordinates(self):
        """Reads the coordinates as given by the xred keyword."""

        with open(self.abinit_output, "r") as rf:
            data = rf.read()
            coordinate_block = re.findall(r"xred\s*([+-.0-9E\s]*)", data)[-1].split()
            coordinate_list = np.array([float(x) for x in coordinate_block])
            self.coordinates = coordinate_list.reshape(len(coordinate_list) // 3, 3)

            return

    def _readLattice(self):
        """Reads the lattice vectors from rprim keyword and scales with acell."""

        with open(self.abinit_output, "r") as rf:
            data = rf.read()

            # acell
            acell = re.findall(r"acell\s*([+-.0-9E\s]*)", data)[-1].split()
            acell = np.array([float(x) for x in acell])

            # convert acell from Bohr to Angstrom
            acell = 0.529177 * acell

            # rprim
            rprim_block = re.findall(r"rprim\s*([+-.0-9E\s]*)", data)[-1].split()
            rprim_list = np.array([float(x) for x in rprim_block])
            rprim = rprim_list.reshape(len(rprim_list) // 3, 3)

            # lattice
            self.lattice = np.zeros(shape=(3, 3))
            for i in range(len(acell)):
                self.lattice[i, :] = acell[i] * rprim[i, :]

            return

    def _readAtoms(self):
        """Reads atomic elements used and puts them in an array according to their composition."""

        with open(self.abinit_output, "r") as rf:
            data = rf.read()

            # Getting typat and znucl
            typat = re.findall(r"typat\s*([+-.0-9E\s]*)", data)[-1].split()
            typat = [int(x) for x in typat]

            znucl = re.findall(r"znucl\s*([+-.0-9E\s]*)", data)[-1].split()
            znucl = [int(float(x)) for x in znucl]

            self.atoms = [elements.atomic_symbol(znucl[x - 1]) for x in typat]

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class Kpoints(collections.abc.Mapping):
    """This class parses the k-point information.

    Attributes
    ----------
        file_str :
        kgrid :
        mode :
        cartesian :
        ngrids :
        knames :
        kshift :
        variables :
        metadata :
        comment :
        special_kpoints :
        automatic :
        filename :

    """

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

        # calling parser
        self._parse_kpoints()
        self.kpath = KPath(
            knames=self.knames,
            special_kpoints=self.special_kpoints,
            ngrids=self.ngrids,
            has_time_reversal=has_time_reversal,
        )

    def _parse_kpoints(self):
        """This method parses a VASP-type KPOINTS file for plotting band structures
        for Abinit calculations. Abinit has multiple ways of providing k-points for
        band structure calculations, therefore we need to come up with a general way
        to tackle this.

        TODO:
        - Generalize a method for obtaining k-point information from Abinit.

        """

        # with open(self.filename, "r") as rf:
        #     data = rf.read()

        #     self.kptbounds = findall(
        #         r"kptbounds[0-9]*\s*([+-\/.0-9A-Za-z#E\s]*)\n", data
        #     )[0].split()

        #     kptbounds_array = np.array(kptbounds)
        #     kptbounds_array2 = kptbounds_array.reshape(len(kptbounds_array) // 5, 5)

        with open(self.filename, "r") as rf:
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

            elif self.mode == "line":
                if rf.readline()[0].lower() == "c":
                    self.cartesian = True
                else:
                    self.cartesian = False
                self.file_str = rf.read()

                temp = np.array(
                    re.findall(
                        "([0-9.-]+)\s*([0-9.-]+)\s*([0-9.-]+)(.*)", self.file_str
                    )
                )
                temp_special_kp = temp[:, :3].astype(float)
                temp_knames = temp[:, -1]
                nsegments = temp_special_kp.shape[0] // 2
                if len(self.ngrids) == 1:
                    self.ngrids = [self.ngrids[0]] * nsegments
                self.knames = np.reshape(
                    [x.replace("!", "").strip() for x in temp_knames], (nsegments, 2)
                )
                self.special_kpoints = temp_special_kp.reshape(nsegments, 2, 3)

        return

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class Procar(collections.abc.Mapping):
    """This class has functions to parse the PROCAR file
    generated from Abinit. Unlike, VASP here the PROCAR files
    need to be merged and fixed for formatting issues prior to
    further processing.

    Attributes
    ----------
        file_str :
        spd :
        structure :
        orbitalName_short :
        kpoints :
        occupancies :
        spd_phase :
        ebs :
        variables :
        ispin :
        orbitalName :
        filename :
        bandsCount :
        ionsCount :
        has_phase :
        labels :
        carray :
        meta_lines :
        orbitalName_old :
        reciprocal_lattice :
        bands :
        kpointsCount :

    """

    def __init__(
        self,
        filename="PROCAR",
        filelist=None,
        abinit_output=None,
        structure=None,
        reciprocal_lattice=None,
        kpath=None,
        kpoints=None,
        efermi=None,
        interpolation_factor=1,
    ):
        self.variables = {}
        self.filename = filename
        self.filelist = filelist
        self.abinit_output = abinit_output
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

        # calling functions

        self._read()

        if self.has_phase:
            self.carray = self.spd_phase[:, :, :, :-1, 1:-1]

        # self.bands += efermi
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

    def _mergeparallel(self, inputfiles=None, outputfile=None, abinit_output=None):
        """This merges Procar files seperated between k-point ranges.
        Happens with parallel Abinit runs.
        """
        print("Merging parallel files...")
        filenames = sorted(inputfiles)
        print(filenames)

        # creating an instance of the AbinitParser class
        if abinit_output:
            abinitparserobject = Output(abinit_output=abinit_output)
            nspin = int(abinitparserobject.nspin)
        else:
            raise IOError("Abinit output file not found.")

        if nspin != 2:
            with open(outputfile, "w") as outfile:
                for fname in filenames:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)

        elif nspin == 2:
            # for spin polarized calculations the spin down segments are saved in the
            # second half of the PROCAR's but in reverse k-point order. So we have to
            # fix the order and merge the second half of the PROCAR's.
            spinup_list = filenames[: int(len(filenames) / 2)]
            spindown_list = filenames[int(len(filenames) / 2) :]

            # reading the second line of the header to set as the separating line
            # in the colinear spin PROCAR.
            fp = open(spinup_list[0], "r")
            header1 = fp.readline()
            header2 = fp.readline()
            fp.close()

            # second half of PROCAR files in reverse order.
            spindown_list.reverse()

            # Writing new PROCAR with first spin up, header2 and then
            # spin down (reversed).
            with open(outputfile, "w") as outfile:
                for spinupfile in spinup_list:
                    with open(spinupfile) as infile:
                        for line in infile:
                            outfile.write(line)
                outfile.write("\n")
                outfile.write(header2)
                outfile.write("\n")
                for spindownfile in spindown_list:
                    with open(spindownfile) as infile:
                        for line in infile:
                            outfile.write(line)

    def _fixformat(self, inputfile=None, outputfile=None):

        """Fixes the formatting of Abinit's Procar
        when the tot projection is not summed and spin directions
        not seperated.
        """
        print("Fixing formatting errors...")
        # removing existing temporary fixed file
        if os.path.exists(outputfile):
            os.remove(outputfile)

        ####### Fixing the parallel PROCARs from Abinit ##########

        rf = open(inputfile, "r")
        data = rf.read()
        rf.close()

        # reading headers
        rffl = open(inputfile, "r")
        first_line = rffl.readline()
        rffl.close()

        # header
        header = re.findall("#\sof\s.*", data)[0]

        # writing to PROCAR
        fp = open(outputfile, "a")
        fp.write(first_line)
        fp.write(str(header) + "\n\n")

        # get all the k-point line headers
        kpoints_raw = re.findall("k-point\s*[0-9]\s*:*.*", data)

        for kpoint_counter in range(len(kpoints_raw)):

            if kpoint_counter == (len(kpoints_raw) - 1):
                # get bands of last k point
                bands_raw = re.findall(
                    kpoints_raw[kpoint_counter] + "([a-z0-9\s\n.+#-]*)", data
                )[0]

            else:
                # get bands between k point n and n+1
                bands_raw = re.findall(
                    kpoints_raw[kpoint_counter]
                    + "([a-z0-9\s\n.+#-]*)"
                    + kpoints_raw[kpoint_counter + 1],
                    data,
                )[0]

            # get the bands headers for a certain k point
            raw_bands = re.findall("band\s*[0-9]*.*", bands_raw)

            # writing k point header to file
            fp.write(kpoints_raw[kpoint_counter] + "\n\n")

            for band_counter in range(len(raw_bands)):

                if band_counter == (len(raw_bands) - 1):
                    # the last band
                    single_band = re.findall(
                        raw_bands[band_counter] + "([a-z0-9.+\s\n-]*)", bands_raw
                    )[0]

                else:
                    # get a single band
                    single_band = re.findall(
                        raw_bands[band_counter]
                        + "([a-z0-9.+\s\n-]*)"
                        + raw_bands[band_counter + 1],
                        bands_raw,
                    )[0]

                # get the column headers for ion, orbitals and total
                column_header = re.findall("ion\s.*tot", single_band)[0]

                # get number of ions using PROCAR file
                nion_raw = re.findall("#\s*of\s*ions:\s*[0-9]*", data)[0]
                nion = int(nion_raw.split(" ")[-1])

                # the first column of the band. Same as ions
                first_column = []
                for x in single_band.split("\n"):
                    if x != "":
                        if x != " ":
                            if x.split()[0] != "ion":
                                first_column.append(x.split()[0])

                # number of spin orientations
                norient = int(len(first_column) / nion)

                # calculate the number of orbital headers (s,p,d etc.)
                for x in single_band.split("\n"):
                    if x != "":
                        if x != " ":
                            if x.split()[0] == "ion":
                                norbital = len(x.split()) - 2

                # array to store a single band data as seperate lines
                single_band_lines = []
                for x in single_band.split("\n"):
                    if x != "":
                        if x != " ":
                            if x.split()[0] != "ion":
                                single_band_lines.append(x)

                # create empty array to store data (the orbitals + tot)
                bands_orb = np.zeros(shape=(norient, nion, norbital + 1))

                # enter data into bands_orb
                iion = 0
                iorient = 0
                for x in single_band.split("\n"):
                    if x != "" and x != " " and x.split()[0] != "ion":
                        iline = x.split()
                        if iion > 1:
                            iion = 0
                            iorient += 1
                        for iorb in range(0, norbital + 1):
                            bands_orb[iorient, iion, iorb] = float(iline[iorb + 1])
                        iion += 1

                # create an array to store the total values
                tot = np.zeros(shape=(norient, norbital + 1))

                # entering data into tot array
                for iorient in range(norient):
                    tot[iorient, :] = np.sum(bands_orb[iorient, :, :], axis=0)

                # writing data
                fp.write(raw_bands[band_counter] + "\n\n")
                fp.write(column_header + "\n")

                band_iterator = 0
                total_count = 0
                for orientations_count in range(norient):
                    for ions_count in range(nion):
                        fp.write(single_band_lines[band_iterator] + "\n")
                        band_iterator += 1
                    fp.write("tot  " + " ".join(map(str, tot[total_count, :])) + "\n\n")
                    total_count += 1
        fp.close()

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
        Tries to open a file, it has suitable values for PROCAR and can
        handle gzipped files. A directory containing the PROCAR file can be provided
        as well.
        Updated to handle Abinit PROCAR files. Merging and fixing of format is
        performed if necessary.

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

        # checking if filename is just a path and needs a "PROCAR to " be
        # appended
        if os.path.isdir(self.filename):
            self.filename_tmp = self.filename
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
            # Check if a filelist of PROCAR files is provided. Otherwise check
            # for PROCAR_* files in a given directory and attempt to merge them.

            if self.filelist is not None:
                inFiles = sorted(self.filelist)

                if isinstance(inFiles, list):
                    self._mergeparallel(
                        inputfiles=inFiles,
                        outputfile=self.filename,
                        abinit_output=self.abinit_output,
                    )
                    in_file = open(self.filename, "r")
                else:
                    raise IOError("Files not found")

            else:
                if os.path.isdir(self.filename_tmp):
                    if self.filename_tmp[-1] != r"/":
                        self.filename_tmp += "/"
                    self.filename_tmp += "PROCAR_*"

                    if glob.glob(self.filename_tmp):
                        inFiles = sorted(glob.glob(self.filename_tmp))
                    if isinstance(inFiles, list):
                        self._mergeparallel(
                            inputfiles=inFiles,
                            outputfile=self.filename,
                            abinit_output=self.abinit_output,
                        )
                        in_file = open(self.filename, "r")
                    else:
                        raise IOError("Files not found")

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
        StdOrbs_short = self.orbitalName_short[: size - 1] + self.orbitalName_short[-1:]
        StdOrbs_old = self.orbitalName_old[: size - 1] + self.orbitalName_old[-1:]
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
                    "Incompatible data: self.ispin= " + str(self.ispin) + ". Now is 4"
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
                self.bandsCount,
                1,
                self.ionsCount,
                self.orbitalCount + 1,
            )
            down = down.reshape(
                self.kpointsCount,
                self.bandsCount,
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
            self.spd_phase = re.findall(r"^(\s*1\s+.+)$", self.file_str, re.MULTILINE)
        else:
            # Added by Francisco to speed up filtering on June 4th, 2019
            # get rid of phase factors
            self.spd_phase = re.findall(
                r"ion.+\n([-.\d\se]+charge[-.\d\se]+)", self.file_str
            )
            self.spd_phase = "".join(self.spd_phase)
            self.spd_phase = re.findall(r"([-.\d\se]+charge.+)\n", self.spd_phase)
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
            temp[:, :, :, :, (i + 1) // 2].imag = self.spd_phase[:, :, :, :, i + 1]
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

                # R = structure.reciprocal_lattice.dot(R).dot(np.linalg.inv(structure.reciprocal_lattice))
                R = (
                    np.linalg.inv(self.reciprocal_lattice.T)
                    .dot(R)
                    .dot(self.reciprocal_lattice.T)
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
                sympoint_vector = rotations[i].dot(self.kpoints[j])
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
