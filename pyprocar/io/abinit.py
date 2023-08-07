import re
from numpy import array
from ..core import Structure, DensityOfStates, ElectronicBandStructure, KPath
from ..utils import elements
from .vasp import Procar, Kpoints
import numpy as np
import glob
import collections
import os
import sys


class Output(collections.abc.Mapping):
    """This class contains methods to parse the fermi energy, reciprocal
    lattice vectors and structure from the Abinit output file.
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

            # for single atom at (0,0,0) xred is not printed out at the end.
            # So we use this workaround.
            if not coordinate_block:
                coordinate_block = re.findall(
                    r"reduced\scoordinates\s\(array\sxred\)\sfor\s*[1-9]\satoms\n([+-.0-9E\s]*)\n",
                    data,
                )[-1].split()

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


class AbinitKpoints(collections.abc.Mapping):
    """This class parses the k-point information."""

    def __init__(self, filename="KPOINTS", has_time_reversal=True):
        self.variables = {}
        self.filename=filename
        # calling parser
        self._parse_kpoints()

        self.kpath=None
        if self.abinitkpointsobject.knames:
            self.kpath = KPath(
                knames=self.abinitkpointsobject.knames,
                special_kpoints=self.abinitkpointsobject.special_kpoints,
                ngrids=self.abinitkpointsobject.ngrids,
                has_time_reversal=has_time_reversal,
            )

    def _parse_kpoints(self):
        """Reads KPOINTS file.
        Abinit has multiple way of defining the
        k-path. For the time being, we suggest to use
        a KPOINTS file similar to VASP to obtain the
        k-path for PyProcar from an Abinit calculation.
        """

        
        self.abinitkpointsobject = Kpoints(filename=self.filename, has_time_reversal=True)

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class AbinitProcar(collections.abc.Mapping):
    """This class has functions to parse the PROCAR file
    generated from Abinit. Unlike, VASP here the PROCAR files
    need to be merged and fixed for formatting issues prior to
    further processing.
    """

    def __init__(
        self,
        dirname=None,
        inFiles=None,
        abinit_output=None,
        reciprocal_lattice=None,
        kpath=None,
        efermi=None,
    ):
        self.variables = {}
        self.inFiles = inFiles
        self.abinit_output = abinit_output
        self.reciprocal_lattice = reciprocal_lattice
        self.kpath = kpath
        self.efermi = efermi

        # Preparing files for merging
        # reading in all PROCAR_* files and putting it into a list if not provided.
        if inFiles is None:
            tmp_str = os.path.join(dirname, "PROCAR_*")
            inFiles = sorted(glob.glob(tmp_str))
        else:
            inFiles = inFiles
        filename = os.path.join(dirname, "PROCAR")
        if isinstance(inFiles, list):
            self._mergeparallel(
                inputfiles=inFiles,
                outputfile=filename,
                abinit_output=self.abinit_output,
            )
        else:
            pass


        # Use VASP Procar parser following PROCAR merge
        self.abinitprocarobject = Procar(
            filename=filename,
            structure=None,
            reciprocal_lattice=self.reciprocal_lattice,
            kpath=self.kpath,
            efermi=self.efermi,
            interpolation_factor=1,
        )

    def _mergeparallel(self, inputfiles=None, outputfile=None, abinit_output=None):
        """This merges Procar files seperated between k-point ranges.
        Happens with parallel Abinit runs.
        """
        print("Merging parallel files...")
        filenames = sorted(inputfiles)

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

    def __contains__(self, x):
        return x in self.variables

    def __getitem__(self, x):
        return self.variables.__getitem__(x)

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self):
        return self.variables.__len__()


class AbinitDOSParser:
    def __init__(
        self,
        dirname: str = "",
    ):
        self.dirname = dirname

        self.dos_total, self.energies, self.fermi = self._parse_total_dos()
        self.projected = self._parse_projected_dos_files()

        self.dos = DensityOfStates(
            energies=self.energies,
            total=self.dos_total,
            projected=self.projected,
            interpolation_factor=1,
        )

    def _parse_total_dos(self):
        self.total_dos_filename = glob.glob(self.dirname + "/abinito_DOS_TOTAL*")[0]

        with open(self.total_dos_filename) as f:
            text_lines = f.readlines()
            header_text = "".join(text_lines[:13])
            dos_text = text_lines[13:]

        nsppol = int(re.findall("nsppol\s=\s(\d)", header_text)[0])
        fermi = float(re.findall("Fermi energy\s:\s*([-\d*.]*)", header_text)[0])

        energy_details = re.findall(
            "between\s*([-\d*.]*)\s*and\s*([-\d*.]*)\s*Hartree\s*by\s*steps\s*of\s*([-\d*.]*)",
            header_text,
        )[0]
        energy_details = [float(x) for x in energy_details]
        e_min, e_max, e_step = energy_details

        energies = np.arange(e_min, e_max + e_step, e_step)
        n_energies = energies.shape[0]

        if nsppol == 2:
            n_spin_header = 3
            n_up_start = n_spin_header
            n_up_end = n_up_start + n_energies
            dos_up = dos_text[n_up_start:n_up_end]

            n_block_spacing = 3
            n_down_start = n_spin_header + n_energies + n_block_spacing
            n_down_end = n_down_start + n_energies
            dos_down = dos_text[n_down_start:n_down_end]

            # converting to floats
            dos_down = np.array(
                [[float(value) for value in line.split()] for line in dos_down]
            )
            dos_up = np.array(
                [[float(value) for value in line.split()] for line in dos_up]
            )

            # Keep dos column
            dos_down = dos_down[:, 1]
            dos_up = dos_up[:, 1]

            dos_total = np.vstack([dos_up, dos_down])

        else:
            n_header = 2
            n_up_start = n_header
            n_up_end = n_up_start + n_energies
            dos_up = dos_text[n_up_start:n_up_end]

            # converting to floats
            dos_up = np.array(
                [[float(value) for value in line.split()] for line in dos_up]
            )

            # Keep dos column
            dos_up = dos_up[:, 1]

            dos_total = dos_up[None, :]

        energies -= fermi
        return dos_total, energies, fermi

    def _parse_projected_dos_files(self):
        self.projected_dos_filenames = glob.glob(self.dirname + "/abinito_DOS_AT*")

        n_atoms = len(self.projected_dos_filenames)
        projected = [0] * n_atoms
        for filename in self.projected_dos_filenames:
            dos_atom_projections, atom_index = self._parse_projected_dos_file(filename)

            projected[atom_index - 1] = dos_atom_projections
            projected.append(dos_atom_projections)
        projected = np.array(projected)

        # organizing the projection array in the appropiate formate
        # (n_atoms, n_principals, n_orbitals, n_spins, n_dos)
        projected = np.transpose(projected, (0, 2, 3, 1))

        # This is adding for the principle quantum number. Throughout the code this is unecessary, but puting here for consitency
        projected = projected[:, None, :, :, :]
        return projected

    def _parse_projected_dos_file(self, filename):
        with open(filename) as f:
            text_lines = f.readlines()
            header_text = "".join(text_lines[:13])
            dos_text = text_lines[13:]

        nsppol = int(re.findall("nsppol\s=\s(\d)", header_text)[0])

        energy_details = re.findall(
            "between\s*([-\d*.]*)\s*and\s*([-\d*.]*)\s*Hartree\s*by\s*steps\s*of\s*([-\d*.]*)",
            header_text,
        )[0]
        energy_details = [float(x) for x in energy_details]
        e_min, e_max, e_step = energy_details

        atom_detail_text = "".join(dos_text[:4])
        atom_index = int(re.findall("iatom=\s*(\d)", atom_detail_text)[0])

        energies = np.arange(e_min, e_max + e_step, e_step)
        n_energies = energies.shape[0]

        if nsppol == 2:
            n_spin_header = 7
            n_up_start = n_spin_header
            n_up_end = n_up_start + n_energies
            dos_up = dos_text[n_up_start:n_up_end]

            n_block_spacing = 7
            n_down_start = n_up_end + n_block_spacing
            n_down_end = n_down_start + n_energies
            dos_down = dos_text[n_down_start:n_down_end]

            # converting to floats
            dos_down = np.array(
                [[float(value) for value in line.split()] for line in dos_down]
            )
            dos_up = np.array(
                [[float(value) for value in line.split()] for line in dos_up]
            )

            # Keep only s,p,d projections
            dos_down = dos_down[:, 11:20]
            dos_up = dos_up[:, 11:20]

            dos_atom_projections = np.dstack([dos_up, dos_down])

        else:
            n_spin_header = 6
            n_up_start = n_spin_header
            n_up_end = n_up_start + n_energies
            dos_up = dos_text[n_up_start:n_up_end]

            # converting to floats
            dos_up = np.array(
                [[float(value) for value in line.split()] for line in dos_up]
            )

            # Keep only s,p,d projections
            dos_up = dos_up[:, 11:20]

            dos_atom_projections = dos_up[:, :, None]

        return dos_atom_projections, atom_index
