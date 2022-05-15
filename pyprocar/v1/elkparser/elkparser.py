"""
Created on Fri March 2 2020
@author: Pedram Tavadze

"""

from re import findall

import numpy as np


class ElkParser:
    def __init__(self, elkin="elk.in", kdirect=True):

        # elk specific inp
        self.fin = elkin
        self.file_names = []
        self.elkin = None
        self.nbands = None

        self.nspin = None
        self.kticks = None

        self.kdirect = kdirect

        self.kpoints = None
        self.bands = None

        self.spd = None
        self.cspd = None

        # spin polarized parameters
        self.spinpol = None

        self.orbitalName = [
            "Y00",
            "Y1-1",
            "Y10",
            "Y11",
            "Y2-2",
            "Y2-1",
            "Y20",
            "Y21",
            "Y22",
            "Y3-3",
            "Y3-2",
            "Y3-1",
            "Y30",
            "Y3-1",
            "Y3-2",
            "Y3-3",
        ]
        self.orbital_name_short = ["s", "p", "d", "f"]
        self.norbital = None  # number of orbitals

        # number of spin components (blocks of data), 1: non-magnetic non
        # polarized, 2: spin polarized collinear, 4: non-collinear
        # spin.
        # NOTE: before calling to `self._readOrbital` the case '4'
        # is marked as '1'

        self._read_elkin()
        self._read_bands()

        return

    #    @property
    #    def nspin(self):
    #        """
    #        number of spin, default is 1.
    #        """
    #        nspindict = {1: 1, 2: 2, 4: 2, None: 1}
    #        return nspindict[self.ispin]

    @property
    def spd_orb(self):
        # indices: ikpt, iband, ispin, iion, iorb
        # remove indices and total from iorb.
        return self.spd[:, :, :, 1:-1]

    @property
    def nhigh_sym(self):
        """
        Returns nu,ber of high symmetry points
        """
        return int(findall("plot1d\n\s*([0-9]*)\s*[0-9]*", self.elkin)[0])

    @property
    def nkpoints(self):
        """
        Returns total number of kpoints
        """
        return int(findall("plot1d\n\s*[0-9]*\s*([0-9]*)", self.elkin)[0])

    @property
    def tasks(self):
        """
        Returns the tasks calculated by elk
        """
        return [int(x) for x in findall("tasks\s*([0-9\s\n]*)", self.elkin)[0].split()]

    @property
    def high_symmetry_points(self):
        """
        Returns the corrdinates of high symmtery points provided in elk.in
        """
        if 20 in self.tasks or 21 in self.tasks or 22 in self.tasks:
            raw_hsp = findall(
                "plot1d\n\s*[0-9]*\s*[0-9]*.*\n" + self.nhigh_sym * "([0-9\s\.-]*).*\n",
                self.elkin,
            )[0]
            high_symmetry_points = np.zeros(shape=(self.nhigh_sym, 3))
            for ihs in range(self.nhigh_sym):
                high_symmetry_points[ihs, :] = [
                    float(x) for x in raw_hsp[ihs].split()[0:3]
                ]
            return high_symmetry_points

    @property
    def nspecies(self):
        """
        Returns number of species
        """

        return int(findall("atoms\n\s*([0-9]*)", self.elkin)[0])

    @property
    def natom(self):
        """
        Returns number of atoms in the structure
        """
        natom = 0
        for ispc in findall("'([A-Za-z]*).in'.*\n\s*([0-9]*)", self.elkin):
            natom += int(ispc[1])
        return natom

    @property
    def composition(self):
        """
        Returns the composition of the structure
        """
        composition = {}
        for ispc in findall("'([A-Za-z]*).in'.*\n\s*([0-9]*)", self.elkin):
            composition[ispc[0]] = int(ispc[1])
        return composition

    @property
    def knames(self):
        """
        Returns the names of the high symmetry points(x ticks) provided in elk.in

        """
        raw_ticks = findall(
            "plot1d\n\s*[0-9]*\s*[0-9]*.*\n" + self.nhigh_sym * ".*:(.*)\n", self.elkin
        )[0]
        if len(raw_ticks) != self.nhigh_sym:
            knames = [str(x) for x in range(self.nhigh_sym)]
        else:
            knames = [
                "$%s$" % (x.replace(",", "").replace("vlvp1d", "").replace(" ", ""))
                for x in findall(
                    "plot1d\n\s*[0-9]*\s*[0-9]*.*\n" + self.nhigh_sym * ".*:(.*)\n",
                    self.elkin,
                )[0]
            ]
        return knames

    def _read_elkin(self):
        """
        Reads and parses elk.in
        """
        rf = open(self.fin, "r")
        self.elkin = rf.read()
        rf.close()

        if 20 in self.tasks:
            self.file_names.append("BANDS.OUT")
        if 21 in self.tasks or 22 in self.tasks:
            ispc = 1
            for spc in self.composition:
                for iatom in range(self.composition[spc]):
                    self.file_names.append(
                        "BAND_S{:02d}_A{:04d}.OUT".format(ispc, iatom + 1)
                    )
                ispc += 1

        # Checking if spinpol = .true. in elk.in
        try:
            self.spinpol = findall(r"spinpol\s*([.a-zA-Z]*)", self.elkin)[0]
        except IndexError:
            self.spinpol = None

        if self.spinpol:
            if self.spinpol == ".true.":
                print("\nElk colinear spin calculation detected.\n")
                self.spinpol = True
                self.nspin = 2
            else:
                print("\nElk non spin calculation detected.\n")
                self.nspin = 1
        else:
            print(
                "\nNo spinpol keyword found in elk.in. Assuming non spin calculation.\n"
            )
            self.nspin = 1

    def _read_bands(self):
        """
        if the task is any of 20,21,22 it parses the BANDS*.OUT files
        and prepares the spd, bands and kpoints for bandsplot
        """
        if np.any([x in self.tasks for x in [20, 21, 22]]):

            rf = open(self.file_names[0], "r")
            lines = rf.readlines()
            rf.close()

            self.nbands = int(len(lines) / (self.nkpoints + 1))
            self.bands = np.zeros(shape=(self.nkpoints, self.nbands))

            rf = open("BANDLINES.OUT", "r")
            bandLines = rf.readlines()
            rf.close()

            tick_pos = []
            # using strings for a better comparision and avoiding rounding by python
            for iline in range(0, len(bandLines), 3):
                tick_pos.append(bandLines[iline].split()[0])
            x_points = []
            for iline in range(self.nkpoints):
                x_points.append(lines[iline].split()[0])

            self.kpoints = np.zeros(shape=(self.nkpoints, 3))
            x_points = np.array(x_points)
            tick_pos = np.array(tick_pos)

            self.kticks = []
            for ihs in range(1, self.nhigh_sym):
                start = np.where(x_points == tick_pos[ihs - 1])[0][0]
                end = np.where(x_points == tick_pos[ihs])[0][0] + 1
                self.kpoints[start:end][:] = np.linspace(
                    self.high_symmetry_points[ihs - 1],
                    self.high_symmetry_points[ihs],
                    end - start,
                )

                self.kticks.append(start)
            self.kticks.append(self.nkpoints - 1)
            if not self.kdirect:
                self.kpoints = np.dot(self.kpoints, self.reclat)

            rf = open(self.file_names[0], "r")
            lines = rf.readlines()
            rf.close()

            iline = 0
            for iband in range(self.nbands):
                for ikpoint in range(self.nkpoints):
                    self.bands[ikpoint, iband] = float(lines[iline].split()[1])
                    iline += 1
                if ikpoint == self.nkpoints - 1:
                    iline += 1
            self.bands *= 27.21138386

            self.norbital = 16
            self.spd = np.zeros(
                shape=(
                    self.nkpoints,
                    self.nbands,
                    self.nspin,
                    self.natom + 1,
                    self.norbital + 2,
                )
            )
            idx_bands_out = None
            for ifile in range(len(self.file_names)):
                if self.file_names[ifile] == "BANDS.OUT":
                    idx_bands_out = ifile
            if idx_bands_out != None:
                del self.file_names[idx_bands_out]

            for ifile in range(self.natom):
                rf = open(self.file_names[ifile], "r")
                lines = rf.readlines()
                rf.close()
                iline = 0

                for iband in range(self.nbands):
                    for ikpoint in range(self.nkpoints):
                        temp = np.array([float(x) for x in lines[iline].split()])
                        self.spd[ikpoint, iband, 0, ifile, 0] = ifile + 1
                        self.spd[ikpoint, iband, 0, ifile, 1:-1] = temp[2:]
                        iline += 1
                    if ikpoint == self.nkpoints - 1:
                        iline += 1
            # self.spd[:,:,:,-1,:] = self.spd.sum(axis=3)
            self.spd[:, :, :, :, -1] = np.sum(self.spd[:, :, :, :, 1:-1], axis=4)
            self.spd[:, :, :, -1, :] = self.spd.sum(axis=3)
            self.spd[:, :, 0, -1, 0] = 0

            if self.nspin == 2:
                # spin up block for spin = 1
                self.spd[:, : self.nbands // 2, 1, :, :] = self.spd[
                    :, : self.nbands // 2, 0, :, :
                ]
                # spin down block for spin = 1
                self.spd[:, self.nbands // 2 :, 1, :, :] = (
                    -1 * self.spd[:, self.nbands // 2 :, 0, :, :]
                )

                # manipulating spd array for spin polarized calculations.
                # The shape is (nkpoints,2*nbands,2,natoms,norbitals)
                # The third dimension is for spin.
                # When this is zero, the bands*2 (all spin up and down bands) have positive projections.
                # When this is one, the the first half of bands (spin up) will have positive projections
                # and the second half (spin down) will have negative projections. This is to adhere to
                # the convention used in PyProcar to obtain spin density and spin magnetization.

                # spin up and spin down block for spin = 0
                # spd2[:, :, 0, :, :] = self.spd[:, :, 0, :, :]

    @property
    def fermi(self):
        """
        Returns the fermi energy read from FERMI.OUT
        """
        rf = open("EFERMI.OUT", "r")
        fermi = float(rf.readline().split()[0])
        rf.close()
        return fermi

    @property
    def reclat(self):
        """
        Returns the reciprocal lattice read from LATTICE.OUT
        """
        rf = open("LATTICE.OUT", "r")
        data = rf.read()
        rf.close()
        lattice_block = findall(r"matrix\s*:([-+\s0-9.]*)Inverse", data)
        lattice_array = np.array(lattice_block[1].split(), dtype=float)
        reclat = lattice_array.reshape((3, 3))
        return reclat
