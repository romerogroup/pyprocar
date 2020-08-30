# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 03:12:50 2020
@author: Logan Lang
"""


from re import findall, search, match, DOTALL, MULTILINE, finditer, compile, sub

from numpy import array, dot, linspace, sum, where, zeros, pi, concatenate
import logging


class LobsterParser:
    def __init__(
        self,
        lobsterin="lobsterin",
        lobsterout="lobsterout",
        scfin="scf.in",
        outfile="scf.out",
        kdirect=True,
        lobstercode="qe",
    ):

        self.lobsterin = lobsterin
        self.scfin = scfin
        self.outfile = outfile
        self.lobsterout = lobsterout
        self.dosin = "DOSCAR.lobster"

        self.file_names = []

        self.discontinuities = []
        self.kdirect = kdirect
        ############################
        self.high_symmetry_points = None
        self.nhigh_sym = None
        self.nspecies = None

        self.composition = {}

        self.kticks = None
        self.knames = None

        self.speciesList = None

        self.kpoints = None
        self.kpointsCount = None
        self.bands = None
        self.bandsCount = None
        self.ionsList = None
        self.ionsCount = None
        self.spinCount = None

        # Used to store atomic states at the top of the kpdos.out file
        self.states = None

        

        self.spd = None
        """ for l=1:
              1 pz     (m=0)
              2 px     (real combination of m=+/-1 with cosine)
              3 py     (real combination of m=+/-1 with sine)
            for l=2:
              1 dz2    (m=0)
              2 dzx    (real combination of m=+/-1 with cosine)
              3 dzy    (real combination of m=+/-1 with sine)
              4 dx2-y2 (real combination of m=+/-2 with cosine)
              5 dxy    (real combination of m=+/-2 with sine)"""
        # Oribital order. This is the way it goes into the spd Array. index 0 is resered for totals
        self.orbitals = [
            "s",
            "p_y",
            "p_z",
            "p_x",
            "d_xy",
            "d_yz",
            "d_z^2",
            "d_x^2-y^2",
            "d_xz",
        ]

        # These are not used
        self.orbitalName_short = ["s", "p", "d", "tot"]
        self.orbitalCount = None

        # Opens the needed files
        rf = open(self.lobsterin, "r")
        self.lobsterin = rf.read()
        rf.close()

        rf = open(self.lobsterout, "r")
        self.lobsterout = rf.read()
        rf.close()

        rf = open(self.scfin, "r")
        self.scfin = rf.read()
        rf.close()
        self.dos = None

        self._readFileNames()
        self._readProjFiles()
        self._readHighKpoint()

        return

    ##########################################################################################
    # Finding high symmetry points
    ##########################################################################################

    def _readHighKpoint(self):
        numK = int(findall("K_POINTS.*\n([0-9]*)", self.scfin)[0])

        raw_khigh_sym = findall(
            "K_POINTS.*\n\s*[0-9]*.*\n" + numK * "(.*)\n", self.scfin,
        )[0]

        tickCountIndex = 0
        raw_high_symmetry = []
        self.knames = []
        self.kticks = []

        for x in raw_khigh_sym:
            if len(x.split()) == 5:

                raw_high_symmetry.append(
                    (float(x.split()[0]), float(x.split()[1]), float(x.split()[2]))
                )
                self.knames.append("$%s$" % x.split()[4].replace("!", ""))
                self.kticks.append(tickCountIndex)
            if float(x.split()[3]) == 0:
                tickCountIndex += 1
        self.high_symmetry_points = array(raw_high_symmetry)
        self.nhigh_sym = len(self.knames)



        # finds discontinuities 
        for i in range(len(self.kticks)):
            if(i < len(self.kticks)-1):
                diff = self.kticks[ i+1 ] - self.kticks[i]
                if diff == 1 :
                    
                    self.discontinuities.append(self.kticks[i])

                    
                    discon_name = "$" + self.knames[i].replace("$","") +"|"+ self.knames[i+1].replace("$","") + "$"
                    self.knames.pop(i+1)
                    self.knames[i] = discon_name
                    self.kticks.pop(i+1)


    ##########################################################################################
    # Finding file names
    ##########################################################################################

    def _readFileNames(self):

        self.ionsList = findall(
            "calculating FatBand for Element: (.*) Orbital.*", self.lobsterout
        )
        raw_speciesList = findall(
            "calculating FatBand for Element: ([a-zA-Z]*)[0-9] Orbital.*",
            self.lobsterout,
        )
        self.speciesList = []
        for x in raw_speciesList:
            if x not in self.speciesList:
                self.speciesList.append(x)
                self.composition.update({str(x): 0})
        self.nspecies = len(self.speciesList)
        for x in raw_speciesList:
            if x in self.speciesList:
                self.composition[x] += 1

        self.ionsCount = len(self.ionsList)

        proj_orbitals = findall(
            "calculating FatBand for Element: (.*) Orbital\(s\):\s*(.*)",
            self.lobsterout,
        )
        for proj in proj_orbitals:
            orbitals = proj[1].split()
            for orbital in orbitals:
                fileName = "FATBAND_" + proj[0] + "_" + orbital + ".lobster"
                self.file_names.append(fileName)

    ##########################################################################################
    # Reading projection files
    ##########################################################################################

    def _readProjFiles(self):
        rf = open(self.file_names[0], "r")
        projFile = rf.read()
        rf.close()

        ##########################################################################################
        # kpoints
        ##########################################################################################

        raw_kpoints = findall(
            "# K-Point \d+ :\s*([-\.\\d]*)\s*([-\.\\d]*)\s*([-\.\\d]*)", projFile
        )

        self.kpointsCount = len(raw_kpoints)
        self.kpoints = zeros(shape=(self.kpointsCount, 3))
        for ik in range(len(raw_kpoints)):
            for coord in range(3):
                self.kpoints[ik][coord] = raw_kpoints[ik][coord]
        # raw_kpoints = findall("(# K-Point \d+ :\s*[-\.\\d]*\s*[-\.\\d]*\s*[-\.\\d]*)", projFile)

        # If kdirect=False, then the kgrid will be in cartesian coordinates.
        # Requires the reciprocal lattice vectors to be parsed from the output.
        if not self.kdirect:
            self.kpoints = dot(self.kpoints, self.reclat)

        #########################################################################################
        # bands
        #########################################################################################

        raw_kpoints = findall(
            "# K-Point \d+ :\s*([-\.\\d]*\s*[-\.\\d]*\s*[-\.\\d]*)", projFile
        )

        # Checks for spin polarization
        if len(findall("spillings for spin channel", self.lobsterout)) == 2:
            self.spinCount = 2
            bandsCount = int(findall("NBANDS\s*(\d*)", projFile)[0]) * self.spinCount
            self.bandsCount = bandsCount // 2
        else:
            self.spinCount = 1
            bandsCount = int(findall("NBANDS\s*(\d*)", projFile)[0])
            self.bandsCount = bandsCount

        band_info = []
        for ik in range(len(raw_kpoints)):
            expression = "# K-Point \d+ :\s*" + raw_kpoints[ik] + ".*\n"
            kpoint_bands = findall(expression + bandsCount * "(.*)\n", projFile)[0]
            for ikband in kpoint_bands:
                band_info.append(ikband)

        #        if len(band_info) == self.bandsCount * self.kpointsCount:
        #            print("Number of bands headers match")
        raw_bands = zeros(shape=(self.kpointsCount, bandsCount))
        
        ik = 0
        ib = 0
        for i in range(len(band_info)):
            raw_bands[ik, ib] = float(band_info[i].split()[1])
            ib += 1
            if int(ib == bandsCount):
                ik += 1
                ib = 0
        # Checks for spin polarization
        if self.spinCount == 2:

            self.bands = zeros(
                shape=(self.kpointsCount, self.bandsCount, self.spinCount)
            )
            self.bands[:, 0 : self.bandsCount, 0] = raw_bands[:, 0 : self.bandsCount]
            self.bands[:, 0 : self.bandsCount, 1] = raw_bands[
                :, self.bandsCount : self.bandsCount * 2
            ]
        else:
            
            self.bands = zeros(
                shape=(self.kpointsCount, self.bandsCount, self.spinCount)
            )
            self.bands[:, 0 : self.bandsCount, 0] = raw_bands[:, 0 : self.bandsCount]
        #########################################################################################
        # Forming SPD array
        #########################################################################################

        # Checks for spin polarization
        if len(findall("spillings for spin channel", self.lobsterout)) == 2:
            self.spinCount = 2
        else:
            self.spinCount = 1
        
        self.orbitalCount = 10
        self.spd = zeros(
            shape=(
                self.kpointsCount,
                self.bandsCount,
                self.spinCount,
                self.ionsCount + 1,
                len(self.orbitals) + 2,
            )
        )

        for file in range(len(self.file_names)):
            rf = open(self.file_names[file], "r")
            projFile = rf.read()
            rf.close()

            raw_kpoints = findall(
                "# K-Point \d+ :\s*([-\.\\d]*\s*[-\.\\d]*\s*[-\.\\d]*)", projFile
            )
            ionIndex = 0
            orbitalIndex = 0
            current_orbital = findall("(# FATBAND.*)", projFile)[0].split()[4]
            for i in range(len(self.ionsList)):
                if self.ionsList[i] == self.file_names[file].split("_")[1]:
                    ionIndex = i
                    # print(ionIndex)
            for i in range(len(self.orbitals)):
                if self.orbitals[i] == sub("[0-9]", "", current_orbital):
                    orbitalIndex = i + 1
                    # print(orbitalIndex)
                    # print(self.orbitals[orbitalIndex])

            band_info = []
            for ik in range(len(raw_kpoints)):
                expression = "# K-Point \d+ :\s*" + raw_kpoints[ik] + ".*\n"
                bands_wSpin = self.bandsCount * self.spinCount
                band = findall(expression + bands_wSpin * "(.*)\n", projFile)[0]
                for iband in range(self.bandsCount):
                    if self.spinCount == 2:
                        self.spd[ik, iband, 0, ionIndex, orbitalIndex] += float(
                            band[iband].split()[2]
                        )
                        self.spd[ik, iband, 0, ionIndex, 0] = ionIndex + 1
                        self.spd[ik, iband, 1, ionIndex, orbitalIndex] += float(
                            band[iband + self.bandsCount].split()[2]
                        )
                        self.spd[ik, iband, 1, ionIndex, 0] = ionIndex + 1
                    else:
                        self.spd[ik, iband, 0, ionIndex, orbitalIndex] += float(
                            band[iband].split()[2]
                        )
                        self.spd[ik, iband, 0, ionIndex, 0] = ionIndex + 1

            self.spd[:, :, :, :, -1] = sum(self.spd[:, :, :, :, 1:-1], axis=4)
            self.spd[:, :, :, -1, :] = sum(self.spd[:, :, :, 0:-1, :], axis=3)
            self.spd[:, :, :, -1, 0] = 0

        # colinear spin polarized case

        # manipulating spd array for spin polarized calculations.
        # The shape is (nkpoints,2*nbands,2,natoms,norbitals)
        # The third dimension is for spin.
        # When this is zero, the bands*2 (all spin up and down bands) have positive projections.
        # When this is one, the the first half of bands (spin up) will have positive projections
        # and the second half (spin down) will have negative projections. This is to adhere to
        # the convention used in PyProcar to obtain spin density and spin magnetization.

        if self.spinCount == 2:
            print("\nLobster colinear spin calculation detected.\n")
            self.spd2 = zeros(
                shape=(
                    self.kpointsCount,
                    self.bandsCount * 2,
                    self.spinCount,
                    self.ionsCount + 1,
                    len(self.orbitals) + 2,
                )
            )

            # spin up block for spin=0
            self.spd2[:, : self.bandsCount, 0, :, :] = self.spd[:, :, 0, :, :]

            # spin down block for spin=0
            self.spd2[:, self.bandsCount :, 0, :, :] = self.spd[:, :, 1, :, :]

            # spin up block for spin=1
            self.spd2[:, : self.bandsCount, 1, :, :] = self.spd[:, :, 0, :, :]

            # spin down block for spin=1
            self.spd2[:, self.bandsCount :, 1, :, :] = -1 * self.spd[:, :, 1, :, :]

            self.spd = self.spd2

            # Reshaping bands array to inlcude all bands (spin up and down)
            # self.bands = concatenate((self.bands, self.bands), axis=1)

            # Reshaping bands array to inlcude all bands (spin up and down)
            self.bands = self.bands.reshape(
                self.kpointsCount, self.bandsCount * 2, order="F"
            )
        else:
            self.bands = self.bands.reshape(
                self.kpointsCount, self.bandsCount, order="F"
            )
    @property
    def fermi(self):
        """
        Returns the fermi energy read from .out
        """

        fi = open(self.outfile, "r")
        data = fi.read()
        fi.close()
        fermi = float(findall(r"the\s*Fermi\s*energy\s*is\s*([\s\d.]*)ev", data)[0])
        return fermi

    @property
    def reclat(self):
        """
        Returns the reciprocal lattice read from .out
        """
        rf = open(self.outfile, "r")
        data = rf.read()
        rf.close()

        alat = float(findall(r"alat\)\s*=\s*([\d.e+-]*)", data)[0])

        b1 = array(
            findall(r"b\(1\)\s*=\s*\(([\d.\s+-e]*)", data)[0].split(), dtype="float64"
        )
        b2 = array(
            findall(r"b\(2\)\s*=\s*\(([\d.\s+-e]*)", data)[0].split(), dtype="float64"
        )
        b3 = array(
            findall(r"b\(3\)\s*=\s*\(([\d.\s+-e]*)", data)[0].split(), dtype="float64"
        )

        reclat = (2 * pi / alat) * (array((b1, b2, b3)))

        # Transposing to get the correct format
        reclat = reclat.T

        return reclat
