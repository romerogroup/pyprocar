"""
Created on Sunday, May 24th
@author: Logan Lang
"""

from re import findall, search, match, DOTALL, MULTILINE, finditer, compile

from numpy import array, dot, linspace, sum, where, zeros, pi, matmul, round, linalg
import logging


class QEFermiParser:
    def __init__(
        self, nscfin="nscf.in", kpdosin="kpdos.in", outfile="scf.out", kdirect=True
    ):

        # Qe inputs
        self.nscfin = nscfin
        self.kfin = kpdosin
        self.outfile = outfile

        # This is not used since the extra files are useless for the parser
        # self.file_names = []
        # Not used
        self.kdirect = kdirect

        ############################
        self.high_symmetry_points = None
        self.nhigh_sym = None
        self.nspecies = None

        self.composition = {}

        self.kpoints = None
        self.kpointsCount = None
        self.bands = None
        self.bandsCount = None
        self.ionsCount = None

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

        # These are not used
        self.orbitalName_short = ["s", "p", "d", "tot"]
        self.orbitalCount = None

        # Opens the needed files
        rf = open(self.nscfin, "r")
        self.nscfIn = rf.read()
        rf.close()

        rf = open(self.kfin, "r")
        self.kpdosIn = rf.read()
        rf.close()

        self.test = None

        # The only method this parser takes. I could make more methods to increase its modularity
        self._readQEin()

        return

    def _readQEin(self):

        ###############################################

        spinCalc = False
        if len(findall("\s*nspin=(.*)", self.nscfIn)) != 0:
            spinCalc = True

        #######################################################################
        # Finding composition and specie data
        #######################################################################

        self.nspecies = int(findall("ntyp\s*=\s*([0-9]*)", self.nscfIn)[0])
        self.ionsCount = int(findall("nat\s*=\s*([0-9]*)", self.nscfIn)[0])
        raw_species = findall(
            "ATOMIC_SPECIES.*\n" + self.nspecies * "(.*).*\n", self.nscfIn
        )[0]
        species_list = []
        if self.nspecies == 1:
            self.composition[raw_species.split()[0]] = 0
            species_list.append(raw_species.split()[0])
        else:
            for nspec in range(self.nspecies):
                species_list.append(raw_species[nspec].split()[0])
                self.composition[raw_species[nspec].split()[0]] = 0

        raw_ions = findall(
            "ATOMIC_POSITIONS.*\n" + self.ionsCount * "(.*).*\n", self.nscfIn
        )[0]

        if self.ionsCount == 1:
            self.composition[raw_species.split()[0]] = 1
        else:
            for ions in range(self.ionsCount):
                for species in range(len(species_list)):
                    if raw_ions[ions].split()[0] == species_list[species]:
                        self.composition[raw_ions[ions].split()[0]] += 1
                    

        #######################################################################
        # Reading the kpdos.out for outputfile labels
        #######################################################################
        rf = open(self.kfin.split(".")[0] + ".out", "r")
        kpdosout = rf.read()
        rf.close()

        # The following lines get the number of states in kpdos.out and stores them in a list, in which their information is stored in a dictionary
        raw_natomwfc = findall(
            r"(?<=\(read from pseudopotential files\)\:)[\s\S]*?(?=k)", kpdosout
        )[0]

        natomwfc = len(findall("state[\s#]*(\d*)", raw_natomwfc))

        raw_wfc = findall("\(read from pseudopotential files\)\:.*\n\n\s*" + natomwfc * "(.*)\n",
            kpdosout,
        )[0]

        raw_states = []

        states_list = []
        state_dict = {}
 
        # Read in raw states
        for state in raw_wfc:
                #print(state)

                state_index = int(state.split('#')[1].split(':')[0])
                atom_index  = int(state.split('atom')[1].split('(')[0])
                atom_name   = str(state.split('(')[1].split(')')[0])
                wfc_index   = int(state.split('wfc')[1].split('(')[0])
                l_index     = int(state.split('wfc')[1].split('l=')[1].split('m=')[0])
                m_index     = int(state.split('wfc')[1].split('m=')[1].split(')')[0])

                append = [state_index, atom_index, atom_name, wfc_index, l_index, m_index]
                #append = (findall("state #\s*([0-9]+):\s*atom\s*([0-9]+)\s*\((.*)\s\),\s*wfc\s*([0-9]+)\s\(l=\s*([0-9]+)\s*m=\s*([0-9]+)\)", state)[0])
                #print(append)
                raw_states.append(append)
                #int(state.split('#')[1].split(':')[0])
                #findall("state #\s*([0-9]*):\s*atom\s*([0-9]*)\s*\((.*)\s\),\s*wfc\s*([0-9]*)\s\(l=\s*([0-9])\s*m=\s*([0-9])\)", state)[0])
            #)
     
        #print(raw_states)
        for state in raw_states:
            state_dict = {}
            state_dict = {
                "state_num": int(state[0]),
                "species_num": int(state[1]),
                "specie": state[2],
                "atm_wfc": int(state[3]),
                "l": int(state[4]),
                "m": int(state[5]),
            }
            states_list.append(state_dict)

        self.states = states_list

        """
        This section was used to parse those extra files we discussed. It is not needed, I kept it just incase
        output_prefix = findall("filpdos\s*=\s*'(.*)'", self.kpdosIn)[0]
         kpdos_files=[]
        atmNum = list(range(1, self.ionsCount+1))
        wfc_file_label = []
        atm_file_label = []
        states_dict ={}
        #Find unique raw file labels
        for state1 in raw_states:
            wfc_file_label.append((state1[3],state1[4])) if (state1[3],state1[4]) not in wfc_file_label else None
            atm_file_label.append((state1[1],state1[2])) if (state1[1],state1[2]) not in atm_file_label else None
        #Combine raw file labels inside a tuple
        for atm in atm_file_label:
            for wfc in wfc_file_label:
                file = (int(atm[0]),atm[1],int(wfc[0]),int(wfc[1]))
                kpdos_files.append(file)
                file = None
        #Convert raw file labels to final file string
        for file in kpdos_files:
            if(file[3] == 0 ):
                self.file_names.append(output_prefix + ".pdos_atm#" + str(file[0]) + "(" + file[1] + ")_wfc#" + str(file[2]) +  "(s)")
            elif (file[3] == 1 ):
                self.file_names.append(output_prefix + ".pdos_atm#" + str(file[0]) + "(" + file[1] + ")_wfc#" + str(file[2]) +  "(p)")
            elif (file[3] == 2 ):
                self.file_names.append(output_prefix + ".pdos_atm#" + str(file[0]) + "(" + file[1] + ")_wfc#" + str(file[2]) +  "(d)")
            elif (file[3] == 3 ):
                self.file_names.append(output_prefix + ".pdos_atm#" + str(file[0]) + "(" + file[1] + ")_wfc#" + str(file[2]) +  "(f)")"""

        #######################################################################
        # Finds kpoints in kpdosout file. They are the k points in the reciprocal space
        #######################################################################

        raw_kpoints = []
        if spinCalc == True:

            self.kpointsCount = int(
                len(findall("k\s*=\s*(.*)\s*(.*)\s*(.*)", kpdosout)) / 2
            )
            for k in range(len(findall("k\s*=\s*(.*)\s*(.*)\s*(.*)", kpdosout))):
                if k < self.kpointsCount:
                    raw_kpoints.append(
                        findall("k\s*=\s*(.*)\s*(.*)\s*(.*)", kpdosout)[k][0]
                    )
            totK = len(findall("k\s*=\s*(.*)\s*(.*)\s*(.*)", kpdosout))
        else:
            for k in range(len(findall("k\s*=\s*(.*)\s*(.*)\s*(.*)", kpdosout))):
                raw_kpoints.append(
                    findall("k\s*=\s*(.*)\s*(.*)\s*(.*)", kpdosout)[k][0]
                )
            self.kpointsCount = len(raw_kpoints)

        self.kpoints = zeros(shape=(self.kpointsCount, 3))
        for ik in range(len(raw_kpoints)):
            for coord in range(3):
                self.kpoints[ik][coord] = raw_kpoints[ik].split()[coord]

        # If kdirect=False, then the kgrid will be in cartesian coordinates.
        # Requires the reciprocal lattice vectors to be parsed from the output.
        if not self.kdirect:
            self.kpoints = dot(self.kpoints, self.reclat)
        self.kpoints = matmul(self.kpoints, linalg.inv(self.reclat))
        self.kpoints  = round( self.kpoints , 5)
        for ik in range(len(self.kpoints[:,0])):
            for ix in range(len(self.kpoints[0,:])):
                if self.kpoints[ik,ix] < 0 :
                    self.kpoints[ik,ix] += 1
        #######################################################################
        # Finds the band count and makes a band array
        #######################################################################

        band_info = findall(r"====\se\(\s*(\d+)\)\s=\s*([-.\d]+)", kpdosout)
        
        if spinCalc == True:
            self.bandsCount = int(len(band_info) / totK)
            self.bands = zeros(shape=(self.kpointsCount, self.bandsCount, 2))
            ik = 0
            for band in range(len(band_info)):
                if ik < self.kpointsCount:
                    self.bands[ik, int(band_info[band][0]) - 1, 0] = float(
                        band_info[band][1]
                    )
                else:
                    self.bands[
                        ik - self.kpointsCount, int(band_info[band][0]) - 1, 1
                    ] = float(band_info[band][1])
                if int(band_info[band][0]) == self.bandsCount:
                    ik += 1
        else:
            self.bandsCount = int(len(band_info) / self.kpointsCount)
            self.bands = zeros(shape=(self.kpointsCount, self.bandsCount))
            ik = 0
            for band in range(len(band_info)):
                self.bands[ik, int(band_info[band][0]) - 1] = float(band_info[band][1])
                if int(band_info[band][0]) == self.bandsCount:
                    ik += 1

        #######################################################################
        # Filling the spd array
        #######################################################################
        if spinCalc == True:
            spinCount = 2
        else:
            spinCount = 1
        k_info = None
        self.orbitalCount = len(self.orbitals)
        self.spd = zeros(
            shape=(
                self.kpointsCount,
                self.bandsCount,
                spinCount,
                self.ionsCount + 1,
                len(self.orbitals) + 2,
            )
        )

        k_string = findall(r"\s?k =[\s-]{3}?.+", kpdosout)

        """First loop goes through the kpoints and matches band information that follows the specific k point. The if else statment is used to catch unique cases, namely the end of the kpoints
            Also sometimes there is a duplicate final k point so findall is used twice to deal with this case
            Second loop goes through band information of a kpoint.The if else statments here again catches unique cases. This is when it hits the last the band or when it is the last band and the last kpoint
            The other if else statement here catches the cases when there are no contributions at all and would return nothing
            Third loop parses the projections of a band
            The fourth loop then goes through the known possible states. The if statment ensures the there is a projection and matchs a projection to a known state.
            The fifth loop then goes throguh the known possible orbitals. The if statment then matches the projection with a specific orbital.
            Finally the projection is put into the spd array
        """

        for kp in range(len(k_string)):
            if kp == len(k_string) - 1:
                expression = "(?<=" + k_string[kp] + ")[\s\S]*?(?=Lowdin Charges:)"
                
                expression_final_k = "(?<=" + k_string[kp - 1] + ")[\s\S]*?(?=$)"
                
                
                k_info1 = findall(expression_final_k, kpdosout)[0]

                k_info = findall(expression, k_info1)[0]
            else:
                expression = (
                    "(?<=" + k_string[kp] + ")[\s\S]*?(?=" + k_string[kp + 1] + ")"
                )
                # print(expression)
                k_info = findall(expression, kpdosout)[0]
                # print(k_info)
                # print(k_info)
                # print(self.bandsCount)
            for iband in range(self.bandsCount):
                if iband == self.bandsCount - 1 and kp == len(k_string) - 1:
                    # print("hi")
                    expression2 = "==== e\(\s*" + str(iband + 1) + "\)[\s\S]*?(?=$)"
                    # print(k_info)
                    iband_proj = findall(expression2, k_info)[0]
                    # print(iband_proj)
                elif iband == self.bandsCount - 1:
                    expression2 = "==== e\(\s*" + str(iband + 1) + "\).*\n([\s\S]*)"
                    iband_proj = findall(expression2, k_info)[0]

                else:
                    # print("hi")
                    expression2 = (
                        "==== e\(\s*"
                        + str(iband + 1)
                        + "\)[\s\S]*?(?===== e\(\s*"
                        + str(iband + 2)
                        + "\))"
                    )
                    iband_proj = findall(expression2, k_info)[0]
                    # print(iband_proj)
                expression3 = "(?<=psi = )[\s\S]*?(?=\s*\|psi\|\^2)"
                wfc = None

                if len(findall(expression3, iband_proj)) == 0:
                    pass
                else:
                    wfc = findall(expression3, iband_proj)[0]
                    # print(wfc)
                    state_proj = wfc.split("+")

                    for proj in state_proj:
                        for known_states in self.states:
                            if (
                                len(findall("#\s*([0-9]*)", proj)) != 0
                                and int(findall("#\s*([0-9]*)", proj)[0])
                                == known_states["state_num"]
                            ):
                                for iorbitals in range(len(self.orbitals)):
                                    if (
                                        known_states["l"]
                                        == self.orbitals[iorbitals]["l"]
                                        and known_states["m"]
                                        == self.orbitals[iorbitals]["m"]
                                    ):
                                        if spinCalc == True:
                                            if kp < self.kpointsCount:
                                                # print(known_states["species_num"])
                                                self.spd[
                                                    kp,
                                                    iband,
                                                    0,
                                                    known_states["species_num"] - 1,
                                                    iorbitals + 1,
                                                ] += float(proj.split("*")[0])
                                                self.spd[
                                                    kp,
                                                    iband,
                                                    0,
                                                    known_states["species_num"] - 1,
                                                    0,
                                                ] = known_states["species_num"]
                                            else:
                                                self.spd[
                                                    kp - self.kpointsCount,
                                                    iband,
                                                    1,
                                                    known_states["species_num"] - 1,
                                                    iorbitals + 1,
                                                ] += float(proj.split("*")[0])
                                                self.spd[
                                                    kp - self.kpointsCount,
                                                    iband,
                                                    1,
                                                    known_states["species_num"] - 1,
                                                    0,
                                                ] = known_states["species_num"]

                                        else:
                                            self.spd[
                                                kp,
                                                iband,
                                                0,
                                                known_states["species_num"] - 1,
                                                iorbitals + 1,
                                            ] += float(proj.split("*")[0])
                                            self.spd[
                                                kp,
                                                iband,
                                                0,
                                                known_states["species_num"] - 1,
                                                0,
                                            ] = known_states["species_num"]

        for ions in range(self.ionsCount):
            self.spd[:, :, :, ions, 0] = ions + 1

        # The following fills the totals for the spd array

        self.spd[:, :, :, :, -1] = sum(self.spd[:, :, :, :, 1:-1], axis=4)
        self.spd[:, :, :, -1, :] = self.spd.sum(axis=3)
        self.spd[:, :, :, -1, 0] = 0

        # colinear spin polarized case

        # manipulating spd array for spin polarized calculations.
        # The shape is (nkpoints,2*nbands,2,natoms,norbitals)
        # The third dimension is for spin.
        # When this is zero, the bands*2 (all spin up and down bands) have positive projections.
        # When this is one, the the first half of bands (spin up) will have positive projections
        # and the second half (spin down) will have negative projections. This is to adhere to
        # the convention used in PyProcar to obtain spin density and spin magnetization.

        if spinCalc:
            print("\nQuantum Espresso colinear spin calculation detected.\n")
            self.spd2 = zeros(
                shape=(
                    self.kpointsCount,
                    self.bandsCount * 2,
                    spinCount,
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
            self.bands = self.bands.reshape(
                self.kpointsCount, self.bandsCount * 2, order="F"
            )

    @property
    def fermi(self):
        """
        Returns the fermi energy read from .out
        """

        fi = open(self.outfile, "r")
        data = fi.read()
        fi.close()

        data = data.split('the Fermi energy is')[1].split('ev')[0]
        fermi = float(data)

        #print((findall(r"the\s*Fermi\s*energy\s*is\s*([\s\d.]*)ev", data)))
        #fermi = float(findall(r"the\s*Fermi\s*energy\s*is\s*([\s\d.]*)ev", data)[0])
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

        # reclat = (2 * pi / alat) * (array((b1, b2, b3)))
        reclat = array((b1, b2, b3))
        # Transposing to get the correct format
        reclat = reclat

        return reclat
