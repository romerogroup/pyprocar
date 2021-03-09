# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 05:22:04 2020

@author: lllan
"""

"""
Created on Sunday, May 24th
@author: Logan Lang
"""

from re import findall

from numpy import array, dot, zeros, add, delete, append, arange, pi
from pyprocar.core import DensityOfStates, Structure

import logging
import os


class QEDOSParser:
    def __init__(
        self, nscfin="nscf.in", pdosin="pdos.in", scfOut="scf.out", dos_interpolation_factor = None 
    ):

        # Qe inputs
        self.nscfin = nscfin
        self.pdosin = pdosin
        self.scfOut = scfOut
   
        # This is not used since the extra files are useless for the parser
        self.file_names = []
        self.filpdos = None
        # Not used

        ############################

        self.nspecies = None

        self.composition = {}

        self.dos_interpolation_factor = dos_interpolation_factor
        self.bands = None
        self.bandsCount = None
        self.ionsCount = None

        # Used to store atomic states at the top of the kpdos.out file
        self.states = None
        self.test = None
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
 
        rf = open(self.scfOut, "r")
        self.scfOut = rf.read()
        rf.close()
        
        rf = open(self.pdosin, "r")
        self.pdosIn = rf.read()
        rf.close()
        
        
        self.test =None
        
        self.spinCalc = False
        if len(findall("\s*nspin=(.*)",self.nscfIn)) != 0:
            self.spinCalc =  True
            self.is_spin_polarized = True
        else:
           self.is_spin_polarized = False 
        # The only method this parser takes. I could make more methods to increase its modularity
        self._readQEin()
        self.data = self.read()

        
        
        
        return
    def read(self):
        """
        Read and parse vasprun.xml.
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        return self.parse_pdos()

    def _get_dos_total(self):
        
        energies = self.data['total'][:, 0]
        dos_total = {'energies': energies}
        
        if self.is_spin_polarized:
            dos_total['Spin-up'] = self.data['total'][:, 1]
            dos_total['Spin-down'] = self.data['total'][:, 2]
            #dos_total['integrated_dos_up'] = self.data['total'][:, 3]
            #dos_total['integrated_dos_down'] = self.data['total'][:, 4]
        else:
            dos_total['Spin-up'] = self.data['total'][:, 1]
            #dos_total['integrated_dos'] = self.data['total'][:, 2]

        return dos_total,list(dos_total.keys())


    def _get_dos_projected(self, atoms=[]):

        if len(atoms) == 0:
            atoms = arange(self.initial_structure.natoms)
      
 
        if 'projected' in list(self.data.keys()):
            
            dos_projected = {}
            ion_list = ["ion %s" % str(x + 1) for x in atoms
                        ]  # using this name as vasrun.xml uses ion #
            for i in range(len(ion_list)):
                iatom = ion_list[i]
                #name = self.initial_structure.atoms[atoms[i]]
                name = self.initial_structure.atoms[atoms[i]] + str(atoms[i])
             
                energies = self.data['projected'][i][:,0]
                
                dos_projected[name] = {'energies': energies}
                if self.is_spin_polarized:
                    dos_projected[name]['Spin-up'] = self.data['projected'][i][:, 1:,0]
                    dos_projected[name]['Spin-down'] = self.data['projected'][i][:, 1:,1]
                else:
                    dos_projected[name]['Spin-up'] = self.data['projected'][i][:, 1:,0]
            
            return dos_projected, self.data['projected_labels_info']
        else:
            print(
                "This calculation does not include partial density of states")
            return None, None
    
    @property
    def dos(self):
        energies = self.dos_total['energies'] - self.fermi
        total = []
        for ispin in self.dos_total:
            if ispin == 'energies':
                continue
            total.append(self.dos_total[ispin])
        # total = np.array(total).T
        return DensityOfStates(
            energies=energies,
            total=total,
            projected=self.dos_projected,
            interpolation_factor=self.dos_interpolation_factor)
  
    @property
    def dos_to_dict(self):
        """
        Returns the complete density (total,projected) of states as a python dictionary
        """
        return {
            'total': self._get_dos_total(),
            'projected': self._get_dos_projected()
        }
    
    @property
    def dos_total(self):
        """
        Returns the total density of states as a pychemia.visual.DensityOfSates object
        """
       
        dos_total, labels = self._get_dos_total()
        # dos_total['energies'] -= self.fermi

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
                    
                    if key == 'energies':
                        continue
                    temp_spin.append(dos_projected[iatom][key][:, iorbital])
                temp_atom.append(temp_spin)
            ret.append([temp_atom])
        return ret   
    
    
#     ###########################################################################
#     # This section parses for the projected density of states and puts it in a 
#     # Pychemia Density of States Object
#     ###########################################################################

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
        # symbols = [x.strip() for x in self.data['ions']]
        symbols = [x.strip() for x in self.ions]
        structures = []

        st = Structure(atoms=symbols)  #lattice = self.lattice)#, #fractional_coordinates = )
                      
        structures.append(st)
        return structures

    @property
    def structure(self):
        """
        crystal structure of the last step
        """
        return self.structures[-1]
    
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
    def fermi(self):
        """
        Returns the fermi energy read from .out
        """

        # fi = open(self.scfOut, "r")
        # data = fi.read()
        # fi.close()

        data = self.scfOut.split('the Fermi energy is')[1].split('ev')[0]
        fermi = float(data)

        #print((findall(r"the\s*Fermi\s*energy\s*is\s*([\s\d.]*)ev", data)))
        #fermi = float(findall(r"the\s*Fermi\s*energy\s*is\s*([\s\d.]*)ev", data)[0])
        
        return fermi
    
#     ###########################################################################
#     ###########################################################################
#     ###########################################################################
    
    def dos_parametric(self,atoms=None,orbitals=None,spin=None,title=None):
        """
        This function sums over the list of atoms and orbitals given 
        for example dos_paramateric(atoms=[0,1,2],orbitals=[1,2,3],spin=[0,1])
        will sum all the projections of atoms 0,1,2 and all the orbitals of 1,2,3 (px,py,pz)
        and return separatly for the 2 spins as a DensityOfStates object from pychemia.visual.DensityofStates
        
        :param atoms: list of atom index needed to be sumed over. count from zero with the same 
                      order as POSCAR
        
        :param orbitals: list of orbitals needed to be sumed over 
        |  s  ||  py ||  pz ||  px || dxy || dyz || dz2 || dxz ||x2-y2||
        |  0  ||  1  ||  2  ||  3  ||  4  ||  5  ||  6  ||  7  ||  8  ||
        
        :param spin: which spins to be included. count from 0
                      There are no sum over spins
        
        """
        projected = self.dos_projected
        dos_projected,labelsInfo = self._get_dos_projected()
        self.availiableOrbitals = list(labelsInfo.keys())
        self.availiableOrbitals.pop(0)
        if atoms == None :
            atoms = arange(self.nions,dtype=int)
        if spin == None :
            spin = [0,1]
        if orbitals == None :
            orbitals = arange((len(projected[0].labels)-1)//2,dtype=int)
        if title == None:
            title = 'Sum'
        orbitals = array(orbitals)
        
        
        if len(spin) == 2:
            labels = ['Energy','Spin-Up','Spin-Down']
            new_orbitals = []
            for ispin in spin :
                new_orbitals.append(list(orbitals+ispin*(len(projected[0].labels)-1)//2))
                
            orbitals = new_orbitals
            
        else : 
            
            for x in orbitals:
                
                if (x+1 > (len(projected[0].labels)-1)//2 ):
                    print('listed wrong amount of orbitals')
                    print('Only use one or more of the following ' + str(arange((len(projected[0].labels)-1)//2,dtype=int)))
                    print('Only use one or more of the following ' + str(arange((len(projected[0].labels)-1)//2,dtype=int)))
                    print('They correspond to the following orbitals : ' + str(self.availiableOrbitals) )
                    print('Again do not trust the plot that was just produced' )
            if spin[0] == 0:
                labels = ['Energy','Spin-Up']
            elif spin[0] == 1:
                labels = ['Energy','Spin-Down']
            
        
        
        ret = zeros(shape=(len(projected[0].energies),len(spin)+1))
        ret[:,0] = projected[0].energies
        
        for iatom in atoms :
            if len(spin) == 2 :
                ret[:,1:]+=self.dos_projected[iatom].values[:,orbitals].sum(axis=2)
            elif len(spin) == 1 :
                ret[:,1]+=self.dos_projected[iatom].values[:,orbitals].sum(axis=1)
                
        return DensityOfStates(table=ret,title=title,labels=labels)

    def _readQEin(self):

        ###############################################

        spinCalc = False
        if len(findall("\s*nspin=(.*)",self.nscfIn)) != 0:
            spinCalc =  True


        
        #######################################################################
        # Finding composition and specie data
        #######################################################################

        self.nspecies = int(findall("ntyp\s*=\s*([0-9]*)", self.nscfIn)[0])
        self.ionsCount = int(findall("nat\s*=\s*([0-9]*)", self.nscfIn)[0])
        raw_species = findall(
            "ATOMIC_SPECIES.*\n" + self.nspecies * "(.*).*\n", self.nscfIn
        )[0]
        self.species_list = []
        if self.nspecies == 1:
            self.composition[raw_species.split()[0]] = 0
            self.species_list.append(raw_species.split()[0])
        else:
            for nspec in range(self.nspecies):
                self.species_list.append(raw_species[nspec].split()[0])
                self.composition[raw_species[nspec].split()[0]] = 0

        # raw_ions = findall(
        #     "ATOMIC_POSITIONS.*\n" + self.ionsCount * "(.*).*\n", self.nscfIn
        # )[0]
        
        raw_ions = findall( "\s*Cartesian\saxes.*\n.*\n.*\n" + self.ionsCount * "(.*)\n", self.scfOut)[0]
        self.ions = [x.split()[1] for x in raw_ions]
        
        
        
        if self.ionsCount == 1:
            self.composition[raw_species.split()[0]] = 1
        else:
            for ions in range(self.ionsCount):
                for species in range(len(self.species_list)):
                    # if  raw_ions[ions].split()[0] == self.species_list[species]:
                    #     self.composition[raw_ions[ions].split()[0]] += 1
                    if  self.ions[ions] == self.species_list[species]:
                        self.composition[self.ions[ions].split()[0]] += 1

        

        #######################################################################
        # Reading the kpdos.out for outputfile labels
        #######################################################################
        rf = open(self.pdosin.split(".")[0] + ".out", "r")
        pdosout = rf.read()
        rf.close()

        # The following lines get the number of states in kpdos.out and stores them in a list, in which their information is stored in a dictionary
        raw_natomwfc = findall(
            r"(?<=\(read from pseudopotential files\)\:)[\s\S]*?(?=k)", pdosout
        )[0]

        natomwfc = len(findall("state[\s#]*(\d*)", raw_natomwfc))

        raw_wfc = findall(
            "\(read from pseudopotential files\)\:.*\n\n\s*" + natomwfc * "(.*)\n",
            pdosout,
        )[0]

        raw_states = []
        pdos_files=[]
        states_list = []
        state_dict = {}

        # Read in raw states
        for state in raw_wfc:
            raw_states.append(
                findall(
                    "state #\s*([0-9]*):\s*atom\s*([0-9]*)\s*\((.*)\s\),\s*wfc\s*([0-9]*)\s\(l=\s*([0-9])\s*m=\s*([0-9])\)",
                    state,
                )[0]
            )
        
        self.species_list = []
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
            
            if state[2] not in self.species_list:
                self.species_list.append(state[2])
        
        
        
        self.states = states_list

        
        #  This section was used to parse those extra files we discussed. It is not needed, I kept it just incase
        output_prefix = findall("filpdos\s*=\s*'(.*)'", self.pdosIn)[0]
        self.filpdos = output_prefix
        #Find unique raw file labels
        for state in self.states:
            file = (state['species_num'],state['specie'].strip(),state['atm_wfc'],state['l'])
            pdos_files.append(file) if file not in pdos_files else None

        #Convert raw file labels to final file string
        for file in pdos_files:
            if(file[3] == 0 ):
                self.file_names.append(output_prefix + ".pdos_atm#" + str(file[0]) + "(" + file[1] + ")_wfc#" + str(file[2]) +  "(s)")
            elif (file[3] == 1 ):
                self.file_names.append(output_prefix + ".pdos_atm#" + str(file[0]) + "(" + file[1] + ")_wfc#" + str(file[2]) +  "(p)")
            elif (file[3] == 2 ):
                self.file_names.append(output_prefix + ".pdos_atm#" + str(file[0]) + "(" + file[1] + ")_wfc#" + str(file[2]) +  "(d)")
            elif (file[3] == 3 ):
                self.file_names.append(output_prefix + ".pdos_atm#" + str(file[0]) + "(" + file[1] + ")_wfc#" + str(file[2]) +  "(f)")

            
    # @staticmethod  
    def parse_pdos(self):
        
        
        rf = open(self.filpdos+'.pdos_tot')
        data = rf.readlines()
        rf.close()
        
        ###################################################################
        # Getting k point weights
        ###################################################################
        rf = open("nscf.out")
        nscfOut = rf.read()
        rf.close()
        
        #weight = float(findall("wk\s=\s*([-\.\d]*)",nscfOut)[0])

        ###################################################################   
        ####################################################################
        
        iline = 0
        header = [str(x) for x in data[iline].split()[2:]]

        if(self.spinCalc == True):
            header.pop(1)
            header.pop(1)

        else:
            header.pop(1)
        header[0] = "Energy"
        if(self.spinCalc == True):
            header[1] = "Dos-up"
            header[2] = "Dos-down"
        else:
            header[1] = "Dos"
            
        iline += 1
        
        ndos= len(data)-1

        total_dos = [[float(x) for x in y.split()[0:]] for y in data[iline:iline + ndos]]

        # self.test = total_dos
        # total_dos = delete(total_dos,1,1)
        # total_dos = delete(total_dos,1,1)
        if(self.spinCalc == True):
             total_dos = delete(total_dos,1,1)
             total_dos = delete(total_dos,1,1)
        else:
             total_dos = delete(total_dos,1,1)
        # ###################################################################################
        tmp_dict ={}        
       
        for filename in self.file_names:
            if not os.path.isfile(filename):
                raise ValueError('ERROR: DOSCAR file not found')
                
            rf = open(filename)
            data = rf.readlines()
            rf.close()
    
    

    
            atmNum = findall("#(\d*)",filename)[0]
            atmName = findall("atm#\d*\(([a-zA-Z0-9]*)\)",filename)[0]
            orbitalName = findall("wfc#\d*\((\S)\)",filename)[0]
            
            atmNumName = 'ion' + atmNum
            if atmNumName not in list(tmp_dict.keys()):
                tmp_dict[atmNumName] = zeros(shape=[len(total_dos[:,0]),10,2])
            # if atmName not in list(tmp_dict.keys()):
            #     tmp_dict[atmName] = zeros(shape=[len(total_dos[:,0]),10,2])
    
            iline = 0
            
              # Skipping the first lines of header
            iline += 1
         
           
            
            final_dos = [[float(x) for x in y.split()[0:]] for y in data[iline:iline + ndos]]
            iline += 1 
            iline += ndos

            final_labels = data[0].split()

            final_labels.pop(0)

            final_labels.pop(1)
            final_labels.pop(1)
            final_labels.pop(1)

            # final_dos = delete(final_dos,1,1)
            # final_dos = delete(final_dos,1,1)
            
            
            # dos = zeros(shape=[len(final_dos[:,0]),10,2])
            
            if self.is_spin_polarized == False:
                final_dos = delete(final_dos,1,1)
                dos = zeros(shape=[len(final_dos[:,0]),10,2])
            
                
                dos[:,0,0] = final_dos[:,0]
                if 's' == orbitalName:
                    dos[:,1,0] = final_dos[:,1]
                elif 'p' == orbitalName:
                    dos[:,2:5,0] = final_dos[:,1:]
                elif 'd' == orbitalName:
                    dos[:,5:10,0] = final_dos[:,1:]
            else:
                final_dos = delete(final_dos,1,1)
                final_dos = delete(final_dos,1,1)
                dos = zeros(shape=[len(final_dos[:,0]),10,2])
            
           
                dos[:,0,0] = final_dos[:,0]
                dos[:,0,1] = final_dos[:,0]
                if 's' == orbitalName:
                    dos[:,1,0] = final_dos[:,1]
                    dos[:,1,1] = final_dos[:,2]
                elif 'p' == orbitalName:
                    dos[:,2:5,0] = final_dos[:,1::2]
                    dos[:,2:5,1] = final_dos[:,2::2]
                elif 'd' == orbitalName:
                    dos[:,5:10,0] = final_dos[:,1::2]
                    dos[:,5:10,1] = final_dos[:,2::2]
            # tmp_dict[atmName] += dos
            tmp_dict[atmNumName] += dos
           
        projected_dos = []
        for name in list(tmp_dict.keys()):
            for ispin in range(2):
                tmp_dict[name][:,0,ispin] = final_dos[:,0]
                tmp_dict[name][:,[2,3],ispin] = tmp_dict[name][:,[3,2],ispin]
                tmp_dict[name][:,[2,4],ispin] = tmp_dict[name][:,[4,2],ispin]
                
                tmp_dict[name][:,[5,9],ispin] = tmp_dict[name][:,[9,5],ispin]
                tmp_dict[name][:,[6,7],ispin] = tmp_dict[name][:,[7,6],ispin]
                tmp_dict[name][:,[7,9],ispin] = tmp_dict[name][:,[9,7],ispin]
                tmp_dict[name][:,[8,9],ispin] = tmp_dict[name][:,[9,8],ispin]

            projected_dos.append(tmp_dict[name])
            

        project_labels = ['energies','s','p_y', 'p_z','p_x', 'd_xy', 'd_zy', 'd_z^2', 'd_zx','d_x^2-y^2']
        return {'total': total_dos,'projected': projected_dos, 'projected_labels_info':project_labels, 'ions': self.species_list}
        
       
         
    @property
    def lattice(self):
        """
        Returns the reciprocal lattice read from .out
        """
        rf = open(self.outfile, "r")
        data = rf.read()
        rf.close()

        alat = float(findall(r"alat\)\s*=\s*([\d.e+-]*)", data)[0])

        a1 = array(
            findall(r"a\(1\)\s*=\s*\(([\d.\s+-e]*)", data)[0].split(), dtype="float64"
        )
        a2 = array(
            findall(r"a\(2\)\s*=\s*\(([\d.\s+-e]*)", data)[0].split(), dtype="float64"
        )
        a3 = array(
            findall(r"a\(3\)\s*=\s*\(([\d.\s+-e]*)", data)[0].split(), dtype="float64"
        )

        lat = (2 * pi / alat) * (array((a1, a2, a3)))

        # Transposing to get the correct format
        #lat = lat.T

        return lat

            
            
    
