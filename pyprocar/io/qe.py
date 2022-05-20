__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import re
import os 
import math
import xml.etree.ElementTree as ET

import numpy as np
from pyprocar.core import DensityOfStates, Structure, ElectronicBandStructure, KPath


HARTREE_TO_EV = 27.211386245988  #eV/Hartree
class QEParser():
    def __init__(self, scfIn_filename = "scf.in", dirname = "", bandsIn_filename = "bands.in", pdosIn_filename = "pdos.in", kpdosIn_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml", dos_interpolation_factor = None ):
        if dirname != "":
            dirname = dirname + os.sep
        else:
            dirname = ""
            
        rf = open(f"{dirname}{scfIn_filename}", "r")
        self.scfIn = rf.read()
        rf.close()

        xml_filename =  re.findall("prefix\s*=\s*'(.*)'", self.scfIn)[0] + ".xml"

        tree = ET.parse(f"{dirname}{xml_filename}")

        self.root = tree.getroot()
        self.prefix =  self.root.findall(".//input/control_variables/prefix")[0].text
        self.dirname = dirname
        
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
        
        # self.orbitalName = [
        #     "s",
        #     "py",
        #     "pz",
        #     "px",
        #     "dxy",
        #     "dyz",
        #     "dz2",
        #     "dxz",
        #     "x2-y2",
        #     "fy3x2",
        #     "fxyz",
        #     "fyz2",
        #     "fz3",
        #     "fxz2",
        #     "fzx2",
        #     "fx3",
        #     "tot",
        # ]
        
        self.orbitalNames = [
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
        
        self.dos_interpolation_factor = dos_interpolation_factor
        
        
        # Parsing structual information 
               
        
        self.parse_magnetization()
        self.parse_structure()
        self.parse_band_structure_tag()
        self.parse_symmetries()
        
        # Detmines if this is a kresolved calculation or not
        
        
        if os.path.exists(f"{dirname}{kpdosIn_filename}"):
            rf = open(f"{dirname}{kpdosIn_filename}", "r")
            self.kpdosIn = rf.read()
            rf.close()
            
            self.pdos_prefix = re.findall("filpdos\s*=\s*'(.*)'", self.kpdosIn)[0]
            self.proj_prefix = re.findall("filproj\s*=\s*'(.*)'", self.kpdosIn)[0]
            
        if os.path.exists(f"{dirname}{pdosIn_filename}"):
            rf = open(f"{dirname}{pdosIn_filename}", "r")
            self.pdosIn = rf.read()
            rf.close()
            
            self.pdos_prefix = re.findall("filpdos\s*=\s*'(.*)'", self.pdosIn)[0]
            self.proj_prefix = re.findall("filproj\s*=\s*'(.*)'", self.pdosIn)[0]
            
        #Parsing spd array and spd phase arrays
        
        if os.path.exists(f"{dirname}{atomic_proj_xml}"):
            self.parse_projections()
            atmProj_tree = ET.parse(f"{dirname}{atomic_proj_xml}" )
            self.atm_proj_root = atmProj_tree.getroot()
            self.parse_atomic_projections()

        if os.path.exists(f"{dirname}{pdosIn_filename}"):
            self.parse_pdos()
            self.data = self.read()
        
        
        if self.root.findall(".//input/control_variables/calculation")[0].text == "bands":
            # print("This xml is a bands calculation")
            self.isBandsCalc = True
            rf = open(f"{dirname}{bandsIn_filename}", "r")
            self.bandsIn = rf.read()
            rf.close()
            self.getKpointLabels()

            self.ebs = ElectronicBandStructure(
                                    kpoints=self.kpoints,
                                    bands=self.bands,
                                    projected=self._spd2projected(self.spd),
                                    efermi=self.efermi,
                                    kpath=self.kpath,
                                    projected_phase=self._spd2projected(self.spd_phase),
                                    labels=self.orbitalNames[:-1],
                                    reciprocal_lattice=self.reciprocal_lattice,
                                    interpolation_factor=dos_interpolation_factor,
                                    # shifted_to_efermi=True,
                                    # shifted_to_efermi=False,
                                )

        else:
            self.kticks = None
            self.knames = None
       
        if self.root.findall(".//input/control_variables/calculation")[0].text == "nscf":
            # print("This xml is a nscf calculation")
            self.isDosFermiCalc = True
        
        
        
        
        
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

        st = Structure(atoms=symbols, lattice = self.direct_lattice, fractional_coordinates =self.atomic_positions )
                      
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
            atoms = np.arange(self.nions,dtype=int)
        if spin == None :
            spin = [0,1]
        if orbitals == None :
            orbitals = np.arange((len(projected[0].labels)-1)//2,dtype=int)
        if title == None:
            title = 'Sum'
        orbitals = np.array(orbitals)
        
        
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
                    print('Only use one or more of the following ' + str(np.arange((len(projected[0].labels)-1)//2,dtype=int)))
                    print('Only use one or more of the following ' + str(np.arange((len(projected[0].labels)-1)//2,dtype=int)))
                    print('They correspond to the following orbitals : ' + str(self.availiableOrbitals) )
                    print('Again do not trust the plot that was just produced' )
            if spin[0] == 0:
                labels = ['Energy','Spin-Up']
            elif spin[0] == 1:
                labels = ['Energy','Spin-Down']
            
        
        
        ret = np.zeros(shape=(len(projected[0].energies),len(spin)+1))
        ret[:,0] = projected[0].energies
        
        for iatom in atoms :
            if len(spin) == 2 :
                ret[:,1:]+=self.dos_projected[iatom].values[:,orbitals].sum(axis=2)
            elif len(spin) == 1 :
                ret[:,1]+=self.dos_projected[iatom].values[:,orbitals].sum(axis=1)
                
        return DensityOfStates(table=ret,title=title,labels=labels)
       
        
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
        
        dos_total['Spin-up'] = self.data['total'][:, 1]
        dos_total['Spin-down'] = self.data['total'][:, 2]
        # if self.nspin == 2 :
        #     dos_total['Spin-up'] = self.data['total'][:, 1]
        #     dos_total['Spin-down'] = self.data['total'][:, 2]
        #     #dos_total['integrated_dos_up'] = self.data['total'][:, 3]
        #     #dos_total['integrated_dos_down'] = self.data['total'][:, 4]
        # else:
        #     dos_total['Spin-up'] = self.data['total'][:, 1]
        #     dos_total['Spin-down'] = self.data['total'][:, 2]
        #     # dos_total['integrated_dos'] = self.data['total'][:, 2]

        return dos_total,list(dos_total.keys())

    def _get_dos_projected(self, atoms=[]):

        if len(atoms) == 0:
            atoms = np.arange(self.initial_structure.natoms)
      
 
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
                if self.nspin == 2 :
                    dos_projected[name]['Spin-up'] = self.data['projected'][i][:, 1:,0]
                    dos_projected[name]['Spin-down'] = self.data['projected'][i][:, 1:,1]
                else:
                    dos_projected[name]['Spin-up'] = self.data['projected'][i][:, 1:,0]
                    dos_projected[name]['Spin-down'] = self.data['projected'][i][:, 1:,1]
                    
            return dos_projected, self.data['projected_labels_info']
        else:
            print(
                "This calculation does not include partial density of states")
            return None, None
    
    @property
    def dos(self):
        
        energies = self.dos_total['energies'] - self.efermi / 2
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
    
    def parse_pdos(self):
    
    
        rf = open(f"{self.dirname}{os.sep}{self.pdos_prefix}.pdos_tot")
        data = rf.readlines()
        rf.close()
        
        ###################################################################
        # Parsing total density of states
        ###################################################################
        
        iline = 0
        header = [str(x) for x in data[iline].split()[2:]]

        if self.nspin == 2 :
            header.pop(1)
            header.pop(1)

        else:
            header.pop(1)
        header[0] = "Energy"
        if self.nspin == 2 :
            header[1] = "Dos-up"
            header[2] = "Dos-down"
        else:
            header[1] = "Dos"
            
        iline += 1
        
        ndos= len(data)-1

        total_dos = np.array([[float(x) for x in y.split()[0:]] for y in data[iline:iline + ndos]])
        
   
        self.text = total_dos

        if self.nspin == 2 :
             total_dos = np.delete(total_dos,1,1)
             total_dos = np.delete(total_dos,1,1)
        else:
             total_dos = np.delete(total_dos,1,1)
             spin_down = np.array([[0] for x in range(len(total_dos[:,0])) ] )
             total_dos = np.append(  total_dos , spin_down ,axis = 1 )
   
        ####################################################################################
        tmp_dict ={}        
        
        self.wfc_filenames = []
        wfc_filenames = []
        
        atms_wfc_num = []
        for file in os.listdir(f"{self.dirname}"):
            if file.startswith(self.pdos_prefix) and not file.endswith(".pdos_tot"):
                filename = f"{self.dirname}/{file}"
                wfc_filenames.append(filename )
                atm_num = int(re.findall("_atm#([0-9]*)\(.*",filename)[0])
                wfc_num = int(re.findall("_wfc#([0-9]*)\(.*",filename)[0])
                wfc = re.findall("_wfc#[0-9]*\(([A-Za-z]*)\).*",filename)[0]
                atm = re.findall("_atm#[0-9]*\(([A-Za-z]*)\).*",filename)[0]
 
                atms_wfc_num.append((atm_num,atm,wfc_num,wfc))
                
                
        # sort pdos wfc filenames  
        sorted_file_num = sorted(atms_wfc_num, key= lambda a: a[0])
        for index in sorted_file_num:
            self.wfc_filenames.append(f"{self.dirname}/{self.pdos_prefix}.pdos_atm#{index[0]}({index[1]})_wfc#{index[2]}({index[3]})")
            
                
            
        
    
        for filename in self.wfc_filenames:
            if not os.path.exists(filename):
                raise ValueError('ERROR: pdos file not found')
                
            rf = open(filename)
            data = rf.readlines()
            rf.close()
    
            atmNum = re.findall("#(\d*)",filename)[0]
            atmName = re.findall("atm#\d*\(([a-zA-Z0-9]*)\)",filename)[0]
            orbitalName = re.findall("wfc#\d*\((\S)\)",filename)[0]
            
            atmNumName = 'ion' + atmNum
            if atmNumName not in list(tmp_dict.keys()):
                tmp_dict[atmNumName] = np.zeros(shape=[len(total_dos[:,0]),10,2])

            # Skipping the first lines of header
            iline = 0
            iline += 1

            final_dos = [[float(x) for x in y.split()[0:]] for y in data[iline:iline + ndos]]
            iline += 1 
            iline += ndos

            final_labels = data[0].split()
            final_labels.pop(0)
            final_labels.pop(1)
            final_labels.pop(1)
            final_labels.pop(1)


            
            if self.nspin == 1 :
                final_dos = np.delete(final_dos,1,1)
                dos = np.zeros(shape=[len(final_dos[:,0]),10,2])
            
                
                dos[:,0,0] = final_dos[:,0]
                if 's' == orbitalName:
                    dos[:,1,0] = final_dos[:,1]
                elif 'p' == orbitalName:
                    dos[:,2:5,0] = final_dos[:,1:]
                elif 'd' == orbitalName:
                    dos[:,5:10,0] = final_dos[:,1:]
            else:
                final_dos = np.delete(final_dos,1,1)
                final_dos = np.delete(final_dos,1,1)
                dos = np.zeros(shape=[len(final_dos[:,0]),10,2])
            
           
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
                
    def getKpointLabels(self):
        
        # Parsing klabels 
        self.ngrids = []
        kmethod = re.findall("K_POINTS[\s\{]*([a-z_]*)[\s\{]*", self.bandsIn)[0]
        self.discontinuities = []
        if kmethod == "crystal":
            numK = int(re.findall("K_POINTS.*\n([0-9]*)", self.bandsIn)[0])

            raw_khigh_sym = re.findall(
                "K_POINTS.*\n\s*[0-9]*.*\n" + numK * "(.*)\n*", self.bandsIn
            )[0]

            tickCountIndex = 0
            self.knames = []
            self.kticks = []

            for x in raw_khigh_sym:
                if len(x.split()) == 5:
                    self.knames.append("%s" % x.split()[4].replace("!", ""))
                    self.kticks.append(tickCountIndex)

                tickCountIndex += 1

            self.nhigh_sym = len(self.knames)

        elif kmethod == "crystal_b":
            self.nhigh_sym = int(re.findall("K_POINTS.*\n([0-9]*)", self.bandsIn)[0])

            raw_khigh_sym = re.findall(
                "K_POINTS.*\n.*\n" + self.nhigh_sym * "(.*)\n*", 
                self.bandsIn,
            )[0]

            
            self.kticks = []
            self.high_symmetry_points = np.zeros(shape=(self.nhigh_sym, 3))

            
            tick_Count = 1
            for ihs in range(self.nhigh_sym):
                self.kticks.append(tick_Count - 1)
                self.ngrids.append(int(raw_khigh_sym[ihs].split()[3]))
                tick_Count += int(raw_khigh_sym[ihs].split()[3])
                
                
                
                
            raw_ticks = re.findall(
                "K_POINTS.*\n\s*[0-9]*\s*[0-9]*.*\n" + self.nhigh_sym * ".*!(.*)\n*",
                self.bandsIn,
            )[0]
            
            if len(raw_ticks) != self.nhigh_sym:
                self.knames = [str(x) for x in range(self.nhigh_sym)]
                
            else:
                self.knames = [
                    "%s" % (x.replace(",", "").replace("vlvp1d", "").replace(" ", ""))
                    for x in raw_ticks
                ]
            

        # Formating to conform with Kpath class
        self.special_kpoints = np.zeros(shape = (len(self.kticks) -1 ,2,3) )

        self.modified_knames = []
        for itick in range(len(self.kticks)):
            if itick != len(self.kticks) - 1: 
                self.special_kpoints[itick,0,:] = self.kpoints[self.kticks[itick]]
                self.special_kpoints[itick,1,:] = self.kpoints[self.kticks[itick+1]]
                self.modified_knames.append([self.knames[itick], self.knames[itick+1] ])


        
        has_time_reversal = True
        self.kpath = KPath(
                        knames=self.modified_knames,
                        special_kpoints=self.special_kpoints,
                        kticks = self.kticks,
                        ngrids=self.ngrids,
                        has_time_reversal=has_time_reversal,
                    )
                    
    def parse_projections(self):

        rf = open(f"{self.dirname}{self.proj_prefix}.projwfc_up", "r")
        self.projections_file = [rf.readlines()]
        rf.close()
        if self.nspin == 2:
            rf = open(f"{self.dirname}{self.proj_prefix}.projwfc_down", "r")
            self.projections_file.append( rf.readlines())
            rf.close()
            
        
        natm = int(self.projections_file[0][1].split()[-2])
        ntyp = int(self.projections_file[0][1].split()[-1])
        
        ibrav = int(self.projections_file[0][2].split()[0])
        if ibrav == 0:
            n_header_lines = 7
        else:
            n_header_lines = 4
        
        nwfc = int(self.projections_file[0][natm + ntyp+n_header_lines].split()[0])
        nk = int(self.projections_file[0][natm + ntyp+n_header_lines].split()[1])
        nbnd = int(self.projections_file[0][natm + ntyp+n_header_lines].split()[2])
        norb = len(self.orbitals) 
        
        # print(natm)
        # print(norb)
        # print(nbnd)
        # print(nk)
        # print(nwfc)
        # print(ntyp)

        # # In natm "+1" for total. In norb "+2" for total

        self.spd = np.zeros(shape = (nk,self.nbnd , self.nspin ,natm+1,norb + 2,))
        

        self.wfc_mapping = {}
        for ispin, projection_file in enumerate(self.projections_file):
            
            nspin = 2
            starting_line_num = natm + ntyp + n_header_lines + 2
            
            lineCounter = starting_line_num
            for iwfc in range(nwfc):
                if iwfc != 0:
                    lineCounter += 1
                iatm = int(projection_file[lineCounter].split()[1])
                atm = projection_file[lineCounter].split()[2]
                l_orbital_type_index = int(projection_file[lineCounter].split()[5])
                m_orbital_type_index = int(projection_file[lineCounter].split()[6])
                tmp_orb_dict = {"l" : l_orbital_type_index , "m" : m_orbital_type_index}
                iorb = 0
                for iorbital, orb in enumerate(self.orbitals):
                    if tmp_orb_dict == orb:
                        iorb = iorbital
                lOrbtial = projection_file[lineCounter].split()[3][1].lower()
                
                
                
                self.wfc_mapping.update({f"wfc_{iwfc+1}":{"orbital" : iorb, "atom" : iatm}})
                
                
                # self.wfc_filenames.append(f"{self.dirname}/{self.pdos_prefix}.pdos_atm#{iatm}({atm})_wfc#{lOrbtial}({lOrbtial})")

                for ik in range(nk):
                    for iband in range(nbnd):
                        lineCounter += 1
                        # print()
                        if ispin ==0:
                            self.spd[ik,iband,0,iatm - 1,iorb + 1 ] = float(projection_file[lineCounter].split()[2])
                        elif ispin ==1:
                            self.spd[ik,iband,1,iatm - 1,iorb + 1 ] = float(projection_file[lineCounter].split()[2])
                        
        for ions in range(self.ionsCount):
            self.spd[:, :, :, ions, 0] = ions + 1

        # The following fills the totals for the spd array
        self.spd[:, :, :, :, -1] = np.sum(self.spd[:, :, :, :, 1:-1], axis=4)
        self.spd[:, :, :, -1, :] = np.sum(self.spd[:, :, :, :-1, :], axis=3)
        self.spd[:, :, :, -1, 0] = 0

        
        # if self.nspin == 2:
        #     print("\nQuantum Espresso colinear spin calculation detected.\n")
        #     self.spd2 = np.zeros(
        #         shape=(
        #             self.kpointsCount,
        #             self.bandsCount *2,
        #             self.nspin,
        #             self.ionsCount + 1,
        #             len(self.orbitals) + 2,
        #         )
        #     )

        #     # spin up block for spin=0
        #     self.spd2[:, : self.bandsCount, 0, :, :] = self.spd[:, :, 0, :, :]

        #     # spin down block for spin=0
        #     self.spd2[:, self.bandsCount :, 0, :, :] = self.spd[:, :, 1, :, :]

        #     # spin up block for spin=1
        #     self.spd2[:, : self.bandsCount, 1, :, :] = self.spd[:, :, 0, :, :]

        #     # spin down block for spin=1
        #     self.spd2[:, self.bandsCount :, 1, :, :] = -1 * self.spd[:, :, 1, :, :]

        #     self.spd = self.spd2

            # Reshaping bands array to inlcude all bands (spin up and down)
            # self.bands = self.bands.reshape(
            #     self.kpointsCount, self.bandsCount * 2, order="F"
            # )
 
    def parse_atomic_projections(self):
        nbnd = int(self.atm_proj_root.findall(".//HEADER/NUMBER_OF_BANDS")[0].text)
        nk = int(self.atm_proj_root.findall(".//HEADER/NUMBER_OF_K-POINTS")[0].text)
        nwfc = int(self.atm_proj_root.findall(".//HEADER/NUMBER_OF_ATOMIC_WFC")[0].text)
        nspin = int(self.atm_proj_root.findall(".//HEADER/NUMBER_OF_SPIN_COMPONENTS")[0].text)

        # self.complex_projections  = np.zeros(shape = (nk, nwfc, nbnd))
        # self.real_projections  = np.zeros(shape = (nk, nwfc, nbnd))
        
        self.spd_phase = np.zeros(
            shape=(
               self.spd.shape
            ),
            dtype=np.complex_,
        )

        for ik, kpoints_element in enumerate(self.atm_proj_root.findall(".//PROJECTIONS")[0]):
            
            if self.nspin == 1:
                for iwfc, wfc_element in enumerate(kpoints_element):
                    
                    iatm = self.wfc_mapping[f"wfc_{iwfc+1}"]["atom"]
                    iorb = self.wfc_mapping[f"wfc_{iwfc+1}"]["orbital"]
                    
                    projections = wfc_element.text.split("\n")[1:nbnd].pop(-1)
                    for iband, band_projection in enumerate(wfc_element.text.split("\n")[1:-1]):
                        # self.real_projections[ik,iwfc,iband] = float(band_projection.split(",")[0])
                        # self.complex_projections[ik,iwfc,iband] = float(band_projection.split(",")[1])
                        
                        
                        real = float(band_projection.split(",")[0])
                        imag = float(band_projection.split(",")[1])
                        self.spd_phase[ik,iband,0,iatm - 1,iorb + 1] = complex(real , imag)

            elif self.nspin ==2:
                
                for ispin, spin_element in enumerate(kpoints_element):
                    for iwfc, wfc_element in enumerate(spin_element):
                        
                        iatm = self.wfc_mapping[f"wfc_{iwfc+1}"]["atom"]
                        iorb = self.wfc_mapping[f"wfc_{iwfc+1}"]["orbital"]
                        
                        projections = wfc_element.text.split("\n")[1:nbnd].pop(-1)
                        for iband, band_projection in enumerate(wfc_element.text.split("\n")[1:-1]):
                            # self.real_projections[ik,iwfc,iband] = float(band_projection.split(",")[0])
                            # self.complex_projections[ik,iwfc,iband] = float(band_projection.split(",")[1])
                            
                            real = float(band_projection.split(",")[0])
                            imag = float(band_projection.split(",")[1])
                            self.spd_phase[ik,iband,ispin,iatm - 1,iorb + 1] = complex(real , imag)
                               
    def parse_structure(self):
        self.nspecies = len(self.root.findall(".//output/atomic_species")[0])
        
        self.composition = { species.attrib['name'] : 0 for species in  self.root.findall(".//output/atomic_species")[0]  }
        self.species_list = list(self.composition.keys())
        self.ionsCount = int(self.root.findall(".//output/atomic_structure")[0].attrib['nat'])
        self.alat =  float(self.root.findall(".//output/atomic_structure")[0].attrib['alat'])
        # self.bravais_index =  int(self.root.findall(".//output/atomic_structure")[0].attrib['bravais_index'])
        
        self.ions = []
        for ion in self.root.findall(".//output/atomic_structure/atomic_positions")[0]:
            self.ions.append(ion.attrib['name'])
            self.composition[ ion.attrib['name']] += 1
            
        self.atomic_positions = np.array([ ion.text.split() for ion in self.root.findall(".//output/atomic_structure/atomic_positions")[0]],dtype = float)
        # in a.u
        self.direct_lattice = np.array([ acell.text.split() for acell  in self.root.findall(".//output/atomic_structure/cell")[0] ],dtype = float)

    def parse_symmetries(self):
        self.nsym = int(self.root.findall(".//output/symmetries/nsym")[0].text)
        self.nrot = int(self.root.findall(".//output/symmetries/nrot")[0].text)
        self.spg = int(self.root.findall(".//output/symmetries/space_group")[0].text)
        self.nsymmetry = len(self.root.findall(".//output/symmetries/symmetry"))
        
        self.rotation_matricies = np.zeros(shape = (self.nsymmetry ,3,3))
        
        for isymmetry,symmetry_operation in enumerate(self.root.findall(".//output/symmetries/symmetry")):

            symmetry_matrix = np.array(symmetry_operation.findall(".//rotation")[0].text.split(),dtype = float).reshape(3,3)

            self.rotation_matricies[isymmetry,:,:] = symmetry_matrix
            
    def parse_magnetization(self):
        self.non_colinear = str2bool(self.root.findall(".//output/magnetization/noncolin")[0].text)
        self.spinCalc = str2bool(self.root.findall(".//output/magnetization/lsda")[0].text)
        self.spin_orbit = str2bool(self.root.findall(".//output/magnetization/spinorbit")[0].text)
        if self.spinCalc:
            self.nspin = 2
        else:
            self.nspin = 1
               
    def parse_band_structure_tag(self):
        if 'monkhorst_pack' in self.root.findall(".//output/band_structure/starting_k_points")[0].tag:
            self.nk1 = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['nk1']
            self.nk2 = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['nk2']
            self.nk3 = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['nk3']
            
            self.nk1 = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['k1']
            self.nk2 = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['k2']
            self.nk3 = self.root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['k3']

        
        self.nks = int(self.root.findall(".//output/band_structure/nks")[0].text)
        self.atm_wfc = int(self.root.findall(".//output/band_structure/num_of_atomic_wfc")[0].text)
        
        self.nelec = float(self.root.findall(".//output/band_structure/nelec")[0].text)
        if self.nspin == 1:
            
            self.nbnd = int(self.root.findall(".//output/band_structure/nbnd")[0].text)
            self.bands = np.zeros(shape = (self.nks, self.nbnd, 1))
            self.kpoints = np.zeros(shape = (self.nks, 3))
            self.weights = np.zeros(shape = (self.nks))
            self.occupations = np.zeros(shape = (self.nks, self.nbnd))
            for ikpoint, kpoint_element in enumerate(self.root.findall(".//output/band_structure/ks_energies")):
                self.kpoints[ikpoint,:] = np.array(kpoint_element.findall(".//k_point")[0].text.split(),dtype = float)
                self.weights[ikpoint] = np.array(kpoint_element.findall(".//k_point")[0].attrib["weight"], dtype = float)
                self.bands[ikpoint, : , 0]  = HARTREE_TO_EV  * np.array(kpoint_element.findall(".//eigenvalues")[0].text.split(),dtype = float)
                
                self.occupations[ikpoint, : ]  = np.array(kpoint_element.findall(".//occupations")[0].text.split(), dtype = float)
        elif self.nspin == 2:
            
            self.nbnd = int(self.root.findall(".//output/band_structure/nbnd_up")[0].text)
            self.nbnd_up = int(self.root.findall(".//output/band_structure/nbnd_up")[0].text)
            self.nbnd_down = int(self.root.findall(".//output/band_structure/nbnd_dw")[0].text)
            
            self.bands = np.zeros(shape = (self.nks, self.nbnd , 2))
            self.kpoints = np.zeros(shape = (self.nks, 3))
            self.weights = np.zeros(shape = (self.nks))
            self.occupations = np.zeros(shape = (self.nks, self.nbnd,2))
            
            band_structure_element = self.root.findall(".//output/band_structure")[0]

            for ikpoint, kpoint_element in enumerate(self.root.findall(".//output/band_structure/ks_energies")):
                 
                self.kpoints[ikpoint,:] = np.array(kpoint_element.findall(".//k_point")[0].text.split(),dtype = float)
                self.weights[ikpoint] = np.array(kpoint_element.findall(".//k_point")[0].attrib["weight"], dtype = float)
                
                
                self.bands[ikpoint, : ,0]  = HARTREE_TO_EV  * np.array(kpoint_element.findall(".//eigenvalues")[0].text.split(),dtype = float)[:self.nbnd_up]
                
                self.occupations[ikpoint, : ,0]  = np.array(kpoint_element.findall(".//occupations")[0].text.split(), dtype = float)[:self.nbnd_up]
                
                self.bands[ikpoint, : ,1]  = HARTREE_TO_EV  * np.array(kpoint_element.findall(".//eigenvalues")[0].text.split(),dtype = float)[self.nbnd_down:]
                self.occupations[ikpoint, : ,1]  = np.array(kpoint_element.findall(".//occupations")[0].text.split(), dtype = float)[self.nbnd_down:]

        # Converting back to cartesian coordiantes in units of 2pi/alat
        self.raw_kpoints = self.kpoints
        self.kpoints = np.around(np.linalg.inv(self.reciprocal_lattice.T / (2*math.pi / self.alat)).dot(self.kpoints.T).T,decimals=8)

        self.kpointsCount = len(self.kpoints)
        self.bandsCount = self.nbnd

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
        nspins = spd.shape[2]
        
        norbitals = spd.shape[4] - 2
        # if spd.shape[2] == 4:
        #     nspins = 3
        # else:
        #     nspins = spd.shape[2]
        # if nspins == 2:
        #     nbands = int(spd.shape[1] / 2)
        # else:
        #     nbands = spd.shape[1]
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
        # if nspins == 3:
        #     projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :-1]
        # elif nspins == 2:
        #     projected[:, :, :, 0, :, 0] = temp_spd[:, :nbands, :-1, 1:-1, 0]
        #     projected[:, :, :, 0, :, 1] = temp_spd[:, nbands:, :-1, 1:-1, 0]
        # else:
        projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]
        return projected
    
    @property 
    def efermi(self):
        """
        Returns the fermi energy read from .out
        """
        
        return self.nspin * float(self.root.findall(".//output/band_structure/fermi_energy")[0].text) * HARTREE_TO_EV
        
    @property
    def reciprocal_lattice(self):   
        """
        The reciprocal lattice in units of 2 pi / alat
        alat :  (in units of a.u.)
        """

        return  (2 * np.pi / self.alat) * np.array([ acell.text.split() for acell  in self.root.findall(".//output/basis_set/reciprocal_lattice")[0] ],dtype = float)
    
    
def str2bool(v):
  return v.lower() in ("true") 
        

