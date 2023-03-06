__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import re
import copy
import os 
import math
import xml.etree.ElementTree as ET

import numpy as np
from pyprocar.core import DensityOfStates, Structure, ElectronicBandStructure, KPath


HARTREE_TO_EV = 27.211386245988  #eV/Hartree
class QEParser():
    """The class is used to parse Quantum Expresso files. 
        The most important objects that comes from this parser are the .ebs and .dos

        Parameters
        ----------
        dirname : str, optional
            Directory path to where calculation took place, by default ""
        scf_in_filename : str, optional
            The scf filename, by default "scf.in"
        bands_in_filename : str, optional
            The bands filename in the case of a band structure calculation, by default "bands.in"
        pdos_in_filename : str, optional
            The pdos filename in the case of a density ofstates calculation, by default "pdos.in"
        kpdos_in_filename : str, optional
            The kpdos filename, by default "kpdos.in"
        atomic_proj_xml : str, optional
            The atomic projection xml name. This is located in the where the outdir is and in the {prefix}.save directory, by default "atomic_proj.xml"
    """

    def __init__(self,
                        dirname:str = "", 
                        scf_in_filename:str = "scf.in", 
                        bands_in_filename:str = "bands.in", 
                        pdos_in_filename:str = "pdos.in", 
                        kpdos_in_filename:str = "kpdos.in", 
                        atomic_proj_xml:str = "atomic_proj.xml", 
        ):
        

        # Handles the pathing to the files
        self.dirname, prefix, xml_root, atomic_proj_xml_filename, pdos_in_filename,bands_in_filename,proj_out_filename = self._initialize_filenames(dirname, scf_in_filename, bands_in_filename,pdos_in_filename)
        
        # Parsing structual and calculation type information 
        self._parse_efermi(main_xml_root=xml_root)
        self._parse_magnetization(main_xml_root=xml_root)
        self._parse_structure(main_xml_root=xml_root)
        self._parse_band_structure_tag(main_xml_root=xml_root)
        self._parse_symmetries(main_xml_root=xml_root)
        
 
        # Parsing projections spd array and spd phase arrays
        if os.path.exists(atomic_proj_xml_filename):
            self._parse_wfc_mapping(proj_out_filename=proj_out_filename)
            self._parse_atomic_projections(atomic_proj_xml_filename=atomic_proj_xml_filename)

        # Parsing density of states files
        if os.path.exists(pdos_in_filename):
            self.dos = self._parse_pdos(pdos_in_filename=pdos_in_filename,dirname=dirname)
        
        # Parsing information related to the bandstructure calculations kpath and klabels
        self.kticks = None
        self.knames = None
        self.kpath = None
        if xml_root.findall(".//input/control_variables/calculation")[0].text == "bands":
            self.isBandsCalc = True
            with open(bands_in_filename, "r") as f:
                self.bandsIn = f.read()
            self.getKpointLabels()

        # if xml_root.findall(".//input/control_variables/calculation")[0].text == "nscf":
        #     self.is_dos_fermi_calc = True
        
        self.ebs = ElectronicBandStructure(
                                kpoints=self.kpoints,
                                bands=self.bands,
                                projected=self._spd2projected(self.spd),
                                efermi=self.efermi,
                                kpath=self.kpath,
                                projected_phase=self._spd2projected(self.spd_phase),
                                labels=self.orbital_names[:-1],
                                reciprocal_lattice=self.reciprocal_lattice,
                            )

        return None

    def kpoints_cart(self):
        """Returns the kpoints in cartesian coordinates

        Returns
        -------
        np.ndarray
            Kpoints in cartesian coordinates
        """
        # cart_kpoints self.kpoints = self.kpoints*(2*np.pi /self.alat)
        # Converting back to crystal basis
        cart_kpoints = self.kpoints.dot(self.reciprocal_lattice) * (self.alat/ (2*np.pi))

        return cart_kpoints

    @property
    def species(self):
        """Returns the species of the calculation

        Returns
        -------
        List
            Returns a list of string or atomic numbers[int]
        """
        return self.initial_structure.species

    @property
    def structures(self):
        """Returns a list of pyprocar.core.Structure

        Returns
        -------
        List
            Returns a list of pyprocar.core.Structure
        """

        # symbols = [x.strip() for x in self.data['ions']]
        symbols = [x.strip() for x in self.ions]
        structures = []

        st = Structure(atoms=symbols, lattice = self.direct_lattice, fractional_coordinates =self.atomic_positions )
                      
        structures.append(st)
        return structures

    @property
    def structure(self):
        """Returns a the last element of a list of pyprocar.core.Structure

        Returns
        -------
        pyprocar.core.Structure
            Returns a the last element of a list of pyprocar.core.Structure
        """
        return self.structures[-1]
    
    @property
    def initial_structure(self):
        """Returns a the first element of a list of pyprocar.core.Structure

        Returns
        -------
        pyprocar.core.Structure
            Returns a the first element of a list of pyprocar.core.Structure
        """
        return self.structures[0]
    
    @property
    def final_structure(self):
        """Returns a the last element of a list of pyprocar.core.Structure

        Returns
        -------
        pyprocar.core.Structure
            Returns a the last element of a list of pyprocar.core.Structure
        """

        return self.structures[-1]

    def _parse_pdos(self,pdos_in_filename,dirname):
        """Helper method to parse the pdos files

        Parameters
        ----------
        pdos_in_filename : str
            The pdos.in filename
        dirname : str
            The directory path where the calculation took place.

        Returns
        -------
        pyprocar.core.DensityOfStates
            The density of states object for the calculation
        """
        
        with open(pdos_in_filename, "r") as f:
            pdos_in = f.read()

        self.pdos_prefix = re.findall("filpdos\s*=\s*'(.*)'", pdos_in)[0]
        self.proj_prefix = re.findall("filproj\s*=\s*'(.*)'", pdos_in)[0]

        # Parsing total density of states
        energies, total_dos = self._parse_dos_total(dos_total_filename=f"{dirname}{os.sep}{self.pdos_prefix}.pdos_tot")

        # Finding all the density of states projections files
        # print(self.wfc_filenames)
        wfc_filenames = self._parse_available_wfc_filenames(dirname = self.dirname)
        projected_dos, projected_labels = self._parse_dos_projections(wfc_filenames=wfc_filenames, n_energy = len(energies))    
   
        # print(projected_labels)
        dos = DensityOfStates(energies=energies,
                            total=total_dos,
                            projected=projected_dos, 
                            interpolation_factor = 1,
                            interpolation_kind='cubic')
        return dos    
    
    def _parse_dos_total(self, dos_total_filename ):
        """Helper method to parse the dos total file

        Parameters
        ----------
        dos_total_filename : str
            The dos total filename

        Returns
        -------
        Tupole
            Returns a tuple with energies and the total dos arrays
        """
        with open(dos_total_filename) as f:
            tmp_text = f.readlines()
            header = tmp_text[0]
            dos_text = ''.join(tmp_text[1:])

        # Strip ending spaces away. Avoind empty string at the end
        raw_dos_blocks_by_energy = dos_text.rstrip().split('\n')

        n_energies = len(raw_dos_blocks_by_energy)
        energies = np.zeros(shape=(n_energies))
        # total_dos =  np.zeros(shape=(n_energies, self.n_spin))
        total_dos =  np.zeros(shape=(self.n_spin,n_energies))
        for ienergy, raw_dos_block_by_energy in enumerate(raw_dos_blocks_by_energy):
            energies[ienergy] = float(raw_dos_block_by_energy.split()[0])

            # Use if self.n_spin==4
            # Covers colinear spin-polarized. This is because these is a difference in energies
            if self.n_spin == 2:
                total_dos[:,ienergy] = [ float(val) for val in raw_dos_block_by_energy.split()[-self.n_spin:] ] 
            # Covers colinear non-spin-polarized and non-colinear. This is because the energies are the same
            else:
                total_dos[0,ienergy] =  float(raw_dos_block_by_energy.split()[2])
  
        energies -= self.efermi
        return energies, total_dos
   
    def _parse_dos_projections(self,wfc_filenames,n_energy):
        """Helper method to parse the dos projection files

        Parameters
        ----------
        wfc_filenames : List[str]
            A List of projection filenames.
        n_energy : int
            The number of energies for which the density of states is calculated at.

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        n_principal_number = 1
        projected_dos_array =  np.zeros(shape=(self.n_atoms,n_principal_number,self.n_orbitals,self.n_spin,n_energy))
        for filename in wfc_filenames[:]:
            if not os.path.exists(filename):
                raise ValueError('ERROR: pdos file not found')   
            with open(filename) as f:
                pdos_text = ''.join(f.readlines()[1:])

            # -1 because indexing starts at 0
            atom_num = int(re.findall("#(\d*)",filename)[0]) - 1
            wfc_Name = re.findall("atm#\d*\(([a-zA-Z0-9]*)\)",filename)[0]
            orbital_name = re.findall("wfc#\d*\(([_a-zA-Z0-9.]*)\)",filename)[0]
            n_orbitial = len(self.orbitals)
            

            # For noncolinear caluclations
            if self.is_non_colinear:
                # In the noncolinear case there are some files that are not used. They are in the uncoupled basis
                try:
                    tot_ang_mom = float(orbital_name.split('j')[-1])
                except:
                    continue
                
                # This controls the mapping from raw orbital projection data to structured orbital projection data
                # as defined by the order in self.orbitals
                l_orbital_name = orbital_name.split('_')[0]
                if l_orbital_name == 's':
                    if tot_ang_mom == 0.5:
                        m_js = np.linspace(-0.5,0.5,2)
                        orbital_nums = [0,1]
                elif l_orbital_name == 'p':
                    if tot_ang_mom == 0.5:
                        m_js = np.linspace(-0.5,0.5,2)
                        orbital_nums = [2,3]
                    elif tot_ang_mom == 1.5:
                        m_js = np.linspace(-1.5,1.5,4)
                        orbital_nums = [4,5,6,7]
                elif l_orbital_name == 'd':
                    if tot_ang_mom == 0.5:
                        m_js = np.linspace(-0.5,0.5,2)
                        orbital_nums = [2,3]
                    elif tot_ang_mom == 1.5:
                        m_js = np.linspace(-1.5,1.5,4)
                        orbital_nums = [8,9,10,11]
                    elif tot_ang_mom == 2.5:
                        m_js = np.linspace(-2.5,2.5,6)
                        orbital_nums = [12,13,14,15,16,17]

                # Filters unwanted empty character from the regular expression search
                raw_energy_blocks = list(filter(None,re.findall('([\s\S]*?)(?=\n \n)',pdos_text)))

                # Loop through the energies
                for i_energy, raw_energy_block in enumerate(raw_energy_blocks[:]):
                    #Strippng unwanted space characters before and after the block
                    raw_energy_block =raw_energy_block.lstrip().rstrip()
                    # Orbital_num is the index in the final array, i_orbital is the index in the raw_data
                    for i_orbital,orbital_num in enumerate(orbital_nums):
                        # Loop through spins
                        for i_spin in range(self.n_spin):
                            # Totals
                            # if i_spin == 0:
                            #     projected_dos_array[atom_num,0,orbital_num,i_spin,i_energy] += float(raw_energy_block.split('\n')[0].split()[2+i_orbital])
                            # # spin components
                            # else:
                            #     projected_dos_array[atom_num,0,orbital_num,i_spin,i_energy] += float(raw_energy_block.split('\n')[i_spin+1].split()[1+i_orbital])
                                

                            # This should be used if we set self.n_spins=3
                            projected_dos_array[atom_num,0,orbital_num,i_spin,i_energy] += float(raw_energy_block.split('\n')[i_spin+2].split()[1+i_orbital])

            # For colinear calculations
            else:
                current_orbital_name =  orbital_name

                # This controls the mapping from raw orbital projection data to structured orbital projection data
                # as defined by the order in self.orbitals
                if current_orbital_name == 's':
                    m_s = np.linspace(0,0,1)
                    orbital_nums = [0]
                elif current_orbital_name == 'p':
                    m_s = np.linspace(-1,1,3)
                    orbital_nums = [1,2,3]
                elif current_orbital_name == 'd':
                    m_s = np.linspace(-2,2,5)
                    orbital_nums = [4,5,6,7,8]

                # Filters unwanted empty character from the regular expression search
                raw_energy_blocks = pdos_text.rstrip().split('\n')

                # Loop through the energies
                for i_energy, raw_energy_block in enumerate(raw_energy_blocks):
                    raw_energy_block =raw_energy_block.lstrip().rstrip()
                    raw_projections = raw_energy_block.split()[1+self.n_spin:]

                    # Orbital_num is the index in the final array, i_orbital is the index in the raw_data
                    for i_orbital, orbital_num in enumerate(orbital_nums):

                        # Loop through spins
                        for i_spin in range(self.n_spin):
                            projected_dos_array[atom_num,0,orbital_num,i_spin,i_energy] += float(raw_projections[i_spin::self.n_spin][i_orbital])
        return projected_dos_array, self.orbital_names

    def getKpointLabels(self):
        """
        This method will parse the bands.in file to get the kpath information.
        """
        
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

                # In QE cyrstal_b mode, the user is able to specify grid on last high symmetry point. 
                # QE just uses 1 for the last high symmetry point.
                grid_current = int(raw_khigh_sym[ihs].split()[3])
                if ihs < self.nhigh_sym - 2:
                    self.ngrids.append(grid_current)
                elif ihs == self.nhigh_sym - 1:
                    self.ngrids.append(grid_current+1)
                elif ihs == self.nhigh_sym:
                    continue
                self.kticks.append(tick_Count - 1)
                tick_Count += grid_current

                
                
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

    def _initialize_filenames(self, dirname, scf_in, bands_in_filename, pdos_in_filename):
        """This helper method handles pathing to the to locate files

        Parameters
        ----------
        dirname : str
            The directory path where the calculation is
        scf_in : str
            The input scf filename
        bands_in_filename : str
            The input bands filename
        pdos_in_filename : str
            The input pdos filename

        Returns
        -------
        Tuple
            Returns a tuple of important pathing information. 
            Mainly, the directory path is prepended to the filenames.
        """

        
        if dirname != "":
            dirname = dirname + os.sep
        else:
            dirname = ""
            
        with open(f"{dirname}{scf_in}", "r") as f:
            scf_in = f.read()

        outdir = re.findall("outdir\s*=\s*'\S*?([A-Za-z]*)'",  scf_in)[0]
        prefix = re.findall("prefix\s*=\s*'(.*)'", scf_in)[0]
        xml_filename =  prefix + ".xml"
        
        if os.path.exists(f"{dirname}{outdir}"):
            atomic_proj_xml = f"{dirname}{os.sep}{outdir}{os.sep}{prefix}.save{os.sep}atomic_proj.xml"
        else:
            atomic_proj_xml = f"{dirname}atomic_proj.xml"

        tree = ET.parse(f"{dirname}{xml_filename}")
        root = tree.getroot()
        prefix = root.findall(".//input/control_variables/prefix")[0].text

        pdos_in_filename = f"{dirname}{pdos_in_filename}"
        bands_in_filename = f"{dirname}{bands_in_filename}"
        dirname = dirname

        if os.path.exists(f"{dirname}{os.sep}kpdos.out"):
            proj_out_filename = f"{dirname}{os.sep}kpdos.out"
        if os.path.exists(f"{dirname}{os.sep}pdos.out"):
            proj_out_filename = f"{dirname}{os.sep}pdos.out"
        
        return dirname, prefix, root, atomic_proj_xml, pdos_in_filename,bands_in_filename,proj_out_filename

    def _parse_available_wfc_filenames(self, dirname):
        """Helper method to parse the projection filename from the pdos.out file

        Parameters
        ----------
        dirname : str
            The directory name where the calculation is.

        Returns
        -------
        List
            Returns a list of projection file names
        """

        wfc_filenames = []
        tmp_wfc_filenames = []
        atms_wfc_num = []

        # Parsing projection filnames for identification information
        for file in os.listdir(f"{self.dirname}"):
            if (file.startswith(self.pdos_prefix) and not 
                file.endswith(".pdos_tot") and not 
                file.endswith(".lowdin") and not 
                file.endswith(".projwfc_down") and not 
                file.endswith(".projwfc_up")and not 
                file.endswith(".xml")):
                
                filename = f"{self.dirname}{os.sep}{file}"
                tmp_wfc_filenames.append(filename )
            
                atm_num = int(re.findall("_atm#([0-9]*)\(.*",filename)[0])
                wfc_num = int(re.findall("_wfc#([0-9]*)\(.*",filename)[0])
                wfc = re.findall("_wfc#[0-9]*\(([_A-Za-z0-9.]*)\).*",filename)[0]
                atm = re.findall("_atm#[0-9]*\(([A-Za-z]*[0-9]*)\).*",filename)[0]
 
                atms_wfc_num.append((atm_num,atm,wfc_num,wfc))
        
        # sort density of states projections files by atom number
        sorted_file_num = sorted(atms_wfc_num, key= lambda a: a[0])
        for index in sorted_file_num:
            wfc_filenames.append(f"{self.dirname}{os.sep}{self.pdos_prefix}.pdos_atm#{index[0]}({index[1]})_wfc#{index[2]}({index[3]})")

        return wfc_filenames

    def _parse_wfc_mapping(self, proj_out_filename):
        """Helper method which creates a mapping between wfc number and the orbtial and atom numbers

        Parameters
        ----------
        proj_out_filename : str
            The proj out filename

        Returns
        -------
        None
            None
        """
        with open(proj_out_filename) as f:
            proj_out =  f.read()
        raw_wfc  =  re.findall('(?<=read\sfrom\spseudopotential\sfiles).*\n\n([\S\s]*?)\n\n(?=\sk\s=)', proj_out)[0]
        wfc_list = raw_wfc.split('\n')

        self.wfc_mapping={}
        # print(self.orbitals)
        for i, wfc in enumerate(wfc_list):

            iwfc  =  int(re.findall('(?<=state\s#)\s*(\d*)',wfc)[0])
            iatm  =  int(re.findall('(?<=atom)\s*(\d*)',wfc)[0])
            l_orbital_type_index = int(re.findall('(?<=l=)\s*(\d*)',wfc)[0])

            if self.is_non_colinear:
                j_orbital_type_index =  float(re.findall('(?<=j=)\s*([-\d.]*)',wfc)[0])
                m_orbital_type_index =  float(re.findall('(?<=m_j=)\s*([-\d.]*)',wfc)[0])
                tmp_orb_dict = {"l" : self._convert_lorbnum_to_letter(lorbnum=l_orbital_type_index),
                                 "j" : j_orbital_type_index, 
                                 "m" : m_orbital_type_index}
                # print(self._convert_lorbnum_to_letter(lorbnum=l_orbital_type_index))
            else:
                m_orbital_type_index =  int(re.findall('(?<=m=)\s*(\d*)',wfc)[0])
                tmp_orb_dict = {"l" : l_orbital_type_index , "m" : m_orbital_type_index}
            
            iorb = 0
            
            for iorbital, orb in enumerate(self.orbitals):
                if tmp_orb_dict == orb:
                    iorb = iorbital
        
            self.wfc_mapping.update({f"wfc_{iwfc}":{"orbital" : iorb, "atom" : iatm}})

        return None

    def _parse_atomic_projections(self,atomic_proj_xml_filename):
        """A Helper method to parse the atomic projection xml file

        Parameters
        ----------
        atomic_proj_xml_filename : str
            The atomic_proj.xml filename

        Returns
        -------
        None
            None
        """
        atmProj_tree = ET.parse(atomic_proj_xml_filename)
        atm_proj_root = atmProj_tree.getroot()

        root_header = atm_proj_root.findall(".//HEADER")[0]

        nbnd = int(root_header.get("NUMBER_OF_BANDS"))
        nk = int(root_header.get("NUMBER_OF_K-POINTS"))
        nwfc = int(root_header.get("NUMBER_OF_ATOMIC_WFC"))
        norb = len(self.orbitals) 
        natm = len(self.species)

        self.spd = np.zeros(shape = (self.n_k, self.n_band , self.n_spin ,self.n_atoms+1,self.n_orbitals + 2,))

        self.spd_phase = np.zeros(
            shape=(
               self.spd.shape
            ),
            dtype=np.complex_,
        )
        # print(self.wfc_mapping)
        ik = -1
        for ieigenstate, eigenstates_element in enumerate(atm_proj_root.findall(".//EIGENSTATES")[0]):
            # print(eigenstates_element.tag)
            if eigenstates_element.tag == 'K-POINT':

                # sets ik back to zero for other spin channel
                if ik==nk-1:
                    ik=0
                else:
                    ik+=1
                
            if eigenstates_element.tag == 'PROJS':
                
                for i, projs_element in enumerate(eigenstates_element):
                    iwfc = int(projs_element.get('index'))
                    iorb = self.wfc_mapping[f"wfc_{iwfc}"]["orbital"]
                    iatm = self.wfc_mapping[f"wfc_{iwfc}"]["atom"]

                    if self.is_non_colinear:
                        # Skips the total projections
                        if projs_element.tag == "ATOMIC_WFC":
                            ispin = int(projs_element.get('spin'))-1
                            continue
                        # Parse spin components. -1 to align index with spd array
                        if projs_element.tag == "ATOMIC_SIGMA_PHI":
                            ispin = int(projs_element.get('ipol')) -1
                    else:
                        if projs_element.tag == "ATOMIC_WFC":
                            ispin = int(projs_element.get('spin'))-1
                       
                    

                    projections = projs_element.text.split("\n")[1:-1]
                    for iband, band_projection in enumerate(projections):
                        real = float(band_projection.split()[0])
                        imag = float(band_projection.split()[1])
                        comp = complex(real , imag)
                        comp_squared = np.absolute(comp)**2

                        
                        self.spd_phase[ik,iband,ispin,iatm - 1,iorb + 1] = complex(real , imag)

                        # The spd will depend on of the calculation is a non colinear or colinear. Noncolinear
                        if self.is_non_colinear:
                            self.spd[ik,iband,ispin,iatm - 1,iorb + 1] = real
                        else:
                            self.spd[ik,iband,ispin,iatm - 1,iorb + 1] = comp_squared

                        
        for ions in range(self.ionsCount):
            self.spd[:, :, :, ions, 0] = ions + 1

        # The following fills the totals for the spd array
        self.spd[:, :, :, :, -1] = np.sum(self.spd[:, :, :, :, 1:-1], axis=4)
        self.spd[:, :, :, -1, :] = np.sum(self.spd[:, :, :, :-1, :], axis=3)
        self.spd[:, :, :, -1, 0] = 0

        return None
                
    def _parse_structure(self,main_xml_root):
        """A helper method to parse the structure tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """

        self.nspecies = len(main_xml_root.findall(".//output/atomic_species")[0])
        self.composition = { species.attrib['name'] : 0 for species in  main_xml_root.findall(".//output/atomic_species")[0]  }
        self.species_list = list(self.composition.keys())
        self.ionsCount = int(main_xml_root.findall(".//output/atomic_structure")[0].attrib['nat'])
        self.alat =  float(main_xml_root.findall(".//output/atomic_structure")[0].attrib['alat'])

        self.ions = []
        for ion in main_xml_root.findall(".//output/atomic_structure/atomic_positions")[0]:
            self.ions.append(ion.attrib['name'][:2])
            self.composition[ ion.attrib['name']] += 1

        self.n_atoms = len(self.ions)

        self.atomic_positions = np.array([ ion.text.split() for ion in main_xml_root.findall(".//output/atomic_structure/atomic_positions")[0]],dtype = float)
        # in a.u
        self.direct_lattice = np.array([ acell.text.split() for acell  in main_xml_root.findall(".//output/atomic_structure/cell")[0] ],dtype = float)


        self.reciprocal_lattice =  (2 * np.pi / self.alat) * np.array([ acell.text.split() for acell  in main_xml_root.findall(".//output/basis_set/reciprocal_lattice")[0] ],dtype = float)
        return None

    def _parse_symmetries(self,main_xml_root):
        """A helper method to parse the symmetries tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """
        self.nsym = int(main_xml_root.findall(".//output/symmetries/nsym")[0].text)
        self.nrot = int(main_xml_root.findall(".//output/symmetries/nrot")[0].text)
        self.spg = int(main_xml_root.findall(".//output/symmetries/space_group")[0].text)
        self.nsymmetry = len(main_xml_root.findall(".//output/symmetries/symmetry"))
        
        self.rotations = np.zeros(shape = (self.nsymmetry ,3,3))
        
        for isymmetry,symmetry_operation in enumerate(main_xml_root.findall(".//output/symmetries/symmetry")):

            symmetry_matrix = np.array(symmetry_operation.findall(".//rotation")[0].text.split(),dtype = float).reshape(3,3).T

            self.rotations[isymmetry,:,:] = symmetry_matrix
        return None
            
    def _parse_magnetization(self,main_xml_root):
        """A helper method to parse the magnetization tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """
        is_non_colinear = str2bool(main_xml_root.findall(".//output/magnetization/noncolin")[0].text)
        is_spin_calc = str2bool(main_xml_root.findall(".//output/magnetization/lsda")[0].text)
        is_spin_orbit_calc = str2bool(main_xml_root.findall(".//output/magnetization/spinorbit")[0].text)
        
        # The calcuulation is non-colinear 
        if is_non_colinear :
            n_spin = 3

            orbitals = [
                        {"l": 's', "j": 0.5, "m": -0.5},
                        {"l": 's', "j": 0.5, "m": 0.5},

                        {"l": 'p', "j": 0.5, "m": -0.5},
                        {"l": 'p', "j": 0.5, "m": 0.5},

                        {"l": 'p', "j": 1.5, "m": -1.5},
                        {"l": 'p', "j": 1.5, "m": -0.5},
                        {"l": 'p', "j": 1.5, "m": -0.5},
                        {"l": 'p', "j": 1.5, "m": 1.5},

                        {"l": 'd', "j": 1.5, "m": -1.5},
                        {"l": 'd', "j": 1.5, "m": -0.5},
                        {"l": 'd', "j": 1.5, "m": -0.5},
                        {"l": 'd', "j": 1.5, "m": 1.5},

                        {"l": 'd', "j": 2.5, "m": -2.5},
                        {"l": 'd', "j": 2.5, "m": -1.5},
                        {"l": 'd', "j": 2.5, "m": -0.5},
                        {"l": 'd', "j": 2.5, "m": 0.5},
                        {"l": 'd', "j": 2.5, "m": 1.5},
                        {"l": 'd', "j": 2.5, "m": 2.5},
                    ]
            orbitalNames = []
            for orbital in orbitals:
                tmp_name = ''
                for key,value in orbital.items():
                    # print(key,value)
                    if key != 'l':
                        tmp_name = tmp_name + key + str(value)
                    else:
                        tmp_name = tmp_name + str(value) + '_'
                orbitalNames.append(tmp_name)

        # The calcuulation is colinear 
        else:
            # colinear spin or non spin polarized
            if is_spin_calc:
                n_spin = 2
            else:
                n_spin = 1
            orbitals = [
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
            orbitalNames = [
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

        self.is_non_colinear = is_non_colinear
        self.is_spin_calc = is_spin_calc
        self.is_spin_orbit_calc = is_spin_orbit_calc
        self.n_spin = n_spin
        self.orbitals = orbitals
        self.n_orbitals = len(orbitals)
        self.orbital_names = orbitalNames
        return None
               
    def _parse_band_structure_tag(self,main_xml_root):
        """A helper method to parse the band_structure tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """
        if 'monkhorst_pack' in main_xml_root.findall(".//output/band_structure/starting_k_points")[0].tag:
            self.nk1 = main_xml_root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['nk1']
            self.nk2 = main_xml_root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['nk2']
            self.nk3 = main_xml_root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['nk3']
            
            self.nk1 = main_xml_root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['k1']
            self.nk2 = main_xml_root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['k2']
            self.nk3 = main_xml_root.findall(".//output/band_structure/starting_k_points/monkhorst_pack").attrib['k3']

        
        self.nks = int(main_xml_root.findall(".//output/band_structure/nks")[0].text)
        self.atm_wfc = int(main_xml_root.findall(".//output/band_structure/num_of_atomic_wfc")[0].text)
        
        self.nelec = float(main_xml_root.findall(".//output/band_structure/nelec")[0].text)
        if self.n_spin == 2:
            
            self.n_band = int(main_xml_root.findall(".//output/band_structure/nbnd_up")[0].text)
            self.nbnd_up = int(main_xml_root.findall(".//output/band_structure/nbnd_up")[0].text)
            self.nbnd_down = int(main_xml_root.findall(".//output/band_structure/nbnd_dw")[0].text)
            
            self.bands = np.zeros(shape = (self.nks, self.n_band , 2))
            self.kpoints = np.zeros(shape = (self.nks, 3))
            self.weights = np.zeros(shape = (self.nks))
            self.occupations = np.zeros(shape = (self.nks, self.n_band,2))
            
            band_structure_element = main_xml_root.findall(".//output/band_structure")[0]

            for ikpoint, kpoint_element in enumerate(main_xml_root.findall(".//output/band_structure/ks_energies")):
                 
                self.kpoints[ikpoint,:] =  np.array(kpoint_element.findall(".//k_point")[0].text.split(),dtype = float)
                self.weights[ikpoint] = np.array(kpoint_element.findall(".//k_point")[0].attrib["weight"], dtype = float)
                
                
                self.bands[ikpoint, : ,0]  = HARTREE_TO_EV  * np.array(kpoint_element.findall(".//eigenvalues")[0].text.split(),dtype = float)[:self.nbnd_up]
                
                self.occupations[ikpoint, : ,0]  = np.array(kpoint_element.findall(".//occupations")[0].text.split(), dtype = float)[:self.nbnd_up]
                
                self.bands[ikpoint, : ,1]  = HARTREE_TO_EV  * np.array(kpoint_element.findall(".//eigenvalues")[0].text.split(),dtype = float)[self.nbnd_down:]
                self.occupations[ikpoint, : ,1]  = np.array(kpoint_element.findall(".//occupations")[0].text.split(), dtype = float)[self.nbnd_down:]
        # For non-spin-polarized and non colinear
        else:
            self.n_band = int(main_xml_root.findall(".//output/band_structure/nbnd")[0].text)
            self.bands = np.zeros(shape = (self.nks, self.n_band, 1))
            self.kpoints = np.zeros(shape = (self.nks, 3))
            self.weights = np.zeros(shape = (self.nks))
            self.occupations = np.zeros(shape = (self.nks, self.n_band))
            for ikpoint, kpoint_element in enumerate(main_xml_root.findall(".//output/band_structure/ks_energies")):
                self.kpoints[ikpoint,:] = np.array(kpoint_element.findall(".//k_point")[0].text.split(),dtype = float)
                self.weights[ikpoint] = np.array(kpoint_element.findall(".//k_point")[0].attrib["weight"], dtype = float)
                self.bands[ikpoint, : , 0]  = HARTREE_TO_EV  * np.array(kpoint_element.findall(".//eigenvalues")[0].text.split(),dtype = float)
                
                self.occupations[ikpoint, : ]  = np.array(kpoint_element.findall(".//occupations")[0].text.split(), dtype = float)
        # Multiply in 2pi/alat
        self.kpoints = self.kpoints*(2*np.pi /self.alat)
        # Converting back to crystal basis
        self.kpoints = np.around(self.kpoints.dot(np.linalg.inv(self.reciprocal_lattice)),decimals=8)
        self.n_k = len(self.kpoints)

        self.kpointsCount = len(self.kpoints)
        self.bandsCount = self.n_band

        return None

    def _spd2projected(self, spd, nprinciples=1):
        """
        Helpermethod to project the spd array to the projected array 
        which will be fed into pyprocar.coreElectronicBandStructure object

        Parameters
        ----------
        spd : np.ndarray
            The spd array from the earlier parse. This has a structure simlar to the PROCAR output in vasp
            Has the shape [n_kpoints,n_band,n_spins,n-orbital,n_atoms]
        nprinciples : int, optional
            The prinicipal quantum numbers, by default 1

        Returns
        -------
        np.ndarray
            The projected array. Has the shape [n_kpoints,n_band,n_atom,n_principal,n-orbital,n_spin]
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
        if nspins == 3:
            # Used if self.spins==3
            projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]
            # Used if self.spins == 4 
            # projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, 1:]
        elif nspins == 2:
            projected[:, :, :, 0, :, 0] = temp_spd[:, :, :-1, 1:-1, 0]
            projected[:, :, :, 0, :, 1] = temp_spd[:, :, :-1, 1:-1, 1]
        else:
            projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]
        return projected

    def _parse_efermi(self,main_xml_root):
        """A helper method to parse the band_structure tag of the main xml file for the fermi energy

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """
        self.efermi =  float(main_xml_root.findall(".//output/band_structure/fermi_energy")[0].text) * HARTREE_TO_EV
        return None

    def _convert_lorbnum_to_letter(self, lorbnum):
        """A helper method to convert the lorb number to the letter format

        Parameters
        ----------
        lorbnum : int
            The number of the l orbital

        Returns
        -------
        str
            The l orbital name
        """
        lorb_mapping = {0:'s',1:'p',2:'d',3:'f'}
        return lorb_mapping[lorbnum]

def str2bool(v):
    """Converts a string of a boolean to an actual boolean

    Parameters
    ----------
    v : str
        The string of the boolean value

    Returns
    -------
    boolean
        The boolean value
    """
    return v.lower() in ("true") 
        

