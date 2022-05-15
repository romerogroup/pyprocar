# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:50:22 2020

@author: lllan
"""

import os
import numpy as np

from ..core import DensityOfStates, Structure
from re import findall, search, match, DOTALL, MULTILINE, finditer, compile, sub


class LobsterDOSParser:
    def __init__(self, filename='DOSCAR.lobster', dos_interpolation_factor = None ):

        if not os.path.isfile(filename):
            raise ValueError('ERROR: DOSCAR file not found')

        self.filename = filename
        self.data = self.read()
        
        self.dos_interpolation_factor = dos_interpolation_factor
        self.test = None
        self.test2 = None
        
        self.ionsList = None
        if 'projected' in self.data:
            self.has_projected = True
            self.nions = len(self.data['projected'])
        else:
            self.has_projected = False
            self.nions = None

        self.ndos, self.total_ncols = self.data['total'].shape

        if self.total_ncols == 5:
            self.is_spin_polarized = True
            self.spins = ['dos_up','dos_down']
        else:
            self.is_spin_polarized = False

        # self.total_dos = self.dos_to_dict['total']

        # if self.has_projected:
        #     self.projected_dos = self.dos_to_dict['projected']
            
        rf = open('lobsterout', "r")
        self.lobsterout = rf.read()
        rf.close()

    ###########################################################################
    # This section parse for the total density of states and puts it in a 
    #Pychemia Density of States Object
    ###########################################################################
    def read(self):
        """
        Read and parse vasprun.xml.
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        return self.parse_doscar(self.filename)


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
            atoms = np.arange(self.initial_structure.natoms)

        if 'projected' in list(self.data.keys()):
            dos_projected = {}
            ion_list = ["ion %s" % str(x + 1) for x in atoms
                        ]  # using this name as vasrun.xml uses ion #
            for i in range(len(ion_list)):
                iatom = ion_list[i]
                name = self.initial_structure.atoms[atoms[i]]

                energies = self.data['projected'][i][:,0,0]
                
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
        energies = self.dos_total['energies']
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
        #dos_total['energies'] -= self.fermi

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
        symbols = [x.strip() for x in self.data['ions']]
        structures = []

        st = Structure(atoms=symbols)
                      
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
    
    
    ###########################################################################
    # This section parses for the projected density of states and puts it in a 
    # Pychemia Density of States Object
    ###########################################################################


    @staticmethod
    def parse_doscar(filename):

        rf = open(filename)
        data = rf.readlines()
        rf.close()

        rf = open("lobsterout", "r")
        lobsterout = rf.read()
        rf.close()
        

        if len(data) < 5:
            raise ValueError('DOSCAR seems truncated')

        ionsList = findall("calculating FatBand for Element: (.*) Orbital.*", lobsterout)
      
        proj_ions = findall("calculating FatBand for Element: (.*) Orbital\(s\):\s*.*", lobsterout)
  
        # Skipping the first lines of header
        iline = 5

        header = [float(x) for x in data[iline].strip().split()]
        ndos = int(header[2])
        iline += 1

        total_dos = [[float(x) for x in y.split()] for y in data[iline:iline + ndos]]
        total_dos = np.array(total_dos)

        iline += ndos
        ndos, total_ncols = total_dos.shape
        is_spin_polarized = False
        if total_ncols == 5:
            is_spin_polarized = True
            spins = ['dos_up','dos_down']
        # In case there are more lines of data, they are the projected DOS
        if len(data) > iline:

            projected_dos = []
            proj_orbitals = []
            ion_index =0
            
            while iline < len(data):
       
                header = [float(x) for x in data[iline].split(";")[0].split()]
                
                #print(header)
  
                proj_orbitals.append((proj_ions[ion_index],data[iline].split(";")[2]))
                #print(proj_orbitals)
                ion_index += 1    
                
                
                ndos = int(header[2])
                iline += 1
                tmp_dos = [[float(x) for x in y.split()] for y in data[iline:iline + ndos]]
                tmp_dos = np.array(tmp_dos)
                
                projected_dos.append(tmp_dos)
               
                iline += ndos
                
            final_projected = []
            
            for i_ion in range(len(projected_dos)):
                tmp_dos = np.zeros(shape = [len(projected_dos[i_ion][:,0]),10,2])
                for ilabel, label in enumerate(proj_orbitals[i_ion][1].split(),1): 
                    if is_spin_polarized == False :
                        tmp_dos[:,0,0] = projected_dos[i_ion][:,0]
                        if (label.find('s') == True):
                            tmp_dos[:,1,0] += projected_dos[i_ion][:,ilabel] 
                        elif(label.find('p_y') == True):
                            tmp_dos[:,2,0] += projected_dos[i_ion][:,ilabel]
                        elif(label.find('p_z') == True):
                            tmp_dos[:,3,0] += projected_dos[i_ion][:,ilabel]
                        elif(label.find('p_x') == True):
                            tmp_dos[:,4,0] += projected_dos[i_ion][:,ilabel]
                        elif(label.find('d_xy') == True):
                            tmp_dos[:,5,0] += projected_dos[i_ion][:,ilabel]
                        elif(label.find('d_yz') == True):
                            tmp_dos[:,6,0] += projected_dos[i_ion][:,ilabel]
                        elif(label.find('d_z^2') == True):
                            tmp_dos[:,7,0] += projected_dos[i_ion][:,ilabel]
                        elif(label.find('d_xz') == True):
                            tmp_dos[:,8,0] += projected_dos[i_ion][:,ilabel]
                        elif(label.find('d_x^2-y^2') == True):
                            tmp_dos[:,9,0] += projected_dos[i_ion][:,ilabel]
                    else:
                        tmp_dos[:,0,0] = projected_dos[i_ion][:,0]
                        tmp_dos[:,0,1] = projected_dos[i_ion][:,0]
                        if (label.find('s') == True):
                            tmp_dos[:,1,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,1,1] += projected_dos[i_ion][:,2*ilabel]
                        elif(label.find('p_y') == True):
                            tmp_dos[:,2,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,2,1] += projected_dos[i_ion][:,2*ilabel]
                        elif(label.find('p_z') == True):
                            tmp_dos[:,3,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,3,1] += projected_dos[i_ion][:,2*ilabel]
                        elif(label.find('p_x') == True):
                            tmp_dos[:,4,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,4,1] += projected_dos[i_ion][:,2*ilabel]
                        elif(label.find('d_xy') == True):
                            tmp_dos[:,5,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,5,1] += projected_dos[i_ion][:,2*ilabel]
                        elif(label.find('d_yz') == True):
                            tmp_dos[:,6,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,6,1] += projected_dos[i_ion][:,2*ilabel]
                        elif(label.find('d_z^2') == True):
                            tmp_dos[:,7,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,7,1] += projected_dos[i_ion][:,2*ilabel]
                        elif(label.find('d_xz') == True):
                            tmp_dos[:,8,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,8,1] += projected_dos[i_ion][:,2*ilabel]
                        elif(label.find('d_x^2-y^2') == True):
                            tmp_dos[:,9,0] += projected_dos[i_ion][:,2*ilabel-1]
                            tmp_dos[:,9,1] += projected_dos[i_ion][:,2*ilabel]
                final_projected.append(tmp_dos)
                final_labels_index = {'energies':None,'s':0,'p_y':1,'p_z':2, 'p_x':3 , 'd_xy': 4,  'd_yz': 5, 'd_z^2': 6, 'd_xz':7,'d_x^2-y^2':8}
                final_labels = list(final_labels_index.keys())        
                        
            
           
            return {'total': total_dos, 'projected': final_projected, 'projected_labels_info':final_labels , 'ions':ionsList}

        else:
            
            return {'total': total_dos}

    # @staticmethod
    # def parse_doscar(filename):

    #     if not os.path.isfile(filename):
    #         raise ValueError('ERROR: DOSCAR file not found')

    #     rf = open(filename)
    #     data = rf.readlines()
    #     rf.close()

    #     rf = open("lobsterout", "r")
    #     lobsterout = rf.read()
    #     rf.close()
        

    #     if len(data) < 5:
    #         raise ValueError('DOSCAR seems truncated')

    #     ionsList = findall("calculating FatBand for Element: (.*) Orbital.*", lobsterout)

      
    #     proj_ions = findall("calculating FatBand for Element: (.*) Orbital\(s\):\s*.*", lobsterout)
  


    #     # Skipping the first lines of header
    #     iline = 5

    #     header = [float(x) for x in data[iline].strip().split()]
    #     ndos = int(header[2])
    #     iline += 1

    #     total_dos = [[float(x) for x in y.split()] for y in data[iline:iline + ndos]]
    #     total_dos = np.array(total_dos)

    #     iline += ndos

    #     # In case there are more lines of data, they are the projected DOS
        
    #     if len(data) > iline:
    #         projected_dos = []
    #         proj_orbitals = []
    #         ion_index =0
            
    #         while iline < len(data):
       
    #             header = [float(x) for x in data[iline].split(";")[0].split()]
                
    #             #print(header)
  
    #             proj_orbitals.append((proj_ions[ion_index],data[iline].split(";")[2]))
    #             #print(proj_orbitals)
    #             ion_index += 1    
                
                
    #             ndos = int(header[2])
    #             iline += 1
    #             tmp_dos = [[float(x) for x in y.split()] for y in data[iline:iline + ndos]]
    #             tmp_dos = np.array(tmp_dos)
                
    #             projected_dos.append(tmp_dos)
               
    #             iline += ndos
    #         #[int(s) for s in str.split() if s.isdigit()]
    #         final_projected = []
    #         principal_list = []
    #         for ion in proj_orbitals:
    #             principalNum = [int(s) for s in ion[1] if s.isdigit()]
    #             for n in principalNum:
    #                 if n not in principal_list:
    #                     principal_list.append(n)
    #         print(principal_list)
    #         return {'total': total_dos, 'projected': projected_dos, 'proj_labels_info':proj_orbitals, 'ions':ionsList,'principal_num':principal_list }
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        #if len(data) > iline:

        #     projected_dos = []
        #     proj_orbitals = []
        #     ion_index =0
        #     while iline < len(data):
       
        #         header = [float(x) for x in data[iline].split(";")[0].split()]
                
        #         #print(header)
  
        #         proj_orbitals.append((proj_ions[ion_index],data[iline].split(";")[2]))
        #         #print(proj_orbitals)
        #         ion_index += 1    
                
                
        #         ndos = int(header[2])
        #         iline += 1
        #         tmp_dos = [[float(x) for x in y.split()] for y in data[iline:iline + ndos]]
        #         tmp_dos = np.array(tmp_dos)
                
        #         projected_dos.append(tmp_dos)
               
        #         iline += ndos
            #self.test = proj_orbitals
            
            
           
        #     return {'total': total_dos, 'projected': projected_dos, 'proj_labels_info':proj_orbitals, 'ions':ionsList}

        # else:

        #     return {'total': total_dos}
        
