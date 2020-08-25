# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:50:22 2020

@author: lllan
"""

import os
import numpy as np
from ..core import DensityOfStates
from re import findall, search, match, DOTALL, MULTILINE, finditer, compile, sub


class LobsterDOSParser:
    def __init__(self, filename='DOSCAR.lobster'):

        if not os.path.isfile(filename):
            raise ValueError('ERROR: DOSCAR file not found')

        self.filename = filename

        self.doscar = self.parse_doscar(filename)
        self.ionsList = None
        
        if 'projected' in self.doscar:
            self.has_projected = True
            self.nions = len(self.doscar['projected'])
        else:
            self.has_projected = False
            self.nions = None

        self.ndos, self.total_ncols = self.doscar['total'].shape
        
        self.is_spin_polarized = False
        if self.total_ncols == 5:
            self.is_spin_polarized = True
            self.spins = ['dos_up','dos_down']

#        self.total_dos = self.dos_to_dict['total']
#
#        if self.has_projected:
#            self.projected_dos = self.dos_to_dict['projected']
            
        rf = open('lobsterout', "r")
        self.lobsterout = rf.read()
        rf.close()

    ###########################################################################
    # This section parse for the total density of states and puts it in a 
    #Pychemia Density of States Object
    ###########################################################################
    def _dos_dict(self):

        ret = {'energies': self.doscar['total'][:, 0]}
        
        if self.is_spin_polarized:
            ret['dos_up'] = self.doscar['total'][:, 1]
            ret['dos_down'] = self.doscar['total'][:, 2]
            ret['integrated_dos_up'] = self.doscar['total'][:, 3]
            ret['integrated_dos_down'] = self.doscar['total'][:, 4]
        else:
            ret['dos'] = self.doscar['total'][:, 1]
            ret['integrated_dos'] = self.doscar['total'][:, 2]
        return ret

    def _get_dos_total(self):

        dos_total = self._dos_dict()

        #self.test = dos_total
        return dos_total,list(dos_total.keys())

    @property
    def dos_total(self):
        """
        Returns the total density of states as a pychemia.visual.DensityOfSates object
        """
        dos_total,labels = self._get_dos_total()
        
        return DensityOfStates(np.array([dos_total[x] for x in dos_total]).T, 
                                         title='Total Density Of States',labels=[x.capitalize() for x in labels])
    ###########################################################################
    ###########################################################################
    
    ###########################################################################
    # This section parses for the projected density of states and puts it in a 
    # Pychemia Density of States Object
    ###########################################################################

    def _get_dos_projected(self):
        proj_dos = {}
        dos={}
        
        
        """Makes an ion list"""
        ions_list = self.doscar['ion_labels']
        
        
        """Sorts the array from self.doscar['projected'][ions] to put in newly formed final array contatining all possible projected orbitals"""
        for ion in range(len(ions_list)):
            energies =  self.doscar['projected'][ion][:,0,0]
            dos[ions_list[ion]] = {'energies': energies}
            
            
            if self.is_spin_polarized == True:
                for ispin in range(len(self.spins)):
                    proj_array = np.zeros(shape = (len(self.doscar['projected'][0][:,0,0]), len(self.doscar['projected_labels_info'])-1))
                    #self.test = proj_array
                    """Goes through possible labels in self.doscar['projected'] for a given ion then puts columns in the appropiate index in the in final array"""
                    
                    if ispin == 0:
                        proj_array[:,:] = self.doscar['projected'][ion][:,1:,0]
                        dos[ions_list[ion]][self.spins[ispin]] = proj_array
                    else:
                        proj_array[:,:] = self.doscar['projected'][ion][:,1:,1]
                        dos[ions_list[ion]][self.spins[ispin]] = proj_array
                            
            else:
                proj_array = np.zeros(shape = (len(self.doscar['projected'][0][:,0,0]), len(self.doscar['projected_labels_info'])-1))
                proj_array[:,:] = self.doscar['projected'][ion][:,1:,0]
                dos[ions_list[ion]]["dos_up"] = proj_array
        
        final_labels = self.doscar['projected_labels_info']

        return dos, final_labels


    @property
    def dos_projected(self): 
        """
        Returns the a list of projected density of states as a pychemia.visual.DensityOfSates object
        each element refers to each atom
        """
        ret = []
        ions_list = self.doscar['ion_labels']
        
        dos_projected,label_info = self._get_dos_projected()
        
        
        if dos_projected == None:
            return None

        ndos = len(dos_projected[list(dos_projected.keys())[0]]['energies'])
        
        
        label_info[0] = label_info[0].capitalize()
 
        for iatom in range(len(dos_projected)):

            labels = []
            labels.append(label_info[0])
            
            
            if self.is_spin_polarized == True:
                for orbital in label_info[1:]:
                    labels.append(orbital + '-Up')
                for orbital in label_info[1:]:
                    labels.append(orbital + '-Down')   
            else:  
                for orbital in label_info[1:]:
                    labels.append(orbital)

                    
           
            norbital = int((len(labels)-1)/len(self.spins))
            
            table = np.zeros(shape = (ndos,norbital*len(self.spins)+1))
            table[:,0] = dos_projected[list(dos_projected.keys())[iatom]]['energies']
       

            start = 1
            
            for key in dos_projected[list(dos_projected.keys())[iatom]].keys():
                
                if key == 'energies':
                    continue
                
                end = start + norbital
                table[:,start:end] = dos_projected[list(dos_projected.keys())[iatom]][key]
                start = end 
            
            
            temp_dos = DensityOfStates(table,title='Projected Density Of States %s'%iatom,labels=labels)
            ret.append(temp_dos)
        return ret
 

#    ###########################################################################
#    ###########################################################################
    @property
    def dos_to_dict(self):
        """
        Returns the complete density (total,projected) of states as a python dictionary        
        """
        return {'total':self._get_dos_total(),'projected':self._get_dos_projected()}
  
#    ###########################################################################
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
        if atoms == None :
            atoms = np.arange(self.nions,dtype=int)
        if spin == None :
            spin = [0,1]
        if orbitals == None :
            if len(spin) ==2 :
                orbitals = np.arange((len(projected[0].labels)-1)//2,dtype=int)
            else :
                orbitals = np.arange((len(projected[0].labels)-1),dtype=int)
        if title == None:
            title = 'Sum'
        orbitals = np.array(orbitals)
        if len(spin) ==2:
            labels = ['Energy','Spin-Up','Spin-Down']
            new_orbitals = []
            for ispin in spin :
                new_orbitals.append(list(orbitals+ispin*(len(projected[0].labels)-1)//2))
            orbitals = new_orbitals
        else : 
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
                        
            
           
            return {'total': total_dos, 'projected': final_projected, 'projected_labels_info':final_labels , 'ion_labels':ionsList}

        else:

            return {'total': total_dos}
        

              
        
    @property
    def species(self):
        """
        Returns the species in POSCAR
        """

        raw_speciesList = findall(
            "calculating FatBand for Element: ([a-zA-Z]*)[0-9] Orbital.*",
            self.lobsterout,
        )
        self.speciesList = []
        for x in raw_speciesList:
            if x not in self.speciesList:
                self.speciesList.append(x)
        return self.speciesList
    
    @property
    def symbols(self):
        """
        Returns the initial Structure as a pychemia structure
        """
        symbolsList = findall(
            "calculating FatBand for Element: (.*) Orbital.*", self.lobsterout
        )
        return symbolsList