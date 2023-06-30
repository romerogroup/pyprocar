#!/usr/bin/env python3
import re 
import os

class outcarParser:

    def __init__(self, OUTCAR):
        #May be some important values left adding for predicting user intention
        #Currently Finding: ISPIN, NKPOINTS, ORBTIAL_MAG, NBANDS, EMIN, EMAX

        self.outcar = str(OUTCAR)      
        f = open(self.outcar, "r")
        self.outcar = f.readlines()
        #ISPIN
        self.ISPIN = int(re.findall(r'ISPIN\s*=\s*(\d*)', ''.join(self.outcar))[0])
        print('ISPIN',self.ISPIN)


        #NKPOINTS
        self.NKPOINTS = int(re.findall(r'NKPTS\s*=\s*(\d*)', ''.join(self.outcar))[0])
        print('NKPOINTS',self.NKPOINTS)


        #MAGMOM(?)

        #ORBITAL MAG(?)
        self.ORBITAL_MAG = re.findall(r'ORBITALMAG\s*=\s*(\S*)', ''.join(self.outcar))[0]
        print("ORBITAL_MAG", self.ORBITAL_MAG)


        #NBANDS(?)
        self.NBANDS = int(re.findall(r'NBANDS\s*=\s*(\d*)', ''.join(self.outcar))[0])
        print('NBANDS',self.NBANDS)


        #EMIN EMAX(?) dont know if to reverse the values for bandplots, this values are for DOS
        self.EMIN = int(re.findall(r'EMIN\s*=\s*(-*\d*)', ''.join(self.outcar))[0])
        print('EMIN',self.EMIN)
        self.EMAX = int(re.findall(r'EMAX\s*=\s*(-*\d*)', ''.join(self.outcar))[0])
        print('EMAX',self.EMAX)


    def kpointPlotState(self):
        if(self.NKPOINTS == 0):
            state = None
        if(self.NKPOINTS == 1):
            state = 'atomic'
        
        #Here I should also check if it is a KPOINT mesh or a path
        #Need to know where to find path or mesh if added
        else:
            state = 'parametric'


        return state
    
    def IsMagnetic(self):
        if(self.ISPIN == 0):
            value = False
        if(self.ISPIN == 1):
            value = True    
        return value
