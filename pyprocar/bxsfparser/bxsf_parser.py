# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:40:39 2020

@author: lllan
"""

import re
import numpy as np


class BxsfParser:
    def __init__(self, infile = "in.bxsf"):
        
        self.bxsf = infile
        
        rf = open(self.bxsf)
        self.data = rf.read()
        rf.close()
        
        self.rec_lattice = None
        self.numBands = None
        self.origin = None 
        self.numPoints = None
        self.bandLabels = None
        self.bandData = None
        
        self.bands = None
        self.parse_bxsf()
        
    def parse_bxsf(self):
        
        self.numBands = int(re.findall("BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n\s*(\d*)", self.data)[0])
        
        self.origin = re.findall("BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n.*\n.*\n(.*)", self.data)[0].split()
        self.origin = np.array([float(x) for x in self.origin])
        
        self.rec_lattice = re.findall("BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n.*\n.*\n.*\n" + 3 * "\s*(.*)\s*\n", self.data)[0]
        self.rec_lattice = np.array([[float(y) for y in x.split()] for x in self.rec_lattice ])

        self.numPoints =re.findall("BEGIN\_BLOCK\_BANDGRID\_3D\n.*\n.*\n.*\n(.*)", self.data)[0]
        self.numPoints = np.array([int(x) for x in self.numPoints.split()])
        
        self.bandLabels = re.findall("BAND\:\s*(.*)", self.data)
        
        
        self.kpoints = np.zeros(shape = [self.numPoints[0]*self.numPoints[1]*self.numPoints[2],3])
        bandDataDim = list(self.numPoints)
        bandDataDim.insert(0,self.numBands)
        self.bandData = np.zeros(shape = bandDataDim)
        self.bandEnergy = np.zeros(shape = [self.numPoints[0]*self.numPoints[1]*self.numPoints[2],self.numBands])

        for iband in range(self.numBands):
            counter = 0
            if (iband == self.numBands - 1):
                
                expression = "BAND\:\s*" + self.bandLabels[iband] + "[\s\S]*(?=END\_BLOCK\_BANDGRID\_3D)"
                self.bands = re.findall(expression, self.data)[0].split()
                self.bands.pop(0)
                self.bands.pop(0)
                self.bands.pop(-1)
                for k in range(self.numPoints[2]):
                    for j in range(self.numPoints[1]):
                        for i in range(self.numPoints[0]):
                            self.bandData[iband,i,j,k] = self.bands[counter]
                            self.bandEnergy[counter, iband] = self.bands[counter]
                            counter += 1
            else:
                expression = "BAND\:\s*" + self.bandLabels[iband] + "[\s\S]*(?=BAND\:\s*"+ self.bandLabels[iband+1]+ ")"
                self.bands = re.findall(expression, self.data)[0].split()
                self.bands.pop(0)
                self.bands.pop(0)
                self.bands = [float(x) for x in self.bands]
                
                counter = 0
                for i in range(self.numPoints[0]):
                    for j in range(self.numPoints[1]):
                        for k in range(self.numPoints[2]):
                            self.bandData[iband,i,j,k] = self.bands[counter]
                            self.bandEnergy[counter, iband] = self.bands[counter]
                            self.kpoints[counter,:] = np.array([i/self.numPoints[0],j/self.numPoints[1],k/self.numPoints[2]])
                            counter += 1
 