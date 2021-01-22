# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: Pedram Tavadze
"""

from scipy.interpolate import CubicSpline
import numpy as np


class ElectronicBandStructure:
    def __init__(self,
                 kpoints=None,
                 energies=None,
                 projections=None,
                 structure=None,
                 interpolation_factor=None):
        
        
        self.kpoints = kpoints
        self.energies = energies
        self.projections = projections
        self.structure = structure
            