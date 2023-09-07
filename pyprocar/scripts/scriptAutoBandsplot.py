#!/usr/bin/env python

import pyprocar.io as io
from pyprocar.utils.info import orbital_names
from pyprocar.plotter import EBSPlot
from pyprocar.utils import welcome, ROOT
from pyprocar.pyposcar.generalUtils import remove_flat_points
from pyprocar.pyposcar.poscar import Poscar
from pyprocar.pyposcar.defects import FindDefect 
from pyprocar.pyposcar.clusters import Clusters

import numpy as np
import matplotlib.pyplot as plt


try:
    from sklearn.neighbors.kde import KernelDensity
except:
    from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

class AutoBandsPlot:
    def __init__(self, code='vasp', dir='.'):
        self.IPR = None
        self.pIPR = None
        self.parser = io.Parser(code = 'vasp', dir = '.')
        self.ebs = self.parser.ebs
        self.structure = self.parser.structure
        self.kpath = self.parser.kpath
        self.ispin = self.ebs.bands.shape[-1]
        self.bands_up = self.ebs.bands[:,:,0] - self.ebs.efermi
        self.bands_down = None
        if self.ispin == 2:
            self.bands_down = self.ebs.bands[:,:,1] - self.ebs.efermi

        self.IPR = self.ebs.ebs_ipr()
        

        # hard boundaries on the energy, the enegy window must lie
        # within
        self.eBoundaries = self.get_energy_boundaries()
        # Estimation of plotting window
        self.eLim = self.simple_energy_window()
        # Estimation of the window to include bulk states
        self.ipr_threshold = self.ipr_energy_window()

        self.defects = self.get_defects()
        # van der Waals layers perhaps?
        self.clusters = self.get_clusters()

        self.defect_states()
        
        # self.plot()



        
        return
    
    

    def simple_energy_window(self, delta=1.0):
        """Return a energy window within the last occupied / first unoccupied
        state. It considers each spin separately and returns the
        largest interval. The interval is enlarged by adding `delta'
        

        """
        # bands need to have the Fermi energy set to zero
        #
        # Finding the lowest occupied / highest unocuppied  energies
        print(self.bands_up.shape)
        emin_up = []
        emax_up = []
        emin_down = []
        emax_down = []
        # looking for the highest occupied level for each kpoint
        for kpoint in range(self.bands_up.shape[0]):
            b_up = self.bands_up[kpoint]
            emin_up.append(np.max(b_up[b_up<0]))
            emax_up.append(np.min(b_up[b_up>0]))
            if self.ispin == 2:
                b_down = self.bands_up[kpoint]
                emin_down.append(np.max(b_down[b_down<0]))
                emax_down.append(np.min(b_down[b_down>0]))
        emin = min(emin_up)
        emax = max(emax_up)
        if self.ispin == 2:
            emin = min(emin, emin_down)
            emax = max(emax, emax_down)
        print('(without delta) emin, emax', emin, emax)
        # adding a little bit of space to the window
        emax = emax + delta
        emin = emin - delta
        if emin < self.eBoundaries[0]:
            emin = self.eBoundaries[0]
        if emax > self.eBoundaries[1]:
            emax = self.eBoundaries[1]
        print('(with delta) emin, emax', emin, emax)
        return emin, emax


    def get_energy_boundaries(self):
        # what are the maximum energies for each kpoint?
        max_energy_up = np.max(self.bands_up, axis=1)
        # what kpoint as the smaller max energy?
        max_energy_up = np.min(max_energy_up)

        min_energy_up = np.min(self.bands_up, axis=1)
        min_energy_up = np.max(min_energy_up)
        max_energy = max_energy_up
        min_energy = min_energy_up

        if self.ispin == 2:
            max_energy_down = np.max(self.bands_down, axis=1)
            max_energy_down = np.min(max_energy_down)

            min_energy_down = np.min(self.bands_down, axis=1)
            min_energy_down = np.max(min_energy_down)
            max_energy = min(max_energy_up, max_energy_down)
            min_energy = max(min_energy_up, min_energy_down)
        print('Boundaries,' , min_energy, max_energy)
        return min_energy, max_energy
    


    def ipr_energy_window(self):
        # first only spin up
        ipr_up = self.IPR[:,:,0]
        threshold_up = np.percentile(ipr_up, 90)
        threshold = threshold_up
        if self.ispin == 2:
            ipr_down = self.IPR[:,:,1]
            threshold_down = np.percentile(ipr_down, 90)
            threshold = max(threshold_up, threshold_down)
        print('IPR threshold', threshold)

        for kpoint in range(self.bands_up.shape[0]):
            # searching at least one bulk band in valence
            band = self.bands_up[kpoint]
            ipr = ipr_up[kpoint]
            indexes = np.argwhere((band < 0) & (ipr < threshold) )
            max_index = np.max(indexes)
            # print('Highest occupied *bulk* band kpoint, index, energy',
            #       kpoint, max_index, band[max_index])
            emin = band[max_index]
            if emin < self.eLim[0] and emin > self.eBoundaries[0]:
                self.eLim = max(emin-0.5, self.eBoundaries[0]),  self.eLim[1]
            # searching for at least one bulk band in conduction region
            indexes = np.argwhere((band > 0) & (ipr < threshold) )
            min_index = np.min(indexes)
            emax = band[min_index]
            if emax > self.eLim[1] and emax < self.eBoundaries[1]:
                self.eLim = self.eLim[0], min(emax+0.5, self.eBoundaries[1])
                
            if self.ispin == 2:
                # searching at least one bulk band in valence
                band = self.bands_up[kpoint]
                ipr = ipr_up[kpoint]
                indexes = np.argwhere((band < 0) & (ipr < threshold) )
                max_index = np.max(indexes)
                # print('Highest occupied *bulk* band kpoint, index, energy',
                #       kpoint, max_index, band[max_index])
                emin = band[max_index]
                if emin < self.eLim[0] and emin > self.eBoundaries[0]:
                    self.eLim = max(emin-0.5, self.eBoundaries[0]),  self.eLim[1]
                # searching for at least one bulk band in conduction region
                indexes = np.argwhere((band > 0) & (ipr < threshold) )
                min_index = np.min(indexes)
                emax = band[min_index]
                if emax > self.eLim[1] and emax < self.eBoundaries[1]:
                    self.eLim = self.eLim[0], min(emax+0.5, self.eBoundaries[1])
                    
        print('self.eLim (IPR)', self.eLim)
        return threshold

    def get_defects(self):
        self.poscar = Poscar('POSCAR')
        # If loading from other source -such as pychemia do not
        # `parse()` it-, just load the attributes, and set
        # `poscar.loaded = True`
        self.poscar.parse()
        d = FindDefect(self.poscar)
        print('\ndefects:', d.defects)
        # are the defects in a same cluster if extended a little bit?
        c = Clusters(self.poscar, marked = d.all_defects)
        # These are the individual atoms marked as defects. Are part
        # of a single cluster? I just need to add nearest neighbors
        # and test whether they merge. I will do that only twice,
        # otherwise the defects are too separated.
        cluster_0 = c.clusters
        c.extend_clusters(n=1)
        cluster_1 = c.clusters
        c.extend_clusters(n=1)
        cluster_2 = c.clusters
        if len(cluster_2) < len(cluster_1):
            def_cluster = cluster_2
        elif len(cluster_1) < len(cluster_0):
            def_cluster = cluster_1
        else:
            def_cluster = cluster_0
        print('\ndefects:', def_cluster)
        # if all atoms are a single defect, there is no defect
        if len(def_cluster) == 1:
            if len(def_cluster[0]) == self.poscar.Ntotal:
                return []
        print('\ndefects:', def_cluster)
        return def_cluster
        
        
    def get_clusters(self):
        self.poscar = Poscar('POSCAR')
        # If loading from other source -such as pychemia do not
        # `parse()` it-, just load the attributes, and set
        # `poscar.loaded = True`
        self.poscar.parse()
        c = Clusters(self.poscar)
        # only one cluster but it amount the whole cell, there is no cluster.
        if len(c.clusters) == 1:
            if len(c.cluster[0]) == self.poscar.Ntotal:
                return []
        print('clusters', c.clusters)
        return c.clusters
    

    def defect_states(self):
        p_ipr = self.ebs.ebs_ipr_atom()
        print('p_ipr.shape', p_ipr.shape)

        # spin up first
        p_ipr = p_ipr[:,:,:,0]
        for defect in self.defects:
            print('defect', defect)
    
    def plot(self):
        if self.bands_up.shape[0] == 1:
            b_up = np.concatenate((self.bands_up,self.bands_up), axis = 0)
            if self.ispin == 2:
                b_down = np.concatenate((self.bands_down,self.bands_down), axis = 0)
        else:
            b_up = self.bands_up
            if self.ispin == 2:
                b_up = self.bands_up
                
        plt.plot(b_up, '-r')
        if self.ispin == 2:
            plt.plot(b_down, '-b')
        plt.ylim(self.eLim)
        plt.show()
        
    
# # getting the atom resolved IPR
# pIPR = ebs.ebs_ipr_atom()
# shape = pIPR.shape
# if shape[-1] == 2:
#     ispin = 2
#     pIPR_up = pIPR[:,:,:,0]
#     pIPR_down = pIPR[:,:,:,1]
# else:
#     ispin = 1
#     pIPR_up = pIPR[:,:,:,0]
#     pIPR_down = None
# print('pIPR_up', pIPR_up.shape)
# print('Any nan?', np.isnan(pIPR_up).any(), np.isnan(pIPR_down).any())

# # I need an energy window to work. The data from the higher
# # eigenvalues is nonsense (unless exact diagonalization was used in
# # DFT)
# if ispin == 2:
#     b_up = ebs.bands[:,:,0] - ebs.efermi
#     b_down = ebs.bands[:,:,1] - ebs.efermi
# else:
#     b_up = ebs.bands[:,:,0] - ebs.efermi
#     b_down = 0

# e_min_up, e_max_up = get_energy_window(b_up)
# if ispin == 2:
#     e_min_down, e_max_down = get_energy_window(b_down)
# e_min = min(e_min_up, e_min_down)
# e_max = max(e_max_up, e_max_down)

# plt.plot(np.concatenate((b_up, b_up), axis=0), '-r')
# plt.plot(np.concatenate((b_down, b_down), axis=0), '-b')
# plt.ylim(e_min, e_max)
# plt.show()
    
# Looking for pIPR values within the desired energy window
# w_ipr_up = pIPR[b_up<e_max and b_up>e_max]
# print('ipr_window.shape', ipr_window.shape)
# samples = np.linspace(0, np.max(pIPR), 100)
# kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X.reshape(-1, 1))
# scores = kde.score_samples(samples.reshape(-1,1))

# print(scores.shape)
# plt.plot(samples, scores)
# plt.show()

a = AutoBandsPlot()
