#!/usr/bin/env python3
from . import latticeUtils
from . import poscar
import numpy as np

###
### Phys. Rev. Lett. 131, 108001 
###


class globalConectivity:

    def __init__(self, POSCAR, custom_nn_dist = None, filter_Neighbors = True, custom_Radii = None, fcc_scaling = False, rdf = False):
        self.poscar = POSCAR
        self.Neighbors = latticeUtils.Neighbors(self.poscar,custom_nn_dist=custom_nn_dist, customDB=custom_Radii, FCC_Scaling=fcc_scaling, RDF = rdf)
        #Re-obtaining neighbors allowing for pbc neighbors
        self.Neighbors.set_neighbors(allow_self = True)
        if(filter_Neighbors):
            self.Neighbors._filter_exclusiveSpNeighbors()
                    
        self.nn_list = self.Neighbors.nn_list
        self.N = len(self.nn_list)
        self.Laplacian = self.getLaplacian()
        self.GC = self.getGC()
        return
    def getGC(self):

        eigenvalues, eigenvectors = np.linalg.eigh(self.Laplacian)
        #Getting rid of -0 type results
        eigenvalues = [np.abs(x) for x in eigenvalues]
        for index, eigen in enumerate(eigenvalues):
            if eigen <= 10E-4:
                eigenvalues[index] = 0
        #Getting LEL
        LEL = 0
        for eigen in eigenvalues:
            LEL += np.sqrt(eigen)
        #Getting l_1
        sorted_eigen = sorted(eigenvalues)
        l_1 = sorted_eigen[1]
        #Getting l_e
        for eigen in sorted_eigen:
            if(eigen != 0):
                l_e = eigen
                break
        Omega = (np.sqrt(l_1) + np.sqrt(l_e))/LEL

        return Omega

    def getLaplacian(self):
        #Calculating L
        L = np.zeros((self.N,self.N))
        for u in range(self.N):
            for v in range(self.N):
                if(u == v):
                    L[u][v] = 1.
                elif(self._isAdjacent(u,v)):
                    times = self.nn_list[u].count(v)
                    d_u = len(self.nn_list[u])
                    d_v = len(self.nn_list[v])
                    if(d_u == 0 or d_v == 0):
                        pass
                    else:
                            for i in range(times):
                                L[u][v] += -(1.)/(np.sqrt(d_u*d_v)) 
                else:
                    continue
        return L

    def _isAdjacent(self, i,j):
        if(j in self.nn_list[i]):
            return True
        else:
            return False



