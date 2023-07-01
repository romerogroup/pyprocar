#!/usr/bin/env python

import numpy as np
from poscar import Poscar

np.set_printoptions(precision=4, linewidth=160, suppress=True)

class Neighbors:
  def __init__(self, poscar=None, verbose=True):
    """ poscar can be the filename or a Poscar file, already parsed"""
    self.verbose = verbose
    self.p = None # a poscar-object
    self.nn_list = None
    self.distances = None
    self.clusters = None # a list of clusters, each cluster has a list with its atoms
    self.nn_dict_clusters = None # a list of dict with the nearest
                                 # neighbors, an dict for each cluster
                                 # getting the poscar object. The keys
                                 # are the atomic index of each atom
                                 # blonging to the cluster
    if isinstance(poscar, str):
      self.p = Poscar(poscar, verbose=False)
      self.p.parse()
      if self.verbose:
        print('Loaded the file:', poscar)
    else:
      self.p = poscar
    
    return
  # there should be severals ways to define nearest neighbors
  def set_neighbors(self, method, parameters):
    """methods should have several posibilities, each `method` has a
respective set of `parameters`. Currently supported:

1) method='distance' (a single cutoff distance) 
   parameters = {'max_dis':1.5,     # maximum distance in Angstroms. No default
                 'allow_self:True'} # can an atom be its own neighbor? Default: False

    """
    # checking whether  the method exists and has the rigth parameters
    if method == 'distance':
      try:
        dCutoff = parameters['max_dis']
      except KeyError:
        raise RuntimeError('the parameter argument doesnt have the key `max_dis`')
      
      d = distances(self.p.cpos, lattice=self.p.lat)
      self.distances = d
      if self.verbose:
        print('distances:')
        print(d)
      N = len(d)
      # setting an empty list to contain the list of neighbors
      self.nn_list = []
      for i in range(N):
        temp = []
        # is j a neighbor of i?
        for j in range(N):
          if d[i,j] <  dCutoff:
            # print(i,j)
            # if they are different atoms, always accept
            if i!=j:
              temp.append(j)
            # if allow_self is True, also accept (False by default)
            elif parameters.get('allow_self', False) is True:
              temp.append(j)
            # otherwise, not a neighbor
        #print('loop over j done, neighbors found:', temp)
        self.nn_list.append(temp)
        
      if self.verbose:
        print('list of first neighbors:')
        print(self.nn_list)
    else:
      raise RuntimeError('set_neighbors() does not support the method ' + str(method))
    return

  def find_clusters(self):
    N = len(self.nn_list)
    if self.verbose:
      print(N, 'atoms')
    # I will start assuming every atom has its own cluster, the
    # interaction will join the sets
    clusters = [set([i]) for i in range(N)]
    nn_list = self.nn_list
    
    # which atoms are connected to atom 0
    for atom in range(N):
      # I need to search for all the pairs of interacctions
      neighbors = nn_list[atom]
      for neighbor in neighbors:
        # print(atom, neighbor)
        # finding the clusters with 'atom' and 'neighbor', removing
        # them and then appending its union
        for c in clusters:
          if atom in c:
            c1 = c
          if neighbor in c:
            c2 = c
        clusters.remove(c1)
        # maybe c1 and c2 are the same (both atoms are already in the
        # same cluster)
        if c2 in clusters:
          clusters.remove(c2)
        clusters.append( c1.union(c2) )
        # print(clusters)
    self.clusters = [list(x) for x in clusters]
    if self.verbose:
      print('clusters:', clusters)
    self._set_nn_clusters()
    return

  def _set_nn_clusters(self):
    # a list of nearest neighbors dict, one per cluster
    nn_dicts = []
    for cluster in self.clusters:
      nn_dict = dict()
      for atom in cluster:
        nn_dict[atom] = self.nn_list[atom]
      nn_dicts.append(nn_dict)
    self.nn_dict_clusters = nn_dicts
    if self.verbose:
      print('clusters found:', len(self.clusters))
      for cluster, nn_dict in zip(self.clusters, self.nn_dict_clusters):
        print('cluster', cluster)
        print('nearest neighbors', nn_dict)
        
    return
  
class Graphene:
  def __init__(self, poscarFile, verbose=False):
    self.verbose = verbose
    if self.verbose:
      print('loading file ', poscarFile)
    self.p = Poscar(poscarFile, verbose=False) # a Poscar class
    self.p.parse()
    self.nn_list = None  # a lsit with the nearest neighbors, just by
                         # distance, ignoring any bonding scheme.
    self.neighbors = None # a `Neighbors class`
    self.clusters = None # a list disjoint group of atoms
    self.sublattices = None # [dict('A':[...], 'B':[...])] one dict
                            # with the sublattices A,B by each
                            # cluster, with the same order
    self.edges = None # a list of edges, one list by each
                      # cluster. [[edge1_cluster1, e2_c2],[e1_c2],[]]
                      # (no edge in cluster3)
    self.bonding = None # It is sp1-bonded C atom?

    # setting all the fast things

    self.set_sublattices()
    self.find_edges()
    
    return
  def set_sublattices(self):
    # getting the nearest neighbors
    n = Neighbors(self.p, verbose=False)
    n.set_neighbors(method='distance', parameters={'max_dis':1.6})
    self.nn_list = n.nn_list
    # if there are N layers of graphene, we need to treat them separately
    n.find_clusters()
    self.clusters = n.clusters
    self.neighbors = n
    
    if self.verbose:
      print('\nclusters found:', len(n.clusters), '\n', n.clusters)

    self.sublattices = []
    # looping over the clusters
    for icluster in range(len(n.clusters)):
      cluster = n.clusters[icluster]
      nn_dict = n.nn_dict_clusters[icluster]
      # two empty sublattices
      sublatticeA = set()
      sublatticeB = set()
      # I need a first element
      current = cluster[0]
      # and it must have a sublattice, lets say `A`
      sublatticeA.add(current)
      # looping over all atoms, if the atom is in a sublattice, then
      # mark its neighbors and remove it from nn_dict. Otherwise ignore
      # it.
      while len(nn_dict) > 0:
        for key in list(nn_dict): # cast to list to allow removing keys during iteration
          if key in sublatticeA:        
            for neighbor in nn_dict[key]:
              sublatticeB.add(neighbor)
            del nn_dict[key]
            #print('key found in sublattice A', key, 'sublattice B is', sublatticeB)
    
          elif key in sublatticeB:
            for neighbor in nn_dict[key]:
              sublatticeA.add(neighbor)
            del nn_dict[key]
            #print('key found in sublattice B', key, 'sublattice A is', sublatticeA)
      if self.verbose:
        print('\nSublatticeA ', sublatticeA)
        print('SublatticeB ', sublatticeB)
      # finally storing the sublattices
      self.sublattices.append({'A':sublatticeA, 'B':sublatticeB})

  def find_edges(self, ignoreH=True):
    self.edges = []
    for icluster in range(len(self.clusters)):
      cluster = self.clusters[icluster]
      # an edge is a C atom with less than 3 nearest neighbors, or an H
      # atom next to a C atom with coor
      edge = [] 
      for i in range(self.p.Ntotal):
        if ignoreH and self.p.elm[i] == 'H':
          pass # ignoring an H atom
        else:
          neighbors = self.nn_list[i]
          counter = 0
          for neighbor in neighbors:
            if self.p.elm[neighbor] != 'H' or ignoreH == False:
              counter += 1
        if counter < 3:
          edge.append(i)
      if self.verbose:
        print('edge:', edge)
      self.edges.append(edge)
    
  def hidrogenate(self, outFile='POSCAR.hydrogenated'):

    newHatoms = []
    for icluster in range(len(self.clusters)):
      for iatom in self.edges[icluster]:
        elem = self.p.elm[iatom]
        neighbors = self.nn_list[iatom]
        # Only undercoordinate C atoms are needed
        if elem in ['C','B','N'] and len(neighbors) < 3:
          print(iatom, elem, neighbors, end=' ')
          # it is easier to attach one H
          if len(neighbors) == 2:
            # I need the positions of the shortest distance within the
            # PBC
            n1, n2 = neighbors[0], neighbors[1]
            # these are the distances within the PBC
            d1 = self.neighbors.distances[iatom, n1]
            d2 = self.neighbors.distances[iatom, n2]
            print('d1', d1, 'd2', d2)
            p0 = self.p.cpos[iatom]
            lat = self.p.lat
            # getting all the periodic disatances, if one of them
            # fits, I will store it
            for i in [-1, 0, 1]:
              for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                  temp_p1 = self.p.cpos[n1] + i*lat[0] + j*lat[1] + k*lat[2]
                  temp_p2 = self.p.cpos[n2] + i*lat[0] + j*lat[1] + k*lat[2]
                  # if it match with the minimum distance, store it
                  if np.abs(np.linalg.norm(p0 - temp_p1) - d1) < 0.01:
                    p1 = temp_p1
                    #print('1:[',i,j,k,']', end=' ')
                  if np.abs(np.linalg.norm(p0 - temp_p2) - d2) < 0.01:
                    p2 = temp_p2
                    #print('2:[',i,j,k,']',end=' ')
            # Now I have the Cartesian position of each neighbor, within the PBC
            r1, r2  = p0 - p1, p0 - p2
            # I only need the angle, normalizing
            r1, r2 = r1/np.linalg.norm(r1), r2/np.linalg.norm(r2)
            # the new position has to be
            r3 = r1+r2
            r3 = r3/np.linalg.norm(r3) * 1.09 # C-H distance 1.09 Ang
            newHatoms.append(r3+p0)

    # I need to add all the newHatoms to poscar
    print(newHatoms)
    for newH in newHatoms:
      self.p.add(newH, element='H', direct=False)
    # writing the data to file
    self.p.write(outFile, direct=False)
  def sublattice_polarization(self):
    import pyprocar
    pro = pyprocar.ProcarParser()
    pro.readFile('PROCAR')
    # just gamma
    spd = pro.spd[0]
    # all bands all atoms, spin 0, just 'tot' orbital value
    # spd = spd[bands, atoms]
    spd = spd[:,0,:,-1]
    # bands with almost no projection at all can bring numerical
    # problems. I will normalize them to have a `tot` value of 1
    spd = (spd.T/spd[:,-1]).T # all bands, just sum of all atoms
    print(spd.shape)
    
    nn_list = self.nn_list
    print('nearest neighbors',nn_list)
    # an empty array
    overlap = 0*spd[:,0]
    counter = 0
    for atom in range(len(nn_list)):
      for neighbor in nn_list[atom]:
        # print('\natom', spd[:,atom])
        # print('neighbor', spd[:,neighbor])
        overlap += spd[:,atom]*spd[:,neighbor]
        # print(overlap)
        counter += 1
          
    import matplotlib.pyplot as plt
    x = np.arange(len(overlap)) + 1
    plt.plot(x, 1/overlap/counter, 'ro--')
    plt.show()
  
if __name__ == '__main__':
  g = Graphene('POSCAR')
  print("Clusters found", len(g.clusters))
  for i in range(len(g.clusters)):
    print('cluster ' +str(i)+ ':')
    print(g.clusters[i])
    print('Sublattices:')
    [print(x, y) for x,y in g.sublattices[i].items() ]
    print('edges:')
    print(g.edges[i])
    
  g.hidrogenate()
  

  
