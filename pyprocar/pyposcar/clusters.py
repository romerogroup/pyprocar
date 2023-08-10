

import numpy as np
from .poscar import Poscar
from .latticeUtils import Neighbors
from  .poscarUtils  import poscar_modify
import copy
from .db import DB
import warnings



class Clusters:
  def __init__(self, poscar, verbose=False, neighbors=None, marked=None):
    """Class to find and define clusters. There are "marked atoms",
    defined by the used, and the idea is to find, create, increase a
    cluster respect to the marked atoms. Always, the marked atoms
    belongs to a host lattice

    `poscar` is an instance of the class `poscar.Poscar`
    `neighbors` is an instance of `latticeUtils.neighbors`
    `marked` what atoms are marked. If absent, it implies all atoms

    """
    # The poscar object shouldn't be modified within this class. It
    # has info about the underlying lattice. Any 'changes' are
    # virtual, in 'self.marked'
    self.p = copy.deepcopy(poscar)
    self.db = DB()
    self.clusters = [] # it a list of lists. One list by each cluster
    self.verbose = verbose
    self.disable_warning = False
    # calculating the neighbors can be demanding, using them if given
    self.neighbors = neighbors
    if self.neighbors is None:
      self.neighbors = Neighbors(self.p, verbose=False)
    if marked is None:
      self.marked = set(range(self.p.Ntotal))
    else:
      self.marked = set(marked)
      
    # the initial_marked atoms should not be removed by
    # 'smooth_edges'
    self.initial_marked = set(self.marked)
    if self.verbose:
      print('\n\nclusters.Clusters.__init__():')
      print('atoms marked', self.marked)
    self.find_clusters()
    return

  def find_clusters(self):
    """It find the clusters (in the crystal's lattice) within the
    "marked" atoms

    """
    # I will start assuming every atom has its own cluster, the
    # interaction will join the sets
    clusters = [set([i]) for i in self.marked]
    nn_list = self.neighbors.nn_list

    # In the main iteration I will modify the contents of `clusters`,
    # then I need to loop over another, immutable object.
    
    # which atoms are connected to atom 0
    for atom in self.marked:
      # I need to search for all the pairs of interacctions
      neighbors = nn_list[atom]
      for neighbor in neighbors:
        # print(atom, neighbor)
        # finding the clusters with 'atom' and 'neighbor', removing
        # them and then appending its union.
        # The las term in the if is to avoid checking: 1<->2, 2<->1
        if neighbor in self.marked and neighbor < atom:
          for c in clusters:
            if atom in c:
              c1 = c
            if neighbor in c:
              c2 = c
          # I am going to append `c1` (perhaps united to `c2`) to
          # `clusters` later
          clusters.remove(c1)
          if c1 != c2:
            clusters.remove(c2)
          clusters.append(c1.union(c2))
          # print(atom, neighbor, clusters)
    self.clusters = [list(x) for x in clusters]
    if self.verbose:
      print('clusters.Clusters.find_clusters(), clusters:', self.clusters)
    # self._set_nn_clusters()
    return

  def extend_clusters(self, n=1):
    """
    It 'marks' the first neighbors of a cluster (ie: they are added to
    the cluster). It is performed `n` times
    """
    # the self.marked object is going to be updated, so better to
    # create a new -static- object list(...)
    for i in range(n):
      if self.verbose:
        print('clusters.Clusters.extend_clusters()... Iteration', i)
      for atom in list(self.marked):
        for neighbor in self.neighbors.nn_list[atom]:
          self.marked.add(neighbor)
      self.find_clusters()
      if self.verbose:
        print('clusters.Clusters.extend_clusters()... clusters:', self.clusters)

  def write(self, filename):
    # the poscar object should not be modified
    pu = poscar_modify(copy.deepcopy(self.p), verbose=False)
    # a set with all atoms
    to_remove = set(list(range(pu.p.Ntotal)))
    to_remove = list(to_remove - self.marked)
    if self.verbose:
      print('cluster.Cluster.write() ... atoms to remove')
      print(to_remove)
    pu.remove(to_remove)
    if self.verbose:
      print('cluster.Cluster.write() ... going to write ', filename)
    pu.write(filename)

    
    
    
  def smooth_edges(self, ignoreH=False, coordination=1, preserve_original=True):
    """It removes all the 'marked' atoms with coordination equal or lower
    than `coordination`. It is useful to invoke after
    Clusters.extend_clusters()

    `ignoreH`: If True, the H atoms with coordination 1 or larger
    won't be unmarked, regardeless `coordination`. Default: False

    `preserve_original`: If True, the atoms marked when the class was
    defined, won't be unmarked, regardless the coordination. For
    instance, the defect wont be touched, only the dangling bonds due
    to clustering

    """
    nn_list = self.neighbors.nn_list
    # not removing while iteration
    to_unmark = []
    if self.verbose:
      print('clusters.Cluster.smooth_edges(): ... looking for undercoordinate edges')
    for i in self.marked:
      # I need to count how many neighbors are marked
      nn = set(nn_list[i])
      cluster_coord = len(nn & self.marked)
      if ignoreH is True and self.p.elm[i] == 'H':
        cutoff = 1
      else:
        cutoff = coordination
        
      if cluster_coord <= cutoff:
        to_unmark.append(i)
        if self.verbose:
          print('atom', i,nn, '. cluster coordination',cluster_coord)
    # The atoms marked as "defects" should not be unmarked, it would
    # change the physics
    if self.verbose:
      print('undercoordinate atoms:', to_unmark)
    if preserve_original:
      to_unmark = set(to_unmark) - self.initial_marked
    if self.verbose:
      print('excluding the initial set of marked atoms,')
      print('undercoordinate atoms:', to_unmark)
    
    self.marked = self.marked - to_unmark
    if self.verbose:
      print('clusters.Clusters.smooth_edges() ... atoms removed (dangling bonds)')
      print(to_unmark)
      

      
  def hydrogenate(self, filename=None):
    """It replaces a 'non-marked' nearest neighbor by of the lattice by a H atom.
    
    The angles (directions) of the bonds are the same of the
    underlying lattice, but the distances are scaled to better reflect
    the real distance (inaccurately)
    
    It returns a `poscar_modify` object with (only) the hydrogented
    cluster

    """
    # first, detect what atoms need to be attached a H atom
    if self.verbose:
      print('\n\nclusters.py -> Clusters.hydrogenate():...')
    # I need to iterate over static elements, so better to use a list
    # instead of a set.
    marked = list(self.marked)
    # the nearest neighbors need to be converted to sets. Only for
    # marked atoms.
    nn_set = [set(self.neighbors.nn_list[atom]) for atom in marked]
    if self.verbose:
      print('marked atoms and their neigbors')
      print(list(zip(marked, nn_set)))
    # to use '-' marked needs to be a set. Anyways, mutability is not an issue here
    missing_atoms = [x - self.marked for x in nn_set]
    if self.verbose:
      print('missing_atoms', missing_atoms)
    
    # second, adding the H atoms
    new_H_atoms = []
    #  Suppose that atoms 1,2 have a common missing neighbor. The
    #  following procedure could have two very close H atoms. I don't
    #  know what to do. But at least print a warning to the user
    missing_used = []
    for atom, mas in zip(marked, missing_atoms):
      if len(mas) > 0 and self.verbose:
        print('atom', atom, 'missing atom', mas)
      for ma in mas:
        # adding the atom `ma` to the list of used neighbors
        missing_used.append(ma)
        # It might happen that the neigbor atom belongs to a different
        # lattice (i.e. [0,0,0.1] and [0,0,0.9]) the H atom should be
        # at [0,0,0.1-delta], not in [0,0,0.1+delta]. Notice, it is
        # not necessary to compare all the 3*3*3 possibilities to get
        # the smallest distance. If the value of any coordinate,
        # |pos1_i-pos2_i|< 1/2, it is in the rigth cell. This is
        # because we are working in a large supercell and the error
        # for wrong PBCs is large too.
        #
        # We will start working in direct cordinates, to get the
        # correct lattice vectors shift. Afterwards, we will add the H
        # atom in cartesian
        p0 = self.p.dpos[atom]
        p1 = self.p.dpos[ma]
        # print('atom', atom, 'missing', ma, 'p0', p0, 'p1', p1)
        # delta is the vector to put the H atom
        delta = p1-p0
        shift = np.array([0,0,0])
        # shifted = False
        for i in [0,1,2]:
          if delta[i] > 0.5:
            shifted = True
            shift[i] = -1
          elif delta[i] < -0.5:
            shifted = True
            shift[i] =  1
        # if shifted:
        #   print('delta (rec)', delta)
        #   print('shift direct',shift)
        # Now that we have the rigth lattice shift, we will aply it to
        # the cartesian positions.
        # following the previous example:
        # p1 = [0,0,.9]
        # p0 = [0,0,.1]
        # p1-p0 = [0,0,.8] -> delta
        # shift = [0,0,-1]
        # p1-p0-shift = [0,0,-.2]
        # now doing the same in cartesian
        p0 = self.p.cpos[atom]
        p1 = self.p.cpos[ma]
        shift = np.dot(shift, self.p.lat)
        delta = p1 - p0 + shift
        # print('p0', p0, 'p1', p1, 'shift', shift, 'delta', delta)
        # if shifted:
        #   print('shift cart', shift)
        #   print('p1', p1)
        #   print('p0', p0)
        #   print('p1-p0+shift', delta)
        # normalizing the direction delta:
        delta = delta/np.linalg.norm(delta)
        # bond_length
        bond_length = 1.08 #self.db.estimateBond(self.p.elm[atom], self.p.elm[ma])
        new_H_pos = p0 + delta*bond_length
        # print('new_H_pos',new_H_pos )
        # if shifted:
        #   print('adding H at', atom, p0,p1, new_H_pos)
        # only left to add a 'H' atom to the poscar, and mark it
        new_H_atoms.append(new_H_pos)
    # warning the user if there an missing atom was replaced by two or more H.
    if self.verbose:
      print('going to serach for duplicates (only reporting if found)')
    if len(set(missing_used)) != len(missing_used) and self.disable_warning == False:
      print('\nclusters.Clusters.hydrogrnate(): at least one atom was replaced'
            ' more than than once by an H. This could be unphysical. It'
            ' requires to be checked by the user ')
      print('list of missing atoms replaced by H', missing_used)
      from collections import Counter
      print('missing atoms usage', Counter(missing_used))
      warnings.warn('At least one atom was replaced twice (or more times)'
                    ' by an H. Check wether this makes sense')
    pu = poscar_modify(copy.deepcopy(self.p), verbose=False)
    # a set with all atoms, then removing all non-marked atoms
    to_remove = set(list(range(pu.p.Ntotal)))
    to_remove = list(to_remove - self.marked)
    pu.remove(to_remove)
    #adding the new H atoms
    [pu.add('H', x, cartesian=True) for x in new_H_atoms]
    if filename:
      pu.write(filename)
    return pu.p
    
          
        
