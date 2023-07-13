"""General lattice related utilities.

-distances(positions, lattice=None, verbose=False):
 calculates the PBC-aware distances among all positions.


add RDF here

"""
import numpy as np
from . import db
from . import rdf
np.set_printoptions(precision=4, linewidth=160, suppress=True)


def distances(positions, lattice=None, allow_self=True, verbose=False):
  """Calculates all the pasirwise distances. The `positions` have to be
  in cartesian coordinates, size Nx3. The lattice should be a 3x3
  array-like.

  `allow_self`: it allows to an atom to be its own neighbor, in
  another lattice

  return: a NxN array with distances.

  """ 
  
  # I will calculate the whole distance matrix, first I need to expand
  # the arrays by replicating the data
  
  pbc_list = []
  if lattice is not None:
    for i in [-1, 0, 1]:
      for j in [-1, 0, 1]:
        for k in [-1, 0, 1]:
          pbc_list.append(i*lattice[0] + j*lattice[1] + k*lattice[2])
  else:
    pbc_list = [np.array([0,0,0])]
  pbc_list = np.array(pbc_list)
  if verbose:
    print('PBC lists:')
    print(pbc_list)
    
  N = len(positions)
  if verbose:
    print('positions')
    print(positions, positions.shape)
    
  # d is a list of NxN distance-matrices (one per pbc_list)
  d = []
  for vector in pbc_list:
    # rows have the position in the central cell
    rows = positions.repeat(N,axis=0)
    rows.shape = (N, N, 3)
    # columns have position on the extended cells
    npos = positions + vector
    columns = npos.repeat(N,axis=0)
    columns.shape = (N, N, 3)
    columns = np.transpose(columns, axes=(1,0,2))
    if verbose:
      print('rows[0]')
      print(rows[0], rows.shape)
      print('columns[0]')
      print(columns[0], columns.shape)
      # calculating the distance
    d.append( np.linalg.norm(rows-columns, axis=2) )
    
  # lets assume the first distance is the minimum distance
  dist = d.pop(0)
  for matrix in d:
    # np.minimum is element-wise
    dist = np.minimum(dist, matrix)

  # now we are looking for the self-distances in another cell
  if allow_self:
    # I popped one entry of `d` before, but it can't be the nearest
    # self-image (it is shited in [-1,-1,-1], and the nearest
    # neighborh are shifted in only one lattice vector)
    for matrix in d:
      for i in range(N):
        # any non-zero value is a good guess
        if dist[i,i] == 0 and matrix[i,i] != 0:
          dist[i,i] = matrix[i,i]
        # but I want the smallest non-sero value
        elif matrix[i,i] < dist[i,i]:
          dist[i,i] = matrix[i,i]          
    
  if verbose:
    print('distances')
    print(dist, dist.max())
  return dist

class Neighbors:
  def __init__(self, poscar, verbose=False):
    self.poscar = poscar
    self.verbose = verbose
    self.nn_list = None # a list of N lists with neighbor indexes
    self.nn_elem = None # the atomic elements of the nn_list

    self.db = db.atomicDB # database with atomic info
    self.distances = distances(positions=self.poscar.cpos,
                               lattice=self.poscar.lat)
    # Maximum distance of a nearest neighbor NxN matrix
    self.estimateMaxBondDist()
    self.nn_list = self.set_neighbors()
    self.d_Max = None # maximum distance for a nearest neighbor (a
                      # dict, for all interactions)
    return

  def estimateMaxBondDist(self):
    """Based on the covalent radii, it estimates (crudely) the maximum
    distance to be regarded as a first neighbor.

    The covalent bond distance d0 is:
    d0 = radii_1 + radii_2

    The maximum bond distance is defined as:
    d_Max = (1+sqrt(2))/2 * d0
    
    which is half way between the first and second neighbors in a FCC lattice

    It returns a Natoms x Natoms matrix with d_Max for each pair of atoms

    """
    if self.verbose:
      print('Find_neighbors.estimateMaxBondDist:')
    elements = self.poscar.elm
    # removing duplicates, and going to write a dictionary with
    # distances
    nelems = list(set(elements))
    
    if self.verbose:
      print('elements to use:', nelems)
    names = [x+y for x in nelems for y in nelems]
    values = [self.db.estimateBond(x,y) for x in nelems for y in nelems]
    max_dist = dict(zip(names, values))
    if self.verbose:
      print('Estimated covalent radius (not maximum yet) ', max_dist)
    
    d_Max = [[max_dist[x+y] for x in elements] for y in elements]
    # rescaling to allow intermediate distances (FCC-like)
    d_Max = np.array(d_Max)*(1+np.sqrt(2))/2
    self.d_Max = d_Max
    if self.verbose:
      print('Estimation of the Maximum bond length:')
      print(self.d_Max)
    return self.d_Max

  def set_neighbors(self,allow_self=True):
    """setting the nearest neighbors by using self.d_Max as cutoff
    distance

     Arguments: 

     `allow_self`: allows to an atom to be its own neighbor, likely
    to be useless in a large supercell.  Beware, if the
    self-distance is zero, this could be troublesome

     """
    self.nn_list = []
    N = self.poscar.Ntotal
    
    my_RDF = rdf.RDF(self.poscar)
    
    self.d_Max = np.minimum(my_RDF.CutoffMatrix, self.d_Max)
    
    if self.verbose:
      print(self.d_Max)
      
    ### Añadir MIS minimos
    
    
    for i in range(N):
      # to store all defects
      temp = []
      for j in range(N):
        if self.distances[i,j] < self.d_Max[i,j]:
          # The self-neighbors may/maynot be included
          if i!=j:
            temp.append(j)
            # if allow_self is True, also accept
          elif 'allow_self':
            temp.append(j)
      self.nn_list.append(temp)

    self._set_nn_elem()
    if self.verbose:
      print('list of first neighbors:')
      print(list(zip(self.nn_list, self.nn_elem)))
    return self.nn_list
    
  def _set_nn_elem(self):
    """ sets the elements of the list of nearest neighbors  """
    nn_elem = []
    for nns in self.nn_list:
      temp = [self.poscar.elm[x] for x in nns]
      nn_elem.append(temp)
    self.nn_elem = nn_elem
    
