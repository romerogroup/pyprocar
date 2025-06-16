"""General lattice related utilities.

-distances(positions, lattice=None, verbose=False):
 calculates the PBC-aware distances among all positions.


add RDF here

"""
import numpy as np
from . import db
from . import rdf
from .poscarUtils import poscar_supercell
from .poscar import Poscar
from typing import Union, List
np.set_printoptions(precision=4, linewidth=160, suppress=True)

import copy

def distances(positions:np.ndarray,
              lattice: Union[np.ndarray, None] = None,
              allow_self:bool=True,
              full_pbc_matrix:bool=False,
              verbose:bool=False) -> np.ndarray: 
  """Calculates all the pasirwise distances. The `positions` have to be
  in cartesian coordinates, size Nx3. The lattice should be a 3x3
  array-like.

  Parameters
  ----------

  positions : np.ndarray
      Cartesian positions, shape (N,3).
  lattice: Union[np.ndarray, None]
      3x3 lattice array. Default is None, a non-periodic system.
  allow_self : bool = True
      If True an atom can be its own neighbor, but in another lattice. Default is False
  full_pbc_matrix : bool + False
      If True, and lattice != None, it returns the whole distance matrix, 
      including all distances with neighbor cells. For instance, one atom 
      in a FCC lattice should be 12 times its own neighbor. It is pretty big.
  verbose : bool
      verbosity. Default is False

  Returns
  -------

  np.ndarray:
      NxN array with distances. The main diagonal will be zero unless you set `allow_self=True`

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
    
  # no further processing if `full_pbc_matrix` is set
  if full_pbc_matrix == True:
    return np.array(d)
  
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
  """Class to detect neighbors of each atom.  the neighbors are stored
    in `self.nn_list`, and the corresponding elements in
    `self.nn_elem`.

  Methods
  -------

  estimateMaxBondDist

  """

  def __init__(self, poscar:Poscar,
               verbose:bool=False,
               custom_nn_dist:dict = None,
               customDB:dict = None,
               FCC_Scaling:bool = True,
               RDF:bool = True):
    """
    It estimates the nearest neighbors of a poscar-class. They are element-dependent!

    Parameters
    ----------
    
    poscar : Poscar
        A poscar instance, to claculate its neighbors
    verbose : bool
        verbosity. Defaults to False
       custon_nn_dist: dict
        Allows the user to input custom cutoff distances values for determining NN's 
    customDB : dict
        Allows the user to input custom atomic radii for the data base as a dictionary eg: {"H":0.31}
        Defaults to None
    FCC_Scaling : bool
        If True, re-scales to allow intermidiate distances between atoms (FCC-like)
        Defaults to True
    RDF : bool
        If True, allows for the radial density function (RDF) to assist in the calculation of Neighbors
        Defaults to True

    """
    
    self.poscar = poscar
    self.verbose = verbose
    self.nn_list = None # a list of N lists with neighbor indexes
    self.nn_elem = None # the atomic elements of the nn_list

    self.custom_nn_dist = custom_nn_dist

    if(customDB != None):
      self.db = db.DB(customRadii=customDB)
    else:
      self.db = db.atomicDB # database with atomic info
    self.fcc = FCC_Scaling
    self.rdf = RDF

    self.distances = None # set when needed
    
    # Maximum distance of a nearest neighbor NxN matrix
    self.d_Max = None # maximum distance for a nearest neighbor (a
                      # dict, for all interactions)
    self.estimateMaxBondDist()
    self.nn_list = self.set_neighbors()
    
    return

  def estimateMaxBondDist(self)-> np.ndarray:
    """Based on the covalent radii, it estimates (crudely) the maximum
    distance to be regarded as a first neighbor.

    The covalent bond distance d0 is:
    d0 = radii_1 + radii_2

    The maximum bond distance is defined as:
    d_Max = (1+sqrt(2))/2 * d0
    
    which is half way between the first and second neighbors in a FCC lattice

    Returns
    -------

    np.ndarray:
        Natoms x Natoms matrix with d_Max for each pair of atoms.

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
    
    
    if(self.custom_nn_dist != None):
      #Clean max_dists values
      for key,value in max_dist.items():
        max_dist[key] = 0
      #Replace with informed values
      for interaction , value in self.custom_nn_dist.items():
        elems = interaction.split('-')
        max_dist[elems[0]+elems[1]] = value
        max_dist[elems[1]+elems[0]] = value


      d_Max = [[max_dist[x+y] for x in elements] for y in elements]

      d_Max = np.array(d_Max)

      self.d_Max = d_Max

      return  self.d_Max

    if self.verbose:
      print('Estimated covalent radius (not maximum yet) ', max_dist)
    
    d_Max = [[max_dist[x+y] for x in elements] for y in elements]
    # rescaling to allow intermediate distances (FCC-like)

    if(self.fcc):
      d_Max = np.array(d_Max)*(1+np.sqrt(2))/2


    d_Max = np.array(d_Max)

    self.d_Max = d_Max
    if self.verbose:
      print('Estimation of the Maximum bond length:')
      print(self.d_Max)
    return self.d_Max

  def set_neighbors(self, allow_self : bool = True) -> List[List[int]]:
    """It updates the reference distance for nearest neighbor according to
    the the lower between:

    (i) first minimum of the radial density function (RDF) 
    (ii) estimated value of bonding distance (see `self.estimateMaxBondDist`)

    Then it creates the nearest neighbor list, stored in `self.nn_list`

    Parameters
    ----------

    allow_self : bool
        If True an atom can be its own neighbor, likely to be useless in a large supercell.
        Beware, if the self-distance is zero, this could be troublesome.

    Returns
    -------

    List[List[int]]:
        0-based indexes of nearest neighbors

    """
    self.nn_list = []
    N = self.poscar.Ntotal

    # if a lattice vector is too small, smaller than twice
    # `self.d_Max` from `self.estimateMaxBondDist()`, we should make a
    # supercell before calculating the rdf
    a = b = c = 1
    if np.linalg.norm(self.poscar.lat[0]) < 2*np.max(self.d_Max):
      a = 2
    if np.linalg.norm(self.poscar.lat[1]) < 2*np.max(self.d_Max):
      b = 2
    if np.linalg.norm(self.poscar.lat[2]) < 2*np.max(self.d_Max):
      c = 2
    if a*b*c > 1:
      if self.verbose:
        print('The unit cell is too small, going to use a larger one for the RDF.')
        print('Supercell size: a,b,c = ', a,b,c)
      sposcar = poscar_supercell(self.poscar)
      sposcar = sposcar.supercell(np.diag([a,b,c]))
      my_RDF = rdf.RDF(sposcar)
      
      # the RDF matrix is of the size of the supercell, I need a new
      # one of original cell size
      elements = self.poscar.elm
      # 'H-B' is the same as 'B-H', just compare them sorted and concatenated ('BH')
      interactions = [str(x[0]+x[1]) for x in np.sort(my_RDF.interactions)]
      cutoff_d = my_RDF.neighbor_thresholdSp
      dict_dmax = dict(zip(interactions, cutoff_d))
      d_max = [[dict_dmax[''.join(sorted([x,y]))] for x in elements] for y in elements]
      d_max = np.array(d_max)
    else:
      my_RDF = rdf.RDF(self.poscar)
      d_max = my_RDF.CutoffMatrix

      
    if(self.rdf):
      self.d_Max = np.minimum(d_max, self.d_Max)


      
    if self.verbose:
      print(self.d_Max)

    if allow_self is False:
      self.distances = distances(positions=self.poscar.cpos,
                                 lattice=self.poscar.lat,
                                 full_pbc_matrix=False)
      for i in range(N):
        # to store all neighbors
        temp = []
        for j in range(N):
          if self.distances[i,j] < self.d_Max[i,j]:
            if i!=j:
              temp.append(j)
        self.nn_list.append(temp)

    else:
      self.distances = distances(positions=self.poscar.cpos,
                                 lattice=self.poscar.lat,
                                 full_pbc_matrix=True)
      # print('self.distances.shape', self.distances.shape)
      for i in range(N):
        temp = []
        for j in range(N):
          for k in range(27): # there are 3x3x3 cells.
            if self.distances[k,i,j] < self.d_Max[i,j]:
              if self.distances[k,i,j] > 0:
                temp.append(j)
              elif i != j:
                temp.append(j)
        self.nn_list.append(temp)
        
    self._set_nn_elem()
    if self.verbose:
      print('list of first neighbors:')
      print(list(zip(self.nn_list, self.nn_elem)))
    return self.nn_list
    
  def _set_nn_elem(self):
    """ sets the elements of the list of nearest neighbors.  """
    nn_elem = []
    for nns in self.nn_list:
      temp = [self.poscar.elm[x] for x in nns]
      nn_elem.append(temp)
    self.nn_elem = nn_elem
    
  def _filter_exclusiveSpNeighbors(self):
    #Mask
    elem_mask = copy.deepcopy(self.nn_list)
    #Copy for not modifying
    new_list = copy.deepcopy(self.nn_list)

    #Creating a Mask to only allow for different species neighbors
    for i, elem in enumerate(self.nn_elem):
      for j, index in enumerate(self.nn_list[i]):
        if(self.poscar.elm[i] != self.poscar.elm[index]):
          elem_mask[i][j] = True
        else:
          elem_mask[i][j] = False
    #Filtering
    result = []
    for value, mask in zip(new_list, elem_mask):
      temp = []
      for v,m in zip(value, mask):
        if(m):
          temp.append(v)
      result.append(temp)

    self.nn_list = result
    self._set_nn_elem()
