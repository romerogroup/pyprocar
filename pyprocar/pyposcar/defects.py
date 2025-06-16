""" 
Defect-related utilities

class
FindDefect
- It tries to identify any defect by statistical means    


"""
from .latticeUtils import Neighbors
from  .poscarUtils import poscar_modify
import copy
from .generalUtils import remove_flat_points
import itertools
import numpy as np
try:
  from sklearn.neighbors.kde import KernelDensity
except:
  from sklearn.neighbors import KernelDensity

from scipy.signal import argrelextrema

import matplotlib.pyplot as plt


class FindDefect:
  """Tries to identify a defect from statistics of the crystal strcuture

  Methods
  -------

  self.__init__(self, poscar, verbose):
      it already invokes the other methods. The results are in `self.defects` and `self.all_defects`
  self.find_forgein_atoms:
      finds atoms from different species (e.g. substitutions)
  self.nearest_neighbors_environment:
      finds atoms with different environment (e.g. edges)
  

  """
  def __init__(self, poscar, verbose=False):
    """It searches for defects atoms, that is atoms which are
    statistically different from the others. This only is valid when
    the simulation cell is big enough to get statistics.


    """
    # avoiding to modify the original poscar
    self.p = copy.deepcopy(poscar)
    self.verbose = verbose
    self.defects = {} # all the defects from the different methods
                      # should be here. It is a dictionary of lists
    self.all_defects = [] # a simple list with all the defects found
    self.nn_elem = [] #a descriptive list of all nearest neighbors clusters
    self.neighbors = Neighbors(self.p, verbose=False)
    self.find_forgein_atoms()
    self.nearest_neighbors_environment()

    ## Work in progress
    #self.local_geometry()
    return
  
  def local_geometry(self):
    #print("Defects", self.defects)
    #print("Defects", self.defects['find_forgein_atoms'])
    #print("Defects", self.defects['nearest_neighbors_environment'])
    #From all the atoms that aren't defects (Have to take them out)
    #Should i take out its neighbors as well? 
    #Compare same cluster types
    #Compare same distances, have to "count" them "C_1-C_2" w "C_1-C_2" not "C_1-C_3"
    defects = self.defects['find_forgein_atoms']
    defects.sort(reverse = True)
    Cluster_types = self.nn_elem
    Neighbor_Class_elements = self.neighbors.nn_list
    D_Matrix = self.neighbors.distances
    #We get rid of defects already found
    for i in defects:
      Cluster_types.pop(i)
      Neighbor_Class_elements.pop(i)
    Total_ClusterDistanceMatrix = []
    #Creates distances Matrix, with each atom and its distances
    #Using the order [(a,a)(a,b)(a,c)],[(b,a),(b,b),(b,c)...] all sorted (so that (a,a) and (b,b) are always the first entries)
    #This should count all distances inside the clusters in the same order
    for atom in Neighbor_Class_elements:
      index = atom
      all_index = list(itertools.product(index,index))
      grouped_index = []
      group = []
      counter = 0
      #Grouping indexes like "[(a,a), (a,b), (a,c)...][(b,a),(b,b),(b,c),...]""
      for value in all_index:
        if(value[0] == index[counter]):
          group.append(value)
        else:
          grouped_index.append(group)
          group = []
          group.append(value)
          counter += 1
      grouped_index.append(group)
      ClusterDistanceMatrix = []
      #Takes the sorted distances
      for distance in grouped_index:
        Cluster = [D_Matrix[x] for x in distance]
        Cluster.sort()
        ClusterDistanceMatrix.append(Cluster) 
      Total_ClusterDistanceMatrix.append(ClusterDistanceMatrix)
    #Now I need to compare these distances accordingly 
    #Comparing each cluster type with itself
    print("Cluster types ", self.nn_elem)
    Cluster_set = list(set(self.nn_elem))
    # print("Cluster set ", Cluster_set)
    Cluster_group_index = []
    for group in Cluster_set:
      indexes = []
      for idx, cluster in enumerate(self.nn_elem):
        if(cluster == group):
          indexes.append(idx)
      Cluster_group_index.append(indexes)
    print("Group Cluster Indexes ", Cluster_group_index)
    #Comparing
    #I want to use a delta matrix which cointains all distance matrix differences
    #Do some "Machine Learning" with this delta matrix and get a common value for each type of cluster
    #Use this metric to decide if a geometrical defect exists
    Geom_defects = {}
    Total_ClusterDistanceMatrix = np.array(Total_ClusterDistanceMatrix)
    #Here will be all norms 
    Delta_General_Norms = []
    #Here will be all norms separated by type of cluster
    All_type_norms = []
    # print("Cluster_total ", Total_ClusterDistanceMatrix)
    for type, idxs in zip(Cluster_set, Cluster_group_index):
      Delta_type_Norms = []
      idx_to_compare = itertools.combinations(idxs, 2)
      for i,j in idx_to_compare:
        delta = Total_ClusterDistanceMatrix[i] - Total_ClusterDistanceMatrix[j]
        #Maybe better to use the norm of this matrix than the whole matrix 
        delta_norm = np.linalg.norm(delta)
        Delta_type_Norms.append(delta_norm)
        Delta_General_Norms.append(delta_norm)
        
        if(delta_norm > 0.1):
          print("Geometrical Defect Found")
      
      All_type_norms.append([type,Delta_type_Norms])
    
    for type, norm in All_type_norms:
      if len(norm) == 1:
        continue
      #May need to add more exceptions when a certain type of Cluster is found in low amounts
      #Might consider just evoiding Cluster types that include defect atoms even if its not the center atom
      norm_array = np.array(norm)
      norm_ml = norm_array.reshape(-1,1)
      kde = KernelDensity(kernel='gaussian',bandwidth=3).fit(norm_ml)
      #Dont remember the idea of Delta in last implementation
      samples = np.linspace(np.min(norm_array)*0.9, np.max(norm_array)*1.1)
      scores = kde.score_samples(samples.reshape(-1,1))
      samples, scores = remove_flat_points(samples, scores)
      maxima = argrelextrema(scores, np.greater)[0]
      minima = argrelextrema(scores, np.less)[0]
      print("Cluster Tipo ", type)
      print("Promedio" , np.average(norm_array))
      print("Desviación estandar", np.std(norm_array))
      print("Max", np.max(norm_array))
      print("Promedio metodo con KDE", samples[maxima])
      print("Minimo KDE", samples[minima])
      # plt.plot(samples,scores)
      # plt.show()
      

      

    #print("All norms", All_type_norms)
    #print("Neighbor Class element ", self.neighbors.nn_list)
    #print("Distance Matrix ", self.neighbors.distances)



    

    
    
  

  def _set_all_defects(self):
      """
      Updates the `self.all_defects` list from `self.defects`. It
      should be run after any update to `self.defects`
      """
      indexes = list(self.defects.values())
      indexes = [set(x) for x in indexes]
      indexes = list(set.union(*indexes))
      self.all_defects = indexes

  def find_forgein_atoms(self):
    numberSp = self.p.numberSp # number of atoms per element
    if len(set(numberSp)) > 1:
      if self.verbose:
        print("\nFindDefect.find_forgein_atoms()")
        print("Number of atoms per element", self.p.numberSp)
    else:
      self.defects['find_forgein_atoms'] = []
      return

    # If there are just two atom types both have a comparable amount,
    # just ignore them and return
    if len(set(numberSp)) == 2 and max(numberSp)/min(numberSp) <= 2.0:
      if self.verbose:
        print('Two atom types with similar ratio, returning ')
      self.defects['find_forgein_atoms'] = []
      return
        
    
    # reshaping the data for machine learning
    numberSp = np.array(numberSp)
    numberSp = numberSp.reshape(-1, 1)
    # print(numberSp)
    #
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(numberSp)
    # The samples are chosen to have a `max-min-max` pattern (maybe
    # with extra -min-max blocks)
    delta = max(int(self.p.Ntotal*0.1),10) # to have a local maximum at start/end
    samples = np.linspace(-delta, max(numberSp.flatten())+delta)
    scores = kde.score_samples(samples.reshape(-1,1))
    samples, scores = remove_flat_points(samples, scores)
    # print(scores)
    # plt.plot(samples, scores)
    # plt.show()
    #
    # The local minima of the scores denotes the groups. argrelextrema
    # returns a tuple, only first entry is useful
    minima = argrelextrema(scores, np.less)[0]
    # print(argrelextrema(scores, np.less))
    maxima = argrelextrema(scores, np.greater)[0]
    if self.verbose:
      print('local max:',maxima,'  localmin:', minima)
    if len(maxima) <= len(minima):
      print('Maxima, ', maxima)
      print('Minima, ', minima)
      raise RuntimeError('FindDefect.find_forgein_atoms error: '
                         'the local min/max doesnt follows '
                         'the expected order')
    # The threshlod to determine if an atom is forgein.
    try: # perhaps there is no minumum
      lower_min = minima[0]
    except IndexError:
      if self.verbose:
        print('\n\ndefects.FindDefect.find_forgein_atoms(): No defect found')
      self.defects['find_forgein_atoms'] = []
      self._set_all_defects()
      return
    # likely only the smallest cluster of atoms are defects, but if
    # there are three or more cluster, I am not so sure, and the user
    # should be warned
    if len(minima) > 1: # printing regardless verbosity
      print("\n\nWARNING: in FindDefect.find_forgein_atoms() more than "
            "two sets of atoms were found. Cluster delimited by "
            "`minima`= ", minima, ', `maxima=`', maxima)
      print('Only elements with less than ', lower_min, 'atoms are regarded as defects')
    
    defect_elements = []
    # detecting what elements are defects
    for (natoms, element) in zip(self.p.numberSp, self.p.typeSp):
      if self.verbose:
        print('natoms,', natoms, 'element,', element)
      if natoms <= lower_min:
        defect_elements.append(element)
    defects = []
    for i in range(len(self.p.elm)):
      if self.p.elm[i] in defect_elements:
        defects.append(i)

    if self.verbose:
      print('list of defects: ')
      print([(i,self.p.elm[i]) for i in defects])
    self.defects['find_forgein_atoms'] = defects
    self._set_all_defects()
    return defects
    
  def nearest_neighbors_environment(self):
    """This method looks for atoms with an statistically different
    environment (nearest neighbors).

    The enviornment of each atom (i.e. the number and type of
    elements) are compared, and those statiscally different from the
    rest are dubbed as defects.

    A good nearest neighbors list is a must for this method.

    """
    # self.verbose = True
    nn_elem = self.neighbors.nn_elem
    # Building a single string with the environment, it needs to be
    # sorted, for taking statistics
    nn_elem = [''.join(sorted(x)) for x in nn_elem]
    
    # Assume in hBN a B->N defect, its environment is NNN, which seems
    # fine, but for a B atom, not when surrounding a N. This means
    # that the atom at which its environment is being proccesed also
    # matters. And it can be distinguihed from its environment (no
    # sorting)
    
    
    #I need to save this as a class var
    #For cluster comparison
    nn_elem = [x[0]+x[1] for x in zip(self.p.elm, nn_elem)]
    self.nn_elem = nn_elem
    # counting the frequency of unique elements
    from collections import Counter
    uniques = Counter(nn_elem)
    if self.verbose:
      print('\nFindDefect.nearest_neighbors_environment()')
      print('Atomic environments and their frequency:')
      print(list(zip(uniques.keys(), uniques.values())))
    # now determining which of them are defects
    
    data = np.array(list(uniques.values())).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(data)
    # The samples are chosen to have a `max-min-max` pattern (maybe
    # with extra -min-max blocks)
    delta = max(int(self.p.Ntotal*0.1),10) # to have a local maximum at start/end
    samples = np.linspace(-delta, max(data.flatten())+delta)
    scores = kde.score_samples(samples.reshape(-1,1))
    # print(scores)
    #
    # The local minima of the scores denotes the groups. argrelextrema
    # returns a tuple, only first entry is useful
    minima = argrelextrema(scores, np.less)[0] 
    maxima = argrelextrema(scores, np.greater)[0]

    if self.verbose:
      print('local max:',maxima,'  localmin:', minima)
    if len(maxima) <= len(minima):
      print('Maxima, ', maxima)
      print('Minima, ', minima)
      raise RuntimeError('FindDefect.nearest_neighbors_environment error: '
                         'the local min/max doesnt follows '
                         'the expected order')
    # The threshlod to determine if an atom is forgein.
    try:
      lower_min = minima[0]
    except IndexError:
      if self.verbose:
        print('\n\ndefects.FindDefect.nearest_neighbors_environment(): No defect found')
      self.defects['nearest_neighbors_environment'] = []
      self._set_all_defects()
      return

    # likely only the atoms with an environment less abundant than
    # `lower_min` are to be regarded as defects. But if there are
    # three or more statistically different types of environment, the
    # user should be warned 
    if len(minima) > 1: # printing regardless verbosity
      print("\n\nWARNING: in FindDefect.nearest_neighbors_environment() more than "
            "two sets of atoms were found. Cluster delimited by "
            "`minima`= ", minima, ', `maxima=`', maxima)
      print('Only elements with environments less abundant than ', lower_min,
            ' are regarded as defects')
    
    defects = []
    # detecting what atoms are defects
    # nn_elem is ['CCC', 'CC' CCH, 'CCH', ...]
    for i in range(len(nn_elem)):
      environment = nn_elem[i]
      if uniques[environment] < lower_min:
        defects.append(i)

    if self.verbose:
      print('list of defects: ')
      print([(i,self.p.elm[i]) for i in defects])
    self.defects['nearest_neighbors_environment'] = defects
    self._set_all_defects()
    return defects

  def write_defects(self, method='any', filename='defects.vasp'):
    """
    Writes a POSCAR file with the defects marked as dummy atoms
    
    `method` can be:
    'find_forgein_atoms' -> see self.find_forgein_atoms()
    'nearest_neighbors_environment' -> see self.nearest_neighbors_environment()
    'any', any method will do

    """
    if method == 'any':
      indexes = self.all_defects
    else:
      indexes = self.defects[method]

    N = len(indexes)
    newElements = ['D']*N
    
    newP = poscar_modify(self.p, verbose=False)
    newP.change_elements(indexes=indexes, newElements=newElements)
    newP.write(filename=filename)
    # print(indexes)
    
