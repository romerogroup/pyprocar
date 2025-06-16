from typing import List

from .poscar import Poscar
from .defects import FindDefect
from .clusters import Clusters


class autoPlot:
  # Goals:
  # - get_defect indexes, all of them
  def __init__(self, poscar:Poscar,
               verbose:bool = False):
    self.poscar = poscar
    self.verbose = verbose
    if self.poscar.loaded == False:
      self.poscar.parse()
    self.defects = None
    self.clusters = None

  def get_defects(self) -> List[int]:
    d = FindDefect(self.poscar)
    if self.verbose:
      print(d.all_defects)
    self.defects = d.all_defects

  def get_clusters(self) -> List[List[int]]:
    c = Clusters(self.poscar, marked = self.defects)
    # These are the individual atoms marked as defects. Are part of a
    # single cluster? I just need to add nearest neighbors and test
    # whether they merge. I will do that only twice, otherwise the
    # defects are too separated.
    cluster_0 = c.clusters
    c.extend_clusters(n=1)
    cluster_1 = c.clusters
    c.extend_clusters(n=1)
    cluster_2 = c.clusters

    if len(cluster_2) < len(cluster_1):
      self.clusters = cluster_2
    elif len(cluster_1) < len(cluster_0):
      self.clusters = cluster_1
    else:
      self.clusters = cluster_0

  
