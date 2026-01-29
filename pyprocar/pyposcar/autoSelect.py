from .clusters import Clusters
from .defects import FindDefect
from .poscar import Poscar


class autoPlot:
    # Goals:
    # - get_defect indexes, all of them

    poscar: Poscar
    verbose: bool
    defects: list[int] | None
    clusters: list[list[int]] | None

    def __init__(self, poscar: Poscar, verbose: bool = False):
        self.poscar = poscar
        self.verbose = verbose
        if not self.poscar.loaded:
            self.poscar.parse()
        self.defects = None
        self.clusters = None

    def get_defects(self) -> None:
        d = FindDefect(self.poscar)
        all_defects = d.all_defects
        if self.verbose:
            print(all_defects)
        self.defects = all_defects

    def get_clusters(self) -> None:
        c = Clusters(self.poscar, marked=set(self.defects) if self.defects is not None else None)
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
