from __future__ import annotations

import copy
import warnings
from collections.abc import Set

import numpy as np

from .db import DB
from .latticeUtils import Neighbors
from .poscar import Poscar
from .poscarUtils import poscar_modify


class Clusters:
    p: Poscar
    db: DB
    clusters: list[list[int]]
    verbose: bool
    disable_warning: bool
    neighbors: Neighbors
    marked: set[int]
    initial_marked: set[int]

    def __init__(
        self,
        poscar: Poscar,
        verbose: bool = False,
        neighbors: Neighbors | None = None,
        marked: Set[int] | None = None,
    ) -> None:
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
        self.clusters = []  # it a list of lists. One list by each cluster
        self.verbose = verbose
        self.disable_warning = False
        # calculating the neighbors can be demanding, using them if given
        if neighbors is None:
            self.neighbors = Neighbors(self.p, verbose=False)
        else:
            self.neighbors = neighbors
        if marked is None:
            assert self.p.Ntotal is not None
            self.marked = set(range(self.p.Ntotal))
        else:
            self.marked = set(marked)

        # the initial_marked atoms should not be removed by
        # 'smooth_edges'
        self.initial_marked = set(self.marked)
        if self.verbose:
            print("\n\nclusters.Clusters.__init__():")
            print("atoms marked", self.marked)
        self.find_clusters()

    def find_clusters(self) -> None:
        """It find the clusters (in the crystal's lattice) within the
        "marked" atoms

        """
        # I will start assuming every atom has its own cluster, the
        # interaction will join the sets
        clusters: list[set[int]] = [set([i]) for i in self.marked]
        nn_list = self.neighbors.nn_list
        assert nn_list is not None

        # In the main iteration I will modify the contents of `clusters`,
        # then I need to loop over another, immutable object.

        # which atoms are connected to atom 0
        c1: set[int] = set()
        c2: set[int] = set()
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
            print("clusters.Clusters.find_clusters(), clusters:", self.clusters)
        # self._set_nn_clusters()

    def extend_clusters(self, n: int = 1) -> None:
        """
        It 'marks' the first neighbors of a cluster (ie: they are added to
        the cluster). It is performed `n` times
        """
        nn_list = self.neighbors.nn_list
        assert nn_list is not None
        # the self.marked object is going to be updated, so better to
        # create a new -static- object list(...)
        for i in range(n):
            if self.verbose:
                print("clusters.Clusters.extend_clusters()... Iteration", i)
            for atom in list(self.marked):
                for neighbor in nn_list[atom]:
                    self.marked.add(neighbor)
            self.find_clusters()
            if self.verbose:
                print("clusters.Clusters.extend_clusters()... clusters:", self.clusters)

    def write(self, filename: str) -> None:
        # the poscar object should not be modified
        pu = poscar_modify(copy.deepcopy(self.p), verbose=False)
        # a set with all atoms
        assert pu.p.Ntotal is not None
        to_remove_set = set(list(range(pu.p.Ntotal)))
        to_remove = list(to_remove_set - self.marked)
        if self.verbose:
            print("cluster.Cluster.write() ... atoms to remove")
            print(to_remove)
        pu.remove(to_remove)
        if self.verbose:
            print("cluster.Cluster.write() ... going to write ", filename)
        pu.write(filename)

    def smooth_edges(
        self,
        ignoreH: bool = False,
        coordination: int = 1,
        preserve_original: bool = True,
    ) -> None:
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
        assert nn_list is not None
        assert self.p.elm is not None
        # not removing while iteration
        to_unmark_list: list[int] = []
        if self.verbose:
            print("clusters.Cluster.smooth_edges(): ... looking for undercoordinate edges")
        for i in self.marked:
            # I need to count how many neighbors are marked
            nn = set(nn_list[i])
            cluster_coord = len(nn & self.marked)
            if ignoreH is True and self.p.elm[i] == "H":
                cutoff = 1
            else:
                cutoff = coordination

            if cluster_coord <= cutoff:
                to_unmark_list.append(i)
                if self.verbose:
                    print("atom", i, nn, ". cluster coordination", cluster_coord)
        # The atoms marked as "defects" should not be unmarked, it would
        # change the physics
        if self.verbose:
            print("undercoordinate atoms:", to_unmark_list)
        to_unmark: set[int]
        if preserve_original:
            to_unmark = set(to_unmark_list) - self.initial_marked
        else:
            to_unmark = set(to_unmark_list)
        if self.verbose:
            print("excluding the initial set of marked atoms,")
            print("undercoordinate atoms:", to_unmark)

        self.marked = self.marked - to_unmark
        if self.verbose:
            print("clusters.Clusters.smooth_edges() ... atoms removed (dangling bonds)")
            print(to_unmark)

    def hydrogenate(self, filename: str | None = None) -> Poscar:
        """It replaces a 'non-marked' nearest neighbor by of the lattice by a H atom.

        The angles (directions) of the bonds are the same of the
        underlying lattice, but the distances are scaled to better reflect
        the real distance (inaccurately)

        It returns a `poscar_modify` object with (only) the hydrogented
        cluster

        """
        # first, detect what atoms need to be attached a H atom
        if self.verbose:
            print("\n\nclusters.py -> Clusters.hydrogenate():...")
        # I need to iterate over static elements, so better to use a list
        # instead of a set.
        marked = list(self.marked)
        nn_list = self.neighbors.nn_list
        assert nn_list is not None
        assert self.p.dpos is not None
        assert self.p.cpos is not None
        assert self.p.lat is not None
        assert self.p.elm is not None
        # the nearest neighbors need to be converted to sets. Only for
        # marked atoms.
        nn_set = [set(nn_list[atom]) for atom in marked]
        if self.verbose:
            print("marked atoms and their neigbors")
            print(list(zip(marked, nn_set)))
        # to use '-' marked needs to be a set. Anyways, mutability is not an issue here
        missing_atoms = [x - self.marked for x in nn_set]
        if self.verbose:
            print("missing_atoms", missing_atoms)

        # second, adding the H atoms
        new_H_atoms: list[np.ndarray] = []
        #  Suppose that atoms 1,2 have a common missing neighbor. The
        #  following procedure could have two very close H atoms. I don't
        #  know what to do. But at least print a warning to the user
        missing_used: list[int] = []
        for atom, mas in zip(marked, missing_atoms):
            if len(mas) > 0 and self.verbose:
                print("atom", atom, "missing atom", mas)
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
                delta = p1 - p0
                shift = np.array([0, 0, 0])
                # _shifted = False
                for i in [0, 1, 2]:
                    if delta[i] > 0.5:
                        _shifted = True
                        shift[i] = -1
                    elif delta[i] < -0.5:
                        _shifted = True
                        shift[i] = 1
                # if _shifted:
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
                # if _shifted:
                #   print('shift cart', shift)
                #   print('p1', p1)
                #   print('p0', p0)
                #   print('p1-p0+shift', delta)
                # normalizing the direction delta:
                delta = delta / np.linalg.norm(delta)
                # bond_length
                bond_length = 1.08  # self.db.estimateBond(self.p.elm[atom], self.p.elm[ma])
                new_H_pos = p0 + delta * bond_length
                # print('new_H_pos',new_H_pos )
                # if _shifted:
                #   print('adding H at', atom, p0,p1, new_H_pos)
                # only left to add a 'H' atom to the poscar, and mark it
                new_H_atoms.append(new_H_pos)
        # warning the user if there an missing atom was replaced by two or more H.
        if self.verbose:
            print("going to serach for duplicates (only reporting if found)")
        if len(set(missing_used)) != len(missing_used) and self.disable_warning == False:
            print(
                "\nclusters.Clusters.hydrogrnate(): at least one atom was replaced"
                " more than than once by an H. This could be unphysical. It"
                " requires to be checked by the user "
            )
            print("list of missing atoms replaced by H", missing_used)
            from collections import Counter

            print("missing atoms usage", Counter(missing_used))
            warnings.warn(
                "At least one atom was replaced twice (or more times)"
                " by an H. Check wether this makes sense"
            )
        pu = poscar_modify(copy.deepcopy(self.p), verbose=False)
        # a set with all atoms, then removing all non-marked atoms
        assert pu.p.Ntotal is not None
        to_remove_set = set(list(range(pu.p.Ntotal)))
        to_remove = list(to_remove_set - self.marked)
        pu.remove(to_remove)
        # adding the new H atoms
        [pu.add("H", x, cartesian=True) for x in new_H_atoms]
        if filename:
            pu.write(filename)
        return pu.p
