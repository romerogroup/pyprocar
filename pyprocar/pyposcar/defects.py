"""
Defect-related utilities

class
FindDefect
- It tries to identify any defect by statistical means


"""

from __future__ import annotations

import copy
import itertools
from typing import Any

import numpy as np

from .generalUtils import remove_flat_points
from .latticeUtils import Neighbors
from .poscar import Poscar
from .poscarUtils import poscar_modify

try:
    from sklearn.neighbors.kde import KernelDensity  # pyright: ignore[reportMissingModuleSource]
except Exception:
    from sklearn.neighbors import KernelDensity

from scipy.signal import argrelextrema


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

    p: Poscar
    verbose: bool
    defects: dict[str, list[int]]
    all_defects: list[int]
    nn_elem: list[str]
    neighbors: Neighbors

    def __init__(self, poscar: Poscar, verbose: bool = False) -> None:
        """It searches for defects atoms, that is atoms which are
        statistically different from the others. This only is valid when
        the simulation cell is big enough to get statistics.


        """
        # avoiding to modify the original poscar
        self.p = copy.deepcopy(poscar)
        self.verbose = verbose
        self.defects = {}  # all the defects from the different methods
        # should be here. It is a dictionary of lists
        self.all_defects = []  # a simple list with all the defects found
        self.nn_elem = []  # a descriptive list of all nearest neighbors clusters
        self.neighbors = Neighbors(self.p, verbose=False)
        self.find_forgein_atoms()
        self.nearest_neighbors_environment()

        ## Work in progress
        # self.local_geometry()

    def local_geometry(self) -> None:
        defects = self.defects["find_forgein_atoms"]
        defects.sort(reverse=True)
        Cluster_types = list(self.nn_elem)
        assert self.neighbors.nn_list is not None
        assert self.neighbors.distances is not None
        Neighbor_Class_elements: list[list[int]] = list(self.neighbors.nn_list)
        D_Matrix: np.ndarray = self.neighbors.distances
        # We get rid of defects already found
        for i in defects:
            Cluster_types.pop(i)
            Neighbor_Class_elements.pop(i)
        Total_ClusterDistanceMatrix_list: list[list[list[Any]]] = []
        # Creates distances Matrix, with each atom and its distances
        for atom in Neighbor_Class_elements:
            index = atom
            all_index = list(itertools.product(index, index))
            grouped_index: list[list[tuple[int, int]]] = []
            group: list[tuple[int, int]] = []
            counter = 0
            for value in all_index:
                if value[0] == index[counter]:
                    group.append(value)
                else:
                    grouped_index.append(group)
                    group = []
                    group.append(value)
                    counter += 1
            grouped_index.append(group)
            ClusterDistanceMatrix: list[list[Any]] = []
            # Takes the sorted distances
            for distance in grouped_index:
                Cluster: list[Any] = [D_Matrix[x] for x in distance]
                Cluster.sort()
                ClusterDistanceMatrix.append(Cluster)
            Total_ClusterDistanceMatrix_list.append(ClusterDistanceMatrix)
        # Now I need to compare these distances accordingly
        # Comparing each cluster type with itself
        print("Cluster types ", self.nn_elem)
        Cluster_set = list(set(self.nn_elem))
        Cluster_group_index: list[list[int]] = []
        for group_str in Cluster_set:
            indexes: list[int] = []
            for idx, cluster in enumerate(self.nn_elem):
                if cluster == group_str:
                    indexes.append(idx)
            Cluster_group_index.append(indexes)
        print("Group Cluster Indexes ", Cluster_group_index)
        _Geom_defects: dict[str, object] = {}
        Total_ClusterDistanceMatrix: np.ndarray = np.array(Total_ClusterDistanceMatrix_list)
        # Here will be all norms
        Delta_General_Norms: list[np.floating[Any]] = []
        # Here will be all norms separated by type of cluster
        All_type_norms: list[tuple[str, list[np.floating[Any]]]] = []
        for cluster_type, idxs in zip(Cluster_set, Cluster_group_index):
            Delta_type_Norms: list[np.floating[Any]] = []
            idx_to_compare = itertools.combinations(idxs, 2)
            for i, j in idx_to_compare:
                delta = Total_ClusterDistanceMatrix[i] - Total_ClusterDistanceMatrix[j]
                delta_norm: np.floating[Any] = np.linalg.norm(delta)
                Delta_type_Norms.append(delta_norm)
                Delta_General_Norms.append(delta_norm)

                if delta_norm > 0.1:
                    print("Geometrical Defect Found")

            All_type_norms.append((cluster_type, Delta_type_Norms))

        for cluster_type_val, norm_list in All_type_norms:
            if len(norm_list) == 1:
                continue
            norm_array: np.ndarray = np.array(norm_list, dtype=np.float64)
            norm_ml = norm_array.reshape(-1, 1)
            kde = KernelDensity(kernel="gaussian", bandwidth=3).fit(norm_ml)
            samples = np.linspace(float(np.min(norm_array)) * 0.9, float(np.max(norm_array)) * 1.1)
            scores = kde.score_samples(samples.reshape(-1, 1))
            samples, scores = remove_flat_points(samples, scores)
            maxima = argrelextrema(scores, np.greater)[0]
            minima = argrelextrema(scores, np.less)[0]
            print("Cluster Tipo ", cluster_type_val)
            print("Promedio", np.average(norm_array))
            print("Desviación estandar", np.std(norm_array))
            print("Max", np.max(norm_array))
            print("Promedio metodo con KDE", samples[maxima])
            print("Minimo KDE", samples[minima])

    def _set_all_defects(self) -> None:
        """
        Updates the `self.all_defects` list from `self.defects`. It
        should be run after any update to `self.defects`
        """
        index_sets: list[list[int]] = list(self.defects.values())
        merged: set[int] = set()
        for idx_list in index_sets:
            merged.update(idx_list)
        self.all_defects = list(merged)

    def find_forgein_atoms(self) -> list[int] | None:
        assert self.p.numberSp is not None
        assert self.p.Ntotal is not None
        assert self.p.typeSp is not None
        assert self.p.elm is not None
        numberSp = list(self.p.numberSp)
        if len(set(numberSp)) > 1:
            if self.verbose:
                print("\nFindDefect.find_forgein_atoms()")
                print("Number of atoms per element", self.p.numberSp)
        else:
            self.defects["find_forgein_atoms"] = []
            return None

        # If there are just two atom types both have a comparable amount,
        # just ignore them and return
        if len(set(numberSp)) == 2 and max(numberSp) / min(numberSp) <= 2.0:
            if self.verbose:
                print("Two atom types with similar ratio, returning ")
            self.defects["find_forgein_atoms"] = []
            return None

        # reshaping the data for machine learning
        numberSp_arr = np.array(numberSp, dtype=np.float64)
        numberSp_ml = numberSp_arr.reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=3).fit(numberSp_ml)
        # The samples are chosen to have a `max-min-max` pattern (maybe
        # with extra -min-max blocks)
        delta = max(int(self.p.Ntotal * 0.1), 10)  # to have a local maximum at start/end
        samples = np.linspace(float(-delta), float(max(numberSp_arr.flatten()) + delta))
        scores = kde.score_samples(samples.reshape(-1, 1))
        samples, scores = remove_flat_points(samples, scores)
        #
        # The local minima of the scores denotes the groups. argrelextrema
        # returns a tuple, only first entry is useful
        minima = argrelextrema(scores, np.less)[0]
        maxima = argrelextrema(scores, np.greater)[0]
        if self.verbose:
            print("local max:", maxima, "  localmin:", minima)
        if len(maxima) <= len(minima):
            print("Maxima, ", maxima)
            print("Minima, ", minima)
            raise RuntimeError(
                "FindDefect.find_forgein_atoms error: "
                "the local min/max doesnt follows "
                "the expected order"
            )
        # The threshlod to determine if an atom is forgein.
        try:  # perhaps there is no minumum
            lower_min = minima[0]
        except IndexError:
            if self.verbose:
                print("\n\ndefects.FindDefect.find_forgein_atoms(): No defect found")
            self.defects["find_forgein_atoms"] = []
            self._set_all_defects()
            return None
        # likely only the smallest cluster of atoms are defects, but if
        # there are three or more cluster, I am not so sure, and the user
        # should be warned
        if len(minima) > 1:  # printing regardless verbosity
            print(
                "\n\nWARNING: in FindDefect.find_forgein_atoms() more than "
                "two sets of atoms were found. Cluster delimited by "
                "`minima`= ",
                minima,
                ", `maxima=`",
                maxima,
            )
            print("Only elements with less than ", lower_min, "atoms are regarded as defects")

        defect_elements: list[str] = []
        # detecting what elements are defects
        for natoms, element in zip(self.p.numberSp, self.p.typeSp):
            if self.verbose:
                print("natoms,", natoms, "element,", element)
            if natoms <= lower_min:
                defect_elements.append(element)
        defects_found: list[int] = []
        for i in range(len(self.p.elm)):
            if self.p.elm[i] in defect_elements:
                defects_found.append(i)

        if self.verbose:
            print("list of defects: ")
            print([(i, self.p.elm[i]) for i in defects_found])
        self.defects["find_forgein_atoms"] = defects_found
        self._set_all_defects()
        return defects_found

    def nearest_neighbors_environment(self) -> list[int] | None:
        """This method looks for atoms with an statistically different
        environment (nearest neighbors).

        The enviornment of each atom (i.e. the number and type of
        elements) are compared, and those statiscally different from the
        rest are dubbed as defects.

        A good nearest neighbors list is a must for this method.

        """
        assert self.neighbors.nn_elem is not None
        assert self.p.elm is not None
        assert self.p.Ntotal is not None
        # self.verbose = True
        nn_elem_raw: list[list[str]] = self.neighbors.nn_elem
        # Building a single string with the environment, it needs to be
        # sorted, for taking statistics
        nn_elem_sorted: list[str] = ["".join(sorted(x)) for x in nn_elem_raw]

        # Assume in hBN a B->N defect, its environment is NNN, which seems
        # fine, but for a B atom, not when surrounding a N. This means
        # that the atom at which its environment is being proccesed also
        # matters. And it can be distinguihed from its environment (no
        # sorting)

        # I need to save this as a class var
        # For cluster comparison
        nn_elem: list[str] = [x[0] + x[1] for x in zip(self.p.elm, nn_elem_sorted)]
        self.nn_elem = nn_elem
        # counting the frequency of unique elements
        from collections import Counter

        uniques = Counter(nn_elem)
        if self.verbose:
            print("\nFindDefect.nearest_neighbors_environment()")
            print("Atomic environments and their frequency:")
            print(list(zip(uniques.keys(), uniques.values())))
        # now determining which of them are defects

        data = np.array(list(uniques.values()), dtype=np.float64).reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=3).fit(data)
        # The samples are chosen to have a `max-min-max` pattern (maybe
        # with extra -min-max blocks)
        delta = max(int(self.p.Ntotal * 0.1), 10)  # to have a local maximum at start/end
        samples = np.linspace(float(-delta), float(max(data.flatten()) + delta))
        scores = kde.score_samples(samples.reshape(-1, 1))
        #
        # The local minima of the scores denotes the groups. argrelextrema
        # returns a tuple, only first entry is useful
        minima = argrelextrema(scores, np.less)[0]
        maxima = argrelextrema(scores, np.greater)[0]

        if self.verbose:
            print("local max:", maxima, "  localmin:", minima)
        if len(maxima) <= len(minima):
            print("Maxima, ", maxima)
            print("Minima, ", minima)
            raise RuntimeError(
                "FindDefect.nearest_neighbors_environment error: "
                "the local min/max doesnt follows "
                "the expected order"
            )
        # The threshlod to determine if an atom is forgein.
        try:
            lower_min = minima[0]
        except IndexError:
            if self.verbose:
                print("\n\ndefects.FindDefect.nearest_neighbors_environment(): No defect found")
            self.defects["nearest_neighbors_environment"] = []
            self._set_all_defects()
            return None

        # likely only the atoms with an environment less abundant than
        # `lower_min` are to be regarded as defects. But if there are
        # three or more statistically different types of environment, the
        # user should be warned
        if len(minima) > 1:  # printing regardless verbosity
            print(
                "\n\nWARNING: in FindDefect.nearest_neighbors_environment() more than "
                "two sets of atoms were found. Cluster delimited by "
                "`minima`= ",
                minima,
                ", `maxima=`",
                maxima,
            )
            print(
                "Only elements with environments less abundant than ",
                lower_min,
                " are regarded as defects",
            )

        defects_found: list[int] = []
        # detecting what atoms are defects
        # nn_elem is ['CCC', 'CC' CCH, 'CCH', ...]
        for i in range(len(nn_elem)):
            environment = nn_elem[i]
            if uniques[environment] < lower_min:
                defects_found.append(i)

        if self.verbose:
            print("list of defects: ")
            print([(i, self.p.elm[i]) for i in defects_found])
        self.defects["nearest_neighbors_environment"] = defects_found
        self._set_all_defects()
        return defects_found

    def write_defects(self, method: str = "any", filename: str = "defects.vasp") -> None:
        """
        Writes a POSCAR file with the defects marked as dummy atoms

        `method` can be:
        'find_forgein_atoms' -> see self.find_forgein_atoms()
        'nearest_neighbors_environment' -> see self.nearest_neighbors_environment()
        'any', any method will do

        """
        if method == "any":
            indexes = self.all_defects
        else:
            indexes = self.defects[method]

        N = len(indexes)
        newElements = ["D"] * N

        newP = poscar_modify(self.p, verbose=False)
        newP.change_elements(indexes=indexes, newElements=newElements)
        newP.write(filename=filename)
