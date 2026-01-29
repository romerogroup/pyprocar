#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .latticeUtils import Neighbors

if TYPE_CHECKING:
    from .poscar import Poscar

###
### Phys. Rev. Lett. 131, 108001
###


class globalConectivity:
    poscar: Poscar
    Neighbors: Neighbors
    nn_list: list[list[int]]
    N: int
    Laplacian: npt.NDArray[np.float64]
    GC: np.floating[Any]

    def __init__(
        self,
        POSCAR: Poscar,
        custom_nn_dist: dict[str, float] | None = None,
        filter_Neighbors: bool = True,
        custom_Radii: dict[str, float] | None = None,
        fcc_scaling: bool = False,
        rdf: bool = False,
    ) -> None:
        self.poscar = POSCAR
        self.Neighbors = Neighbors(
            self.poscar,
            custom_nn_dist=custom_nn_dist,
            customDB=custom_Radii,
            FCC_Scaling=fcc_scaling,
            RDF=rdf,
        )
        # Re-obtaining neighbors allowing for pbc neighbors
        self.Neighbors.set_neighbors(allow_self=True)
        if filter_Neighbors:
            self.Neighbors._filter_exclusiveSpNeighbors()

        assert self.Neighbors.nn_list is not None
        self.nn_list = self.Neighbors.nn_list
        self.N = len(self.nn_list)  # pyright: ignore[reportConstantRedefinition]
        self.Laplacian = self.getLaplacian()
        self.GC = self.getGC()  # pyright: ignore[reportConstantRedefinition]

    def getGC(self) -> np.floating[Any]:
        eigenvalues_raw = np.linalg.eigvalsh(self.Laplacian)
        # Getting rid of -0 type results
        eigenvalues = [np.abs(x) for x in eigenvalues_raw]
        for index, eigen in enumerate(eigenvalues):
            if eigen <= 10e-4:
                eigenvalues[index] = np.float64(0)
        # Getting LEL
        lel: np.floating[Any] = np.float64(0)
        for eigen in eigenvalues:
            lel += np.sqrt(eigen)
        # Getting l_1
        sorted_eigen = sorted(eigenvalues)
        l_1 = sorted_eigen[1]
        # Getting l_e
        l_e: np.floating[Any] = np.float64(0)
        for eigen in sorted_eigen:
            if eigen != 0:
                l_e = eigen
                break
        omega: np.floating[Any] = (np.sqrt(l_1) + np.sqrt(l_e)) / lel

        return omega

    def getLaplacian(self) -> npt.NDArray[np.float64]:
        # Calculating L
        L = np.zeros((self.N, self.N))
        for u in range(self.N):
            for v in range(self.N):
                if u == v:
                    L[u][v] = 1.0
                elif self._isAdjacent(u, v):
                    times = self.nn_list[u].count(v)
                    d_u = len(self.nn_list[u])
                    d_v = len(self.nn_list[v])
                    if d_u == 0 or d_v == 0:
                        pass
                    else:
                        for _ in range(times):
                            L[u][v] += -(1.0) / (np.sqrt(d_u * d_v))
                else:
                    continue
        return L

    def _isAdjacent(self, i: int, j: int) -> bool:
        if j in self.nn_list[i]:
            return True
        return False
