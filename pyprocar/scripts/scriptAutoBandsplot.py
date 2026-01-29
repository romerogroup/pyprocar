#!/usr/bin/env python
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from pyprocar.core import ElectronicBandStructure
from pyprocar.core.structure import Structure
from pyprocar.io import Parser
from pyprocar.pyposcar.clusters import Clusters
from pyprocar.pyposcar.defects import FindDefect
from pyprocar.pyposcar.poscar import Poscar
from pyprocar.scripts.scriptBandsplot import bandsplot

logger = logging.getLogger(__name__)


class AutoBandsPlot:
    parser: Parser
    code: str
    ebs: ElectronicBandStructure
    fermi: float
    dirname: str
    structure: Structure | None
    ispin: int
    bands_up: npt.NDArray[np.float64]
    bands_down: npt.NDArray[np.float64] | None
    IPR: npt.NDArray[np.float64]
    pIPR: npt.NDArray[np.float64]
    eBoundaries: tuple[float, float]
    eLim: tuple[float, float]
    ipr_threshold: float
    poscar: Poscar
    defects: list[list[int]]
    clusters: list[list[int]]
    defect_states: list[
        tuple[npt.NDArray[np.intp] | list[object], npt.NDArray[np.intp] | list[object]]
    ]
    cluster_states: list[
        tuple[npt.NDArray[np.intp] | list[object], npt.NDArray[np.intp] | list[object]]
    ]
    defect_clim: list[list[float]]
    cluster_clim: list[list[float]]

    def __init__(
        self,
        code: str = "vasp",
        dirname: str = ".",
        fermi: float | None = None,
        use_cache: bool = False,
    ) -> None:
        self.parser = Parser(code=code, dirpath=dirname)
        self.code = code
        self.ebs = ElectronicBandStructure.from_code(code, dirname, use_cache=use_cache)

        codes_with_scf_fermi = ["qe", "elk"]
        if code in codes_with_scf_fermi and fermi is None:
            logger.info(f"No fermi given, using the found fermi energy: {self.ebs.fermi}")
            fermi = self.ebs.fermi
        elif fermi is None:
            fermi = 0.0
        self.fermi = fermi

        self.dirname = dirname
        self.structure = self.ebs.structure

        bands_prop = self.ebs.bands
        if bands_prop is None:
            msg = "bands property is not set on ElectronicBandStructure"
            raise ValueError(msg)
        bands_array = bands_prop.to_array()

        self.ispin = bands_array.shape[-1]
        self.bands_up = bands_array[:, :, 0] - fermi
        self.bands_down = None
        if self.ispin == 2:
            self.bands_down = bands_array[:, :, 1] - fermi

        ipr_prop = self.ebs.ebs_ipr
        if ipr_prop is None:
            msg = "ebs_ipr property is not set on ElectronicBandStructure"
            raise ValueError(msg)
        self.IPR = ipr_prop.to_array()  # pyright: ignore[reportConstantRedefinition]

        pipr_prop = self.ebs.ebs_ipr_atom
        if pipr_prop is None:
            msg = "ebs_ipr_atom property is not set on ElectronicBandStructure"
            raise ValueError(msg)
        self.pIPR = pipr_prop.to_array()

        print(self.pIPR.shape)
        print(self.IPR.shape)
        #
        # Setting the energy window for plotting
        #

        # hard boundaries on the energy, the enegy window must lie within
        self.eBoundaries = self.get_energy_boundaries()
        # Estimation of plotting window
        self.eLim = self.simple_energy_window()
        # Estimation of the window to include bulk states in conduction and valence regions
        self.ipr_threshold = self.ipr_energy_window()

        #
        # Guessing relevant atoms
        #
        self.poscar = Poscar()
        if self.structure is not None:
            elements_list: list[str] = (
                self.structure.atoms.tolist() if self.structure.atoms is not None else []
            )
            self.poscar.load_from_data(
                direct_positions=self.structure.fractional_coordinates
                if self.structure.fractional_coordinates is not None
                else np.empty((0, 3)),
                lattice=self.structure.lattice if self.structure.lattice is not None else np.eye(3),
                elements=elements_list,
            )

        self.defects = self.get_defects()
        # van der Waals layers perhaps?
        self.clusters = self.get_clusters()
        #
        # correlating defects with electronic structure within the
        # energy window
        #
        self.defect_states = self.find_defect_states(defects=self.defects)
        self.cluster_states = self.find_defect_states(defects=self.clusters)

        # getting an estimation of clim (actually the max value). One
        # value for each defect and cluster
        self.defect_clim = self.get_clim(self.defects)
        self.cluster_clim = self.get_clim(self.clusters)
        print(self.defect_clim)
        print(self.cluster_clim)

        self.write_report(verbosity=False, filename="report.txt")

        self.plot()

    def simple_energy_window(self, delta: float = 1.0) -> tuple[float, float]:
        """Return a energy window within the last occupied / first unoccupied
        state. It considers each spin separately and returns the
        largest interval. The interval is enlarged by adding `delta'


        """
        # bands need to have the Fermi energy set to zero
        #
        # Finding the lowest occupied / highest unocuppied  energies
        emin_up: list[np.floating[Any]] = []
        emax_up: list[np.floating[Any]] = []
        emin_down: list[np.floating[Any]] = []
        emax_down: list[np.floating[Any]] = []
        # looking for the highest occupied level for each kpoint
        for kpoint in range(self.bands_up.shape[0]):
            b_up = self.bands_up[kpoint]
            emin_up.append(np.max(b_up[b_up < 0]))
            emax_up.append(np.min(b_up[b_up > 0]))
            if self.ispin == 2:
                b_down = self.bands_up[kpoint]
                emin_down.append(np.max(b_down[b_down < 0]))
                emax_down.append(np.min(b_down[b_down > 0]))
        emin: float = float(min(emin_up))
        emax: float = float(max(emax_up))
        if self.ispin == 2:
            emin = min(emin, float(min(emin_down)))
            emax = max(emax, float(min(emax_down)))
        # adding a little bit of space to the window
        emax = emax + delta
        emin = emin - delta
        emin = max(emin, self.eBoundaries[0])
        emax = min(emax, self.eBoundaries[1])
        return emin, emax

    def get_energy_boundaries(self) -> tuple[float, float]:
        # what are the maximum energies for each kpoint?
        max_energy_up = float(np.min(np.max(self.bands_up, axis=1)))
        min_energy_up = float(np.max(np.min(self.bands_up, axis=1)))
        max_energy: float = max_energy_up
        min_energy: float = min_energy_up

        if self.ispin == 2:
            assert self.bands_down is not None
            max_energy_down = float(np.min(np.max(self.bands_down, axis=1)))
            min_energy_down = float(np.max(np.min(self.bands_down, axis=1)))
            max_energy = min(max_energy_up, max_energy_down)
            min_energy = max(min_energy_up, min_energy_down)
        return min_energy, max_energy

    def ipr_energy_window(self) -> float:
        # first only spin up
        ipr_up = self.IPR[:, :, 0]
        threshold_up = float(np.percentile(ipr_up, 90))
        threshold: float = threshold_up
        if self.ispin == 2:
            ipr_down = self.IPR[:, :, 1]
            threshold_down = float(np.percentile(ipr_down, 90))
            threshold = max(threshold_up, threshold_down)

        for kpoint in range(self.bands_up.shape[0]):
            # searching at least one bulk band in valence
            band = self.bands_up[kpoint]
            ipr = ipr_up[kpoint]
            indexes = np.argwhere((band < 0) & (ipr < threshold))
            max_index = int(np.max(indexes))
            emin = float(band[max_index])
            if emin < self.eLim[0] and emin > self.eBoundaries[0]:
                self.eLim = max(emin - 0.5, self.eBoundaries[0]), self.eLim[1]
            # searching for at least one bulk band in conduction region
            indexes = np.argwhere((band > 0) & (ipr < threshold))
            min_index = int(np.min(indexes))
            emax = float(band[min_index])
            if emax > self.eLim[1] and emax < self.eBoundaries[1]:
                self.eLim = self.eLim[0], min(emax + 0.5, self.eBoundaries[1])

            if self.ispin == 2:
                # searching at least one bulk band in valence
                band = self.bands_up[kpoint]
                ipr = ipr_up[kpoint]
                indexes = np.argwhere((band < 0) & (ipr < threshold))
                max_index = int(np.max(indexes))
                emin = float(band[max_index])
                if emin < self.eLim[0] and emin > self.eBoundaries[0]:
                    self.eLim = max(emin - 0.5, self.eBoundaries[0]), self.eLim[1]
                # searching for at least one bulk band in conduction region
                indexes = np.argwhere((band > 0) & (ipr < threshold))
                min_index = int(np.min(indexes))
                emax = float(band[min_index])
                if emax > self.eLim[1] and emax < self.eBoundaries[1]:
                    self.eLim = self.eLim[0], min(emax + 0.5, self.eBoundaries[1])

        return threshold

    def get_defects(self) -> list[list[int]]:
        d = FindDefect(self.poscar)
        # are the defects in a same cluster if extended a little bit?
        c = Clusters(self.poscar, marked=set(d.all_defects))
        # These are the individual atoms marked as defects. Are part
        # of a single cluster? I just need to add nearest neighbors
        # and test whether they merge. I will do that only twice,
        # otherwise the defects are too separated.
        cluster_0 = c.clusters
        c.extend_clusters(n=1)
        cluster_1 = c.clusters
        c.extend_clusters(n=1)
        cluster_2 = c.clusters
        if len(cluster_2) < len(cluster_1):
            def_cluster = cluster_2
        elif len(cluster_1) < len(cluster_0):
            def_cluster = cluster_1
        else:
            def_cluster = cluster_0
        # if all atoms are a single defect, there is no defect
        if len(def_cluster) == 1:
            if len(def_cluster[0]) == self.poscar.Ntotal:
                return []
        return def_cluster

    def get_clusters(self) -> list[list[int]]:
        c = Clusters(self.poscar)
        # only one cluster but it amount the whole cell, there is no cluster.
        if len(c.clusters) == 1:
            if len(c.clusters[0]) == self.poscar.Ntotal:
                return []
        return c.clusters

    def find_defect_states(
        self,
        defects: list[list[int]] | None = None,
        factor: float = 0.70,
        IPR_threshold: float | None = None,
        k_threshold: float = 0.25,
    ) -> list[tuple[npt.NDArray[np.intp] | list[object], npt.NDArray[np.intp] | list[object]]]:
        """Find those localized states which correlate with any given defect.

        Returns
        -------
        list : It has one entry for each defect, each entry is a tuple
        (spin_up, spin_down). Inside there is a Nx2 numpy array, with
        [kpoint_index, band_index] for each defect state. If there is
        only spin_up, the spin_down contents are []. All values are
        zero-based

        """
        if defects is None:
            defects = self.defects
        if IPR_threshold is None:
            IPR_threshold = self.ipr_threshold

        # are the defects active within the desired region?
        defect_states_up: list[npt.NDArray[np.intp] | list[object]] = []
        defect_states_down: list[npt.NDArray[np.intp] | list[object]] = []
        for defect in defects:
            _Natoms = self.poscar.Ntotal
            _Ndefect = len(defect)
            Nratio = _Ndefect / _Natoms if _Natoms else 0
            # spin up first
            pipr = self.pIPR[:, :, 0, :]
            ipr = self.IPR[:, :, 0]
            bands = self.bands_up
            pipr = np.sum(pipr[:, :, defect], axis=-1)
            # for the defect to be regarded as localized within the
            # energy window, it must
            # 1) be more localized than its size.
            # 2) that should be within the energy window
            # 3) be a localized state (IPR_threshhold)
            localized_def = pipr / ipr > factor
            within_energy = (bands < self.eLim[1]) & (bands > self.eLim[0])
            above_th = ipr > IPR_threshold
            indexes: npt.NDArray[np.intp] | list[object] = np.argwhere(
                localized_def & within_energy & above_th
            )
            # as a final requirement is to need to cover a finite
            # region of the K-space.
            k_fraction = len(indexes) / bands.shape[0]
            if k_fraction < k_threshold:
                indexes = []
            defect_states_up.append(indexes)

            if self.ispin == 2:
                pipr = self.pIPR[:, :, :, 1]
                ipr = self.IPR[:, :, 1]
                bands_d = self.bands_down
                assert bands_d is not None
                pipr = np.sum(pipr[:, :, defect], axis=-1)
                localized_def = pipr / ipr > Nratio * factor
                within_energy = (bands_d < self.eLim[1]) & (bands_d > self.eLim[0])
                above_th = ipr > IPR_threshold
                indexes = np.argwhere(localized_def & within_energy & above_th)
                k_fraction = len(indexes) / bands_d.shape[0]
                if k_fraction < k_threshold:
                    indexes = []
                defect_states_down.append(indexes)
            else:
                defect_states_down.append([])
        defect_states = list(zip(defect_states_up, defect_states_down))
        return defect_states

    def write_report(self, verbosity: bool = False, filename: str = "report.txt") -> None:
        f = open(filename, "w")
        f.write("code = " + self.code + "\n")
        if self.ispin == 2:
            f.write("Spin polarized (collinear) = Yes\n")
        else:
            f.write("Spin polarized (collinear) = No\n")
        f.write("Energy window (guessed): " + str(self.eLim) + "\n")
        f.write("-----\n\n")
        f.write("Defects?\n")
        f.writelines(str(i) + " " + str(self.defects[i]) + "\n" for i in range(len(self.defects)))
        if len(self.defects) == 0:
            f.write("None\n")
        f.write("\nClusters? (including van der Waals layers)\n")
        f.writelines(str(i) + " " + str(self.clusters[i]) + "\n" for i in range(len(self.clusters)))
        if len(self.clusters) == 0:
            f.write("None\n")

        f.write("----\n\n")
        f.write("Defects states within the energy window\n\n")
        for i in range(len(self.defects)):
            states_up = self.defect_states[i][0]
            if len(states_up) > 0:
                f.write("Spin 0, defect " + str(i) + " " + str(self.defects[i]) + " \n")
                if verbosity:
                    f.write("[kpoint index, band_index]\n")
                    f.write(str(states_up) + "\n\n")
                else:
                    states_up_arr = np.asarray(states_up)
                    states_up_sorted = sorted(set(states_up_arr[:, 1].tolist()))
                    f.write("band_indexes " + str(states_up_sorted) + "\n\n")
            if self.ispin == 2:
                states_down = self.defect_states[i][0]
                if len(states_down) > 0:
                    f.write("Spin 1, defect " + str(i) + " " + str(self.defects[i]) + " \n")
                    if verbosity:
                        f.write("[kpoint index, band_index]\n")
                        f.write(str(states_down) + "\n\n")
                    else:
                        states_down_arr = np.asarray(states_down)
                        states_down_sorted = sorted(set(states_down_arr[:, 1].tolist()))
                        f.write("band_indexes " + str(states_down_sorted) + "\n\n")
        f.write("----\n\n")

        f.write("Clusters states within the energy window\n\n")
        for i in range(len(self.clusters)):
            states_up = self.cluster_states[i][0]
            if len(states_up) > 0:
                f.write("Spin 0, cluster " + str(i) + " " + str(self.clusters[i]) + " \n")
                if verbosity:
                    f.write("[kpoint index, band_index]\n")
                    f.write(str(states_up) + "\n\n")
                else:
                    states_up_arr = np.asarray(states_up)
                    states_up_sorted = sorted(set(states_up_arr[:, 1].tolist()))
                    f.write("band_indexes " + str(states_up_sorted) + "\n\n")
            if self.ispin == 2:
                states_down = self.cluster_states[i][0]
                if len(states_down) > 0:
                    f.write("Spin 1, cluster " + str(i) + " " + str(self.clusters[i]) + " \n")
                    if verbosity:
                        f.write("[kpoint index, band_index]\n")
                        f.write(str(states_down) + "\n\n")
                    else:
                        states_down_arr = np.asarray(states_down)
                        states_down_sorted = sorted(set(states_down_arr[:, 1].tolist()))
                        f.write("band_indexes " + str(states_down_sorted) + "\n\n")
        f.write("----\n\n")

        f.close()

    def get_clim(self, atoms_list: list[list[int]]) -> list[list[float]]:
        emin, emax = self.eLim
        clim: list[list[float]] = []

        for atoms in atoms_list:
            p_up = self.ebs.ebs_sum(atoms=atoms)[:, :, 0]
            values = p_up[(self.bands_up > emin) & (self.bands_up < emax)]
            vmax_up = float(np.max(values))
            vmax_down = 0.0
            if self.ispin == 2:
                p_down = self.ebs.ebs_sum(atoms=atoms)[:, :, 1]
                assert self.bands_down is not None
                values = p_down[(self.bands_down > emin) & (self.bands_down < emax)]
                vmax_down = float(np.max(values))
            vmax = max(vmax_up, vmax_down)

            clim.append([0.0, vmax])
        return clim

    def plot(self) -> None:
        spins = [0]
        if self.ispin == 2:
            spins = [0, 1]

        active_clusters: list[int] = []
        for i in range(len(self.clusters)):
            cs = self.cluster_states[i]
            if len(cs[0]) > 0 or len(cs[1]) > 0:
                active_clusters.append(i)

        active_defects: list[int] = []
        for i in range(len(self.defects)):
            cs = self.defect_states[i]
            if len(cs[0]) > 0 or len(cs[1]) > 0:
                active_defects.append(i)

        if len(active_clusters) == 0 and len(active_defects) == 0:
            bandsplot(
                code=self.code,
                dirname=self.dirname,
                fermi=self.fermi,
                mode="plain",
                spins=spins,
                elimit=list(self.eLim),
            )
            return

        for index in active_defects:
            atoms = self.defects[index]
            clim = self.defect_clim[index]
            bandsplot(
                code=self.code,
                dirname=self.dirname,
                mode="parametric",
                fermi=self.fermi,
                spins=spins,
                elimit=list(self.eLim),
                atoms=atoms,
                title="Defect " + str(index),
                clim=clim,
                cmap="plasma_r",
            )
        for index in active_clusters:
            atoms = self.clusters[index]
            clim = self.cluster_clim[index]
            bandsplot(
                code=self.code,
                dirname=self.dirname,
                fermi=self.fermi,
                mode="parametric",
                spins=spins,
                elimit=list(self.eLim),
                atoms=atoms,
                title="Cluster " + str(index),
                clim=clim,
                cmap="plasma_r",
            )


def autobandsplot(
    code: str = "vasp",
    dirname: str = ".",
    fermi: float | None = None,
    use_cache: bool = False,
) -> None:
    AutoBandsPlot(code=code, dirname=dirname, fermi=fermi, use_cache=use_cache)
