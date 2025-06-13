#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from .. import io
from ..pyposcar.clusters import Clusters
from ..pyposcar.defects import FindDefect
from ..pyposcar.generalUtils import remove_flat_points
from ..pyposcar.poscar import Poscar
from ..scripts.scriptBandsplot import bandsplot

try:
    from sklearn.neighbors.kde import KernelDensity
except:
    from sklearn.neighbors import KernelDensity

from scipy.signal import argrelextrema


class AutoBandsPlot:
    def __init__(self, code="vasp", dirname=".", fermi: int = None):

        self.parser = io.Parser(code="vasp", dirpath=dirname)
        self.code = code
        self.ebs = self.parser.ebs

        codes_with_scf_fermi = ["qe", "elk"]
        if code in codes_with_scf_fermi and fermi is None:
            logger.info(
                f"No fermi given, using the found fermi energy: {self.ebs.efermi}"
            )
            fermi = self.ebs.efermi
        elif fermi is None:
            fermi = 0
        self.fermi = fermi

        self.dirname = dirname
        self.structure = self.parser.structure
        self.kpath = self.ebs.kpath
        self.ispin = self.ebs.bands.shape[-1]
        self.bands_up = self.ebs.bands[:, :, 0] - fermi
        self.bands_down = None
        if self.ispin == 2:
            self.bands_down = self.ebs.bands[:, :, 1] - fermi

        self.IPR = self.ebs.ebs_ipr()
        self.pIPR = self.ebs.ebs_ipr_atom()

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
        self.poscar.load_from_data(
            direct_positions=self.structure.fractional_coordinates,
            lattice=self.structure.lattice,
            elements=self.structure.atoms,
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

        return

    def simple_energy_window(self, delta=1.0):
        """Return a energy window within the last occupied / first unoccupied
        state. It considers each spin separately and returns the
        largest interval. The interval is enlarged by adding `delta'


        """
        # bands need to have the Fermi energy set to zero
        #
        # Finding the lowest occupied / highest unocuppied  energies
        # print(self.bands_up.shape)
        emin_up = []
        emax_up = []
        emin_down = []
        emax_down = []
        # looking for the highest occupied level for each kpoint
        for kpoint in range(self.bands_up.shape[0]):
            b_up = self.bands_up[kpoint]
            emin_up.append(np.max(b_up[b_up < 0]))
            emax_up.append(np.min(b_up[b_up > 0]))
            if self.ispin == 2:
                b_down = self.bands_up[kpoint]
                emin_down.append(np.max(b_down[b_down < 0]))
                emax_down.append(np.min(b_down[b_down > 0]))
        emin = min(emin_up)
        emax = max(emax_up)
        if self.ispin == 2:
            emin = min(emin, min(emin_down))
            emax = max(emax, min(emax_down))
        # print('(without delta) emin, emax', emin, emax)
        # adding a little bit of space to the window
        emax = emax + delta
        emin = emin - delta
        if emin < self.eBoundaries[0]:
            emin = self.eBoundaries[0]
        if emax > self.eBoundaries[1]:
            emax = self.eBoundaries[1]
        # print('(with delta) emin, emax', emin, emax)
        return emin, emax

    def get_energy_boundaries(self):
        # what are the maximum energies for each kpoint?
        max_energy_up = np.max(self.bands_up, axis=1)
        # what kpoint as the smaller max energy?
        max_energy_up = np.min(max_energy_up)

        min_energy_up = np.min(self.bands_up, axis=1)
        min_energy_up = np.max(min_energy_up)
        max_energy = max_energy_up
        min_energy = min_energy_up

        if self.ispin == 2:
            max_energy_down = np.max(self.bands_down, axis=1)
            max_energy_down = np.min(max_energy_down)

            min_energy_down = np.min(self.bands_down, axis=1)
            min_energy_down = np.max(min_energy_down)
            max_energy = min(max_energy_up, max_energy_down)
            min_energy = max(min_energy_up, min_energy_down)
        # print('Boundaries,' , min_energy, max_energy)
        return min_energy, max_energy

    def ipr_energy_window(self):
        # first only spin up
        ipr_up = self.IPR[:, :, 0]
        threshold_up = np.percentile(ipr_up, 90)
        threshold = threshold_up
        if self.ispin == 2:
            ipr_down = self.IPR[:, :, 1]
            threshold_down = np.percentile(ipr_down, 90)
            threshold = max(threshold_up, threshold_down)
        # print('IPR threshold', threshold)

        for kpoint in range(self.bands_up.shape[0]):
            # searching at least one bulk band in valence
            band = self.bands_up[kpoint]
            ipr = ipr_up[kpoint]
            indexes = np.argwhere((band < 0) & (ipr < threshold))
            max_index = np.max(indexes)
            # print('Highest occupied *bulk* band kpoint, index, energy',
            #       kpoint, max_index, band[max_index])
            emin = band[max_index]
            if emin < self.eLim[0] and emin > self.eBoundaries[0]:
                self.eLim = max(emin - 0.5, self.eBoundaries[0]), self.eLim[1]
            # searching for at least one bulk band in conduction region
            indexes = np.argwhere((band > 0) & (ipr < threshold))
            min_index = np.min(indexes)
            emax = band[min_index]
            if emax > self.eLim[1] and emax < self.eBoundaries[1]:
                self.eLim = self.eLim[0], min(emax + 0.5, self.eBoundaries[1])

            if self.ispin == 2:
                # searching at least one bulk band in valence
                band = self.bands_up[kpoint]
                ipr = ipr_up[kpoint]
                indexes = np.argwhere((band < 0) & (ipr < threshold))
                max_index = np.max(indexes)
                # print('Highest occupied *bulk* band kpoint, index, energy',
                #       kpoint, max_index, band[max_index])
                emin = band[max_index]
                if emin < self.eLim[0] and emin > self.eBoundaries[0]:
                    self.eLim = max(emin - 0.5, self.eBoundaries[0]), self.eLim[1]
                # searching for at least one bulk band in conduction region
                indexes = np.argwhere((band > 0) & (ipr < threshold))
                min_index = np.min(indexes)
                emax = band[min_index]
                if emax > self.eLim[1] and emax < self.eBoundaries[1]:
                    self.eLim = self.eLim[0], min(emax + 0.5, self.eBoundaries[1])

        # print('self.eLim (IPR)', self.eLim)
        return threshold

    def get_defects(self):
        d = FindDefect(self.poscar)
        # print('\ndefects:', d.defects)
        # are the defects in a same cluster if extended a little bit?
        c = Clusters(self.poscar, marked=d.all_defects)
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
        # print('\ndefects:', def_cluster)
        return def_cluster

    def get_clusters(self):
        c = Clusters(self.poscar)
        # only one cluster but it amount the whole cell, there is no cluster.
        if len(c.clusters) == 1:
            if len(c.clusters[0]) == self.poscar.Ntotal:
                return []
        # print('clusters', c.clusters)
        return c.clusters

    def find_defect_states(
        self, defects=None, factor=0.70, IPR_threshold=None, k_threshold=0.25
    ):
        """Find those localized states which correlate with any given defect.

        Returns
        -------

        list : It has one entry for each defect, each entry is a tuple
        (spin_up, spin_down). Inside there is a Nx2 numpy array, with
        [kpoint_index, band_index] for each defect state. If there is
        only spin_up, the spin_down contents are []. All values are
        zero-based

        """
        if defects == None:
            defects = self.defects
        if IPR_threshold == None:
            IPR_threshold = self.ipr_threshold

        # print('pIPR.shape', self.pIPR.shape)
        # are the defects active within the desired region?
        defect_states_up = []
        defect_states_down = []
        for defect in defects:
            # print('defect', defect)
            Natoms = self.poscar.Ntotal
            Ndefect = len(defect)
            Nratio = Ndefect / Natoms
            # spin up first
            pipr = self.pIPR[:, :, :, 0]
            ipr = self.IPR[:, :, 0]
            bands = self.bands_up
            pipr = np.sum(pipr[:, :, defect], axis=-1)
            # print(ipr.shape, pipr.shape)
            # for the defect to be regarded as localized within the
            # energy window, it must
            # 1) be more localized than its size.
            # 2) that should be within the energy window
            # 3) be a localized state (IPR_threshhold)
            localized_def = pipr / ipr > factor
            within_energy = (bands < self.eLim[1]) & (bands > self.eLim[0])
            above_th = ipr > IPR_threshold
            indexes = np.argwhere(localized_def & within_energy & above_th)
            # as a final requirement is to need to cover a finite
            # region of the K-space.
            k_fraction = len(indexes) / bands.shape[0]
            if k_fraction < k_threshold:
                indexes = []
            defect_states_up.append(indexes)

            if self.ispin == 2:
                pipr = self.pIPR[:, :, :, 1]
                ipr = self.IPR[:, :, 1]
                bands = self.bands_down
                pipr = np.sum(pipr[:, :, defect], axis=-1)
                localized_def = pipr / ipr > Nratio * factor
                within_energy = (bands < self.eLim[1]) & (bands > self.eLim[0])
                above_th = ipr > IPR_threshold
                indexes = np.argwhere(localized_def & within_energy & above_th)
                k_fraction = len(indexes) / bands.shape[0]
                if k_fraction < k_threshold:
                    indexes = []
                defect_states_down.append(indexes)
            else:
                defect_states_down.append([])
            # if len(defect_states_up[-1]) or len(defect_states_down[-1]):
            #     print('Defect states found')
        defect_states = list(zip(defect_states_up, defect_states_down))
        # print('defect_states', defect_states)
        return defect_states

    def write_report(self, verbosity=False, filename="report.txt"):
        f = open(filename, "w")
        f.write("code = " + self.code + "\n")
        if self.ispin == 2:
            f.write("Spin polarized (collinear) = Yes\n")
        else:
            f.write("Spin polarized (collinear) = No\n")
        f.write("Energy window (guessed): " + str(self.eLim) + "\n")
        f.write("-----\n\n")
        f.write("Defects?\n")
        for i in range(len(self.defects)):
            f.write(str(i) + " " + str(self.defects[i]) + "\n")
        if len(self.defects) == 0:
            f.write("None\n")
        f.write("\nClusters? (including van der Waals layers)\n")
        for i in range(len(self.clusters)):
            f.write(str(i) + " " + str(self.clusters[i]) + "\n")
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
                    states_up = sorted(set(states_up[:, 1]))
                    f.write("band_indexes " + str(states_up) + "\n\n")
            if self.ispin == 2:
                states_down = self.defect_states[i][0]
                if len(states_down) > 0:
                    f.write(
                        "Spin 1, defect " + str(i) + " " + str(self.defects[i]) + " \n"
                    )
                    if verbosity:
                        f.write("[kpoint index, band_index]\n")
                        f.write(str(states_down) + "\n\n")
                    else:
                        states_down = sorted(set(states_down[:, 1]))
                        f.write("band_indexes " + str(states_down) + "\n\n")
        f.write("----\n\n")

        f.write("Clusters states within the energy window\n\n")
        for i in range(len(self.clusters)):
            states_up = self.cluster_states[i][0]
            if len(states_up) > 0:
                f.write(
                    "Spin 0, cluster " + str(i) + " " + str(self.clusters[i]) + " \n"
                )
                if verbosity:
                    f.write("[kpoint index, band_index]\n")
                    f.write(str(states_up) + "\n\n")
                else:
                    states_up = sorted(set(states_up[:, 1]))
                    f.write("band_indexes " + str(states_up) + "\n\n")
            if self.ispin == 2:
                states_down = self.cluster_states[i][0]
                if len(states_down) > 0:
                    f.write(
                        "Spin 1, cluster "
                        + str(i)
                        + " "
                        + str(self.clusters[i])
                        + " \n"
                    )
                    if verbosity:
                        f.write("[kpoint index, band_index]\n")
                        f.write(str(states_down) + "\n\n")
                    else:
                        states_down = sorted(set(states_down[:, 1]))
                        f.write("band_indexes " + str(states_down) + "\n\n")
        f.write("----\n\n")

        f.close()

    def get_clim(self, atoms_list):
        emin, emax = self.eLim
        clim = []

        for atoms in atoms_list:
            p_up = self.ebs.ebs_sum(atoms=atoms)[:, :, 0]
            values = p_up[(self.bands_up > emin) & (self.bands_up < emax)]
            vmax_up = np.max(values)
            vmax_down = 0
            if self.ispin == 2:
                p_down = self.ebs.ebs_sum(atoms=atoms)[:, :, 1]
                values = p_down[(self.bands_down > emin) & (self.bands_down < emax)]
                vmax_down = np.max(values)
            vmax = max(vmax_up, vmax_down)

            clim.append([0, vmax])
        return clim

    def plot(self):
        spins = [0]
        if self.ispin == 2:
            spins = [0, 1]

        active_clusters = []
        for i in range(len(self.clusters)):
            cs = self.cluster_states[i]
            if len(cs[0]) > 0 or len(cs[1]) > 0:
                active_clusters.append(i)

        active_defects = []
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
                elimit=self.eLim,
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
                elimit=self.eLim,
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
                elimit=self.eLim,
                atoms=atoms,
                title="Cluster " + str(index),
                clim=clim,
                cmap="plasma_r",
            )


def autobandsplot(code="vasp", dirname=".", fermi: int = None):
    a = AutoBandsPlot(code=code, dirname=dirname, fermi=fermi)
