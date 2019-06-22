import numpy as np
import re
from ase.io import read
from .unfolder import Unfolder
import matplotlib.pyplot as plt
from .fatband import plot_band_weight
from ..procarparser import ProcarParser


class ProcarUnfolder(object):
    def __init__(self, procar, poscar, supercell_matrix):
        self.fname = procar
        self.supercell_matrix = supercell_matrix
        self._parse_procar()
        self.atoms = read(poscar)
        self.basis = []
        self.positions = []

    def _parse_procar(self):
        self.procar = ProcarParser()
        self.procar.readFile2(self.fname, phase=True)

    def _prepare_unfold_basis(self, ispin):
        # basis, which are the name of the bands e.g. 'Ti|dxy|0'
        # self.eigenvectors = np.zeros(
        #    (self.procar.kpointsCount, self.procar.bandsCount,
        #     (self.procar.orbitalCount - 1) * (self.procar.ionsCount - 1) *
        #     self.procar.ispin), dtype='complex')
        self.eigenvectors = np.reshape(
            self.procar.carray[:, :, ispin, :, :],
            (self.procar.kpointsCount, self.procar.bandsCount,
             self.procar.ionsCount * self.procar.orbitalCount))
        norm = np.linalg.norm(self.eigenvectors, ord=2, axis=2)
        self.eigenvectors /= norm[:, :, None]

        for iatom, chem in enumerate(self.atoms.get_chemical_symbols()):
            for iorb, orb in enumerate(self.procar.orbitalName):
                for spin in range(self.procar.nspin):
                    # todo: what about spin?
                    self.basis.append("%s|%s|%s" % (None, orb, spin))
                    self.positions.append(
                        self.atoms.get_scaled_positions()[iatom])

    def unfold(self, ispin=0):
        # spd: spd[kpoint][band][ispin][atom][orbital]
        # bands[kpt][iband]
        # to unfold,
        # unfolder:
        #def __init__(self, cell, basis, positions , supercell_matrix, eigenvectors, qpoints, tol_r=0.1, compare=None):
        self._prepare_unfold_basis(ispin=ispin)
        self.unfolder = Unfolder(
            self.atoms.cell,
            self.basis,
            self.positions,
            self.supercell_matrix,
            self.eigenvectors,
            self.procar.kpoints,
            phase=False)
        w = self.unfolder.get_weights()
        return w

    def plot(self,
             efermi=5.46,
             ylim=(-5, 10),
             ktick=[0, 41, 83, 125, 200],
             kname=['$\Gamma$', 'X', 'M', 'R', '$\Gamma$'],
             show_band=True,
             shift_efermi=True,
             width=4.0,
             color='blue',
             axis=None, 
             savetab=None,
             ):
        xlist = [list(range(self.procar.kpointsCount))]
        uf = self.unfold()
        if savetab is not None:
            nk, nb=uf.shape
            tab=np.zeros((nb, nk*2), dtype=float)
            tab[:, ::2]=self.procar.bands.T
            tab[:, 1::2]=uf.T
            np.savetxt(savetab, tab, delimiter=',', fmt="%10.4f", header='# nkpoints: %s   nbands:%s \n#E(k1) w(k1) E(k2) w(k2) E(k3) w(k3)...'%(nk, nb))
        axes = plot_band_weight(
            xlist * self.procar.bandsCount,
            self.procar.bands.T,
            np.abs(uf.T),
            xticks=[kname, ktick],
            efermi=efermi,
            shift_efermi=shift_efermi,
            fatness=width,
            color=color,
            axis=axis)
        axes.set_ylim(ylim)
        axes.set_xlim(0, self.procar.kpointsCount - 1)
        if shift_efermi:
            shift=-efermi
        else:
            shift=0.0
        if show_band:
            for i in range(self.procar.bandsCount):
                axes.plot(self.procar.bands[:,i]+shift, color='gray', linewidth=1, alpha=0.3)
        return axes

