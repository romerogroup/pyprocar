import numpy as np
import re
from ase.io import read
from unfolder import Unfolder
import matplotlib.pyplot as plt
from plot import plot_band_weight
from pyprocar import ProcarParser


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
        print("read File")
        self.procar.readFile2(self.fname, phase=True)
        print("File read")

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
             axis=None, 
             ):
        xlist = [list(range(self.procar.kpointsCount))]
        uf = self.unfold()
        axes = plot_band_weight(
            xlist * self.procar.bandsCount,
            self.procar.bands.T,
            np.abs(uf.T),
            xticks=[kname, ktick],
            efermi=efermi,
            axis=axis)
        axes.set_ylim(ylim)
        axes.set_xlim(0, self.procar.kpointsCount - 1)
        if show_band:
            for i in range(self.procar.bandsCount):
                axes.plot(self.procar.bands[:,i], color='gray', linewidth=1, alpha=0.3)
        return axes


def run_unfolding(
        fname='PROCAR',
        poscar='POSCAR',
        supercell_matrix=np.diag([2, 2, 2]),
        efermi=4.298,
        ylim=(-5, 15),
        ktick=[0, 36, 54, 86, 110, 147, 165, 199],
        knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
        print_kpts=False,
        show_band=True,
        figname='unfolded_band.png'):
    uf = ProcarUnfolder(
        procar=fname,
        poscar=poscar,
        supercell_matrix=supercell_matrix,
    )
    if print_kpts:
        for ik, k in enumerate(uf.procar.kpoints):
            print(ik, k)
    axes = uf.plot(efermi=efermi, ylim=ylim, ktick=ktick, kname=knames, show_band=show_band)
    plt.savefig(figname)
    plt.show()


def test():
    import os
    path = '/media/hexu/Data/Projects/pyprocar_unfolding/MgB2/MgB2_sc222_Aldoping'
    print(os.path.exists(os.path.join(path, 'PROCAR')))
    run_unfolding(
        fname=os.path.join(path, 'PROCAR'),
        poscar=os.path.join(path, 'POSCAR'))


if __name__ == '__main__':
    test()
