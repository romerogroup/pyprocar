import numpy as np
import re
from ase.io import read
from unfolder import Unfolder
import matplotlib.pyplot as plt
from plot import plot_band_weight


class ProcarParser():
    def __init__(self, fname):
        self.fname = fname
        self.nkpt, self.bandsCount, self.nion, self.norb = (None, None, None,
                                                            None)
        self.kpoints = None
        self.kweights = None
        self.bands = None
        self.projections = None
        self.carray = None
        self.orbital_names = None

    def plot_band(self, axes):
        for i in range(self.bandsCount):
            axes.plot(self.bands[:, i], color='gray', linewidth=0.9, alpha=0.6)

    def read_procar(self):
        # read header
        ikpt = 0
        iband = 0
        with open(self.fname) as myfile:
            for line in myfile:
                if line.startswith("# of k-points"):
                    a = re.findall(":\s*([0-9]*)", line)
                    self.kpointsCount, self.bandsCount, self.nion = map(int, a)
                    print("kpointsCount: ", self.kpointsCount)
                    print("bandsCount: ", self.bandsCount)
                    print("nion: ", self.nion)
                    self.kpoints = np.zeros([self.kpointsCount, 3])
                    self.kweights = np.zeros(self.kpointsCount)
                    self.bands = np.zeros([self.kpointsCount, self.bandsCount])
                if line.strip().startswith('k-point'):
                    ss = line.strip().split()
                    ikpt = int(ss[1]) - 1
                    k0 = float(ss[3])
                    k1 = float(ss[4])
                    k2 = float(ss[5])
                    w = float(ss[-1])
                    self.kpoints[ikpt, :] = [k0, k1, k2]
                    self.kweights[ikpt] = w
                if line.strip().startswith('band'):
                    ss = line.strip().split()
                    iband = int(ss[1]) - 1
                    e = float(ss[4])
                    occ = float(ss[-1])
                    self.bands[ikpt, iband] = e
                if line.strip().startswith('ion'):
                    if line.strip().endswith('tot'):
                        self.orbital_names = line.strip().split()[1:-1]
                        self.norb = len(self.orbital_names)
                    if self.projections is None:
                        self.projections = np.zeros([
                            self.kpointsCount, self.bandsCount, self.nion,
                            self.norb
                        ])
                        self.carray = np.zeros([
                            self.kpointsCount, self.bandsCount, self.nion,
                            self.norb
                        ],
                                               dtype='complex')
                    for i in range(self.nion):
                        line = next(myfile)
                        t = line.strip().split()
                        if len(t) == self.norb + 2:
                            self.projections[ikpt, iband, i, :] = [
                                float(x) for x in t[1:-1]
                            ]
                        elif len(t) == self.norb * 2 + 2:
                            self.carray[ikpt, iband, i, :] += np.array(
                                [float(x) for x in t[1:-1:2]])

                            self.carray[ikpt, iband, i, :] += 1j * np.array(
                                [float(x) for x in t[2::2]])
                        else:
                            raise Exception(
                                "Cannot parse line to projection: %s" % line)
        self._post_proc()

    def _post_proc(self):
        self.projections = np.reshape(
            self.projections,
            (self.kpointsCount, self.bandsCount, self.nion * self.norb))
        self.carray = np.reshape(
            self.carray.transpose((0, 1, 2, 3)),
            (self.kpointsCount, self.bandsCount, self.nion * self.norb))
        self._normalize_carray()

    def _normalize_carray(self):
        norm = np.linalg.norm(self.carray, ord=2, axis=2)
        self.carray /= norm[:, :, None]


class ProcarUnfolder(object):
    def __init__(self, procar, poscar, supercell_matrix):
        self.fname = procar
        self.supercell_matrix = supercell_matrix
        self._parse_procar()
        self.atoms = read(poscar)
        self.basis = []
        self.positions = []
        self._prepare_unfold_basis()

    def _parse_procar(self):
        self.procar = ProcarParser(self.fname)
        self.procar.read_procar()

    def _prepare_unfold_basis(self):
        # basis, which are the name of the bands e.g. 'Ti|dxy|0'
        # self.eigenvectors = np.zeros(
        #    (self.procar.kpointsCount, self.procar.bandsCount,
        #     (self.procar.orbitalCount - 1) * (self.procar.ionsCount - 1) *
        #     self.procar.ispin), dtype='complex')
        self.eigenvectors = self.procar.carray
        for iatom, chem in enumerate(self.atoms.get_chemical_symbols()):
            #for iorb, orb in enumerate(self.procar.orbitalNames[:-1]):
            for iorb, orb in enumerate(self.procar.orbital_names):
                #for spin in range(self.procar.ispin):
                for spin in range(1):
                    # todo: what about spin?
                    self.basis.append("%s|%s|%s" % (None, orb, spin))
                    pos = self.atoms.get_scaled_positions()[iatom]
                    self.positions.append(
                        self.atoms.get_scaled_positions()[iatom])
                    #for ikpt, kpt in enumerate(self.procar.kpoints):
                    #    self.eigenvectors[ikpt, :,
                    #                  counter] = np.sqrt(self.procar.spd[ikpt, :, spin,
                    #iatom, iorb+1]) *np.exp(-2j* np.vdot(kpt, pos)*np.pi)
                    #counter += 1
        #self._add_phase()

    def unfold(self):
        # spd: spd[kpoint][band][ispin][atom][orbital]
        # bands[kpt][iband]
        # to unfold,
        # unfolder:
        #def __init__(self, cell, basis, positions , supercell_matrix, eigenvectors, qpoints, tol_r=0.1, compare=None):
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
             axis=None):
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
        return axes


def run_unfolding(
        fname='PROCAR',
        supercell_matrix=np.diag([2, 2, 2]),
        efermi=4.298,
        ylim=(-5, 15),
        ktick=[0, 36, 54, 86, 110, 147, 165, 199],
        knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A']):
    fname = 'PROCAR'
    uf = ProcarUnfolder(
        fname,
        poscar='POSCAR',
        supercell_matrix=supercell_matrix,
    )
    for ik, k in enumerate(uf.procar.kpoints):
        print(ik, k)
    axes = uf.plot(efermi=efermi, ylim=ylim, ktick=ktick, kname=knames)
    uf.procar.plot_band(axes=axes)
    plt.savefig('MgB2_unfold.png')
    plt.show()


if __name__ == '__main__':
    run_unfolding()
