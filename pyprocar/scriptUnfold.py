import numpy as np
import matplotlib.pyplot as plt
from .unfold import ProcarUnfolder
from .utilsprocar import UtilsProcar

def run_unfolding(
        fname='PROCAR',
        poscar='POSCAR',
        outcar='OUTCAR',
        supercell_matrix=np.diag([2, 2, 2]),
        efermi=None,
        ylim=(-5, 15),
        ktick=[0, 36, 54, 86, 110, 147, 165, 199],
        knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
        print_kpts=False,
        show_band=True,
        figname='unfolded_band.png'):
    """
    Params:
    ==============================================
    fname: PROCAR filename.
    poscar: POSCAR filename
    outcar: OUTCAR filename, for reading fermi energy. You can also use efermi and set outcar=None
    supercell_matrix: supercell matrix from primitive cell to supercell
    efermi: Fermi energy
    ylim: range of energy to be plotted.
    ktick: the indices of K points which has labels given in knames.
    knames: see ktick
    print_kpts: print all the kpoints to screen. This is to help find the ktick and knames.
    show_band: whether to plot the bands before unfolding.
    figname: the file name of which the figure will be saved.
    """

    if efermi is not None:
        fermi = efermi
    elif outcar is not None:
        outcarparser = UtilsProcar()
        fermi = outcarparser.FermiOutcar(outcar)
    else:
        raise Warning(
            "Fermi energy is not given, neither an OUTCAR contains it.")

    uf = ProcarUnfolder(
        procar=fname,
        poscar=poscar,
        supercell_matrix=supercell_matrix,
    )
    if print_kpts:
        for ik, k in enumerate(uf.procar.kpoints):
            print(ik, k)
    axes = uf.plot(
        efermi=fermi,
        ylim=ylim,
        ktick=ktick,
        kname=knames,
        show_band=show_band)
    plt.savefig(figname)
    plt.show()


if __name__ == '__main__':
    run_unfolding(
        fname='PROCAR',
        poscar='POSCAR',
        outcar='OUTCAR',
        supercell_matrix=np.diag([2, 2, 2]),
        efermi=None,
        ylim=(-5, 15),
        ktick=[0, 36, 54, 86, 110, 147, 165, 199],
        knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
        print_kpts=False,
        show_band=True,
        figname='unfolded_band.png')
