import numpy as np
import matplotlib.pyplot as plt
from .procarunfold import ProcarUnfolder
from .utilsprocar import UtilsProcar

def unfold(
        fname='PROCAR',
        poscar='POSCAR',
        outcar='OUTCAR',
        supercell_matrix=np.diag([2, 2, 2]),
        efermi=None,
        shift_efermi=True,
        elimit=(-5, 15),
        kticks=[0, 36, 54, 86, 110, 147, 165, 199],
        knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
        print_kpts=False,
        show_band=True,
        savefig='unfolded_band.png'):
    """
    Params:
    ==============================================
    fname: PROCAR filename.
    poscar: POSCAR filename
    outcar: OUTCAR filename, for reading fermi energy. You can also use efermi and set outcar=None
    supercell_matrix: supercell matrix from primitive cell to supercell
    efermi: Fermi energy
    elimit: range of energy to be plotted.
    kticks: the indices of K points which has labels given in knames.
    knames: see kticks
    print_kpts: print all the kpoints to screen. This is to help find the kticks and knames.
    show_band: whether to plot the bands before unfolding.
    savefig: the file name of which the figure will be saved.
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
        shift_efermi=shift_efermi,
        ylim=elimit,
        ktick=kticks,
        kname=knames,
        show_band=show_band)
    plt.savefig(savefig)
    plt.show()


# if __name__ == '__main__':
#     """
#     An example of how to use
#     """
#     import pyprocar
#     import numpy as np
#     pyprocar.unfold(
#         fname='PROCAR',
#         poscar='POSCAR',
#         outcar='OUTCAR',
#         supercell_matrix=np.diag([2, 2, 2]),
#         efermi=None,
#         shift_efermi=True,
#         elimit=(-5, 15),
#         kticks=[0, 36, 54, 86, 110, 147, 165, 199],
#         knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
#         print_kpts=False,
#         show_band=True,
#         savefig='unfolded_band.png')
