import numpy as np
import matplotlib.pyplot as plt
from .procarunfold import ProcarUnfolder
from .utilsprocar import UtilsProcar

def unfold(
        fname='PROCAR',
        poscar='POSCAR',
        outcar='OUTCAR',
        supercell_matrix=np.diag([2, 2, 2]),
        ispin=None,
        efermi=None,
        shift_efermi=True,
        elimit=(-5, 15),
        kticks=[0, 36, 54, 86, 110, 147, 165, 199],
        knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
        print_kpts=False,
        show_band=True,
        width=4,
        color='blue',
        savetab='unfold_result.csv',
        savefig='unfolded_band.png'):

    """

    Parameters
    ---------- 
    fname: PROCAR filename.
    poscar: POSCAR filename
    outcar: OUTCAR filename, for reading fermi energy. You can also use efermi and set outcar=None
    supercell_matrix: supercell matrix from primitive cell to supercell
    ispin: For non-spin polarized system, ispin=None. 
           For spin polarized system: ispin=1 is spin up, ispin=2 is spin down.
    efermi: Fermi energy
    elimit: range of energy to be plotted.
    kticks: the indices of K points which has labels given in knames.
    knames: see kticks
    print_kpts: print all the kpoints to screen. This is to help find the kticks and knames.
    show_band: whether to plot the bands before unfolding.
    width: the width of the unfolded band. 
    color: color of the unfoled band.
    savetab: the csv file name of which  the table of unfolding result will be written into.
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
        ispin=ispin
    )
    if print_kpts:
        for ik, k in enumerate(uf.procar.kpoints):
            print(ik, k)
    axes = uf.plot(
        efermi=fermi,
        ispin=ispin,
        shift_efermi=shift_efermi,
        ylim=elimit,
        ktick=kticks,
        kname=knames,
        color=color,
        width=width,
        savetab=savetab,
        show_band=show_band)
    plt.savefig(savefig,bbox_inches='tight')
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
#         ispin=0,
#         elimit=(-5, 15),
#         kticks=[0, 36, 54, 86, 110, 147, 165, 199],
#         knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
#         print_kpts=False,
#         show_band=True,
#         savefig='unfolded_band.png')
