import numpy as np
import matplotlib.pyplot as plt
from .splash import welcome


def unfold(
    procar="PROCAR",
    poscar="POSCAR",
    outcar="OUTCAR",
    procar="PROCAR",
    abinit_output="abinit.out",
    transformation_matrix=np.diag([2, 2, 2]),
    kpoints=None,
    elkin="elk.in",
    code="vasp",
    mode="plain",
    spins=None,
    atoms=None,
    orbitals=None,
    items=None,
    projection_mask=None,
    unfold_mask=None,
    fermi=None,
    interpolation_factor=1,
    interpolation_type="cubic",
    vmax=None,
    vmin=None,
    kticks=None,
    knames=None,
    kdirect=True,
    elimit=None,
    ax=None,
    show=True,
    savefig=None,
    old=False,
    savetab="unfold_result.csv",
    **kwargs,
):
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
        exportplt: flag to export plot as matplotlib.pyplot object.

        """
    welcome()

    structure = None
    reciprocal_lattice = None
    kpath = None
    ebs = None
    kpath = None
    structure = None

    settings.modify(kwargs)

    if code == "vasp":
        if outcar is not None:
            outcar = io.vasp.Outcar(outcar)
        if fermi is None:
            fermi = outcar.efermi
            reciprocal_lattice = outcar.reciprocal_lattice
        if poscar is not None:
            poscar = io.vasp.Poscar(poscar)
            structure = poscar.structure
        if reciprocal_lattice is None:
            reciprocal_lattice = poscar.structure.reciprocal_lattice

        if kpoints is not None:
            kpoints = io.vasp.Kpoints(kpoints)
            kpath = kpoints.kpath

            procar = io.vasp.Procar(
                procar,
                structure,
                reciprocal_lattice,
                kpath,
                fermi,
                interpolation_factor=interpolation_factor,
            )
            ebs = procar.ebs

            ebs_plot = EBSPlot(ebs, kpath, ax, spins)


#     if efermi is not None:
#         fermi = efermi
#     elif outcar is not None:
#         outcarparser = UtilsProcar()
#         fermi = outcarparser.FermiOutcar(outcar)
#     else:
#         raise Warning("Fermi energy is not given, neither an OUTCAR contains it.")

#     uf = ProcarUnfolder(
#         procar=fname, poscar=poscar, supercell_matrix=supercell_matrix, ispin=ispin
#     )
#     if print_kpts:
#         for ik, k in enumerate(uf.procar.kpoints):
#             print(ik, k)
#     axes = uf.plot(
#         efermi=fermi,
#         ispin=ispin,
#         shift_efermi=shift_efermi,
#         ylim=elimit,
#         ktick=kticks,
#         kname=knames,
#         color=color,
#         width=width,
#         savetab=savetab,
#         show_band=show_band,
#     )

#     if exportplt:
#         return plt

#     else:
#         if savefig:
#             plt.savefig(savefig, bbox_inches="tight")
#             plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
#         else:
#             plt.show()
#         return


# # if __name__ == '__main__':
# #     """
# #     An example of how to use
# #     """
# #     import pyprocar
# #     import numpy as np
# #     pyprocar.unfold(
# #         fname='PROCAR',
# #         poscar='POSCAR',
# #         outcar='OUTCAR',
# #         supercell_matrix=np.diag([2, 2, 2]),
# #         efermi=None,
# #         shift_efermi=True,
# #         ispin=0,
# #         elimit=(-5, 15),
# #         kticks=[0, 36, 54, 86, 110, 147, 165, 199],
# #         knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
# #         print_kpts=False,
# #         show_band=True,
# #         savefig='unfolded_band.png')
