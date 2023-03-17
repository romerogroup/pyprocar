import numpy as np
import matplotlib.pyplot as plt
from ..utils import welcome
from ..utils.defaults import settings
from ..utils.info import orbital_names
from ..plotter import EBSPlot
from .. import io

def unfold(
    procar="PROCAR",
    poscar="POSCAR",
    outcar="OUTCAR",
    abinit_output="abinit.out",
    transformation_matrix=np.diag([2, 2, 2]),
    kpoints=None,
    elkin="elk.in",
    code="vasp",
    mode="plain",
    unfold_mode="both",
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
    # welcome()

    structure = None
    reciprocal_lattice = None
    kpath = None
    ebs = None
    kpath = None
    structure = None
    labels=None
    settings.general.modify(kwargs)
    settings.unfold.modify(kwargs)
    settings.ebs.modify(settings.unfold.config)

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
    print(ebs)
    print(kpath)
    ebs_plot = EBSPlot(ebs, kpath, ax, spins)
    ebs_plot.ebs.unfold(transformation_matrix=transformation_matrix, structure=structure)
    if unfold_mode == 'both':
        width_weights = ebs.weights
        width_mask = unfold_mask
        color_weights = ebs.weights
        color_mask = unfold_mask
    elif unfold_mode == 'thickness':
        width_weight = ebs.weights
        width_mask = unfold_mask
    elif unfold_mode == 'color':
        color_weights = ebs.weights
        color_mask = unfold_mask
    else :
        raise Exception("Invalid unfold_mode was selected: %s please select from the following 'both', 'thickness','color'" % unfold_mode)
    if mode == "plain":
        ebs_plot.plot_bands()
        ebs_plot.plot_parameteric(color_weights=color_weights,
                                  width_weights=width_weights,
                                  color_mask=color_mask,
                                  width_mask=width_mask,
                                  vmin=vmin,
                                  vmax=vmax)
        ebs_plot.handles = ebs_plot.handles[:ebs_plot.nspins]

    ebs_plot.set_xticks(kticks, knames)
    ebs_plot.set_yticks(interval=elimit)
    ebs_plot.set_xlim()
    ebs_plot.set_ylim(elimit)
    ebs_plot.draw_fermi(
        color=settings.ebs.fermi_color,
    linestyle=settings.ebs.fermi_linestyle,
        linewidth=settings.ebs.fermi_linewidth,
    )
    ebs_plot.set_ylabel()
    if settings.ebs.grid:
        ebs_plot.grid()
    if settings.ebs.legend:
        ebs_plot.legend(labels)
    if savefig is not None:
        ebs_plot.save(savefig)
    if show:
        ebs_plot.show()
    return ebs_plot
#     if efermi is not None:
#         fermi = efermi7
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
