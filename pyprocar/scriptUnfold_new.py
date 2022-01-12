import os
import re
import matplotlib.pyplot as plt
import numpy as np
from .io import vasp
from .plotter import EBSPlot
from .abinitparser import AbinitParser
from .elkparser import ElkParser
from .qeparser import QEParser
from .lobsterparser import LobsterParser
from .procarparser import ProcarParser
from .procarplot import ProcarPlot
from .procarselect import ProcarSelect
from .splash import welcome
from .utilsprocar import UtilsProcar


def unfold(
        procar="PROCAR",
        abinit_output="abinit.out",
        poscar=None,
        outcar=None,
        kpoints=None,
        elkin="elk.in",
        mode="plain",
        linestyles=None,
        spins=None,
        atoms=None,
        orbitals=None,
        fermi=None,
        interpolation_factor=1,
        projection_mask=None,
        unfold_mask=None,
        colors=None,
        weighted_width=False,
        weighted_color=True,
        cmap="viridis",
        marker="o",
        markersize=0.02,
        linewidths=None,
        opacities=None,
        labels=None,
        vmax=None,
        vmin=None,
        grid=False,
        kticks=None,
        knames=None,
        elimit=None,
        ax=None,
        show=True,
        legend=False,
        savefig=None,
        plot_color_bar=True,
        title=None,
        kdirect=True,
        code="vasp",
        lobstercode="qe",
        unfold_mode=None,
        transformation_matrix=None,
        verbose=True,
):
    """


    Parameters
    ----------
    procar : TYPE, optional
        DESCRIPTION. The default is "PROCAR".
    abinit_output : TYPE, optional
        DESCRIPTION. The default is "abinit.out".
    outcar : TYPE, optional
        DESCRIPTION. The default is "OUTCAR".
    kpoints : TYPE, optional
        DESCRIPTION. The default is "KPOINTS".
    elkin : TYPE, optional
        DESCRIPTION. The default is "elk.in".
    mode : TYPE, optional
        DESCRIPTION. The default is "plain".
    spin_mode : TYPE, optional
        plain, magnetization, density, "spin_up", "spin_down", "both", "sx",
        "sy", "sz", "spin_texture"
        DESCRIPTION. The default is "plain".
    spins : TYPE, optional
        DESCRIPTION.
    atoms : TYPE, optional
        DESCRIPTION. The default is None.
    orbitals : TYPE, optional
        DESCRIPTION. The default is None.
    fermi : TYPE, optional
        DESCRIPTION. The default is None.
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    colors : TYPE, optional
        DESCRIPTION.
    cmap : TYPE, optional
        DESCRIPTION. The default is "jet".
    marker : TYPE, optional
        DESCRIPTION. The default is "o".
    markersize : TYPE, optional
        DESCRIPTION. The default is 0.02.
    linewidth : TYPE, optional
        DESCRIPTION. The default is 1.
    vmax : TYPE, optional
        DESCRIPTION. The default is None.
    vmin : TYPE, optional
        DESCRIPTION. The default is None.
    grid : TYPE, optional
        DESCRIPTION. The default is False.
    kticks : TYPE, optional
        DESCRIPTION. The default is None.
    knames : TYPE, optional
        DESCRIPTION. The default is None.
    elimit : TYPE, optional
        DESCRIPTION. The default is None.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    show : TYPE, optional
        DESCRIPTION. The default is True.
    savefig : TYPE, optional
        DESCRIPTION. The default is None.
    plot_color_bar : TYPE, optional
        DESCRIPTION. The default is True.
    title : TYPE, optional
        DESCRIPTION. The default is None.
    kdirect : TYPE, optional
        DESCRIPTION. The default is True.
    code : TYPE, optional
        DESCRIPTION. The default is "vasp".
    lobstercode : TYPE, optional
        DESCRIPTION. The default is "qe".
    verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """

    # import matplotlib
    # Roman ['rm', 'cal', 'it', 'tt', 'sf',
    plt.rcParams["mathtext.default"] = "regular"
    #                                                   'bf', 'default', 'bb', 'frak',
    #                                                   'circled', 'scr', 'regular']
    plt.rcParams["font.family"] = "Arial"
    plt.rc("font", size=22)  # controls default text sizes
    plt.rc("axes", titlesize=22)  # fontsize of the axes title
    plt.rc("axes", labelsize=22)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=22)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=22)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=22)    # legend fontsize
    # plt.rc('figure', titlesize=22)  # fontsize of the figure title

    # Turn interactive plotting off
    plt.ioff()

    # Verbose section

    # First handling the options, to get feedback to the user and check
    # that the input makes sense.
    # It is quite long
    structure = None
    reciprocal_lattice = None
    kpath = None

    if code == "vasp":
        if outcar is not None:
            outcar = vasp.Outcar(outcar)
            if fermi is None:
                fermi = outcar.efermi
            reciprocal_lattice = outcar.reciprocal_lattice
        if poscar is not None:
            poscar = vasp.Poscar(poscar)
            structure = poscar.structure
            if reciprocal_lattice is None:
                reciprocal_lattice = poscar.reciprocal_lattice

        if kpoints is not None:
            kpoints = vasp.Kpoints(kpoints)
            kpath = kpoints.kpath

        procar = vasp.Procar(procar, structure, reciprocal_lattice,
                             kpath, fermi, interpolation_factor=interpolation_factor)
        ebs_plot = EBSPlot(ebs = procar.ebs, kpath = kpath, ax = ax, spins=spins,
                           color=colors, opacity=opacities, linestyle=linestyles, linewidth=linewidths, label=labels)

    if unfold_mode is not None:
        if procar.has_phase and unfold_mode != "kpath":
            ebs_plot.ebs.unfold(
                transformation_matrix=transformation_matrix, structure=structure)
        # elif unfold_mode != "kpath":
        #     print("unfolding mode chosen : {}".format(unfold_mode))
        #     raise Exception(
        #         "Unfolding requires phases of band projections. If using VASP, use LORBIT=12")
        for isegment in range(ebs_plot.kpath.nsegments):
            for ip in range(2):
                ebs_plot.kpath.special_kpoints[isegment][ip] = np.dot(
                    np.linalg.inv(transformation_matrix),ebs_plot.kpath.special_kpoints[isegment][ip])

    if mode == "plain":
        ebs_plot.plot_bands()
    else:
        if atoms is not None and type(atoms[0]) is str:
            atoms_str = atoms
            atoms = []
            for iatom in np.unique(atoms_str):
                atoms = np.append(atoms, np.where(
                    structure.atoms == iatom)[0]).astype(np.int)

        if orbitals is not None and type(orbitals[0]) is str:
            orbital_str = orbitals
            orbital_names = {'s': 0,
                             'p': [1, 2, 3],
                             'd': [4, 5, 6, 7, 8],
                             'f': [9, 10, 11, 12, 13, 14, 15],
                             "py": 1,
                             "pz": 2,
                             "px": 3,
                             "dxy": 4,
                             "dyz": 5,
                             "dz2": 6,
                             "dxz": 7,
                             "x2-y2": 8,
                             "fy3x2": 9,
                             "fxyz": 10,
                             "fyz2": 11,
                             "fz3": 12,
                             "fxz2": 13,
                             "fzx2": 14,
                             }
            orbitals = []
            for iorb in orbital_str:
                orbitals = np.append(
                    orbitals, orbital_names[iorb]).astype(np.int)
        weights = ebs_plot.ebs.ebs_sum(
            atoms=atoms, principal_q_numbers=[-1], orbitals=orbitals, spins=spins)

        if weighted_color:
            color_weights = weights
        else:
            color_weights = None
        if weighted_width:
            width_weights = weights
        else:
            width_weights = None
        color_mask = projection_mask
        width_mask = projection_mask
        if unfold_mode == "thickness":
            width_weights = ebs_plot.ebs.weights
            width_mask = unfold_mask
        elif unfold_mode == "color":
            color_weights = ebs_plot.ebs.weights
            color_mask = unfold_mask
        elif unfold_mode is not None:
            if weighted_width:
                width_weights = ebs_plot.ebs.weights
                width_mask = unfold_mask
            if weighted_color:
                color_weights = ebs_plot.ebs.weights
                color_mask = unfold_mask

        if mode == "parametric":
            ebs_plot.plot_parameteric(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                # color_map=cmap,
                # plot_color_bar=plot_color_bar,
                vmin=vmin,
                vmax=vmax)
        elif mode == "scatter":
            ebs_plot.plot_scatter(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                # color_map=cmap,
                # plot_color_bar=plot_color_bar,
                vmin=vmin,
                vmax=vmax)
        else:
            print("Selected mode %s not valid. Please check the spelling " % mode)

    ebs_plot.set_xticks()
    ebs_plot.set_yticks(interval=elimit)
    ebs_plot.set_xlim()
    ebs_plot.set_ylim(elimit)
    ebs_plot.draw_fermi()
    ebs_plot.set_ylabel()
    if legend:
        ebs_plot.legend()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    return ebs_plot

    # if code == "vasp" or code == "abinit":
    #     if repair:
    #         repairhandle = UtilsProcar()
    #         repairhandle.ProcarRepair(procarfile, procarfile)

#     if atoms is None:
#         atoms = [-1]
#         if human is True:
#             print("WARNING: `--human` option given without atoms list!")
#             print("--human will be set to False (ignored)\n ")
#             human = False
#     if orbitals is None:
#         orbitals = [-1]
#     if verbose:
#         welcome()
#         print("Script initiated...")
#         if repair:
#             print("PROCAR repaired. Run with repair=False next time.")
#         print("code           : ", code)
#         print("input file     : ", procarfile)
#         print("mode           : ", mode)
#         print("spin comp.     : ", spin)
#         print("atoms list     : ", atoms)
#         print("orbs. list     : ", orbitals)

#     if (
#         fermi is None
#         and outcar is None
#         and abinit_output is None
#         and (code != "elk" and code != "qe" and code != "lobster")
#     ):
#         print(
#             "WARNING : Fermi Energy not set! Please set manually or provide output file and set code type."
#         )
#         fermi = 0

#     elif fermi is None and code == "elk":
#         fermi = None

#     elif fermi is None and code == "qe":
#         fermi = None

#     elif fermi is None and code == "lobster":
#         fermi = None

#     print("fermi energy   : ", fermi)
#     print("energy range   : ", elimit)

#     if mask is not None:
#         print("masking thres. : ", mask)

#     print("colormap       : ", cmap)
#     print("markersize     : ", markersize)
#     print("permissive     : ", permissive)
#     if permissive:
#         print("INFO : Permissive flag is on! Be careful")
#     print("vmax           : ", vmax)
#     print("vmin           : ", vmin)
#     print("grid enabled   : ", grid)
#     if human is not None:
#         print("human          : ", human)
#     print("savefig        : ", savefig)
#     print("title          : ", title)
#     print("outcar         : ", outcar)

#     if kdirect:
#         print("k-grid         :  reduced")
#     else:
#         print(
#             "k-grid         :  cartesian (Remember to provide an output file for this case to work.)"
#         )

#     if discontinuities is None:
#         discontinuities = []

#     #### READING KPOINTS FILE IF PRESENT ####

#     # If KPOINTS file is given:
#     if kpointsfile is not None:
#         # Getting the high symmetry point names from KPOINTS file
#         f = open(kpointsfile)
#         KPread = f.read()
#         f.close()

#         KPmatrix = re.findall("reciprocal[\s\S]*", KPread)
#         tick_labels = np.array(re.findall("!\s(.*)", KPmatrix[0]))
#         knames = []
#         knames = [tick_labels[0]]

#         ################## Checking for discontinuities ########################
#         discont_indx = []
#         icounter = 1
#         while icounter < len(tick_labels) - 1:
#             if tick_labels[icounter] == tick_labels[icounter + 1]:
#                 knames.append(tick_labels[icounter])
#                 icounter = icounter + 2
#             else:
#                 discont_indx.append(icounter)
#                 knames.append(tick_labels[icounter] +
#                               "|" + tick_labels[icounter + 1])
#                 icounter = icounter + 2
#         knames.append(tick_labels[-1])
#         discont_indx = list(dict.fromkeys(discont_indx))

#         ################# End of discontinuity check ##########################

#         # Added by Nicholas Pike to modify the output of seekpath to allow for
#         # latex rendering.
#         for i in range(len(knames)):
#             if knames[i] == "GAMMA":
#                 knames[i] = "\Gamma"
#             else:
#                 pass

#         knames = [str("$" + latx + "$") for latx in knames]

#         # getting the number of grid points from the KPOINTS file
#         f2 = open(kpointsfile)
#         KPreadlines = f2.readlines()
#         f2.close()
#         numgridpoints = int(KPreadlines[1].split()[0])

#         kticks = [0]
#         gridpoint = 0
#         for kt in range(len(knames) - 1):
#             gridpoint = gridpoint + numgridpoints
#             kticks.append(gridpoint - 1)

#         print("knames         : ", knames)
#         print("kticks         : ", kticks)

#         # creating an array for discontunuity k-points. These are the indexes
#         # of the discontinuity k-points.
#         for k in discont_indx:
#             discontinuities.append(kticks[int(k / 2) + 1])
#         if discontinuities:
#             print("discont. list  : ", discontinuities)

#     #### END OF KPOINTS FILE DEPENDENT SECTION ####

#     # The spin argument should be a number (index of an array), or
#     # 'st'. In the last case it will be handled separately (later)

#     spin = {"0": 0, "1": 1, "2": 2, "3": 3, "st": "st"}[str(spin)]

#     #### parsing the PROCAR file or equivalent to retrieve spd data ####
#     code = code.lower()
#     if code == "vasp":
#         procarFile = ProcarParser()

#     elif code == "abinit":
#         procarFile = ProcarParser()
#         abinitFile = AbinitParser(abinit_output=abinit_output)

#     elif code == "elk":
#         # reciprocal lattice already taken care of
#         procarFile = ElkParser(kdirect=kdirect)

#         # Retrieving knames and kticks from Elk
#         if kticks is None and knames is None:
#             if procarFile.kticks and procarFile.knames:
#                 kticks = procarFile.kticks
#                 knames = procarFile.knames

#     elif code == "qe":
#         # reciprocal lattice already taken care of
#         procarFile = QEParser(kdirect=kdirect)

#         # Retrieving knames and kticks from QE
#         if kticks is None and knames is None:
#             if procarFile.kticks and procarFile.knames:
#                 kticks = procarFile.kticks
#                 knames = procarFile.knames

#             # Retrieving discontinuities if present
#             if procarFile.discontinuities:
#                 discontinuities = procarFile.discontinuities

#     elif code == "lobster":
#         # reciprocal lattice already taken care of
#         procarFile = LobsterParser(kdirect=kdirect, lobstercode=lobstercode)

#         # Retrieving knames and kticks from Lobster
#         if kticks is None and knames is None:
#             if procarFile.kticks and procarFile.knames:
#                 kticks = procarFile.kticks
#                 knames = procarFile.knames

#             # Retrieving discontinuities if present
#             if procarFile.discontinuities:
#                 discontinuities = procarFile.discontinuities

#     # If ticks and names are given by the user manually:
#     if kticks is not None and knames is not None:
#         ticks = list(zip(kticks, knames))
#     elif kticks is not None:
#         ticks = list(zip(kticks, kticks))
#     else:
#         ticks = None

#     if kpointsfile is None and kticks and knames:
#         print("knames         : ", knames)
#         print("kticks         : ", kticks)
#         if discontinuities:
#             print("discont. list  : ", discontinuities)

#     # The second part of this function is parse/select/use the data in
#     # OUTCAR (if given) and PROCAR

#     # first parse the outcar if given, to get Efermi and Reciprocal lattice
#     recLat = None
#     if code == "vasp":
#         if outcar:
#             outcarparser = UtilsProcar()
#             if fermi is None:
#                 fermi = outcarparser.FermiOutcar(outcar)
#                 print("Fermi energy   :  %s eV (from OUTCAR)" % str(fermi))
#             recLat = outcarparser.RecLatOutcar(outcar)

#     elif code == "elk":
#         if fermi is None:
#             fermi = procarFile.fermi
#             print("Fermi energy    :  %s eV (from Elk output)" % str(fermi))

#     elif code == "qe":
#         if fermi is None:
#             fermi = procarFile.fermi
#             print("Fermi energy   :  %s eV (from Quantum Espresso output)" %
#                   str(fermi))

#     elif code == "lobster":
#         if fermi is None:
#             fermi = procarFile.fermi
#             print("Fermi energy   :  %s eV (from Lobster output)" % str(fermi))
#             # lobster already shifts fermi so we set it to zero here.
#             fermi = 0.0

#     elif code == "abinit":
#         if fermi is None:
#             fermi = abinitFile.fermi
#             print("Fermi energy   :  %s eV (from Abinit output)" % str(fermi))
#         recLat = abinitFile.reclat

#     # if kdirect = False, then the k-points will be in cartesian coordinates.
#     # The output should be read to find the reciprocal lattice vectors to transform
#     # from direct to cartesian

#     if code == "vasp":
#         if kdirect:
#             procarFile.readFile(procarfile, permissive)
#         else:
#             procarFile.readFile(procarfile, permissive, recLattice=recLat)

#     elif code == "abinit":
#         if kdirect:
#             procarFile.readFile(procarfile, permissive)
#         else:
#             procarFile.readFile(procarfile, permissive, recLattice=recLat)

#     # processing the data, getting an instance of the class that reduces the data
#     data = ProcarSelect(procarFile, deepCopy=True, mode=mode)
#     numofbands = int(data.spd.shape[1] / 2)

#     # Unit conversions

#     # Abinit PROCAR has band energy units in Hartree. We need it in eV.
#     if code == "abinit":
#         data.bands = data.bands * 27.211396641308

#     # handling the spin, `spin='st'` is not straightforward, needs
#     # to calculate the k vector and its normal. Other `spin` values
#     # are trivial.
#     if spin == "st":
#         # two `ProcarSelect` instances, to store temporal values: spin_x, spin_y
#         dataX = ProcarSelect(procarFile, deepCopy=True)
#         dataX.selectIspin([1])
#         dataX.selectAtoms(atoms, fortran=human)
#         dataX.selectOrbital(orbitals)
#         dataY = ProcarSelect(procarFile, deepCopy=True)
#         dataY.selectIspin([2])
#         dataY.selectAtoms(atoms, fortran=human)
#         dataY.selectOrbital(orbitals)
#         # getting the signed angle of each K-vector
#         angle = np.arctan2(dataX.kpoints[:, 1],
#                             (dataX.kpoints[:, 0] + 0.000000001))
#         sin = np.sin(angle)
#         cos = np.cos(angle)
#         sin.shape = (sin.shape[0], 1)
#         cos.shape = (cos.shape[0], 1)
#         # print sin, cos
#         # storing the spin projection into the original array
#         data.spd = -sin * dataX.spd + cos * dataY.spd
#     else:
#         data.selectIspin([spin], separate=separate)
#         data.selectAtoms(atoms, fortran=human)
#         data.selectOrbital(orbitals)

#     # Plotting the data
#     if separate:
#         if spin == 0:
#             # plotting spin up bands separately
#             data.bands = (
#                 data.bands[:, :numofbands].transpose() - np.array(fermi)
#             ).transpose()
#             print("Plotting spin up bands...")

#         elif spin == 1:
#             # plotting spin down bands separately
#             data.bands = (
#                 data.bands[:, numofbands:].transpose() - np.array(fermi)
#             ).transpose()
#             print("Plotting spin down bands...")

#         plot = ProcarPlot(data.bands, data.spd, data.kpoints)

#     else:
#         # Regular plotting method. For spin it plots density or magnetization.
#         data.bands = (data.bands.transpose() - np.array(fermi)).transpose()
#         plot = ProcarPlot(data.bands, data.spd, data.kpoints)

#     ###### start of mode dependent options #########

#     if mode == "scatter":
#         fig, ax1 = plot.scatterPlot(
#             mask=mask,
#             size=markersize,
#             cmap=cmap,
#             vmin=vmin,
#             vmax=vmax,
#             marker=marker,
#             ticks=ticks,
#             discontinuities=discontinuities,
#             ax=ax,
#         )
#         if fermi is not None:
#             ax1.set_ylabel(r"$E-E_f$ [eV]")
#         else:
#             ax1.set_ylabel(r"Energy [eV]")
#         if elimit is not None:
#             ax1.set_ylim(elimit)

#     elif mode == "plain":
#         fig, ax1 = plot.plotBands(
#             markersize,
#             marker=marker,
#             ticks=ticks,
#             color=color,
#             discontinuities=discontinuities,
#             ax=ax,
#         )
#         if fermi is not None:
#             ax1.set_ylabel(r"$E-E_f$ [eV]")
#         else:
#             ax1.set_ylabel(r"Energy [eV]")
#         if elimit:
#             ax1.set_ylim(elimit)

#     elif mode == "parametric":
#         fig, ax1 = plot.parametricPlot(
#             cmap=cmap,
#             vmin=vmin,
#             vmax=vmax,
#             mask=mask,
#             ticks=ticks,
#             discontinuities=discontinuities,
#             plot_bar=plot_color_bar,
#             ax=ax,
#             linewidth=linewidth,
#         )
#         if fermi is not None:
#             ax1.set_ylabel(r"$E-E_f$ [eV]")
#         else:
#             ax1.set_ylabel(r"Energy [eV]")
#         if elimit is not None:
#             ax1.set_ylim(elimit)

#     elif mode == "atomic":
#         fig, ax1 = plot.atomicPlot(cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
#         if fermi is not None:
#             ax1.set_ylabel(r"$E-E_f$ [eV]")
#         else:
#             ax1.set_ylabel(r"Energy [eV]")
#         if elimit is not None:
#             ax1.set_ylim(elimit)
#     ###### end of mode dependent options ###########

#     if grid:
#         ax1.grid()

#     if title:
#         ax1.set_title(title)

#     else:
#         if savefig:
#             plt.savefig(savefig, bbox_inches="tight")
#             plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
#             return None, None
#         else:
#             if show:
#                 plt.show()
#             else:
#                 pass
#         return fig, ax1

# # if __name__ == "__main__":
# # bandsplot(mode='parametric',elimit=[-6,6],orbitals=[4,5,6,7,8],vmin=0,vmax=1, code='elk')
# # knames=['$\Gamma$', '$X$', '$M$', '$\Gamma$', '$R$','$X$'],
# # kticks=[0, 8, 16, 24, 38, 49])
