"""
Created on May 17 2020
@author: Pedram Tavadze
"""
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .doscarplot import DosPlot
from .vaspxml import VaspXML
from .lobsterparser import LobsterDOSParser, LobsterParser
from .qeparser import QEDOSParser, QEParser

from .abinitparser import AbinitParser
from .doscarplot import DosPlot
from .elkparser import ElkParser
from .procarparser import ProcarParser
from .procarplot import ProcarPlot
from .procarselect import ProcarSelect
from .splash import welcome
from .utilsprocar import UtilsProcar

plt.rcParams["mathtext.default"] = "regular"
# Roman ['rm', 'cal', 'it', 'tt', 'sf',
#        'bf', 'default', 'bb', 'frak',
#        'circled', 'scr', 'regular']
plt.rcParams["font.family"] = "Arial"
plt.rc("font", size=18)  # controls default text sizes
plt.rc("axes", titlesize=22)  # fontsize of the axes title
plt.rc("axes", labelsize=22)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=22)  # fontsize of the tick labels
plt.rc("ytick", labelsize=22)  # fontsize of the tick labels
plt.rc("legend", fontsize=18)  # legend fontsize
# plt.rc('figure', titlesize=22)  # fontsize of the figure title


def bandsdosplot(
    bands_file="PROCAR",
    dos_file="vasprun.xml",
    outcar="OUTCAR",
    abinit_output=None,
    bands_mode="plain",
    dos_mode="plain",
    plot_total=True,
    fermi=None,
    mask=None,
    markersize=0.02,
    marker="o",
    atoms=None,
    orbitals=None,
    bands_spin=0,
    bands_separate=False,
    dos_spins=None,
    dos_labels=None,
    dos_spin_colors=[(1, 0, 0), (0, 0, 1)],
    dos_colors=None,
    dos_items=None,
    dos_interpolation_factor=None,
    dlimit=None,
    elimit=None,
    vmin=None,
    vmax=None,
    cmap="jet",
    grid=False,
    kpointsfile="KPOINTS",
    code="vasp",
    savefig=None,
    title=None,
    kdirect=True,
    discontinuities=None,
    plot_color_bar=True,
    repair=True,
    show=True,
):
    """This function creates plots containing both DOS and bands."""

    welcome()

    # Repair PROCAR
    if code == "vasp" or code == "abinit":
        if repair:
            repairhandle = UtilsProcar()
            repairhandle.ProcarRepair(bands_file, bands_file)

    fig = plt.figure(figsize=(13, 7), constrained_layout=False)
    widths = [13, 5]
    heights = [9]
    gs = fig.add_gridspec(1, 2, width_ratios=widths, height_ratios=heights)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    total = plot_total

    if atoms is None:
        bands_atoms = [-1]
    else:
        bands_atoms = atoms.copy()
    if orbitals is None:
        bands_orbitals = [-1]
    else:
        bands_orbitals = (
            orbitals.copy()
        )  # these copying is because orbitals select changes shifts orbitals by one

    if dos_spins is None:
        dos_spins = [0, 1]

    if dos_mode not in [
        "plain",
        "parametric_line",
        "parametric",
        "stack_species",
        "stack_orbitals",
        "stack",
    ]:
        raise ValueError(
            "Mode should be choosed from ['plain', 'parametric_line','parametric','stack_species','stack_orbitals','stack']"
        )

    # Verbose section

    print("Script initiated...")
    if repair:
        print("PROCAR repaired. Run with repair=False next time.")
    print("code           : ", code)
    print("bands file     : ", bands_file)
    print("bands mode     : ", bands_mode)
    print("bands spin     : ", bands_spin)
    print("dos file       : ", dos_file)
    print("dos mode       : ", dos_mode)
    print("dos spins      : ", dos_spins)
    print("atoms list     : ", atoms)
    print("orbs. list     : ", orbitals)

    if fermi is None and outcar is None and abinit_output is None:
        print(
            "WARNING : Fermi Energy not set! Please set manually or provide output file and set code type."
        )
        fermi = 0

    print("fermi energy   : ", fermi)
    print("energy range   : ", elimit)

    if mask is not None:
        print("masking thres. : ", mask)

    print("colormap       : ", cmap)
    print("markersize     : ", markersize)
    print("vmax           : ", vmax)
    print("vmin           : ", vmin)
    print("grid enabled   : ", grid)
    print("savefig        : ", savefig)
    print("title          : ", title)
    print("outcar         : ", outcar)

    if kdirect:
        print("k-grid         : reduced")
    else:
        print(
            "k-grid         : cartesian (Remember to provide an output file for this case to work.)"
        )

    if discontinuities is None:
        discontinuities = []

    #### READING KPOINTS FILE IF PRESENT ####

    # If KPOINTS file is given:
    if code == "vasp":

        if kpointsfile is not None:
            # Getting the high symmetry point names from KPOINTS file
            f = open(kpointsfile)
            KPread = f.read()
            f.close()

            KPmatrix = re.findall("reciprocal[\s\S]*", KPread)
            tick_labels = np.array(re.findall("!\s(.*)", KPmatrix[0]))
            knames = []
            knames = [tick_labels[0]]

            ################## Checking for discontinuities ########################
            discont_indx = []
            icounter = 1
            while icounter < len(tick_labels) - 1:
                if tick_labels[icounter] == tick_labels[icounter + 1]:
                    knames.append(tick_labels[icounter])
                    icounter = icounter + 2
                else:
                    discont_indx.append(icounter)
                    knames.append(
                        tick_labels[icounter] + "|" + tick_labels[icounter + 1]
                    )
                    icounter = icounter + 2
            knames.append(tick_labels[-1])
            discont_indx = list(dict.fromkeys(discont_indx))

            ################# End of discontinuity check ##########################

            # Added by Nicholas Pike to modify the output of seekpath to allow for
            # latex rendering.
            for i in range(len(knames)):
                if knames[i] == "GAMMA":
                    knames[i] = "\Gamma"
                else:
                    pass

            knames = [str("$" + latx + "$") for latx in knames]

            # getting the number of grid points from the KPOINTS file
            f2 = open(kpointsfile)
            KPreadlines = f2.readlines()
            f2.close()
            numgridpoints = int(KPreadlines[1].split()[0])

            kticks = [0]
            gridpoint = 0
            for kt in range(len(knames) - 1):
                gridpoint = gridpoint + numgridpoints
                kticks.append(gridpoint - 1)

            print("knames         : ", knames)
            print("kticks         : ", kticks)

            # creating an array for discontunuity k-points. These are the indexes
            # of the discontinuity k-points.
            for k in discont_indx:
                discontinuities.append(kticks[int(k / 2) + 1])
            if discontinuities:
                print("discont. list  : ", discontinuities)

    elif code == "lobster":
        procarFile = LobsterParser()

        kticks = procarFile.kticks
        knames = procarFile.knames
        print("knames         : ", knames)
        print("kticks         : ", kticks)
        if procarFile.discontinuities:
            discontinuities = procarFile.discontinuities
        vaspxml = LobsterDOSParser()
        dos_plot = DosPlot(dos=vaspxml.dos, structure=vaspxml.structure)
        if dos_spins is None:
            dos_spins = np.arange(len(vaspxml.dos.total))

    elif code == "qe":
        procarFile = QEParser()

        kticks = procarFile.kticks
        knames = procarFile.knames
        print("knames         : ", knames)
        print("kticks         : ", kticks)
        # Retrieving knames and kticks from QE
        if procarFile.discontinuities:
            discontinuities = procarFile.discontinuities
        vaspxml = QEDOSParser()
        dos_plot = DosPlot(dos=vaspxml.dos, structure=vaspxml.structure)
        if dos_spins is None:
            dos_spins = np.arange(len(vaspxml.dos.total))
    #### END OF KPOINTS FILE DEPENDENT SECTION ####

    # spin = {"0": 0, "1": 1, "2": 2, "3": 3, "st": "st"}[str(spin)]

    #### parsing the PROCAR file or equivalent to retrieve spd data ####
    code = code.lower()
    if code == "vasp":
        procarFile = ProcarParser()
        vaspxml = VaspXML(
            filename=dos_file, dos_interpolation_factor=dos_interpolation_factor
        )
        dos_plot = DosPlot(dos=vaspxml.dos, structure=vaspxml.structure)

        if dos_spins is None:
            dos_spins = np.arange(len(vaspxml.dos.total))

    # elif code == "lobster":
    #     procarFile = LobsterParser()
    #     vaspxml = LobsterDOSParser()
    #     dos_plot = DosPlot(dos=vaspxml.dos, structure=vaspxml.structure)

    # If ticks and names are given by the user manually:
    if kticks is not None and knames is not None:
        ticks = list(zip(kticks, knames))
    elif kticks is not None:
        ticks = list(zip(kticks, kticks))
    else:
        ticks = None

    if kpointsfile is None and kticks and knames:
        print("knames         : ", knames)
        print("kticks         : ", kticks)
        if discontinuities:
            print("discont. list  : ", discontinuities)

    # The second part of this function is parse/select/use the data in
    # OUTCAR (if given) and PROCAR

    # first parse the outcar if given, to get Efermi and Reciprocal lattice
    recLat = None
    if code == "vasp":
        if outcar:
            outcarparser = UtilsProcar()
            if fermi is None:
                fermi = outcarparser.FermiOutcar(outcar)
                print("Fermi energy   :  %s eV (from OUTCAR)" % str(fermi))
            recLat = outcarparser.RecLatOutcar(outcar)
    elif code == "lobster":
        fermi = procarFile.fermi
        recLat = procarFile.reclat
    elif code == "qe":
        fermi = procarFile.fermi
        recLat = procarFile.reclat
    # if kdirect = False, then the k-points will be in cartesian coordinates.
    # The output should be read to find the reciprocal lattice vectors to transform
    # from direct to cartesian

    if code == "vasp":
        if kdirect:
            procarFile.readFile(bands_file, permissive=False)
        else:
            procarFile.readFile(bands_file, permissive=False, recLattice=recLat)

    # processing the data, getting an instance of the class that reduces the data
    data = ProcarSelect(procarFile, deepCopy=True, mode=bands_mode)
    numofbands = int(data.spd.shape[1] / 2)

    # handling the spin, `spin='st'` is not straightforward, needs
    # to calculate the k vector and its normal. Other `spin` values
    # are trivial.

    if bands_spin == "st":
        # two `ProcarSelect` instances, to store temporal values: spin_x, spin_y
        dataX = ProcarSelect(procarFile, deepCopy=True)
        dataX.selectIspin([1])
        dataX.selectAtoms(bands_atoms, fortran=False)

        dataX.selectOrbital(bands_orbitals)
        dataY = ProcarSelect(procarFile, deepCopy=True)
        dataY.selectIspin([2])
        dataY.selectAtoms(bands_atoms, fortran=False)
        dataY.selectOrbital(bands_orbitals)
        # getting the signed angle of each K-vector
        angle = np.arctan2(dataX.kpoints[:, 1], (dataX.kpoints[:, 0] + 0.000000001))
        sin = np.sin(angle)
        cos = np.cos(angle)
        sin.shape = (sin.shape[0], 1)
        cos.shape = (cos.shape[0], 1)
        # print sin, cos
        # storing the spin projection into the original array
        data.spd = -sin * dataX.spd + cos * dataY.spd
    else:
        data.selectIspin([bands_spin], separate=bands_separate)
        data.selectAtoms(bands_atoms, fortran=False)
        data.selectOrbital(bands_orbitals)

    # Plotting the data
    if bands_separate:
        if bands_spin == 0:
            # plotting spin up bands separately
            data.bands = (
                data.bands[:, :numofbands].transpose() - np.array(fermi)
            ).transpose()
            print("Plotting spin up bands...")

        elif bands_spin == 1:
            # plotting spin down bands separately
            data.bands = (
                data.bands[:, numofbands:].transpose() - np.array(fermi)
            ).transpose()
            print("Plotting spin down bands...")

        plot = ProcarPlot(data.bands, data.spd, data.kpoints, ax=ax1)

    else:
        # Regular plotting method. For spin it plots density or magnetization.
        if code == "lobster" or code == "qe":
            data.bands = (data.bands.transpose()).transpose()
        else:
            data.bands = (data.bands.transpose() - np.array(fermi)).transpose()

        plot = ProcarPlot(data.bands, data.spd, data.kpoints)

    if vmin is None:
        vmin = plot.spd.min()
    if vmax is None:
        vmax = plot.spd.max()

    ###### start of mode dependent options #########
    if bands_mode == "scatter":
        _, ax1 = plot.scatterPlot(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ticks=ticks,
            discontinuities=discontinuities,
            ax=ax1,
            mask=mask,
        )
        if fermi is not None:
            ax1.set_ylabel(r"$E-E_f$ [eV]")
        else:
            ax1.set_ylabel(r"Energy [eV]")
        if elimit is not None:
            ax1.set_ylim(elimit)

    elif bands_mode == "plain":
        _, ax1 = plot.plotBands(ticks=ticks, discontinuities=discontinuities, ax=ax1,)
        if fermi is not None:
            ax1.set_ylabel(r"$E-E_f$ [eV]")
        else:
            ax1.set_ylabel(r"Energy [eV]")
        if elimit:
            ax1.set_ylim(elimit)

    elif bands_mode == "parametric":
        if dos_mode == "parametric":
            plot_color_bar = False
        else:
            plot_color_bar = True
        _, ax1 = plot.parametricPlot(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ticks=ticks,
            discontinuities=discontinuities,
            ax=ax1,
            plot_bar=plot_color_bar,
            mask=mask,
        )

        if fermi is not None:
            ax1.set_ylabel(r"$E-E_f$ [eV]")
        else:
            ax1.set_ylabel(r"Energy [eV]")
        if elimit is not None:
            ax1.set_ylim(elimit)

    if dos_mode == "plain":
        _, ax2 = dos_plot.plot_total(
            spins=dos_spins, ax=ax2, orientation="vertical", labels=dos_labels
        )

    elif dos_mode == "parametric_line":
        if not total:
            _, ax2 = dos_plot.plot_parametric_line(
                atoms=atoms,
                spins=dos_spins,
                orbitals=orbitals,
                spin_colors=dos_spin_colors,
                ax=ax2,
                orientation="vertical",
                labels=dos_labels,
            )

        else:
            _, ax2 = dos_plot.plot_total(
                spins=dos_spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax2,
                orientation="vertical",
            )

            _, ax2 = dos_plot.plot_parametric_line(
                atoms=atoms,
                spins=dos_spins,
                orbitals=orbitals,
                spin_colors=dos_spin_colors,
                ax=ax2,
                orientation="vertical",
                labels=dos_labels,
            )
    elif dos_mode == "parametric":
        if not total:
            _, ax2 = dos_plot.plot_parametric(
                atoms=atoms,
                spins=dos_spins,
                orbitals=orbitals,
                spin_colors=dos_spin_colors,
                cmap=cmap,
                elimit=elimit,
                ax=ax2,
                vmin=vmin,
                vmax=vmax,
                orientation="vertical",
                labels=dos_labels,
            )

        else:
            _, ax1 = dos_plot.plot_parametric(
                atoms=atoms,
                spins=dos_spins,
                orbitals=orbitals,
                spin_colors=dos_spin_colors,
                cmap=cmap,
                elimit=elimit,
                ax=ax2,
                vmin=vmin,
                vmax=vmax,
                orientation="vertical",
                labels=dos_labels,
            )

            _, ax2 = dos_plot.plot_total(
                spins=dos_spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax2,
                orientation="vertical",
            )
        ax2.yaxis.set_visible(False)
    elif dos_mode == "stack_species":
        if not total:
            _, ax2 = dos_plot.plot_stack_species(
                spins=dos_spins,
                spin_colors=dos_spin_colors,
                colors=dos_colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax2,
                orientation="vertical",
            )
            # dos = dos_plot.dos_total
        else:
            _, ax2 = dos_plot.plot_stack_species(
                spins=dos_spins,
                spin_colors=dos_spin_colors,
                colors=dos_colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax2,
                orientation="vertical",
            )
            # dos = dos_plot.dos_total
            _, ax2 = dos_plot.plot_total(
                spins=dos_spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax2,
                orientation="vertical",
            )
    elif dos_mode == "stack_orbitals":
        if not total:
            _, ax2 = dos_plot.plot_stack_orbitals(
                spins=dos_spins,
                spin_colors=dos_spin_colors,
                colors=dos_colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax2,
                orientation="vertical",
            )
        else:
            _, ax2 = dos_plot.plot_stack_orbitals(
                spins=dos_spins,
                spin_colors=dos_spin_colors,
                colors=dos_colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax2,
                orientation="vertical",
            )

            _, ax2 = dos_plot.plot_total(
                spins=dos_spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax2,
                orientation="vertical",
            )
    elif dos_mode == "stack":
        if not total:
            _, ax2 = dos_plot.plot_stack(
                items=dos_items,
                spins=dos_spins,
                spin_colors=dos_spin_colors,
                colors=dos_colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax2,
                orientation="vertical",
            )

        else:
            _, ax2 = dos_plot.plot_stack(
                items=dos_items,
                spins=dos_spins,
                spin_colors=dos_spin_colors,
                colors=dos_colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax2,
                orientation="vertical",
            )

            _, ax2 = dos_plot.plot_total(
                spins=dos_spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax2,
                orientation="vertical",
            )

    ax2.set_ylim(elimit)
    ax2.axhline(color="black", linestyle="--")
    ax2.axvline(color="black", linestyle="--")
    if grid:
        ax1.grid()
        ax2.grid()
    #    ax2.yaxis.set_ticklabels([])

    if dos_labels or "stack" in dos_mode:
        ax2.legend()
    ax2.yaxis.set_visible(False)
    cond1 = vaspxml.dos.energies >= elimit[0]
    cond2 = vaspxml.dos.energies <= elimit[1]
    cond = np.all([cond1, cond2], axis=0)
    if len(dos_spins) > 1:
        ylim = [
            vaspxml.dos.total[1, cond].max() * -1.1,
            vaspxml.dos.total[0, cond].max() * 1.1,
        ]
    else:

        ylim = [0, vaspxml.dos.total[dos_spins[0], cond].max() * 1.1]

    if dlimit is not None:
        if len(dlimit) == 2:
            ax2.set_xlim(dlimit[0], dlimit[1])
        else :
            ax2.set_xlim(dlimit[0],)
    elif (
        dos_mode == "stack_species"
        or dos_mode == "stack_orbitals"
        or dos_mode == "stack"
        or dos_mode == "parametric"
        or dos_mode == "parametric_line"
    ):
        # ax2.set_xlim(ax2.get_xlim()[0], ax2.get_xlim()[1] * 1.1)
        ax2.set_xlim(ylim[0], ylim[1])

    elif dos_mode == "plain":
        ax2.set_xlim(ylim[0], ylim[1])
    if title:
        ax1.set_title(title)

    fig.tight_layout()

    if savefig:
        fig.savefig(savefig, bbox_inches="tight")
        plt.close()
        return None, None
    if show:
        plt.show()
    return fig, ax1, ax2
