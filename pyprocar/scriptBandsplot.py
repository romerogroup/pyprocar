import re

import matplotlib.pyplot as plt
import numpy as np

from .abinitparser import AbinitParser
from .elkparser import ElkParser
from .qeparser import QEParser
from .lobsterparser import LobsterParser
from .procarparser import ProcarParser
from .procarplot import ProcarPlot
from .procarselect import ProcarSelect
from .splash import welcome
from .utilsprocar import UtilsProcar

# import matplotlib
plt.rcParams["mathtext.default"] = "regular"  # Roman ['rm', 'cal', 'it', 'tt', 'sf',
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


def bandsplot(
    procarfile=None,
    mode="plain",
    color="blue",
    abinit_output=None,
    spin=0,
    atoms=None,
    orbitals=None,
    fermi=None,
    elimit=[-2, 2],
    mask=None,
    markersize=0.02,
    cmap="jet",
    vmax=None,
    vmin=None,
    grid=True,
    marker="o",
    permissive=False,
    human=False,
    savefig=None,
    kticks=None,
    knames=None,
    title=None,
    outcar=None,
    kpointsfile=None,
    kdirect=True,
    code="vasp",
    separate=False,
    ax=None,
    discontinuities=None,
    show=True,
    lobstercode="qe",
    plot_color_bar=True,
    verbose=True,
    linewidth=1,
    repair=True,
):
    """This function plots band structures
  """

    # Turn interactive plotting off
    plt.ioff()

    # Verbose section

    # First handling the options, to get feedback to the user and check
    # that the input makes sense.
    # It is quite long

    if code == "vasp" or code == "abinit":
        if repair:
            repairhandle = UtilsProcar()
            repairhandle.ProcarRepair(procarfile, procarfile)

    if atoms is None:
        atoms = [-1]
        if human is True:
            print("WARNING: `--human` option given without atoms list!")
            print("--human will be set to False (ignored)\n ")
            human = False
    if orbitals is None:
        orbitals = [-1]
    if verbose:
        welcome()
        print("Script initiated...")
        if repair:
            print("PROCAR repaired. Run with repair=False next time.")
        print("code           : ", code)
        print("input file     : ", procarfile)
        print("mode           : ", mode)
        print("spin comp.     : ", spin)
        print("atoms list     : ", atoms)
        print("orbs. list     : ", orbitals)

    if (
        fermi is None
        and outcar is None
        and abinit_output is None
        and (code != "elk" and code != "qe" and code != "lobster")
    ):
        print(
            "WARNING : Fermi Energy not set! Please set manually or provide output file and set code type."
        )
        fermi = 0

    elif fermi is None and code == "elk":
        fermi = None

    elif fermi is None and code == "qe":
        fermi = None

    elif fermi is None and code == "lobster":
        fermi = None

    print("fermi energy   : ", fermi)
    print("energy range   : ", elimit)

    if mask is not None:
        print("masking thres. : ", mask)

    print("colormap       : ", cmap)
    print("markersize     : ", markersize)
    print("permissive     : ", permissive)
    if permissive:
        print("INFO : Permissive flag is on! Be careful")
    print("vmax           : ", vmax)
    print("vmin           : ", vmin)
    print("grid enabled   : ", grid)
    if human is not None:
        print("human          : ", human)
    print("savefig        : ", savefig)
    print("title          : ", title)
    print("outcar         : ", outcar)

    if kdirect:
        print("k-grid         :  reduced")
    else:
        print(
            "k-grid         :  cartesian (Remember to provide an output file for this case to work.)"
        )

    if discontinuities is None:
        discontinuities = []

    #### READING KPOINTS FILE IF PRESENT ####

    # If KPOINTS file is given:
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
                knames.append(tick_labels[icounter] + "|" + tick_labels[icounter + 1])
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

    #### END OF KPOINTS FILE DEPENDENT SECTION ####

    # The spin argument should be a number (index of an array), or
    # 'st'. In the last case it will be handled separately (later)

    spin = {"0": 0, "1": 1, "2": 2, "3": 3, "st": "st"}[str(spin)]

    #### parsing the PROCAR file or equivalent to retrieve spd data ####
    code = code.lower()
    if code == "vasp":
        procarFile = ProcarParser()

    elif code == "abinit":
        procarFile = ProcarParser()
        abinitFile = AbinitParser(abinit_output=abinit_output)

    elif code == "elk":
        # reciprocal lattice already taken care of
        procarFile = ElkParser(kdirect=kdirect)

        # Retrieving knames and kticks from Elk
        if kticks is None and knames is None:
            if procarFile.kticks and procarFile.knames:
                kticks = procarFile.kticks
                knames = procarFile.knames

    elif code == "qe":
        # reciprocal lattice already taken care of
        procarFile = QEParser(kdirect=kdirect)

        # Retrieving knames and kticks from QE
        if kticks is None and knames is None:
            if procarFile.kticks and procarFile.knames:
                kticks = procarFile.kticks
                knames = procarFile.knames

            # Retrieving discontinuities if present
            if procarFile.discontinuities:
                discontinuities = procarFile.discontinuities

    elif code == "lobster":
        # reciprocal lattice already taken care of
        procarFile = LobsterParser(kdirect=kdirect, lobstercode=lobstercode)

        # Retrieving knames and kticks from Lobster
        if kticks is None and knames is None:
            if procarFile.kticks and procarFile.knames:
                kticks = procarFile.kticks
                knames = procarFile.knames

            # Retrieving discontinuities if present
            if procarFile.discontinuities:
                discontinuities = procarFile.discontinuities

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

    elif code == "elk":
        if fermi is None:
            fermi = procarFile.fermi
            print("Fermi energy    :  %s eV (from Elk output)" % str(fermi))

    elif code == "qe":
        if fermi is None:
            fermi = procarFile.fermi
            print("Fermi energy   :  %s eV (from Quantum Espresso output)" % str(fermi))

    elif code == "lobster":
        if fermi is None:
            fermi = procarFile.fermi
            print("Fermi energy   :  %s eV (from Lobster output)" % str(fermi))
            # lobster already shifts fermi so we set it to zero here.
            fermi = 0.0

    elif code == "abinit":
        if fermi is None:
            fermi = abinitFile.fermi
            print("Fermi energy   :  %s eV (from Abinit output)" % str(fermi))
        recLat = abinitFile.reclat

    # if kdirect = False, then the k-points will be in cartesian coordinates.
    # The output should be read to find the reciprocal lattice vectors to transform
    # from direct to cartesian

    if code == "vasp":
        if kdirect:
            procarFile.readFile(procarfile, permissive)
        else:
            procarFile.readFile(procarfile, permissive, recLattice=recLat)

    elif code == "abinit":
        if kdirect:
            procarFile.readFile(procarfile, permissive)
        else:
            procarFile.readFile(procarfile, permissive, recLattice=recLat)

    # processing the data, getting an instance of the class that reduces the data
    data = ProcarSelect(procarFile, deepCopy=True, mode=mode)
    numofbands = int(data.spd.shape[1] / 2)

    # Unit conversions

    # Abinit PROCAR has band energy units in Hartree. We need it in eV.
    if code == "abinit":
        data.bands = data.bands * 27.211396641308

    # handling the spin, `spin='st'` is not straightforward, needs
    # to calculate the k vector and its normal. Other `spin` values
    # are trivial.
    if spin == "st":
        # two `ProcarSelect` instances, to store temporal values: spin_x, spin_y
        dataX = ProcarSelect(procarFile, deepCopy=True)
        dataX.selectIspin([1])
        dataX.selectAtoms(atoms, fortran=human)
        dataX.selectOrbital(orbitals)
        dataY = ProcarSelect(procarFile, deepCopy=True)
        dataY.selectIspin([2])
        dataY.selectAtoms(atoms, fortran=human)
        dataY.selectOrbital(orbitals)
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
        data.selectIspin([spin], separate=separate)
        data.selectAtoms(atoms, fortran=human)
        data.selectOrbital(orbitals)

    # Plotting the data
    if separate:
        if spin == 0:
            # plotting spin up bands separately
            data.bands = (
                data.bands[:, :numofbands].transpose() - np.array(fermi)
            ).transpose()
            print("Plotting spin up bands...")

        elif spin == 1:
            # plotting spin down bands separately
            data.bands = (
                data.bands[:, numofbands:].transpose() - np.array(fermi)
            ).transpose()
            print("Plotting spin down bands...")

        plot = ProcarPlot(data.bands, data.spd, data.kpoints)

    else:
        # Regular plotting method. For spin it plots density or magnetization.
        data.bands = (data.bands.transpose() - np.array(fermi)).transpose()
        plot = ProcarPlot(data.bands, data.spd, data.kpoints)

    ###### start of mode dependent options #########

    if mode == "scatter":
        fig, ax1 = plot.scatterPlot(
            mask=mask,
            size=markersize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            ticks=ticks,
            discontinuities=discontinuities,
            ax=ax,
        )
        if fermi is not None:
            ax1.set_ylabel(r"$E-E_F$ (eV)")
        else:
            ax1.set_ylabel(r"Energy (eV)")
        if elimit is not None:
            ax1.set_ylim(elimit)

    elif mode == "plain":
        fig, ax1 = plot.plotBands(
            markersize,
            marker=marker,
            ticks=ticks,
            color=color,
            discontinuities=discontinuities,
            ax=ax,
        )
        if fermi is not None:
            ax1.set_ylabel(r"$E-E_F$ (eV)")
        else:
            ax1.set_ylabel(r"Energy (eV)")
        if elimit:
            ax1.set_ylim(elimit)

    elif mode == "parametric":
        fig, ax1 = plot.parametricPlot(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            mask=mask,
            ticks=ticks,
            discontinuities=discontinuities,
            plot_bar=plot_color_bar,
            ax=ax,
            linewidth=linewidth,
        )
        if fermi is not None:
            ax1.set_ylabel(r"$E-E_F$ (eV)")
        else:
            ax1.set_ylabel(r"Energy (eV)")
        if elimit is not None:
            ax1.set_ylim(elimit)

    elif mode == "atomic":
        fig, ax1 = plot.atomicPlot(cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
        if fermi is not None:
            ax1.set_ylabel(r"$E-E_F$ (eV)")
        else:
            ax1.set_ylabel(r"Energy (eV)")
        if elimit is not None:
            ax1.set_ylim(elimit)
    ###### end of mode dependent options ###########

    if grid:
        ax1.grid()

    if title:
        ax1.set_title(title)

    else:
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
            plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
            return None, None
        else:
            if show:
                plt.show()
            else:
                pass
        return fig, ax1


# if __name__ == "__main__":
# bandsplot(mode='parametric',elimit=[-6,6],orbitals=[4,5,6,7,8],vmin=0,vmax=1, code='elk')
# knames=['$\Gamma$', '$X$', '$M$', '$\Gamma$', '$R$','$X$'],
# kticks=[0, 8, 16, 24, 38, 49])
