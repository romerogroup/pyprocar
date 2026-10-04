#!/usr/bin/env python

"""
Stand alone script for PyProcar.

This calls the modules in the /pyprocar directory.

Based on the original script developed by Aldo Romero (alromero@mail.wvu.edu) and
Francisco Munoz (fvmunoz@gmail.com).
"""

import argparse
from argparse import RawTextHelpFormatter

import numpy as np

import pyprocar
from pyprocar.scripts.scriptFermi2D import Fermi2DMode


def call_bandsplot(args):
    """
    This module calls the band structure plotting function.
    """
    pyprocar.bandsplot(
        code=args.code,
        dirname=args.dirname,
        mode=args.mode,
        spins=args.spins,
        atoms=args.atoms,
        orbitals=args.orbitals,
        fermi=args.fermi,
        elimit=args.elimit,
        kticks=args.kticks,
        knames=args.knames,
        savefig=args.savefig,
        show=args.savefig is None,
        **_given_plot_options(args),
    )


def _given_plot_options(args) -> dict:
    names = ("title", "cmap", "clim")
    return {name: getattr(args, name) for name in names if getattr(args, name, None) is not None}


def call_kpath(args):
    """
    This module calls the k-path generation function.
    """

    pyprocar.kpath(
        args.infile,
        grid_size=args.grid_size,
        with_time_reversal=args.with_time_reversal,
        recipe=args.recipe,
        threshold=args.threshold,
        symprec=args.symprec,
        angle_tolerence=args.angle_tolerence,
        supercell_matrix=args.supercell_matrix,
    )


def call_repair(args):
    """
    This module calls the repair function.
    """
    pyprocar.repair(args.infile, args.outfile)


def call_generate2dkmesh(args):
    """
    This module calls the k-mesh generating function.
    """
    pyprocar.generate2dkmesh(args.x1, args.y1, args.x2, args.y2, args.z, args.nkx, args.nky)


def call_fermi2D(args):
    """
    This module calls the fermi2D plotting function.
    """
    pyprocar.fermi2D(
        args.code,
        args.dirname,
        mode=args.mode,
        fermi=args.fermi,
        spins=args.spins,
        atoms=args.atoms,
        orbitals=args.orbitals,
        energy=args.energy,
        savefig=args.savefig,
        plot_arrows=not args.noarrow,
    )


def call_filter(args):
    """
    This module calls the filter function.
    """

    pyprocar.filter(
        args.inFile,
        args.outFile,
        atoms=args.atoms,
        orbitals=args.orbitals,
        orbital_names=args.orbital_names,
        bands=args.bands,
        spin=args.spin,
        human_atoms=args.human,
    )


def call_cat(args):
    """
    This module calls the cat function.
    """
    pyprocar.cat(inFiles=args.inFiles, outFile=args.outFile, gz=args.gz)


def call_bandgap(args):
    """
    This module calls the mergeabinit function.
    """
    pyprocar.bandgap(args.procar, args.outcar, args.code, args.fermi)


def call_unfold(args):
    """
    This module calls the band unfolding function.
    """
    pyprocar.unfold(
        code=args.code,
        dirname=args.dirname,
        mode=args.mode,
        unfold_mode=args.unfold_mode,
        transformation_matrix=np.diag(args.supercell),
        atoms=args.atoms,
        orbitals=args.orbitals,
        fermi=args.fermi,
        elimit=args.elimit,
        kticks=args.kticks,
        knames=args.knames,
        savefig=args.savefig,
        show=args.savefig is None,
        **_given_plot_options(args),
    )


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if args:
        # Top level parser
        description = "PyProcar: A Python library for analyzing PROCAR files."
        parser = argparse.ArgumentParser(description=description)
        subparsers = parser.add_subparsers(help="sub-command help")

        ############### cat ############################################

        phelp = (
            "concatenation of PROCARs files, they should be compatible (ie: "
            "joining parts of a large bandstructure calculation)."
        )
        parserCat = subparsers.add_parser("cat", help=phelp)

        phelp = "Input files. They can be compressed"
        parserCat.add_argument("inFiles", nargs="+", help=phelp)

        phelp = "Output file."
        parserCat.add_argument("outFile", help=phelp)

        phelp = "Writes a gzipped outfile (if needed a .gz extension automatically will be added)"
        parserCat.add_argument("--gz", help=phelp, action="store_true")

        parserCat.set_defaults(func=call_cat)

        ############### unfold #######################################
        parserunfold = subparsers.add_parser("unfold", help="Band unfolding.")
        parserunfold.add_argument("dirname", help="Supercell calculation directory (LORBIT = 12).")
        parserunfold.add_argument("--code", help="DFT code.", default="vasp")
        parserunfold.add_argument(
            "-m",
            "--mode",
            default="plain",
            choices=["plain", "parametric", "scatter", "overlay_species", "overlay_orbitals"],
        )
        parserunfold.add_argument(
            "--unfold-mode",
            help="How the unfolding weight is drawn.",
            default="both",
            choices=["both", "thickness", "color"],
        )
        parserunfold.add_argument(
            "--supercell",
            help="Diagonal of the primitive-to-supercell matrix.",
            type=int,
            nargs=3,
            default=[2, 2, 2],
        )
        parserunfold.add_argument("-a", "--atoms", type=int, nargs="+", default=None)
        parserunfold.add_argument("-o", "--orbitals", type=int, nargs="+", default=None)
        parserunfold.add_argument("-f", "--fermi", help="Fermi energy.", type=float, default=None)
        parserunfold.add_argument("--elimit", help="Energy range.", type=float, nargs=2)
        parserunfold.add_argument(
            "--kticks", help="k-point indices of the ticks.", type=int, nargs="+"
        )
        parserunfold.add_argument("--knames", help="Names of the ticks.", type=str, nargs="+")
        parserunfold.add_argument("--cmap", help="Colormap.", default=None)
        parserunfold.add_argument("--clim", help="Color range.", type=float, nargs=2, default=None)
        parserunfold.add_argument("-t", "--title", type=str, default=None)
        parserunfold.add_argument("--savefig", help="Save the figure instead of showing it.")
        parserunfold.set_defaults(func=call_unfold)

        ############### filter ##########################################
        phelp = (
            "Filters (manipulates) the data of the input file (PROCAR-like) and"
            " it yields a new file (PROCAR-like too) with the changes. This "
            "method can do only one manipulation at time (ie: spin, atoms, "
            "bands or orbitals)."
        )
        parserFilter = subparsers.add_parser("filter", help=phelp)

        phelp = "Input file. Can be compressed"
        parserFilter.add_argument("inFile", help=phelp)

        phelp = "Output file."
        parserFilter.add_argument("outFile", help=phelp)

        OptFilter = parserFilter.add_mutually_exclusive_group()
        phelp = (
            "List of atoms to group (add) as a new single entry. Each group of"
            " atoms should be specified in a different `--atoms` option. "
            "Example: `procar.py filter in out -a 0 1 -a 2` will group the 1st"
            " and 2nd atoms, while keeping the 3rd atom in `out` (any atom "
            "beyond the 3rd will be discarded). Mind the last atomic field "
            "present on a PROCAR file, is not an atom, is the 'tot' value (sum"
            " of all atoms), this field always is included in the outfile and "
            "it always is the 'tot' value from infile, regardless the selection"
            " of atoms."
        )
        OptFilter.add_argument("-a", "--atoms", type=int, nargs="+", action="append", help=phelp)

        phelp = (
            "List of orbitals to group as a single entry. Each group of "
            "orbitals needs a different `--orbitals` list. By instance, to "
            "group orbitals in 's','p', 'd' it is needed `-o 0 -o 1 2 3 -o 4 5 "
            "6 7 8`. Where 0=s, 1,2,3=px,py,pz, 4...9=dxx...dyz. Mind the last "
            "value (aka) 'tot' always is written."
        )
        OptFilter.add_argument("-o", "--orbitals", help=phelp, type=int, nargs="+", action="append")

        phelp = (
            "Keeps only the bands between `min` and `max` indexes. To keep the "
            "bands from 120 to 150 you should give `-b 120 150 `. It is not "
            "obvious which indexes are in the interest region, therefore I "
            "recommend you trial and error "
        )
        OptFilter.add_argument("-b", "--bands", help=phelp, type=int, nargs=2)

        phelp = (
            "Which spin components should be written: 0=density, 1,2,3=Sx,Sy,Sz."
            " They are not averaged."
        )
        OptFilter.add_argument("-s", "--spin", help=phelp, type=int, nargs="+")

        phelp = (
            "enable to give atoms list in a more human, 1-based order (say the"
            " 1st is 1, 2nd is 2 and so on ). Mind: this only holds for atoms."
        )
        parserFilter.add_argument("--human", help=phelp, action="store_true")

        phelp = (
            "List of names of new 'orbitals' to appear in the new file, eg. "
            "(`--orbital_names s p d` for a 's', 'p', 'd'). Only meaningful "
            "when manipulating the orbitals, ie: using `-o` "
        )
        parserFilter.add_argument("--orbital_names", help=phelp, nargs="+")

        parserFilter.set_defaults(func=call_filter)

        ################ fermi2D ##########################################
        parserFermi2D = subparsers.add_parser(
            "fermi2D",
            help="Plot the Fermi surface in the k_z = 0 plane",
        )

        phelp = "Directory that holds the DFT calculation."
        parserFermi2D.add_argument("dirname", help=phelp)

        phelp = (
            "DFT code of the calculation: vasp, qe, elk, abinit, siesta or lobster. Default: vasp"
        )
        parserFermi2D.add_argument("--code", help=phelp, default="vasp")

        phelp = (
            "plain, plain_bands, parametric or spin_texture. parametric colors the "
            "contours by the '-a', '-o' and '-s' projection. spin_texture needs a "
            "non-collinear calculation. Default: plain"
        )
        parserFermi2D.add_argument(
            "--mode",
            help=phelp,
            choices=[m.value for m in Fermi2DMode],
            default=Fermi2DMode.plain.value,
        )

        phelp = (
            "Spin indices to project onto in parametric mode. For a non-collinear "
            "calculation, 0 is the total and 1, 2, 3 are Sx, Sy, Sz."
        )
        parserFermi2D.add_argument(
            "-s", "--spins", type=int, nargs="+", choices=[0, 1, 2, 3], help=phelp
        )

        phelp = "Atom indices (0-based) to project onto, ie. '-a 0 2'. Default: all atoms"
        parserFermi2D.add_argument("-a", "--atoms", type=int, nargs="+", help=phelp)

        phelp = (
            "Orbital indices (0-based) to project onto: `-o 0`='s', `-o 1 2 3`='p', "
            "`-o 4 5 6 7 8`='d'. Default: all orbitals"
        )
        parserFermi2D.add_argument("-o", "--orbitals", type=int, nargs="+", help=phelp)

        phelp = "Energy of the surface relative to the Fermi energy, in eV. Default: 0"
        parserFermi2D.add_argument("-e", "--energy", help=phelp, type=float, default=0.0)

        phelp = "Fermi energy in eV. Default: the Fermi energy of the calculation"
        parserFermi2D.add_argument("-f", "--fermi", help=phelp, type=float)

        phelp = "Save the figure to this file instead of showing it."
        parserFermi2D.add_argument("--savefig", help=phelp)

        phelp = "In spin_texture mode, draw no spin arrows."
        parserFermi2D.add_argument("--noarrow", help=phelp, action="store_true")

        parserFermi2D.set_defaults(func=call_fermi2D)

        ################# repair ##########################################
        parserrepair = subparsers.add_parser(
            "repair", help="Repairs formatting issues in PROCAR file."
        )
        parserrepair.add_argument("infile", help="Input file. Can be compressed.")
        parserrepair.add_argument("outfile", help="Output file.")
        parserrepair.set_defaults(func=call_repair)

        ################# bandgap ##########################################
        parserbandgap = subparsers.add_parser(
            "bandgap",
            help="Calculate bandgap. procar and outcar needed only for Abinit and VASP.",
        )
        parserbandgap.add_argument("procar", help="PROCAR file.")
        parserbandgap.add_argument("outcar", help="OUTCAR file.")
        parserbandgap.add_argument(
            "code",
            help="code",
            choices=["vasp", "qe", "lobster", "abinit", "elk"],
            default="vasp",
        )
        parserbandgap.add_argument(
            "fermi", help="Fermi energy. Retrived from output if not provided."
        )
        parserbandgap.set_defaults(func=call_bandgap)

        ################## k-mesh ########################################
        parsergenerate2dkmesh = subparsers.add_parser(
            "generate2dkmesh",
            help="Generate a 2D k-meshcentered at a given k-point in a given k-plane.",
        )
        parsergenerate2dkmesh.add_argument("x1", help="x1 coordinate")
        parsergenerate2dkmesh.add_argument("y1", help="y1 coordinate")
        parsergenerate2dkmesh.add_argument("x2", help="x2 coordinate")
        parsergenerate2dkmesh.add_argument("y2", help="y2 coordinate")
        parsergenerate2dkmesh.add_argument("z", help="z plane")
        parsergenerate2dkmesh.add_argument("nkx", help="number of grids in the x direction")
        parsergenerate2dkmesh.add_argument("nky", help="number of grids in the y direction")
        parsergenerate2dkmesh.set_defaults(func=call_generate2dkmesh)

        ################## k-path ####################################################
        parserkpath = subparsers.add_parser(
            "kpath", help="k-path generator.", formatter_class=RawTextHelpFormatter
        )
        parserkpath.add_argument("infile", help="POSCAR file", default="POSCAR")
        parserkpath.add_argument("-grid_size", help="Grid size", default=40, type=int)
        parserkpath.add_argument(
            "-with_time_reversal",
            help="Flag to turn on time reversal symmetry",
            action="store_true",
        )
        parserkpath.add_argument(
            "-recipe",
            help="The algorithm that defines the special points and paths",
            type=str,
            default="hpkot",
        )
        parserkpath.add_argument(
            "-threshold",
            help="The threshold to use to verify if we are in an edge case",
            type=float,
            default=1e-07,
        )
        parserkpath.add_argument(
            "-symprec",
            help="The symmetry precision used internally by SPGLIB",
            type=float,
            default=1e-05,
        )
        parserkpath.add_argument(
            "-angle_tolerence",
            help="Angle_tolerance used internally by SPGLIB",
            type=float,
            default=-1.0,
        )
        parserkpath.add_argument(
            "-supercell_matrix",
            help="The super cell for band unfolding. Default 3x3 identity matrix.",
            type=int,
            default=np.eye(3),
        )
        parserkpath.set_defaults(func=call_kpath)

        ################### bandstructure ######################################################
        parserBandsplot = subparsers.add_parser(
            "bandsplot",
            help="Bandstructure plot.",
            formatter_class=RawTextHelpFormatter,
        )

        parserBandsplot.add_argument("dirname", help="Calculation directory.")
        parserBandsplot.add_argument("--code", help="DFT code.", default="vasp")

        choices = [
            "plain",
            "parametric",
            "scatter",
            "atomic",
            "overlay_species",
            "overlay_orbitals",
            "ipr",
        ]
        parserBandsplot.add_argument("-m", "--mode", default="plain", choices=choices)

        phelp = (
            "Spin channels to plot. A non-collinear calculation takes one\n"
            "component: 0 total, 1 Sx, 2 Sy, 3 Sz.\n\n"
        )
        parserBandsplot.add_argument("-s", "--spins", type=int, nargs="+", help=phelp, default=None)

        phelp = (
            "List of rows (atoms) to be used. This list refers to the rows of\n"
            "(each block of) your PROCAR file. If you haven't manipulated your\n"
            "PROCAR (eg: with the '-a' option of 'filter' mode) each row\n"
            "correspond to the respective atom in the POSCAR.\n\n"
            "Mind: This list is 0-based, ie: the 1st atom is 0, the 2nd is 1,\n"
            "  and so on.\n\n"
            "Example:\n"
            "-a 0 2 :  select the 1st  and 3rd. rows (likely 1st and 3rd atoms)"
            "\n\n"
        )
        parserBandsplot.add_argument("-a", "--atoms", type=int, nargs="+", help=phelp, default=None)

        phelp = (
            "Orbitals index(es) to be used, take a look to the PROCAR file, \n"
            "they are 's py pz px ...', then s->0, py->1, pz->2 and so on. \n"
            "Note that indexes begin at 0!. Its default is the last field (ie:\n"
            "'tot', did you saw the PROCAR?). Some examples:\n\n"
            "-o 0 : s-orbital (unless you modified the orbitals, eg. 'filter')\n"
            "-o 1 2 3 : py+pz+px (unless you modified the orbitals)\n"
            "-o 4 5 6 7 8 : all the d-orbitasl (unless...)\n"
            "-o 2 6 : pz+dzz (did you look at the PROCAR?)\n\n "
        )
        parserBandsplot.add_argument(
            "-o", "--orbitals", type=int, nargs="+", help=phelp, default=None
        )

        phelp = (
            "Set the Fermi energy (or any reference energy) as the zero energy.\n"
            "Mind: The Fermi energy MUST be the one from the self-consistent\n"
            "calculation, not from a Bandstructure calculation!\n\n"
        )
        parserBandsplot.add_argument("-f", "--fermi", type=float, help=phelp, default=None)

        phelp = (
            "Min/Max energy to be ploted. Example:\n "
            "--elimit -1 1 : From -1 to 1 around Fermi energy (if given)\n\n"
        )
        parserBandsplot.add_argument("--elimit", type=float, nargs=2, help=phelp, default=None)

        phelp = (
            "Change the color scheme. Example:\n\n"
            "--cmap  seismic : blue->white->red, useful to see the \n"
            "  spin-polarization of a band (it will blueish or reddish)\n"
            "  depending of spin channel\n"
            "--cmap  seismic_r : the 'seismic' colormap, but reversed.\n\n"
        )
        parserBandsplot.add_argument("--cmap", help=phelp, default=None)

        phelp = "Color range of the projections, for example '--clim 0 1'.\n\n"
        parserBandsplot.add_argument("--clim", type=float, nargs=2, help=phelp, default=None)

        phelp = (
            "Saves the figure, instead of display it on screen. Anyway, you can\n"
            "save from the screen too. Any file extension supported by\n "
            "`matplotlib.savefig` is valid (if you are too lazy to google it,\n"
            "trial and error also works fine)\n\n"
        )
        parserBandsplot.add_argument("--savefig", help=phelp, default=None)

        phelp = (
            "list of ticks along the kpoints axis (x axis). For instance a\n"
            "bandstructure G-X-M with 10 point by segment should be:\n "
            "--kticks 0 9 19\n\n"
        )
        parserBandsplot.add_argument("--kticks", help=phelp, nargs="+", type=int)

        phelp = (
            "Names of the points given in `--kticks`. In the `kticks` example\n"
            'they should be `--knames "\$Gamma\$" X M`. As you can see \n'
            "LaTeX stuff works with a minimal mess (extra \\s)\n\n"
        )
        parserBandsplot.add_argument("--knames", help=phelp, nargs="+", type=str, default=None)

        phelp = (
            "Title, to use several words, use quotation marks\"\" or ''. Latex\n"
            " works if you scape the special characteres, ie: $\\alpha$ -> \n"
            "\$\\\\alpha\$"
        )
        parserBandsplot.add_argument("-t", "--title", help=phelp, type=str, default=None)

        parserBandsplot.set_defaults(func=call_bandsplot)

        args = parser.parse_args()
        args.func(args)

    else:
        print("PyProcar: A Python library for analyzing PROCAR files.\n")
        print("Usage: procar [-h]")
        print("{cat,unfold,filter,fermi2D,repair,generate2dkmesh,kpath,bandsplot,bandscompare}")
