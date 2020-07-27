#!/usr/bin/env python

"""
Stand alone script for PyProcar.

This calls the modules in the /pyprocar directory.

Based on the original script developed by Aldo Romero (alromero@mail.wvu.edu) and
Francisco Munoz (fvmunoz@gmail.com).
"""

import argparse
from argparse import RawTextHelpFormatter
import pyprocar
import numpy as np


def call_bandsplot(args):
    """
	This module calls the band structure plotting function.
	"""

    pyprocar.bandsplot(
        args.file,
        mode=args.mode,
        color=args.color,
        abinit_output=args.abinit,
        spin=args.spin,
        atoms=args.atoms,
        orbitals=args.orbitals,
        fermi=args.fermi,
        elimit=args.elimit,
        mask=args.mask,
        markersize=args.markersize,
        cmap=args.cmap,
        vmax=args.vmax,
        vmin=args.vmin,
        grid=args.grid,
        marker=args.marker,
        permissive=args.permissive,
        human=args.human,
        savefig=args.savefig,
        kticks=args.kticks,
        knames=args.knames,
        title=args.title,
        outcar=args.outcar,
        kpointsfile=args.kpointsfile,
        kdirect=args.kdirect,
    )


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
    pyprocar.generate2dkmesh(
        args.x1, args.y1, args.x2, args.y2, args.z, args.nkx, args.nky
    )


def call_fermi2D(args):
    """
	This module calls the fermi2D plotting function.
	"""
    pyprocar.fermi2D(
        args.file,
        outcar=args.outcar,
        spin=args.spin,
        atoms=args.atoms,
        orbitals=args.orbitals,
        energy=args.energy,
        fermi=args.fermi,
        rec_basis=args.rec_basis,
        rot_symm=args.rot_symm,
        translate=args.translate,
        rotation=args.rotation,
        human=args.human,
        mask=args.mask,
        savefig=args.savefig,
        st=args.st,
        noarrow=args.noarrow,
    )


def call_fermi3D(args):
    """
	This module calls the fermi3D plotting function.
	"""
    pyprocar.fermi3D(
        args.procar,
        args.outcar,
        args.bands,
        scale=args.scale,
        mode=args.mode,
        st=args.st,
        kwargs=args.kwargs,
    )


def call_filter(args):
    """
	This module calls the filter function.
	"""

    pyprocar.filter(
        args.infile,
        args.outfile,
        atoms=args.atoms,
        orbitals=args.orbitals,
        orbital_names=args.orbital_names,
        bands=args.bands,
        spin=args.spin,
        human_atoms=args.human_atoms,
    )


def call_cat(args):
    """
	This module calls the cat function.
	"""
    pyprocar.cat(inFiles=args.inFiles, outFile=args.outFile, gz=args.gz)


def call_mergeabinit(args):
    """
	This module calls the mergeabinit function.
	"""
    pyprocar.mergeabinit(args.outfile)


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
        fname=args.fname,
        poscar=args.poscar,
        outcar=args.outcar,
        supercell_matrix=args.supercell_matrix,
        ispin=args.ispin,
        efermi=args.efermi,
        shift_efermi=args.shift_efermi,
        elimit=args.elimit,
        kticks=args.kticks,
        knames=args.knames,
        print_kpts=args.print_kpts,
        show_band=args.show_band,
        width=args.width,
        color=args.color,
        savetab=args.savetab,
        savefig=args.savefig,
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

        ################ mergeabinit ##################################

        parsermergeabinit = subparsers.add_parser(
            "mergeabinit", help="Merge Abinit PROCARs."
        )
        parsermergeabinit.add_argument("outfile", help="Merged PROCAR")
        parsermergeabinit.set_defaults(func=call_mergeabinit)

        ############### unfold #######################################
        parserunfold = subparsers.add_parser("unfold", help="Band unfolding.")
        parserunfold.add_argument("-fname", help="PROCAR filename.")
        parserunfold.add_argument("-poscar", help="POSCAR filename.")
        parserunfold.add_argument(
            "-outcar", help="OUTCAR filename. Used to get Fermi energy."
        )
        parserunfold.add_argument(
            "-supercell_matrix",
            help="Supercell matrix from primitive cell to supercell.",
        )
        parserunfold.add_argument(
            "-ispin",
            help="None - non spin polarized \n 1 - spin up" "\n 2 - spin down.",
            choices=[None, 1, 2],
        )
        parserunfold.add_argument(
            "-efermi", help="Fermi energy. Only when no OUTCAR is given."
        )
        parserunfold.add_argument("-elimit", help="Range of energy to be plotted.")
        parserunfold.add_argument(
            "-kticks",
            help="The indices of k-points for labels.",
            type=int,
            nargs="+",
            action="append",
        )
        parserunfold.add_argument(
            "-knames", help="Labels of k-points.", type=str, nargs="+", action="append"
        )
        parserunfold.add_argument(
            "-print_kpts", help="Print all the k-points to screen.", action="store_true"
        )
        parserunfold.add_argument(
            "-show_band",
            help="Whether to plot the bands before unfolding.",
            action="store_true",
        )
        parserunfold.add_argument(
            "-width", help="Width of the unfolded band.", type=float
        )
        parserunfold.add_argument(
            "-color", help="Color of the unfolded band.", type=str
        )
        parserunfold.add_argument(
            "-savetab",
            help="The csv file name of which the table of unfolding result will be written into",
        )
        parserunfold.add_argument(
            "-savefig", help="The file name of which the figure will be saved."
        )
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
        OptFilter.add_argument(
            "-a", "--atoms", type=int, nargs="+", action="append", help=phelp
        )

        phelp = (
            "List of orbitals to group as a single entry. Each group of "
            "orbitals needs a different `--orbitals` list. By instance, to "
            "group orbitals in 's','p', 'd' it is needed `-o 0 -o 1 2 3 -o 4 5 "
            "6 7 8`. Where 0=s, 1,2,3=px,py,pz, 4...9=dxx...dyz. Mind the last "
            "value (aka) 'tot' always is written."
        )
        OptFilter.add_argument(
            "-o", "--orbitals", help=phelp, type=int, nargs="+", action="append"
        )

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
            help="Plot the Fermi surface for a 2D Brillouin zone (layer-wise)"
            "along z and just perpendicular to z! (actually k_z)",
        )

        phelp = "Input file. It can be compressed"
        parserFermi2D.add_argument("file", help=phelp)

        phelp = (
            "Spin component to be used: for non-polarized calculations density "
            "is '-s 0'. For spin polarized case test '-s 0' (ignore the spin) or"
            " '-s 1' (assing a sign to the spin channel). For "
            "non-collinear stuff you can use '-s 0', '-s 1', '-s 2', -s "
            "3 for the magnitude, x, y, z components of your spin "
            "vector, in this case you really want to see spin textures "
            "'--st'. Default: s=0"
        )
        parserFermi2D.add_argument(
            "-s", "--spin", type=int, choices=[0, 1, 2, 3], default=0, help=phelp
        )

        phelp = (
            "List of atoms to be used (0-based): ie. '-a 0 2' to select the 1st"
            " and 3rd. It defaults to the last one (-a -1 the 'tot' entry)"
        )
        parserFermi2D.add_argument("-a", "--atoms", type=int, nargs="+", help=phelp)

        phelp = (
            "Orbital index(es) to be used 0-based. Take a look to the PROCAR "
            "file. Its default is the last field (ie: 'tot'). From a standard "
            "PROCAR: `-o 0`='s', `-o 1 2 3`='p', `-o 4 5 6 7 8`='d'."
        )
        parserFermi2D.add_argument("-o", "--orbitals", type=int, nargs="+", help=phelp)

        phelp = (
            "Energy for the surface. To plot the Fermi surface at Fermi Energy `-e 0`"
        )
        parserFermi2D.add_argument(
            "-e", "--energy", help=phelp, type=float, required=True
        )

        phelp = (
            "Set the Fermi energy (or any reference energy) as zero. To get it "
            "you should `grep E-fermi` the self-consistent OUTCAR. See `--outcar`"
        )
        parserFermi2D.add_argument("-f", "--fermi", help=phelp, type=float)

        phelp = (
            "reciprocal space basis vectors. 9 number are required b1x b1y ... "
            " b3z. This option is quite involved, so I recommend you to use `--outcar`"
        )
        parserFermi2D.add_argument("--rec_basis", help=phelp, type=float, nargs=9)

        phelp = (
            "Apply a rotational symmetry to unfold the Kpoints found. If your"
            "PROCAR only has a portion of the Brillouin Zone, you may want to "
            "plot the FULL BZ (ie: a Dirac cone at Gamma will look like a cone "
            "and not like a segment of circle). Supported rotations are "
            "1,2,3,4,6. All of them along Z and centered at Gamma. Consider to "
            "'--translate' your cell to rotate with other origin. This is the "
            "last symmetry operation to be performed."
        )
        parserFermi2D.add_argument("--rot_symm", help=phelp, type=int, default=1)

        phelp = (
            "Translate your mesh to the specified point. The point can be 3 "
            "coordinates (numbers) or the index of one K-point (zero-based, as "
            "usual). This is the first symmetry operation to be performed "
            "(i.e. rotations will take this point as the origin)."
        )
        parserFermi2D.add_argument(
            "--translate", help=phelp, nargs="+", default=[0, 0, 0]
        )

        phelp = (
            "A general rotation is applied to the data in the PROCAR. While this "
            " script has a large bias to work on the 'xy' plane, with this option"
            " you can rotate your whole PROCAR to fit the 'xy' plane. A rotation "
            "is composed by one angle plus one fixed axis, eg: '--rotation 90 1 0"
            " 0' is 90 degrees along the x axis, this changes the 'xy'->'xz'. The"
            " rotation is performed after the translation and before applying "
            "rot_symm. "
        )
        parserFermi2D.add_argument(
            "--rotation", help=phelp, type=float, nargs=4, default=[0, 0, 0, 1]
        )

        phelp = "enable to give atoms list in a more human, 1-based way (say the 1st is 1,2nd is 2 and so on )"
        parserFermi2D.add_argument("-u", "--human", help=phelp, action="store_true")

        phelp = "If set, masks(hides) values lowers than it. Useful to remove unwanted bands."
        parserFermi2D.add_argument("--mask", type=float, help=phelp)

        phelp = (
            "Saves the figure, instead of display it on screen. Anyway, you can save from"
            " the screen too. Any file extension supported by"
            "matplotlib.savefig is valid (if you are too lazy"
            " to google it, trial and error also works)"
        )
        parserFermi2D.add_argument("--savefig", help=phelp)

        phelp = (
            "OUTCAR file where to find the reciprocal lattice vectors and "
            "perhaps E_fermi. Mind: '--fermi' has precedence, remember that the"
            " E-fermi should correspond to a self-consistent run, not a "
            "bandstructure! (however, this is irrelevant for basis vectors)"
        )
        parserFermi2D.add_argument("--outcar", help=phelp)

        phelp = (
            "Plot of the spin texture (ie: spin arrows) on the Fermi's surface."
            " This option works quite indepentent of another options."
        )
        parserFermi2D.add_argument("--st", help=phelp, action="store_true")

        phelp = (
            "Plot of the spin texture without arrows (just intensity) for a "
            "given spin direction on the Fermi's surface.  This option works"
            "quite indepentent of another options but needs to set '--st' and '--spin'."
        )
        parserFermi2D.add_argument("--noarrow", help=phelp, action="store_true")

        parserFermi2D.set_defaults(func=call_fermi2D)

        ################ Fermi3D ##########################################

        parserfermi3D = subparsers.add_parser("fermi3D", help="Plot 3D Fermi surface")
        parserfermi3D.add_argument("procar", help="PROCAR file.")
        parserfermi3D.add_argument("outcar", help="OUTCAR file.")
        parserfermi3D.add_argument(
            "bands", help="Array of bands to be included. -1 considers all.", default=-1
        )
        parserfermi3D.add_argument(
            "scale", help="Interpolation factor", type=float, default=1
        )
        parserfermi3D.add_argument(
            "mode", help="Plot mode.", choices=["plain", "parametric", "external"]
        )
        parserfermi3D.add_argument(
            "-st", help="Flag to set spin texture on.", action="store_true"
        )
        parserfermi3D.add_argument("kwargs", help="Additional arguments.", nargs="*")
        parserfermi3D.set_defaults(func=call_fermi3D)

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
            help="Generate a 2D k-mesh"
            "centered at a given k-point in a given k-plane.",
        )
        parsergenerate2dkmesh.add_argument("x1", help="x1 coordinate")
        parsergenerate2dkmesh.add_argument("y1", help="y1 coordinate")
        parsergenerate2dkmesh.add_argument("x2", help="x2 coordinate")
        parsergenerate2dkmesh.add_argument("y2", help="y2 coordinate")
        parsergenerate2dkmesh.add_argument("z", help="z plane")
        parsergenerate2dkmesh.add_argument(
            "nkx", help="number of grids in the x direction"
        )
        parsergenerate2dkmesh.add_argument(
            "nky", help="number of grids in the y direction"
        )
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

        phelp = "Input file. It can be compressed"
        parserBandsplot.add_argument("file", help=phelp)

        phelp = (
            "Plot mode:\n"
            "-m  scatter : is a points plot with the color given by the chosen\n"
            "  projection. It produces a rather heavy pdf file.\n\n"
            "-m  parametric : like scatter, but with lines instead of points \n"
            "  (bands crossings are not handled, and some  unphysical 'jumps' \n"
            " can be present). Sligthy smaller PDF size.\n\n"
            "-m plain : is a featureless bandstructure ignoring all data about\n"
            "  projections. Rather light-weight\n\n"
            "-m atomic : For non-periodic system, like molecules, rather ugly \n"
            "  but useful to visualize energy level. Only 1 K-point!\n\n"
        )
        choices = ["scatter", "plain", "parametric", "atomic"]
        parserBandsplot.add_argument(
            "-m", "--mode", help=phelp, default="plain", choices=choices
        )

        phelp = "Color of the bands for plain mode."
        parserBandsplot.add_argument("-color", help=phelp, default="blue")

        phelp = "Name of Abinit output file if used."
        parserBandsplot.add_argument("-abinit", help=phelp, default=None)

        phelp = (
            "Spin component to be used (default -s 0): \n\n"
            "Non-polarized calculations density is '-s 0', just ignore it.\n\n"
            "Spin-Polarized (collinear) calculation: \n"
            "-s 0 are the unpolarized bands, the spin-polarization is ignored.\n"
            "-s 1 'spin-polarized' bands, the character of 'up' bands positive,\n"
            "  but negative for 'down' bands, this means that the color of \n"
            "  'down' is negative. Useful together with '--cmap seismic'.\n\n"
            "Non-collinear calculation: \n"
            "-s 0 : density, ie: Spin-orbit-coupling but don't care of spin.\n"
            "-s 1 : Sx, projection along 'x' quantization axis, see SAXIS flag\n"
            "  in the VASP manual\n"
            "-s 2 : Sy, projection along 'y' quantization axis.\n"
            "-s 3 : Sy, projection along 'z' quantization axis.\n"
            "-s st : Spin-texture perpendicular in the plane (kx,ky) to each\n "
            "(kx,ky) vector. Useful for Rashba-like states in surfaces. Use\n "
            "'--cmap seismic'\n\n "
        )
        parserBandsplot.add_argument(
            "-s", "--spin", choices=["0", "1", "2", "3", "st"], default="0", help=phelp
        )

        phelp = (
            "List of rows (atoms) to be used. This list refers to the rows of\n"
            "(each block of) your PROCAR file. If you haven't manipulated your\n"
            "PROCAR (eg: with the '-a' option of 'filter' mode) each row\n"
            "correspond to the respective atom in the POSCAR.\n\n"
            "Mind: This list is 0-based, ie: the 1st atom is 0, the 2nd is 1,\n"
            "  and so on. If you need to be treated like a human, specify '-u'\n"
            "or '--human' and 1st->1, 2nd->2, etc.\n\n"
            "Example:\n"
            "-a 0 2 :  select the 1st  and 3rd. rows (likely 1st and 3rd atoms)"
            "\n\n"
        )
        parserBandsplot.add_argument(
            "-a", "--atoms", type=int, nargs="+", help=phelp, default=None
        )

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
            "See '--outcar', avoids to give it explicitly. A list of \n"
            "k-dependant 'fermi-like energies' are also accepted (useful to\n"
            "compare different systems in one PROCAR made by hand). \n\n"
            "Mind: The Fermi energy MUST be the one from the self-consistent\n"
            "calculation, not from a Bandstructure calculation!\n\n"
        )
        parserBandsplot.add_argument(
            "-f", "--fermi", type=float, help=phelp, nargs="+", default=None
        )

        phelp = (
            "Min/Max energy to be ploted. Example:\n "
            "--elimit -1 1 : From -1 to 1 around Fermi energy (if given)\n\n"
        )
        parserBandsplot.add_argument(
            "--elimit", type=float, nargs=2, help=phelp, default=None
        )

        phelp = (
            "If given, it masks(hides) bands with values lowers than 'mask'.\n"
            "It is useful to remove 'unwanted' bands. For instance, if you\n"
            "project the bandstructure on a 'surface' atom -with the default\n"
            "colormap- some white points can appear, they are bands with \n"
            "almost no contribution to the 'surface': no physics but they \n"
            "still look ugly, to hide those bands use '--mask 0.08' (or some \n"
            "other small value). Mind: it works with the absolute value of\n"
            "projection (no problem with spin polarization)\n\n"
        )
        parserBandsplot.add_argument("--mask", type=float, help=phelp, default=None)

        phelp = (
            "Size of markers, if used. Each mode has it own scale,\n"
            "just test them\n\n"
        )
        parserBandsplot.add_argument(
            "--markersize", type=float, help=phelp, default=0.02
        )

        phelp = (
            "Change the color scheme. Example:\n\n"
            "--cmap  seismic : blue->white->red, useful to see the \n"
            "  spin-polarization of a band (it will blueish or reddish)\n"
            "  depending of spin channel\n"
            "--cmap  seismic_r : the 'seismic' colormap, but reversed.\n\n"
        )
        parserBandsplot.add_argument("--cmap", help=phelp, default="jet")

        phelp = (
            "Do you want to Normalize the plots to the same scale of colors\n"
            "(ie: the numbers on the bar at the right), just try '--vmax'\n\n"
            "--vmax 1 : If you are looking for the s, p or d character.\n"
            "--vmax 0.2: If you want to capture some tiny effect, eg: s-band of\n"
            "  a impurity on a metal\n\n"
        )
        parserBandsplot.add_argument("--vmax", type=float, help=phelp, default=None)

        phelp = (
            "Like '--vmax' (see '--vmax'), However, for spin-polarized \n"
            "(collinear or not) you can set it to a negative value. Actually\n"
            "you can do it for a non-spin-polarized calculation and the \n"
            "effect will be a 'stretching' of the color scheme, try it.\n\n"
        )
        parserBandsplot.add_argument("--vmin", type=float, help=phelp, default=None)

        phelp = "switch on/off the grid. Default is 'on'\n\n"
        parserBandsplot.add_argument("--grid", type=bool, help=phelp, default=True)

        phelp = (
            "set the marker shape, ie: 'o'=circle, 's'=square,\ '-'=line\n"
            "(only mode `plain`, other symbols: google pyplot markers)\n\n"
        )
        parserBandsplot.add_argument("--marker", type=str, help=phelp, default="o")

        phelp = (
            "Some fault tolerance for ill-formatted files (stupid fortran)\n"
            "But be careful, something could be messed up and don't work (at\n"
            "least as expected). Length of K-points paths will be ignored\n\n"
        )
        parserBandsplot.add_argument("--permissive", help=phelp, action="store_true")

        phelp = (
            "Enable human-like 1-based order (ie 1st is 1, 2nd is 2, and so\n"
            "on). Mind: this only works for atoms, not for orbitals or spin\n\n"
        )
        parserBandsplot.add_argument("-u", "--human", help=phelp, action="store_true")

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
        parserBandsplot.add_argument(
            "--knames", help=phelp, nargs="+", type=str, default=None
        )

        phelp = (
            "Title, to use several words, use quotation marks\"\" or ''. Latex\n"
            " works if you scape the special characteres, ie: $\\alpha$ -> \n"
            "\$\\\\alpha\$"
        )
        parserBandsplot.add_argument(
            "-t", "--title", help=phelp, type=str, default=None
        )

        phelp = (
            "OUTCAR file where to find the reciprocal lattice vectors and\n "
            "perhaps E_fermi.\n"
            "Mind: '--fermi' has precedence, remember that the E-fermi should\n"
            "correspond to a self-consistent run, not to a bandstructure!\n"
            "(however, the basis and reciprocal vectors will be safe from the\n"
            "non-self-consistentness)\n\n"
        )
        parserBandsplot.add_argument("--outcar", help=phelp, default=None)

        phelp = "KPOINTS file for bandstructure plotting.\n"
        parserBandsplot.add_argument("--kpointsfile", help=phelp, default=None)

        phelp = "Convert k-points from reduced to cartesian for plot #1?"
        parserBandsplot.add_argument("-kdirect", help=phelp, action="store_false")

        parserBandsplot.set_defaults(func=call_bandsplot)

        args = parser.parse_args()
        args.func(args)

    else:
        print("PyProcar: A Python library for analyzing PROCAR files.\n")
        print("Usage: procar [-h]")
        print(
            "{cat,mergeabinit,unfold,filter,fermi2D,fermi3D,repair,generate2dkmesh,kpath,bandsplot,bandscompare}"
        )
