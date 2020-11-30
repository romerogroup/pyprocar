from .utilsprocar import UtilsProcar
from .procarparser import ProcarParser
from .procarselect import ProcarSelect
from .procarplot import ProcarPlot
from .procarsymmetry import ProcarSymmetry
from .fermisurface import FermiSurface
from .elkparser import ElkParser
from .abinitparser import AbinitParser
import matplotlib.pyplot as plt
from .splash import welcome


def fermi2D(
    file,
    outcar=None,
    abinit_output=None,
    spin=0,
    atoms=None,
    orbitals=None,
    energy=None,
    fermi=None,
    rec_basis=None,
    rot_symm=1,
    translate=[0, 0, 0],
    rotation=[0, 0, 0, 1],
    human=False,
    mask=None,
    savefig=None,
    st=False,
    noarrow=False,
    exportplt=False,
    code="vasp",
    repair=True,
):
    """
  This module plots 2D Fermi surface.
  """

    welcome()

    # Turn interactive plotting off
    plt.ioff()

    # Repair PROCAR
    if code == "vasp" or code == "abinit":
        if repair:
            repairhandle = UtilsProcar()
            repairhandle.ProcarRepair(file, file)
            print("PROCAR repaired. Run with repair=False next time.")

    if atoms is None:
        atoms = [-1]
        if human is True:
            print("WARNING: `--human` option given without atoms list!!!!!")

    if orbitals is None:
        orbitals = [-1]

    if rec_basis != None:
        rec_basis = np.array(rec_basis)
        rec_basis.shape = (3, 3)

    if len(translate) != 3 and len(translate) != 1:
        print("Error: --translate option is invalid! (", translate, ")")
        raise RuntimeError("invalid option --translate")

    print("file            : ", file)
    print("outcar          : ", outcar)
    print("Abinit output   : ", abinit_output)
    print("atoms           : ", atoms)
    print("orbitals        : ", orbitals)
    print("spin comp.      : ", spin)
    print("energy          : ", energy)
    print("fermi energy    : ", fermi)
    print("Rec. basis      : ", rec_basis)
    print("rot. symmetry   : ", rot_symm)
    print("origin (trasl.) : ", translate)
    print("rotation        : ", rotation)
    print("masking thres.  : ", mask)
    print("save figure     : ", savefig)
    print("st              : ", st)
    print("no_arrows       : ", noarrow)

    # first parse the outputs if given
    if code == "vasp":
        if rec_basis is None and outcar:
            outcarparser = UtilsProcar()
            if fermi is None:
                fermi = outcarparser.FermiOutcar(outcar)
                print("Fermi energy found in outcar file = " + str(fermi))
            rec_basis = outcarparser.RecLatOutcar(outcar)
        # Reciprocal lattices are needed!
        elif rec_basis is None and outcar is None:
            print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
            raise RuntimeError("Reciprocal Lattice not found")

        # parsing the file
        procarFile = ProcarParser()
        # permissive incompatible with Fermi surfaces
        procarFile.readFile(file, permissive=False, recLattice=rec_basis)

    elif code == "elk":
        procarFile = ElkParser()
        if rec_basis is None:
            if fermi is None:
                fermi = procarFile.fermi
                print("Fermi energy found in Elk output file = " + str(fermi))
            rec_basis = procarFile.reclat
        # Reciprocal lattices are needed!
        if rec_basis is None:
            print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
            raise RuntimeError("Reciprocal Lattice not found")
        procarFile = Elkparser(kdirect=False)

    elif code == "abinit":
        if rec_basis is None and abinit_output:
            abinitparser = AbinitParser(abinit_output=abinit_output)
            if fermi is None:
                fermi = abinitparser.fermi
                print("Fermi energy found in Abinit ouput file = " + str(fermi))
            rec_basis = abinitparser.reclat
        # Reciprocal lattices are needed!
        elif rec_basis is None and abinit_output is None:
            print("ERROR: Reciprocal Lattice is needed, use --rec_basis or --outcar")
            raise RuntimeError("Reciprocal Lattice not found")
        # parsing the file
        procarFile = ProcarParser()
        # permissive incompatible with Fermi surfaces
        procarFile.readFile(file, permissive=False, recLattice=rec_basis)

    ### End of parsing ###

    if st is not True:
        # processing the data
        data = ProcarSelect(procarFile)
        data.selectIspin([spin])
        # fortran flag is equivalent to human,
        # but the later seems more human-friendly
        data.selectAtoms(atoms, fortran=human)
        data.selectOrbital(orbitals)
    else:
        # first get the sdp reduced array for all spin components.
        stData = []
        for i in [1, 2, 3]:
            data = ProcarSelect(procarFile)
            data.selectIspin([i])
            data.selectAtoms(atoms, fortran=human)
            data.selectOrbital(orbitals)
            stData.append(data.spd)

    # Once the PROCAR is parsed and reduced to 2x2 arrays, we can apply
    # symmetry operations to unfold the Brillouin Zone
    kpoints = data.kpoints
    bands = data.bands
    character = data.spd
    if st is True:
        sx, sy, sz = stData[0], stData[1], stData[2]
        symm = ProcarSymmetry(kpoints, bands, sx=sx, sy=sy, sz=sz, character=character)
    else:
        symm = ProcarSymmetry(kpoints, bands, character=character)

    symm.Translate(translate)
    symm.GeneralRotation(rotation[0], rotation[1:])
    # symm.MirrorX()
    symm.RotSymmetryZ(rot_symm)

    # plotting the data
    print("Bands will be shifted by the Fermi energy = ", fermi)
    fs = FermiSurface(symm.kpoints, symm.bands - fermi, symm.character)
    fs.FindEnergy(energy)

    if not st:
        fs.Plot(mask=mask, interpolation=300)
    else:
        fs.st(sx=symm.sx, sy=symm.sy, sz=symm.sz, noarrow=noarrow, spin=spin)

    if exportplt:
        return plt

    else:
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
            plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
        else:
            plt.show()
        return
