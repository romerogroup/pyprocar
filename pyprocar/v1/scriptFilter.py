from .procarfilefilter import ProcarFileFilter
from .splash import welcome


def filter(
    inFile,
    outFile,
    atoms=None,
    orbitals=None,
    orbital_names=None,
    bands=None,
    spin=None,
    kpoints=None,
    human_atoms=False,
):
    """
    This module filters the PROCAR file and re-write a new one.
    """
    welcome()

    print("Input file  :", inFile)
    print("Output file :", outFile)

    print("atoms       :", atoms)
    if atoms:
        print("human_atoms     :", human_atoms)
    print("orbitals  :", orbitals)
    if orbitals:
        print("orb. names  :", orbital_names)
    print("bands       :", bands)
    print("spins       :", spin)
    print("k-points    :", kpoints)

    # Access init class of ProcarFileFilter and pass two arguments
    FileFilter = ProcarFileFilter(inFile, outFile)

    # for atoms
    if atoms:
        print("Manipulating the atoms")

        if human_atoms:
            atoms = [[y - 1 for y in x] for x in atoms]
            print("new atoms list :", atoms)

        # Now just left to call the driver member
        FileFilter.FilterAtoms(atoms)

    # for orbitals
    elif orbitals:
        print("Manipulating the orbitals")
        # If orbitals orbital_names is None, it needs to be filled
        if orbital_names is None:
            orbital_names = ["o" + str(x) for x in range(len(orbitals))]
            print("New orbitals names (default): ", orbital_names)
        # testing if makes sense
        if len(orbitals) != len(orbital_names):
            raise RuntimeError("length of orbitals and orbitals names do not match")

        FileFilter.FilterOrbitals(orbitals, orbital_names)

    # for bands
    elif bands:
        print("Manipulating the bands")

        bmin = bands[0]
        bmax = bands[1]
        if bmax < bmin:
            bmax, bmin = bmin, bmax
            print("New bands limits: ", bmin, " to ", bmax)

        FileFilter.FilterBands(bmin, bmax)

    # for k-points
    elif kpoints:
        print("Manipulating the k-points")

        kmin = kpoints[0]
        kmax = kpoints[1]
        if kmax < kmin:
            kmax, kmin = kmin, kmax
            print("New k-points limits: ", kmin, " to ", kmax)

        FileFilter.FilterKpoints(kmin, kmax)

    # for spin
    elif spin:
        print("Manipulating the spin")

        FileFilter.FilterSpin(spin)

    return
