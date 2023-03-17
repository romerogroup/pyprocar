
from typing import List

from ..utils import ProcarFileFilter
from ..utils import welcome


def filter(
    inFile:str,
    outFile:str,
    atoms:List[int]=None,
    orbitals:List[int]=None,
    orbital_names:List[str]=None,
    bands:List[int]=None,
    spin=None,
    kpoints=None,
    human_atoms:bool=False,
    ):
    """This module filters the PROCAR file and re-write a new one.

    Parameters
    ----------
    inFile : str
        _description_
    outFile : str
        _description_
    atoms : List[int], optional
        _description_, by default None
    orbitals : List[int], optional
        _description_, by default None
    orbital_names : List[str], optional
        _description_, by default None
    bands : List[int], optional
        _description_, by default None
    spin : _type_, optional
        _description_, by default None
    kpoints : _type_, optional
        _description_, by default None
    human_atoms : bool, optional
        _description_, by default False

    Raises
    ------
    RuntimeError
        _description_
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
