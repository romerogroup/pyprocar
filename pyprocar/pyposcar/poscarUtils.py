#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt

from .latticeUtils import distances as _distances
from .poscar import Poscar


def poscarDiff(
    poscar1: Poscar | str, poscar2: Poscar | str, tolerance: float = 0.01
) -> dict[str, object]:
    """It compares two different Poscar objects. Small numerical errors
    up to `tolerance` are ignored.

    Return: a dictionary with the differences found. If the files are
    the same, it returns an empty dictionary (i.e. a False value)

    The comparison does:
    -comparison between elements
    -comparison between lattices (distances and angles)
    -comparison of the relative distances between atoms

    Parameters
    ----------
    poscar1 : Poscar | str
        the first Poscar filename or object
    poscar2 : Poscar | str
        the first Poscar filename or object
    tolerance : float
        numerical difference to consider both files equal. Default 0.01


    Returns
    -------
    dict:
      the differences are stored with keys 'Elements', 'lattices',
      'distances'. If no differences are found an empty dict is
      returned

    """
    # If poscar1, poscar2 are string a Poscar-object needs to be created
    if isinstance(poscar1, str):
        poscar1 = Poscar(poscar1)
    if isinstance(poscar2, str):
        poscar2 = Poscar(poscar2)

    if poscar1.loaded is False:
        poscar1.parse()
    if poscar2.loaded is False:
        poscar2.parse()

    differences: dict[str, object] = {}
    # Checking for type of elements
    assert poscar1.elm is not None
    assert poscar2.elm is not None
    if list(poscar1.elm) != list(poscar2.elm):
        differences["Elements"] = (list(poscar1.elm), list(poscar2.elm))
        return differences
    # The rest only makes sense to check if elements are the same
    # Checking lattice
    assert poscar1.lat is not None
    assert poscar2.lat is not None
    lat_delta = np.zeros((3, 3))
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            # We compare the module of all lattice vectors
            lat_1 = np.dot(poscar1.lat[i], poscar1.lat[j])
            lat_2 = np.dot(poscar2.lat[i], poscar2.lat[j])
            lat_delta[i, j] = np.abs(lat_1 - lat_2)
    for delta_row in lat_delta:
        if any([x > tolerance for x in delta_row]):
            differences["lattices"] = lat_delta
    # Checking distances
    # We get the distance matrix, wich includes distances between atoms for all atoms
    assert poscar1.cpos is not None
    assert poscar2.cpos is not None
    d1 = _distances(poscar1.cpos, lattice=poscar1.lat)
    d2 = _distances(poscar2.cpos, lattice=poscar2.lat)
    delta = d1 - d2
    # We take the norm of the difference between the distances
    delta_norm = np.linalg.norm(delta)
    if delta_norm > tolerance:
        differences["distances"] = delta_norm
    return differences


class poscar_modify:
    """High-level class to change properties of a Poscar-object.

    Methods
    -------
    write(filename, cartesian, xyz)   # write the poscar with `filename`, in `cartesian`?
    pos_multiply(factor, cartesian)   # multiply the position of each atoms by `factor`
    pos_sum(factor, cartesian)        # sums `factor` to each position
    remove(self, atoms, human)        # removes a list of `atoms`
    add(element, position, cartesian) # add a single atom with  `element` at `position`
    shift(amount, cartesian)          # shift all the positions by `amount`
    scale_lattice(factor, cartesian)  # scale the lattice by `factor`. are `cartesian` fixed?

    """

    p: Poscar
    verbose: bool

    def __init__(self, poscar: Poscar | str, verbose: bool = False):
        """High-level class to change properties of a Poscar-object.

        Parameters
        ----------
        poscar : Poscar|str
            Filename or Poscar instance to be modified
        verbose : bool
            verbosity. Default is False

        """
        if isinstance(poscar, str):
            self.p = Poscar(poscar)
        else:
            self.p = poscar
        if self.p.loaded is False:
            self.p.parse()
        self.verbose = verbose

    def write(self, filename: str, cartesian: bool = False, xyz: bool = False) -> None:
        """Writes the content of this class into a file. It just invokes the
        write method from the `Poscar` class. Is here just for convenience.

        Parameters
        ----------
        filename : str
            the name of the file to be written
        cartesian: bool:
            Cartesian (True) or direct (False) coordinates. Default is True
        xyz: bool
            should a xyz be written too? The '.xyz' extension will be added
            automatically to `filename` Default is False.

        """
        direct = True
        if cartesian:
            direct = False
        self.p.write(filename, direct=direct)
        if xyz:
            self.p.xyz(filename + ".xyz")

        if self.verbose:
            print("new POSCAR:")
            print(self.p.poscar)
            if xyz:
                print("XYZ file written")

    def pos_multiply(self, factor: np.ndarray | list[float], cartesian: bool = True) -> None:
        """Multiplies each (x,y,z) position by the Factor (Fx,Fy,Fz)

        Parameters
        ----------
        factor : np.ndarray|list[float]
            array or list with 3 numbers, the x,y,z factors to scale each position
        cartesian: bool
            should the operation be done in Cartesian (False) or direct (True)
            coordinates? Default is True

        """
        factor = np.array(factor, dtype=float)
        if self.verbose:
            print("Multiply positions, factor = ", factor)
            print("old positions:")
            if cartesian:
                print(self.p.cpos)
            else:
                print(self.p.dpos)

        if cartesian is True:
            self.p.cpos = self.p.cpos * factor
            self.p._set_direct()
        else:
            self.p.dpos = self.p.dpos * factor
            self.p._set_cartesian()

        if self.verbose:
            print("\nnew positions:")
            if cartesian:
                print(self.p.cpos)
            else:
                print(self.p.dpos)

    def pos_sum(self, factor: np.ndarray | list[float], cartesian: bool = False) -> None:
        """Add the Factor (Fx,Fy,Fz) to each position

        Parameters
        ----------
        factor : np.ndarray|list[float]
            3 numbers, the factor to add to each position
        cartesian : bool
            should the operation be done in Cartesian (True) or direct coordinates (False).
            Default is False

        """
        factor = np.array(factor, dtype=float)

        if self.verbose:
            print("summing to positions, factor = ", factor)
            print("old positions:")
            if cartesian:
                print(self.p.cpos)
            else:
                print(self.p.dpos)

        if cartesian is True:
            self.p.cpos = self.p.cpos + factor
            self.p._set_direct()
        else:
            self.p.dpos = self.p.dpos + factor
            self.p._set_cartesian()

        if self.verbose:
            print("\nnew positions:")
            if cartesian:
                print(self.p.cpos)
            else:
                print(self.p.dpos)

    def change_elements(
        self,
        indexes: np.ndarray | list[int] | int,
        newElements: np.ndarray | list[str] | str,
        human: bool = False,
    ) -> None:
        """It changes the Element of one or more atoms in this poscar object.

        Parameters
        ----------
        indexes : np.ndarray | list[int] | int
            the 0-based index(es) of the atom to be replaced.
        newElements : np.ndarray | list[str] | str
            the element(s) to replace. Same size of `indexes`
        human : bool
            if True, the index(es) will be one-based, as humans like to count. Default is False

        """
        # converting anything to arrays
        if isinstance(indexes, int):
            indexes = np.array([indexes], dtype=int)
        # I need a np.ndarray
        indexes = np.array(indexes, dtype=int)
        if human:
            indexes = indexes - 1

        if isinstance(newElements, str):
            newElements = [newElements]

        # first retrive the positions
        assert self.p.dpos is not None
        dpos = self.p.dpos[indexes]

        # then removing
        self.remove(list(indexes), human=False)
        # and finally adding a new atoms, one at a time
        for pos, elem in zip(dpos, newElements):
            self.add(elem, pos, cartesian=False)

        if self.verbose:
            print("Added element ", newElements, "at direct coord:", dpos)

    def remove(self, atoms: list[int] | np.ndarray, human: bool = False) -> None:
        """Removes a list of atoms from the Poscar object. The order of
        removal is not trivial, and it is equivalent to removing all the
        desired atoms at once.

        Parameters
        ----------
        atoms : list[int] | np.ndarray
            a list with the indexes of the atoms to remove
        human : bool
            does `atoms` start from 1 (True) or 0 (False)? Default is False

        """
        atoms_arr = np.array(atoms)
        # the atoms list could be disordered, Poscar.remove is safe
        if human:
            atoms_arr = atoms_arr - 1
        self.p.remove(list(int(x) for x in atoms_arr))
        if self.verbose:
            print("removing the following atoms (0-based indexes):", atoms)
            print(self.p.numberSp, self.p.typeSp)

    def add(
        self, element: str, position: list[float] | np.ndarray, cartesian: bool = False
    ) -> None:
        """Adds a single atom to the Poscar object.

        Parameters
        ----------
        element : str
            a string with the atomic specie, e.g. 'Cu'
        position : list[float] | np.ndarray
            [X, Y, Z]
        cartesian : bool
            are the positions in Cartesian (True) or direct coordiantes (False)? Default is False

        """
        position = np.array(position, dtype=float)
        if self.verbose:
            print("New atom:", element, position)
        direct = True
        if cartesian is True:
            direct = False
        self.p.add(position=position, element=element, direct=direct)

    def shift(self, amount: list[float] | np.ndarray, cartesian: bool = False) -> None:
        """Shift all the positions by `amount`, given in Cartesian or direct
        coordinates. The PBCs are always enforced (i.e. [0,1] in direct
        coords). If amount = [0,0,0] it just applies the perodic boundary
        conditions.

        Parameters
        ----------
        amount : list[float] | np.ndarray
            [X,Y,Z] the shift along each basis vector or along Cartesian axis.
        cartesian : bool
            is the `amount` given in Cartesian (True) or direct (False) coords? Default False

        """
        amount = np.array(amount, dtype=float)
        if cartesian:
            if self.verbose:
                print("\nOriginal Cartesian coords:")
                print(self.p.cpos)
            self.p.cpos = self.p.cpos + amount
            self.p._set_direct()
            if self.verbose:
                print("\nShifted Cartesian coords:")
                print(self.p.cpos)
        else:
            if self.verbose:
                print("\nOriginal Direct coords:")
                print(self.p.dpos)
            self.p.dpos = self.p.dpos + amount
            self.p._set_cartesian()
            if self.verbose:
                print("\nShifted Cartesian coords:")
                print(self.p.cpos)

        # enforcing the PBCs
        assert self.p.dpos is not None
        self.p.dpos = np.mod(self.p.dpos, 1.0)
        self.p._set_cartesian()

    def scale_lattice(self, factor: np.ndarray, keep_cartesian: bool = False) -> None:
        """Scale the lattice vectors by factor [a,b,c]

        Parameters
        ----------
        factor : np.ndarray
           [A,B,C], the first lattice vector is multiplied by A, etc.
        keep_cartesian : bool
            What cooddinates should remain constant? Cartesian or direct? Default is False

        """
        assert self.p.lat is not None
        if self.verbose:
            print("Old lattice")
            print(self.p.lat)

        self.p.lat = (self.p.lat.T * factor).T
        if self.verbose:
            print("New lattice")
            print(self.p.lat)
        # setting the new volume
        self.p.volume = np.linalg.det(self.p.lat)

        if keep_cartesian:
            # if cartesian positions are to remain constant, the direct ones
            # needs to be updated
            self.p._set_direct()
        else:
            self.p._set_cartesian()


class poscar_supercell:
    """class to generate a supercell by providing a supercell matrix."""

    poscar: Poscar
    verbose: bool

    def __init__(self, poscar: Poscar | str, verbose: bool = False):
        """This class created a supercell of `poscar`, see the `supercell`
        method

        Parameters
        ----------
        __________

        poscar : Poscar | str
            Filename or Poscar instance to be modified
        verbose : bool
            verbosity. Default is False

        """
        if isinstance(poscar, str):
            self.poscar = Poscar(poscar)
        else:
            import copy

            self.poscar = copy.deepcopy(poscar)
        if self.poscar.loaded is False:
            self.poscar.parse()

        self.verbose = verbose

    def supercell(self, size: np.ndarray) -> Poscar:
        """Creates a supercell of the given size. The content of the original
        Poscar is overwritten

        size = [[b1x, b1y, b1z],
                [b2x, b2y, b2z],
                [b3x, b3y, b3z]]

        Parameters
        ----------
        size : ndarray
            (3x3) array of integers with the supercell vectors in term of the
            original lattice vectors. The order is [[b1x, b1y, b1z], [b2x, ...] ...]


        Returns
        -------
        Poscar
            A Poscar object with the desired supercell. It is the same instance
            stored in this class. Note, the creation of `poscar_supercell` makes
            a deep copy of the `Poscar` instance provided

        """
        assert self.poscar.lat is not None
        assert self.poscar.dpos is not None
        assert self.poscar.elm is not None
        lat = self.poscar.lat
        pos = self.poscar.dpos
        elem = self.poscar.elm
        scell = np.array(size, dtype=int)

        if self.verbose:
            print("original lattice:\n", lat, "\n")
            print("New lattice (in terms of the original vectors)\n", scell, "\n")
            print("Cartesian coordinates new lattice:\n", np.dot(scell, lat))

        # inverse: a=ocell*b
        ocell = np.linalg.inv(scell)
        if self.verbose:
            print("inverse transformation:\n", ocell)
            print("atoms, direct\n", pos)
            print("atoms in cart\n")
            print(np.einsum("ij,jk", pos, lat))

            print("in terms of new lattice\n")
        spos = np.einsum("ij,jk", pos, ocell)

        # I need to find the values of n_i, a_i *inside* the supercell.
        # n_i*a_i = n_i*ocell_ij*b_j
        # then, the condition is : 0 < n_i*ocell_ij < 1
        b = np.ones(3)
        n_max = int(np.max(np.abs(np.einsum("j,ji", b, scell))))
        if self.verbose:
            print("maximum value of n to search for repetitions : ", n_max)

        n_range = np.arange(-n_max, n_max)

        n_grid = np.array([(x, y, z) for z in n_range for y in n_range for x in n_range])
        nuseful: list[npt.NDArray[np.signedinteger[Any]]] = []
        # checking which of the previous repetitions works
        for trial in n_grid:
            value = np.einsum("i,ij", trial, ocell)
            if value.min() >= 0 and value.max() < 1:
                nuseful.append(trial)

        if self.verbose:
            print("set of new coords\n", nuseful)
        npos_list: list[npt.NDArray[np.floating[Any]]] = []
        for nn in nuseful:
            npos_list.append(spos + np.einsum("i,ij", nn, ocell))

        if self.verbose:
            print("positions:")
        npos = np.concatenate(npos_list)
        npos = np.mod(npos, 1)
        if self.verbose:
            print(npos, npos.shape)

        # I can have repeated elements, such as '0 0 1', and '0 0 0'
        # (the 1 can be 0.9999999 and fail the previous filter)
        tol = 0.001
        temp_list: list[npt.NDArray[np.floating[Any]]] = []
        for i in range(len(npos)):
            repeated = False
            for j in range(i):
                d = np.abs(npos[i] - npos[j])
                for k in range(len(d)):
                    if abs(d[k] - 1) < d[k]:
                        d[k] = d[k] - 1
                if np.linalg.norm(d) < tol and i != j:
                    repeated = True
                    if self.verbose:
                        print(i, j, npos[i], npos[j])
            if not repeated:
                temp_list.append(npos[i])

        temp = np.concatenate(temp_list)
        temp = temp.reshape(-1, 3)
        if self.verbose:
            print(temp.shape)
        npos = temp[:]
        new_elem = list(elem) * len(nuseful)
        self.poscar.elm = new_elem
        self.poscar.lat = np.dot(scell, lat)
        self.poscar.dpos = npos
        self.poscar._set_cartesian()

        self.poscar.sort()

        return self.poscar

    def write(self, filename: str, cartesian: bool = False, xyz: bool = False) -> None:
        """Just a convenience method to save the content into a file.

        Parameters
        ----------
        filename : str
            the name of the file to be written
        cartesian : bool
            Do you prefer the output position in Cartesian coords (True)? Default is False
        xyz : bool
            Do you want to write an .xyz file? the .xyz extension will be added to filename.
            Default is False

        """
        pm = poscar_modify(self.poscar, verbose=False)
        pm.write(filename=filename, cartesian=cartesian, xyz=xyz)


def p_atoms_f(args: argparse.Namespace) -> None:
    print("Operations related with atomic positions")
    verbose: bool = args.verbose
    input_file: str = args.input
    output_file: str = args.output
    sum_val: list[float] | None = args.sum
    multiply_val: list[float] | None = args.multiply
    xyz: bool = args.xyz
    cart: bool = args.cart
    sc: bool = args.sc
    remove_val: list[int] | None = args.remove
    human: bool = args.human
    add_val: list[str] | None = args.add

    if verbose:
        print("Input:     ", input_file)
        print("Output:    ", output_file)
        print("sum:       ", sum_val)
        print("multiply:  ", multiply_val)
        print("xyz:       ", xyz)
        print("cartesian: ", cart)
        print("save_cart: ", sc)
        print("remove:    ", remove_val)
        print("human:     ", human)
        print("add:       ", add_val)

    p = Poscar(input_file, verbose=False)
    p.parse()

    modifier = poscar_modify(p, verbose=verbose)

    # first dealing with the maths
    if multiply_val:
        modifier.pos_multiply(multiply_val, cartesian=cart)

    if sum_val:
        modifier.pos_sum(sum_val, cartesian=cart)

    if remove_val:
        modifier.remove(remove_val, human=human)

    if add_val:
        # parsing the string: 'C 1.2 4 -5.0'
        element = add_val[0]
        position_strs = add_val[1:]
        if len(position_strs) != 3:
            raise RuntimeError("the --add parameter has a wrong format, " + str(add_val))
        position = np.array(position_strs, dtype=float)
        modifier.add(element=element, position=position, cartesian=cart)

    # Now we are done with all modifications

    modifier.write(output_file, cartesian=sc, xyz=xyz)


def p_pbc_f(args: argparse.Namespace) -> None:
    print("PBC-related utilities")
    verbose: bool = args.verbose
    input_file: str = args.input
    output_file: str = args.output
    shift_val: list[float] | None = args.shift
    xyz: bool = args.xyz
    cart: bool = args.cart
    sc: bool = args.sc

    if verbose:
        print("Input:     ", input_file)
        print("Output:    ", output_file)
        print("shift:     ", shift_val)
        print("xyz:       ", xyz)
        print("cartesian: ", cart)
        print("save_cart: ", sc)

    p = Poscar(input_file, verbose=False)
    p.parse()

    modifier = poscar_modify(p, verbose=verbose)

    if shift_val:
        modifier.shift(shift_val, cart)
    modifier.write(output_file, cartesian=sc, xyz=xyz)


def p_lattice_f(args: argparse.Namespace) -> None:
    print("Lattice stuff")
    verbose: bool = args.verbose
    input_file: str = args.input
    output_file: str = args.output
    scale_val: list[float] | None = args.scale
    factor_val: float = args.factor
    xyz: bool = args.xyz
    cart: bool = args.cart
    sc: bool = args.sc

    if verbose:
        print("Input:     ", input_file)
        print("Output:    ", output_file)
        print("scale:     ", scale_val)
        print("factor:    ", factor_val)
        print("xyz:       ", xyz)
        print("cartesian: ", cart)
        print("save_cart: ", sc)

    p = Poscar(input_file, verbose=False)
    p.parse()

    modifier = poscar_modify(p, verbose=verbose)

    # first dealing with the factors, if any
    scale: npt.NDArray[np.floating[Any]]
    if scale_val is None:
        scale = np.array([1.0, 1.0, 1.0])
    else:
        scale = np.array(scale_val)

    factor = factor_val * scale
    # Now changing the lattice vectors
    modifier.scale_lattice(factor=factor, keep_cartesian=cart)
    # and writing
    modifier.write(output_file, cartesian=sc, xyz=xyz)


def p_scell_f(args: argparse.Namespace) -> None:
    print("Supercell creation")
    verbose: bool = args.verbose
    input_file: str = args.input
    output_file: str = args.output
    b1: list[int] = args.b1
    b2: list[int] = args.b2
    b3: list[int] = args.b3
    xyz: bool = args.xyz
    sc: bool = args.sc

    if verbose:
        print("Input:     ", input_file)
        print("Output:    ", output_file)
        print("b1:        ", b1)
        print("b2:        ", b2)
        print("b3:        ", b3)
        print("xyz:       ", xyz)
        print("save_cart: ", sc)

    p = Poscar(input_file)
    p.parse()

    supercell = poscar_supercell(p, verbose=verbose)
    supercell.supercell(size=np.array([b1, b2, b3]))
    supercell.write(
        output_file,
    )

    supercell.write(filename=output_file, cartesian=sc, xyz=xyz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilities related to poscar.")
    subparsers = parser.add_subparsers(help="sub-command")

    # defining subparsers
    p_atoms = subparsers.add_parser(
        "atoms", help="operations related with atomic positions, leaving the lattice unchanged"
    )
    p_atoms.set_defaults(func=p_atoms_f)

    p_pbc = subparsers.add_parser(
        "pbc",
        help="boundaries-related operations. It always applies PCBs at the atomic positions. ",
    )
    p_pbc.set_defaults(func=p_pbc_f)

    p_lattice = subparsers.add_parser(
        "lattice",
        help="lattice-related operations. The positions may/may not be modified with the lattice",
    )
    p_lattice.set_defaults(func=p_lattice_f)

    p_scell = subparsers.add_parser("supercell", help="Creates a supercell")
    p_scell.set_defaults(func=p_scell_f)

    # filing the subparser options
    p_atoms.add_argument("input", type=str, help="Input POSCAR file ")
    p_atoms.add_argument("output", type=str, help="Output POSCAR file")
    p_atoms.add_argument(
        "-s",
        "--sum",
        type=float,
        nargs=3,
        help="Sum (adds) [X Y Z] to each "
        "position. This is in Direct coordinates by default, see `--cart`."
        " The PBC are NOT taken in to account, i.e. [0.5 0.5 0.5] + "
        "[0.0 0.0 0.7] = [0.5 0.5 1.2]. Multiplication precedes addition."
        " Nevertheless it is recommended to separate the operation in "
        "different calls",
    )

    p_atoms.add_argument(
        "-m",
        "--multiply",
        type=float,
        nargs=3,
        help="Multiplies [X Y Z]"
        " element-wise each postion. This is in Direct coords by default,"
        " see `--cart`. PBC are not taken into account, i.e. [0.1 0.2 0.7]"
        " * [0.5 1.0 2.0] = [0.05 0.2 1.4]. Multiplication precedes addition."
        " Nevertheless it is recommended to separate the operation in "
        "different calls",
    )

    p_atoms.add_argument("--xyz", action="store_true", help="writes an xyz file")

    p_atoms.add_argument(
        "-c",
        "--cart",
        action="store_true",
        help="set to perform the "
        "operations in the cartesian positions. By default Direct coords"
        " are used",
    )
    p_atoms.add_argument(
        "--sc",
        action="store_true",
        help="set to save the file in "
        "Cartesian. By default Direct coords are used, even if the "
        "operations are  done in cartesians",
    )
    p_atoms.add_argument("-v", "--verbose", action="store_true")
    p_atoms.add_argument(
        "--remove",
        "-r",
        nargs="+",
        type=int,
        help="removes the atoms"
        " with the given indexes. They indexes start from 0, unless you"
        " set `--human`",
    )
    p_atoms.add_argument(
        "--human",
        "-u",
        action="store_true",
        help="All the indexes "
        "from the input are put in `human` format, starting from 1 (i.e"
        ".: the first atom is 1, and so on)",
    )
    p_atoms.add_argument(
        "--add",
        "-a",
        nargs=4,
        type=str,
        help="`--add C 0.5 0.3 0.1`,"
        " adds an C atom at the given direct coordinate. The coordinates"
        " can be direct (default) or Cartesian, see `--cart`",
    )
    # p_atoms.add_argument('--duplicates', action='store_true', help="removes the duplicate "
    #                      "atoms, i.e. Those that are almost in the same position. Mind the"
    #                      "program will keep only the first occurrence. See `duplicate_tol`")
    # p_atoms.add_argument('--duplicate_tol', type=float, default=0.1, help='Tolerance '
    #                      'critetion for setting an atom as `duplicate`')

    #########

    p_pbc.add_argument("input", type=str, help="input file")
    p_pbc.add_argument("output", type=str, help="output file")
    # p_pbc.add_argument('-p', '--pbc', action='store_true', help='it actually imposes'
    #                    ' PBCs, by moving all the atoms into the [0, 1) interval. ')
    p_pbc.add_argument(
        "-s",
        "--shift",
        type=float,
        nargs=3,
        help="it adds [X, Y ,Z] to each coordinate, and then applies PBCs. see `--cart`, `--sc`",
    )
    p_pbc.add_argument(
        "--cart", action="store_true", help="The shift is given and made in Cartesian coords. "
    )
    p_pbc.add_argument(
        "--sc",
        action="store_true",
        help="sets the output POSCAR file in cartesian coords. The default is direct",
    )
    p_pbc.add_argument("--xyz", action="store_true", help="also saves a XYZ file")
    p_pbc.add_argument("-v", "--verbose", action="store_true")

    #########

    p_lattice.add_argument("input", type=str, help="Input POSCAR file ")
    p_lattice.add_argument("output", type=str, help="Output POSCAR file")
    p_lattice.add_argument(
        "-s",
        "--scale",
        type=float,
        nargs=3,
        help="multiplies each "
        "lattice vector by the [X Y Z] factor. The positions may/may "
        "not be affeted. See `cart`",
    )
    p_lattice.add_argument(
        "-f",
        "--factor",
        type=float,
        default=1.0,
        help="multiplies all the lattice"
        " by the given factor. The position may/may not be affected, "
        "see `--cart`",
    )
    p_lattice.add_argument(
        "-c",
        "--cart",
        action="store_true",
        help="The positions are"
        " keep fixed in cartesian coordinates, their value in direct"
        " coords change with the lattice. The default is to keep the"
        " positions unaltered in direct coords.",
    )
    p_lattice.add_argument(
        "--sc",
        action="store_true",
        help="set it to write the output file in cartesian coords. Te default is in direct coords.",
    )
    p_lattice.add_argument("-v", "--verbose", action="store_true")
    p_lattice.add_argument("--xyz", action="store_true", help="also a XYZ file is writtem")
    # p.lattice.add_argument('-r', '--rotate')

    ##########

    p_scell.add_argument("input", type=str, help="input file")
    p_scell.add_argument("output", type=str, help="output file")

    p_scell.add_argument(
        "--b1", nargs=3, type=int, default=[1, 0, 0], help="first supercell vector, eg '1 2 -1'"
    )
    p_scell.add_argument(
        "--b2", nargs=3, type=int, default=[0, 1, 0], help="first supercell vector, eg '1 1 1'"
    )
    p_scell.add_argument(
        "--b3", nargs=3, type=int, default=[0, 0, 1], help="first supercell vector, eg '0 0 10'"
    )
    p_scell.add_argument("--xyz", action="store_true")
    p_scell.add_argument(
        "--sc",
        action="store_true",
        help="set it to write the output"
        " file in cartesian coords. The default is in direct coords.",
    )
    p_scell.add_argument("-v", "--verbose", action="store_true")

    #########################################

    # there is a python 3 bug. If no argument provided an exception is raised
    # try:
    args = parser.parse_args()

    args.func(args)
    # except AttributeError:
    #  parser.error("too few arguments")
