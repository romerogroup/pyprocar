#!/usr/bin/env python
"""DFTB+ parser for PyProcar."""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from pyprocar.io.base import BaseParser
from pyprocar.pyposcar.poscar import Poscar

if TYPE_CHECKING:
    from pyprocar.core.dos import DensityOfStates
    from pyprocar.core.ebs import ElectronicBandStructure
    from pyprocar.core.kpoints import KPath
    from pyprocar.core.structure import Structure


class DFTB_evec:
    """Class to parse DFTB+ eigenvector files."""

    verbose: bool
    filename: str | Path
    f: str | None
    Nkpoints: int
    Nbands: int
    Natoms: int
    Norbs: int
    spd: npt.NDArray[np.float64] | npt.NDArray[np.complex128] | None
    orbDict: OrderedDict[str, int] | None
    is_complex: bool
    bands: npt.NDArray[np.float64] | None
    kpoints: npt.NDArray[np.float64] | None
    occupancies: npt.NDArray[np.float64] | None

    def __init__(self, filename: str | Path, verbose: bool = False) -> None:
        self.verbose = verbose
        self.filename = filename
        self.f = None  # the whole file
        self.Nkpoints = 0  # How many k-points
        self.Nbands = 0  # How many bands
        self.Natoms = 0
        self.Norbs = 0  # How many orbitals (as a whole quantity, not by atom)
        self.spd = None  # np.array[Nkpoints, Nbands, Natoms, Norbs]
        self.orbDict = None  # An ordered dictionary assigning an index to each orbital
        self.is_complex = False  # Is the data complex or real?
        self.bands = None  # The eigenvalues (Nkpoints, Nbands)
        self.kpoints = None  # The kpoints (Nkpoints, 4)
        self.occupancies = None  # (Nkpoints, Nbands)

    def load(self) -> None:
        with open(self.filename) as file_handle:
            self.f = file_handle.read()
        # The syntax is different whether there are more than one k-point
        # or just a single k-point.
        klist = re.findall(r"K-point\s*:\s*\d+", self.f)
        klist = set(klist)  # using `set` to remove repeated elements
        self.Nkpoints = len(klist)
        # if no kpoints found, there is only one
        if self.Nkpoints == 0:
            self.Nkpoints = 1
        if self.verbose:
            print("Number of k-points: ", self.Nkpoints)
        if self.Nkpoints == 1:
            self.is_complex = False
        else:
            self.is_complex = True
        if self.verbose:
            print("Looking for complex data?", self.is_complex)

        # How many eigenvectors?
        band_list = re.findall(r"Eigenvector\s*:\s*\d+", self.f)
        up_list = re.findall(r"Eigenvector\s*:\s*\d+\s*\(up\)", self.f)
        if len(band_list) != len(up_list):
            print("Number of bands: ", len(band_list), " Number of UP bands: ", len(up_list))
            raise RuntimeError("Spin polarization not supported")
        up_list = set(up_list)
        self.Nbands = len(up_list)
        if self.verbose:
            print("Number of bands: ", self.Nbands)

        # How many atoms?
        atoms_list = re.findall(r"^\s*\d+\s+\w+", self.f, re.MULTILINE)
        atoms_list = set(atoms_list)
        self.Natoms = len(atoms_list)
        if self.verbose:
            print("Number of atoms: ", len(atoms_list))
            # print(atoms_list)

        # How many orbitals?
        orb_list = re.findall(r"[ ]+([a-zA-Z]+[-_a-zA-z\d]*)[ ]+[.(\d\-]+", self.f)
        self.Norbs = len(set(orb_list))
        print("Orbitals found:", set(orb_list))
        if self.verbose:
            print("Number of orbitals: ", self.Norbs)

        # also, we are going to create a Dict of the unique orbitals,
        # assigning them an index to store them into the arrays.
        # We are faking an orderedSet by using an orderedDict
        d: OrderedDict[str, int] = OrderedDict()
        for orb in orb_list:
            if orb not in d:
                d[orb] = len(d)
        self.orbDict = d
        if self.verbose:
            print("Set of orbitals and their indexes: ", self.orbDict)

        # Creating a big array with all the info. It needs to be
        # initialized to zero, since not all fields are present in the
        # file
        if self.is_complex:
            self.spd = np.zeros(
                [self.Nkpoints, self.Nbands, self.Natoms, self.Norbs], dtype=complex
            )
        else:
            self.spd = np.zeros([self.Nkpoints, self.Nbands, self.Natoms, self.Norbs], dtype=float)

        # splitting the file by Eigenvector (and perhaps K-point):
        #
        # The first match is 'Coefficients and Mulliken populations of the
        # atomic orbitals', and needs to be discarded

        if self.Nkpoints == 1:
            # Eigenvector:   1    (up)
            evecs = re.split(r"Eigenvector:\s*\d+\s*\(up\)\s*", self.f)[1:]
        else:
            # K-point:    1    Eigenvector:    1    (up)
            evecs = re.split(r"K-point:\s*\d+\s*Eigenvector:\s*\d+\s*\(up\)\s*", self.f)[1:]
        if self.verbose:
            print(
                "We got :",
                len(evecs),
                "Eigenvector entries (expected: ",
                self.Nkpoints * self.Nbands,
                ")",
            )
        # temporal storage of the evecs as an Nkpoint*Nbands array
        evecs = np.array(evecs).reshape((self.Nkpoints, self.Nbands))

        # going to parse all the data, iterating over:
        # Kpoints->Bands->Atoms (and assignation at orbital level)

        for cKpoint in range(self.Nkpoints):
            for cEvec in range(self.Nbands):
                # just dealing with the data of the current block
                evec = evecs[cKpoint, cEvec]
                # splitting by atom
                evec = re.split(r"\n*\s*\d+\s*[a-zA-Z]+\s+", evec)
                # the first match should be '', and should be discarded
                if evec[0] == "":
                    evec.pop(0)
                # if self.verbose:
                #  print('number of atoms in the block',  len(evec))

                cAtom = 0  # a counter of atoms
                for atom in evec:
                    # next we search for the orbitals involved
                    orbNames = re.findall(r"[a-zA-Z_]+[-_a-zA-z\d]*", atom)
                    # what is the index of each orbital?
                    orbIndexes = [self.orbDict[x] for x in orbNames]
                    # print(orbNames, orbIndexes)
                    # getting all the floating numbers
                    numbers_list = re.findall(r"[\-0-9]+\.\d+", atom)
                    numbers_arr = np.array(numbers_list, dtype=float)
                    # if they are complex, we need to cast them. The last column
                    # needs to be ignored.
                    numbers: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
                    if self.is_complex:
                        numbers_arr = numbers_arr.reshape((len(orbIndexes), 3))  # Re, Im, Mulliken
                        numbers = numbers_arr[:, 0] + 1j * numbers_arr[:, 1]
                    else:
                        numbers_arr = numbers_arr.reshape((len(orbIndexes), 2))  # Re. Mulliken
                        numbers = numbers_arr[:, 0]
                    # print(numbers)
                    self.spd[cKpoint, cEvec, cAtom, orbIndexes] = numbers
                    cAtom = cAtom + 1
        if self.verbose:
            print("file", self.filename, "loaded")

        print("Overlap Populations (Mulliken) are ignored")
        # if self.normalized:
        #   if self.verbose:
        #     print('The projection of eigenvectors will be normalized')
        #   #norms = np.zeros((self.Nkpoints, self.Nbands), dtype=float)
        #   for k in range(self.Nkpoints):
        #     for b in range(self.Nbands):
        #       self.spd[k,b] /= np.linalg.norm(self.spd[k,b])
        #       #norms[k,b] = np.linalg.norm(self.spd[k,b])
        #     #self.spd = self.spd/norms
        return

    def set_bands(
        self, bands: npt.NDArray[np.float64], occupancies: npt.NDArray[np.float64]
    ) -> None:
        if self.verbose:
            print("Setting the bands into the eigenvectors class")
        if bands.shape == (self.Nkpoints, self.Nbands):
            self.bands = bands
        else:
            raise RuntimeError(
                "The number of the bands array doesn't"
                "match: " + str(bands.shape) + " vs " + str((self.Nkpoints, self.Nbands))
            )
        if occupancies.shape == (self.Nkpoints, self.Nbands):
            self.occupancies = occupancies
        else:
            raise RuntimeError(
                "The number of the occupancies array doesn't"
                "match: " + str(occupancies.shape) + " vs " + str((self.Nkpoints, self.Nbands))
            )

    def set_kpoints(self, kpoints: npt.NDArray[np.float64]) -> None:
        if self.verbose:
            print("Setting the kpoints into the eigenvectors class")
        if kpoints.shape == (self.Nkpoints, 4):
            self.kpoints = kpoints
        else:
            raise RuntimeError(
                "The shape of the kpoints array doesn't"
                "match: " + str(kpoints.shape) + " vs " + str((self.Nkpoints, 4))
            )

    def writeProcar(self) -> None:
        if self.spd is None or self.bands is None or self.kpoints is None or self.occupancies is None:
            raise RuntimeError("Data must be loaded before writing PROCAR")
        if self.orbDict is None:
            raise RuntimeError("Orbital dictionary must be loaded before writing PROCAR")

        # changing the coefficients to its module:
        spd_data = np.array((self.spd * np.conjugate(self.spd)).real, dtype=float)

        # I need to sum over orbitals, and over atoms and over both
        tot_orbs = np.sum(spd_data, axis=3)
        tot_atoms = np.sum(spd_data, axis=2)
        tot_oa = np.sum(tot_orbs, axis=2)  # already summed over axis=3
        if self.verbose:
            print("sum over orbitals.shape: ", tot_orbs.shape)
            print("sum over atoms.shape: ", tot_atoms.shape)

        # casting to strings
        if self.verbose:
            print("going to transform all data to strings: ", end="")

        spd_str = np.array(["%.5f" % x for x in spd_data.flatten()]).reshape(
            (self.Nkpoints, self.Nbands, self.Natoms, self.Norbs)
        )

        tot_orbs_str = np.array(["%.5f" % x for x in tot_orbs.flatten()]).reshape(
            (self.Nkpoints, self.Nbands, self.Natoms)
        )

        tot_atoms_str = np.array(["%.5f" % x for x in tot_atoms.flatten()]).reshape(
            (self.Nkpoints, self.Nbands, self.Norbs)
        )

        tot_oa_str = np.array(["%.5f" % x for x in tot_oa.flatten()]).reshape(
            (self.Nkpoints, self.Nbands)
        )

        # also the energies need to be casted to strings safely (i.e. no
        # scientific notation)
        bands_str = np.array(["%.6f" % x for x in self.bands.flatten()]).reshape(
            (self.Nkpoints, self.Nbands)
        )

        # casting kpoints and weights
        kpoints_str = np.array(["%.6f" % x for x in self.kpoints.flatten()]).reshape(
            (self.Nkpoints, 4)
        )

        # and occupancies
        occupancies_str = np.array(["%.5f" % x for x in self.occupancies.flatten()]).reshape(
            (self.Nkpoints, self.Nbands)
        )

        if self.verbose:
            print("done")

        # writing the data

        f = open("PROCAR", "w")
        # the header
        f.write("PROCAR lm decomposed\n")
        # # of k-points:  165         # of bands:    6         # of ions:    10
        f.write(
            "# of k-points:  "
            + str(self.Nkpoints)
            + "         # of bands:    "
            + str(self.Nbands)
            + "         # of ions:    "
            + str(self.Natoms)
            + "\n"
        )

        for ikpoint in range(self.Nkpoints):
            # preparing the K-point line
            index_k = str(ikpoint + 1)
            kpoint_vals = kpoints_str[ikpoint][:3]
            kpoint_line = " ".join([str(x) for x in kpoint_vals])
            weight = str(kpoints_str[ikpoint][3])
            kstr = "\n k-point " + index_k + " :   " + kpoint_line + " weight = " + weight + "\n"
            f.write(kstr)

            for iband in range(self.Nbands):
                # preparing the band line
                index_b = str(iband + 1)
                energy = str(bands_str[ikpoint][iband])
                occ = str(occupancies_str[ikpoint][iband])
                bstr = "\nband " + index_b + " # energy " + energy + " # occ. " + occ + "\n"
                f.write(bstr)

                # preparing atom string
                # ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot
                astr = "\nion   " + "  ".join(self.orbDict.keys()) + " tot\n "
                for iatom in range(self.Natoms):
                    index_a = str(iatom + 1)
                    astr += index_a + " " + " ".join(spd_str[ikpoint, iband, iatom])
                    astr += " " + str(tot_orbs_str[ikpoint, iband, iatom]) + "\n "
                # special line `tot`
                astr += "tot " + " ".join(tot_atoms_str[ikpoint, iband])
                astr += " " + str(tot_oa_str[ikpoint, iband]) + "\n"
                f.write(astr)
        f.close()


class DFTB_utils:
    """Utilities that do not belong to other place."""

    verbose: bool

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def find_fermi(self, filename: str | Path) -> float:
        """Find the Fermi energy from output file."""
        if os.path.isfile(filename):
            with open(filename) as f:
                fermiFile = f.read()
        else:
            raise RuntimeError("File " + str(filename) + " not found")
        # Fermi level:                        -0.1467617917 H           -3.9936 eV
        fermi_match = re.findall(r"Fermi level:\s*[\-\d.]+\s*H\s*([\-\d.]+)", fermiFile)
        fermi = float(fermi_match[0])
        if self.verbose:
            print("Fermi energy found: ", fermi, "eV")
        return fermi

    def get_kpoints(self, filename: str | Path) -> npt.NDArray[np.float64]:
        """Get k-points from XML file in direct coordinates."""
        from xml.dom import minidom

        dom = minidom.parse(str(filename))
        kpoints_elem = dom.getElementsByTagName("kpointsandweights")[0]
        if kpoints_elem.firstChild is None:
            raise RuntimeError("No k-points found in XML")
        # minidom Text node has .data attribute but typestubs don't properly type it
        kpoints_str: str = getattr(kpoints_elem.firstChild, "data", "")
        kpoints_list = re.findall(r"-?\d+\.\d+", kpoints_str)
        kpoints = np.array(kpoints_list, dtype=float)
        Nkpoints = len(kpoints) // 4
        # the last entry is the weight
        kpoints = kpoints.reshape((Nkpoints, 4))
        if self.verbose:
            print("Nkpoints: ", Nkpoints)
        return kpoints

    def find_lattice(self, filename: str | Path) -> npt.NDArray[np.float64]:
        """Get lattice vectors from XML file."""
        from xml.dom import minidom

        dom = minidom.parse(str(filename))
        lat_elem = dom.getElementsByTagName("latticevectors")[0]
        if lat_elem.firstChild is None:
            raise RuntimeError("No lattice vectors found in XML")
        # minidom Text node has .data attribute but typestubs don't properly type it
        lat_str: str = getattr(lat_elem.firstChild, "data", "")
        lat_list = re.findall(r"-?\d+\.\d+", lat_str)
        lat = np.array(lat_list, dtype=float).reshape((3, 3))
        # Bohr to Angstroms
        lat = lat * 0.529177249
        return lat

    def find_atoms(
        self, filename: str | Path
    ) -> tuple[list[str], list[int], npt.NDArray[np.float64]]:
        """Get atom types and positions from XML file. Positions are in Cartesian Angstroms."""
        from xml.dom import minidom

        dom = minidom.parse(str(filename))
        typenames_elem = dom.getElementsByTagName("typenames")[0]
        coords_elem = dom.getElementsByTagName("typesandcoordinates")[0]
        if typenames_elem.firstChild is None or coords_elem.firstChild is None:
            raise RuntimeError("No atom info found in XML")
        # minidom Text node has .data attribute but typestubs don't properly type it
        typenames_str: str = getattr(typenames_elem.firstChild, "data", "")
        coords_str: str = getattr(coords_elem.firstChild, "data", "")

        typenames = re.findall(r'"([\w]+)"\s+', typenames_str)
        ntypes = re.findall(r"\s(\d+)\s", coords_str)
        # I need the number of elements per type
        n = len(typenames)
        types_dict = dict(zip(typenames, range(1, n + 1)))
        occurences = [ntypes.count(str(types_dict[x])) for x in typenames]
        positions_list = re.findall(r"([-\d]+\.[-\d]+)", coords_str)
        positions = np.array(positions_list, dtype=float).reshape((sum(occurences), 3))
        # Bohr to Angstrom
        positions = positions * 0.529177249
        return typenames, occurences, positions

    def writePoscar(self, detailed_xml: str | Path) -> None:
        # the positions are written in Bohrs and Cartesain coordinates

        poscarStr = "automatically created to speedup pyProcar\n"
        poscarStr += "1.0\n"
        lat = self.find_lattice(detailed_xml)
        lat = np.array(lat, dtype=str)
        lat = "\n".join([" ".join(x) for x in lat])
        poscarStr += lat + "\n"

        typenames, occurences, positions = self.find_atoms(detailed_xml)

        poscarStr += " ".join(typenames) + "\n"
        poscarStr += " ".join([str(x) for x in occurences]) + "\n"
        poscarStr += " C \n"

        positions = np.array(positions, dtype=str)
        poscarStr += "\n".join([" ".join(x) for x in positions])
        poscarStr += "\n"

        # print(poscarStr)

        p = Poscar("", verbose=False)
        p.parse(fromString=poscarStr)
        p.write("POSCAR", direct=True)
        return

    def writeOutcar(self, detailed_out: str | Path, detailed_xml: str | Path) -> None:
        """Write OUTCAR file with Fermi energy and reciprocal lattice."""
        with open("OUTCAR", "w") as f:
            fermi = self.find_fermi(detailed_out)
            f.write("E-fermi : " + str(fermi) + " \n")

            lat = self.find_lattice(detailed_xml)
            vol = float(np.dot(lat[0], np.cross(lat[1], lat[2])))
            b0 = np.cross(lat[1], lat[2]) / vol
            b1 = np.cross(lat[2], lat[0]) / vol
            b2 = np.cross(lat[0], lat[1]) / vol

            f.write("\nreciprocal lattice vectors \n")
            f.write(str(lat[0, 0]) + " " + str(lat[0, 1]) + " " + str(lat[0, 2]) + "  ")
            f.write(str(b0[0]) + " " + str(b0[1]) + " " + str(b0[2]) + "\n")
            f.write(str(lat[1, 0]) + " " + str(lat[1, 1]) + " " + str(lat[1, 2]) + "  ")
            f.write(str(b1[0]) + " " + str(b1[1]) + " " + str(b1[2]) + "\n")
            f.write(str(lat[2, 0]) + " " + str(lat[2, 1]) + " " + str(lat[2, 2]) + "  ")
            f.write(str(b2[0]) + " " + str(b2[1]) + " " + str(b2[2]) + "\n")


class DFTB_bands:
    """Parse DFTB+ band output files."""

    verbose: bool
    filename: str | Path
    Nkpoints: int
    Nbands: int
    Bands: npt.NDArray[np.float64] | None
    Occupancies: npt.NDArray[np.float64] | None

    def __init__(self, filename: str | Path, verbose: bool = False) -> None:
        self.verbose = verbose
        self.filename = filename
        self.Nkpoints = 0
        self.Nbands = 0
        self.Bands = None
        self.Occupancies = None

    def load(self) -> None:
        with open(self.filename) as f:
            data = f.read()

        # Spin-polarized data is not supproted yet, if there exist, we
        # need to raise an exception
        spinful = re.findall(r"SPIN\s*2", data)
        if len(spinful) == 2:
            raise RuntimeError("Spinful calculations are not supported")

        # splitting by k-point
        #  KPT            1  SPIN            1  KWEIGHT    1.0000000000000000
        kpoints = re.split(r"\s*KPT\s*\d+\s*\w+\s*1\s*\w+\s*[-.0-9E]+\s*", data)
        # the first value has to be empty ('') there is nothing before the
        # first match
        if kpoints[0] != "":
            print("WARNING, the first match wasn't empty. It content is: ", kpoints[0])
        kpoints.pop(0)
        self.Nkpoints = len(kpoints)
        if self.verbose:
            print("Number of Kpoints (bands): ", self.Nkpoints)

        # splitting the numbers at each kpoint, only the numbers at the
        # middleand end are needed :
        #
        #     1   -20.981  2.00000
        evalues_list: list[list[str]] = []
        occs_list: list[list[str]] = []  # occupancies
        for kpoint in kpoints:
            occ = re.findall(r"\d+\s+[0-9.\-]+\s+([0-9.\-]+)\s*\n?", kpoint)
            kpoint_bands = re.findall(r"\d+\s+([0-9.\-]+)\s+[0-9.\-]+\s*\n?", kpoint)
            evalues_list.append(kpoint_bands)
            occs_list.append(occ)
        evalues = np.array(evalues_list, dtype=float)
        occs = np.array(occs_list, dtype=float)
        if self.verbose:
            print("Bands found (Nkpoints. Nbands): ", evalues.shape)

        self.Bands = evalues
        self.Occupancies = occs


class DFTB_input:
    """Parse DFTB+ input files (dft_in.hsd)."""

    verbose: bool
    filename: str | Path
    finput: str

    def __init__(self, filename: str | Path, verbose: bool = False) -> None:
        self.verbose = verbose
        self.filename = filename
        self.finput = ""
        if self.verbose:
            print("DFTB_input.__init__(): going to open ", self.filename)
        with open(self.filename) as f:
            self.finput = f.read()

    def _remove_comments(self) -> str:
        # going to remove any text following a comment
        pattern = r"(\".*?\"|\'.*?\')|(#[^\r\n]*$)"
        # first group captures quoted strings (double or single)
        # second group captures comments
        regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

        def _replacer(match: re.Match[str]) -> str:
            # if the 2nd group (capturing comments) is not None,
            # it means we have captured a non-quoted (real) comment string.
            if match.group(2) is not None:
                return ""  # so we will return empty to remove the comment
            else:  # otherwise, we will return the 1st group
                result = match.group(1)
                return result if result is not None else ""  # captured quoted-string

        new_file = regex.sub(_replacer, self.finput)
        return new_file

    def _find_block(self, string: str) -> str:
        """It finds a full block named `string` and return the full string of
        the block.
        It doesn't work of individual tags.

        The block (and some tags) are xml-like and parsing them is not direct
        """
        string2 = r"^\s*(" + string + r"\s*=\s*\{.*)"
        occurences = re.findall(string2, self.finput, re.MULTILINE)
        if self.verbose:
            print("searching string:", string2, occurences)
        if self.verbose:
            print("DFTB_input._find_block:", len(occurences), " blocks found")
        if len(occurences) > 1:
            raise RuntimeError("More than one block: " + string)
        if len(occurences) == 0:
            block = ""
            return block

        # Otherwise there is only one occurrence, and I need to find the actual block

        # from the block to EOF, I will isolate the block later
        block = re.findall(string2 + r".*", self.finput, re.DOTALL | re.MULTILINE)[0]
        # if self.verbose:
        #   print('\n\n\n Block')
        #   print(block)

        # counting {,}, to know where to close the block
        pattern = re.compile(r"(\{)|(\})")
        isopen = 0
        end_pos = len(block)
        for m in pattern.finditer(block):
            # print(m.groups(1), m.groups(2))
            if m.group(1):
                isopen += 1
                # print('a { found, isopen = ', isopen)
            if m.group(2):
                isopen -= 1
                # print('a } found, isopen = ', isopen)
            if isopen == 0:
                end_pos = m.start(0) + 1
                break
        block = block[:end_pos]
        if self.verbose is True:
            print("block", string, "found:")
            print(block)
            print("------------")
        return block[:]

    def _set_tag(self, tagName: str, blockName: str, value: str) -> None:
        """Searches for the tag `tagName` in the block `blockName`, and set
        its value to `value`"""

        # First, finding the block, if it doesnt exist, it must be created.
        block = self._find_block(blockName)
        if block == "":
            self.finput += "\n" + blockName + " = {\n}\n"
            # now the block does exist, I need to update the variable
            block = self._find_block(blockName)
        # I need to find/create the tag
        #
        # the tag pattern could be as simple as 'foo = ' or more
        # complicated 'foo [qux] = bar'
        tagPattern = tagName + r"\W+.*"
        tag = re.findall(tagPattern, block)
        if self.verbose:
            print("does the tag exists?")
            print(tag)
        if len(tag) > 1:
            raise RuntimeError("More than one occurrence of " + tagName + " found: " + str(tag))
        elif len(tag) == 0:
            if self.verbose:
                print("The tag does not exist, I am going to create it")
                print("---------")
            # no tag, just create it
            pattern = "(" + blockName + r"\s*=\s*\{.*)"
            newTag = r"\1\n    " + tagName + " = " + re.escape(value)
            newBlock = re.sub(pattern, newTag, block)
        else:
            if self.verbose:
                print("The tag exist")
                print("---------")

            # there is one tag, I need to isolate it. Can be a complex
            # multiline matrix-like field foo = bar { baz = {}, quux = {} }
            # I will analyze the all line below `tag` (including it)
            tag = re.findall(tagPattern, block, re.DOTALL)[0]
            # does the tag have any { within its first line?
            if len(re.findall(r"\{", tag.split("\n")[0])) == 0:
                tag = tag.split("\n")[0]
                if self.verbose:
                    print("The tag is just one line:", tag, "\n")
                newTag = tagName + " = " + value
                # avoid re.sub(), the unescaped {} are troublesome
                newBlock = block.replace(tag, newTag)
                self.finput = self.finput.replace(block, newBlock)
            else:
                # counting {,}, to know where to close the block
                pattern = re.compile(r"(\{)|(\})")
                isopen = 0
                end_pos = len(tag)
                for m in pattern.finditer(tag):
                    if m.group(1):
                        isopen += 1
                    if m.group(2):
                        isopen -= 1
                    if isopen == 0:
                        end_pos = m.start(0) + 1
                        break
                tag = tag[:end_pos]
                if self.verbose:
                    print("tag (complex)")
                    print(tag)
                    print("------------")
            # if tag was found
            newTag = tagName + " = " + value
            newBlock = block.replace(tag, newTag)
        # regardless tag found or created
        self.finput = self.finput.replace(block, newBlock)

        if self.verbose:
            print("Block", blockName, "with tag", tagName, "=", value)
            print(newBlock)
            print("-------\n")
        return

    def add_pyprocar_tags(self, output: str | Path | None = None) -> None:
        """This methods parses the dftb_input (already loaded) and add/changes
        the next tags to make DFTB+ output compatible with pyprocar:

         bandstructure (bands.out)
         kpoints (detailed.xml)
         Fermi energy (detailed.out)
         wavefunctions (eigenvectors.out)

        All changes are done in memory, not to the file, see self.write()
        """

        # removing all comments, they could do nasty things
        self.finput = self._remove_comments()

        # the files `bands.out` and `eigenvectors.out` will be written by
        # default, unless `WriteBandOut = No` is at `Analysis`. Anyway we
        # will overwrite if present or write if missing
        #
        # Analysis = {
        #   ...
        #   WriteBandOut = Yes
        #   WriteEigenvectors = Yes
        #   ... = { ... }
        #   EigenvectorsAsText = Yes
        #  }
        #
        if self.verbose:
            print("DFTB_inout.add_pyprocar_tags: searching tags")

        if self.verbose:
            print("    self._set_tag('WriteBandOut', 'Analysis', value='Yes')")
        self._set_tag("WriteBandOut", "Analysis", value="Yes")

        if self.verbose:
            print("    self._set_tag('WriteEigenvectors', 'Analysis', value='Yes')")
        self._set_tag("WriteEigenvectors", "Analysis", value="Yes")

        if self.verbose:
            print("    self._set_tag('EigenvectorsAsText', 'Analysis', value='Yes')")
        self._set_tag("EigenvectorsAsText", "Analysis", value="Yes")

        if self.verbose:
            print("    self._set_tag('WriteDetailedXML', 'Options', value='Yes')")
        self._set_tag("WriteDetailedXML", "Options", value="Yes")

        if self.verbose:
            print("    self._set_tag('WriteDetailedOut', 'Options', value='Yes')")
        self._set_tag("WriteDetailedOut", "Options", value="Yes")
        print(self.finput)

        if output:
            import shutil

            output_path = Path(output)
            if output_path.is_file():
                shutil.copyfile(output_path, str(output_path) + ".bkp")
            with open(output_path, "w") as f:
                f.write(self.finput)


class DFTBParser(BaseParser):
    """The class parses the input form DFTB+.

    Parameters
    ----------

    dirname : str, optional
        Directory path to where calculation took place, by default ""
    eigenvec_filename : str, optional
        The (plain-text) eigenvectors, by default 'eigenvec.out'
    bands_filename : str, optional
        The file with the bands, by default 'band.out'
    detailed_out : str, optional
        The file with the Fermi energy, by default 'detailed.out'
    detailed_xml : str, optional
        The file with the list of kpoints, by default 'detailed.xml'
    """

    eigenvec_filename: Path
    bands_filename: Path
    detailed_out: Path
    detailed_xml: Path

    def __init__(
        self,
        dirpath: str | Path = "",
        eigenvec_filename: str = "eigenvec.out",
        bands_filename: str = "band.out",
        detailed_out: str = "detailed.out",
        detailed_xml: str = "detailed.xml",
    ) -> None:
        super().__init__(dirpath)
        eigenvec_filepath = Path(eigenvec_filename)
        bands_filepath = Path(bands_filename)
        detailed_out_filepath = Path(detailed_out)
        detailed_xml_filepath = Path(detailed_xml)

        self.eigenvec_filename = self.dirpath / eigenvec_filepath
        self.bands_filename = self.dirpath / bands_filepath
        self.detailed_out = self.dirpath / detailed_out_filepath
        self.detailed_xml = self.dirpath / detailed_xml_filepath

        # Searching for the Fermi level
        utils = DFTB_utils(verbose=False)

        # Loading the bands
        bands_data = DFTB_bands(filename=self.bands_filename, verbose=False)
        bands_data.load()

        # Loading the kpoints
        kpoints: npt.NDArray[np.float64]
        if self.detailed_xml.exists():
            kpoints = utils.get_kpoints(self.detailed_xml)
        else:
            # Fallback: create dummy kpoints (physically meaningless)
            Nkpoints = bands_data.Nkpoints
            kpoints = np.linspace(0, 1, num=Nkpoints).reshape(-1, 1)
            kpoints = np.hstack([kpoints, np.zeros((Nkpoints, 3))])  # Add weight column
            print(
                "A list of k-points was not provided. Creating a fake and meaningless set of K-points"
            )

        utils.writeOutcar(detailed_out=self.detailed_out, detailed_xml=self.detailed_xml)

        utils.writePoscar(detailed_xml=self.detailed_xml)

        # I need to find if I need to create the VASP files, just the
        # PROCAR, the other are trivial

        create_procar = True
        try:
            mtime_procar = os.path.getmtime("PROCAR")
            mtime_evec = os.path.getmtime(self.eigenvec_filename)
            if mtime_evec > mtime_procar:
                create_procar = True
            else:
                create_procar = False
        except OSError:
            # Handle file not found or other errors
            create_procar = True
        if create_procar is False:
            return

        print(
            "Going to create a PROCAR file from eigenvec.txt, it might take a"
            " while but will be done just once"
        )

        evec = DFTB_evec(filename=str(self.eigenvec_filename), verbose=False)
        evec.load()

        # setting the bands and kpoints to the class with eigenvectors
        if bands_data.Bands is not None and bands_data.Occupancies is not None:
            evec.set_bands(bands_data.Bands, bands_data.Occupancies)
        evec.set_kpoints(kpoints)

        evec.writeProcar()

    @property
    @override
    def ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure is not directly available from DFTB+ parser."""
        return None

    @property
    @override
    def dos(self) -> DensityOfStates | None:
        """Density of states is not available from DFTB+ parser."""
        return None

    @property
    @override
    def structure(self) -> Structure | None:
        """Crystal structure is not directly available from DFTB+ parser."""
        return None

    @property
    @override
    def kpath(self) -> KPath | None:
        """K-path is not available from DFTB+ parser."""
        return None
