import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure

from . import qe, vasp

# Physics Contstnats
HARTREE_TO_EV = 27.211386245988  # eV/Hartree


def str2bool(v):
    return v.lower() in ("true")


class LobsterParser:
    """The class helps parse the information from a lobster calculation

    Parameters
    ----------
    dirpath : str or Path, optional
        The directory name for the calculation, by default ""
    code : str, optional
        The code name, by default 'qe'
    lobsterin_filepath : str, optional
        The lobster in put filename, by default 'lobsterin'
    lobsterout_filepath : str, optional
        The lobster output filename, by default 'lobsterout'
    scfin_filepath : str, optional
        The scf.in file for QE calculations, by default "scf.in"
    outcar_filepath : str, optional
        The outcar for vasp calculations, by default 'OUTCAR'
    poscar_filepath : str, optional
        The poscar for vasp calculations, by default 'POSCAR'
    procar_filepath : str, optional
        The procar for vasp calculations, by default 'PROCAR'
    dos_interpolation_factor : _type_, optional
        The interpolation factor on the density of states, by default None

    """

    def __init__(
        self,
        dirpath: Union[str, Path] = "",
        code: str = "qe",
        lobsterin_filepath: Union[str, Path] = Path("lobsterin"),
        lobsterout_filepath: Union[str, Path] = Path("lobsterout"),
        scfin_filepath: Union[str, Path] = Path("scf.in"),
        outcar_filepath: Union[str, Path] = Path("OUTCAR"),
        poscar_filepath: Union[str, Path] = Path("POSCAR"),
        procar_filepath: Union[str, Path] = Path("PROCAR"),
        dos_interpolation_factor: int = None,
    ):

        # Convert dirpath to Path object
        self.dirpath = Path(dirpath)

        self.code = code

        # Use Path for file operations
        lobsterin_filepath = self.dirpath / lobsterin_filepath.name
        with open(lobsterin_filepath, "r") as rf:
            self.lobsterin = rf.read()

        lobsterout_filepath = self.dirpath / lobsterout_filepath.name
        with open(lobsterout_filepath, "r") as rf:
            self.lobsterout = rf.read()

        self.orbitals = [
            "s",
            "p_y",
            "p_z",
            "p_x",
            "d_xy",
            "d_yz",
            "d_z^2",
            "d_xz",
            "d_x^2-y^2",
        ]

        self.dos_interpolation_factor = dos_interpolation_factor

        self.scfin_filepath = scfin_filepath

        self.parse_structure(
            outcar_filepath=outcar_filepath,
            poscar_filepath=poscar_filepath,
            procar_filepath=procar_filepath,
            interpolation_factor=1,
            fermi=None,
        )

        self._getSpecialKpoints()

        # For band structures

        if len(self.kticks) != 0:
            self._readFileNames()
            self._readFatBands()
            self._createKPath()
            self.ebs = ElectronicBandStructure(
                kpoints=self.kpoints,
                bands=self.bands + self.efermi,
                projected=self._spd2projected(self.spd),
                efermi=self.efermi,
                kpath=self.kpath,
                projected_phase=None,
                labels=self.orbitals[:-1],
                reciprocal_lattice=self.reciprocal_lattice,
                interpolation_factor=dos_interpolation_factor,
                # shifted_to_efermi=True,
                # shifted_to_efermi=False,
            )

        doscar_path = self.dirpath / "DOSCAR.lobster"
        if doscar_path.exists():
            self.data = self._parse_doscar(doscar_path)

    @property
    def species(self):
        """Returns the species of the calculation

        Returns
        -------
        List
            Returns a list of string or atomic numbers[int]
        """
        return self.initial_structure.species

    @property
    def structures(self):
        """Returns a list of pyprocar.core.Structure

        Returns
        -------
        List
            Returns a list of pyprocar.core.Structure
        """
        # symbols = [x.strip() for x in self.data['ions']]
        symbols = [x.strip() for x in self.ions]
        structures = []

        st = Structure(
            atoms=symbols,
            lattice=self.direct_lattice,
            fractional_coordinates=self.atomic_positions,
        )

        structures.append(st)
        return structures

    @property
    def structure(self):
        """Returns a the last element of a list of pyprocar.core.Structure

        Returns
        -------
        pyprocar.core.Structure
            Returns a the last element of a list of pyprocar.core.Structure
        """
        return self.structures[-1]

    @property
    def initial_structure(self):
        """Returns a the first element of a list of pyprocar.core.Structure

        Returns
        -------
        pyprocar.core.Structure
            Returns a the first element of a list of pyprocar.core.Structure
        """
        return self.structures[0]

    @property
    def final_structure(self):
        """Returns a the last element of a list of pyprocar.core.Structure

        Returns
        -------
        pyprocar.core.Structure
            Returns a the last element of a list of pyprocar.core.Structure
        """

        return self.structures[-1]

    @property
    def dos(self):
        """Initializes the pyprocar.core.DensityOfStates

        Returns
        -------
        pyprocar.core.DensityOfStates
            Retruns the pyprocar.core.DensityOfStates object
        """
        energies = self.dos_total["energies"]
        total = []
        for ispin in self.dos_total:
            if ispin == "energies":
                continue
            total.append(self.dos_total[ispin])
        # total = np.array(total).T
        return DensityOfStates(
            energies=energies,
            total=total,
            projected=self.dos_projected,
            interpolation_factor=self.dos_interpolation_factor,
        )

    """
        Returns the complete density (total,projected) of states as a python dictionary
        """

    @property
    def dos_to_dict(self):
        """Returns the complete density (total,projected) of states as a python dictionary

        Returns
        -------
        dict
            Returns a dictionary with the projected and total denisity of states
        """

        return {"total": self._get_dos_total(), "projected": self._get_dos_projected()}

    @property
    def dos_total(self):
        """Returns the density of states total as a dict.
        The keys are energies,Spin-up, Spin-down

        Returns
        -------
        dict
            Returns the density of states total as a dict
        """

        dos_total, labels = self._get_dos_total()
        # dos_total['energies'] -= self.fermi

        return dos_total

    @property
    def dos_projected(self):
        """Returns the projected DOS as a multi-dimentional array,
        to be used in the pyprocar.core.DensityOfStates object

        Returns
        -------
        list
            A list of projections per atom
        """

        ret = []
        dos_projected, info = self._get_dos_projected()
        if dos_projected is None:
            return None
        norbitals = len(info) - 1
        info[0] = info[0].capitalize()
        labels = []
        labels.append(info[0])
        ret = []
        for iatom in dos_projected:
            temp_atom = []
            for iorbital in range(norbitals):
                temp_spin = []
                for key in dos_projected[iatom]:
                    if key == "energies":
                        continue
                    temp_spin.append(dos_projected[iatom][key][:, iorbital])
                temp_atom.append(temp_spin)
            ret.append([temp_atom])
        return ret

    def _get_dos_total(self):
        """Helper method to get the total density of state

        Returns
        -------
        dict, list
            Returns the total_dos dict and label names
        """

        energies = self.data["total"][:, 0]
        dos_total = {"energies": energies}

        if self.nspin != 1:
            dos_total["Spin-up"] = self.data["total"][:, 1]
            dos_total["Spin-down"] = self.data["total"][:, 2]
            # dos_total['integrated_dos_up'] = self.data['total'][:, 3]
            # dos_total['integrated_dos_down'] = self.data['total'][:, 4]
        else:
            dos_total["Spin-up"] = self.data["total"][:, 1]
            # dos_total['integrated_dos'] = self.data['total'][:, 2]

        return dos_total, list(dos_total.keys())

    def _get_dos_projected(self, atoms=[]):
        """
        Helper method to get the total density of state

        Parameters
        -------
        atoms: list, optional
            Atoms to to iterate through
        Returns
        -------
        list, list
            Returns the porjected dos istt and label names
        """
        if len(atoms) == 0:
            atoms = np.arange(self.initial_structure.natoms)

        if "projected" in list(self.data.keys()):
            dos_projected = {}
            ion_list = [
                "ion %s" % str(x + 1) for x in atoms
            ]  # using this name as vasrun.xml uses ion #
            for i in range(len(ion_list)):
                iatom = ion_list[i]
                name = self.initial_structure.atoms[atoms[i]]

                energies = self.data["projected"][i][:, 0, 0]

                dos_projected[name] = {"energies": energies}
                if self.nspin != 1:
                    dos_projected[name]["Spin-up"] = self.data["projected"][i][:, 1:, 0]
                    dos_projected[name]["Spin-down"] = self.data["projected"][i][
                        :, 1:, 1
                    ]
                else:
                    dos_projected[name]["Spin-up"] = self.data["projected"][i][:, 1:, 0]

            return dos_projected, self.data["projected_labels_info"]
        else:
            print("This calculation does not include partial density of states")
            return None, None

    def dos_parametric(self, atoms=None, orbitals=None, spin=None, title=None):
        """This function sums over the list of atoms and orbitals given
        for example dos_paramateric(atoms=[0,1,2],orbitals=[1,2,3],spin=[0,1])
        will sum all the projections of atoms 0,1,2 and all the orbitals of 1,2,3 (px,py,pz)
        and return separatly for the 2 spins as a DensityOfStates object from pychemia.visual.DensityofStates

        Parameters
        ----------
        atoms : list[int], optional
            list of atom index needed to be sumed over., by default None
        orbitals : list[int], optional
            list of orbitals needed to be sumed over
        |  s  ||  py ||  pz ||  px || dxy || dyz || dz2 || dxz ||x2-y2||
        |  0  ||  1  ||  2  ||  3  ||  4  ||  5  ||  6  ||  7  ||  8  ||, by default None
        spin : list[int], optional
            which spins to be included, by default None
        title : str, optional
            Title, by default None

        Returns
        -------
        pyprocar.core.DensityOfStates
            Returns the dos object
        """
        projected = self.dos_projected
        dos_projected, labelsInfo = self._get_dos_projected()
        self.availiableOrbitals = list(labelsInfo.keys())
        self.availiableOrbitals.pop(0)
        if atoms == None:
            atoms = np.arange(self.nions, dtype=int)
        if spin == None:
            spin = [0, 1]
        if orbitals == None:
            orbitals = np.arange((len(projected[0].labels) - 1) // 2, dtype=int)
        if title == None:
            title = "Sum"
        orbitals = np.array(orbitals)

        if len(spin) == 2:
            labels = ["Energy", "Spin-Up", "Spin-Down"]
            new_orbitals = []
            for ispin in spin:
                new_orbitals.append(
                    list(orbitals + ispin * (len(projected[0].labels) - 1) // 2)
                )

            orbitals = new_orbitals

        else:

            for x in orbitals:

                if x + 1 > (len(projected[0].labels) - 1) // 2:
                    print("listed wrong amount of orbitals")
                    print(
                        "Only use one or more of the following "
                        + str(np.arange((len(projected[0].labels) - 1) // 2, dtype=int))
                    )
                    print(
                        "Only use one or more of the following "
                        + str(np.arange((len(projected[0].labels) - 1) // 2, dtype=int))
                    )
                    print(
                        "They correspond to the following orbitals : "
                        + str(self.availiableOrbitals)
                    )
                    print("Again do not trust the plot that was just produced")
            if spin[0] == 0:
                labels = ["Energy", "Spin-Up"]
            elif spin[0] == 1:
                labels = ["Energy", "Spin-Down"]

        ret = np.zeros(shape=(len(projected[0].energies), len(spin) + 1))
        ret[:, 0] = projected[0].energies

        for iatom in atoms:
            if len(spin) == 2:
                ret[:, 1:] += self.dos_projected[iatom].values[:, orbitals].sum(axis=2)
            elif len(spin) == 1:
                ret[:, 1] += self.dos_projected[iatom].values[:, orbitals].sum(axis=1)

        return DensityOfStates(table=ret, title=title, labels=labels)

    def _parse_doscar(self, filename):
        """Helper method to parse the DOSCAR

        Parameters
        ----------
        filename : str or Path
            The filename of the DOSCAR

        Returns
        -------
        dict
            Returns a dict with the total, projected, and labels

        Raises
        ------
        ValueError
            DOSCAR seems truncated
        """
        with open(filename, "r") as rf:
            data = rf.readlines()

        if len(data) < 5:
            raise ValueError("DOSCAR seems truncated")

        # Skipping the first lines of header
        iline = 5

        header = [float(x) for x in data[iline].strip().split()]
        ndos = int(header[2])
        iline += 1

        total_dos = [[float(x) for x in y.split()] for y in data[iline : iline + ndos]]
        total_dos = np.array(total_dos)

        iline += ndos
        ndos, total_ncols = total_dos.shape
        is_spin_polarized = False
        if total_ncols == 5:
            is_spin_polarized = True
            spins = ["dos_up", "dos_down"]
        # In case there are more lines of data, they are the projected DOS
        if len(data) > iline:

            projected_dos = []
            proj_orbitals = []
            ion_index = 0

            while iline < len(data):

                header = [float(x) for x in data[iline].split(";")[0].split()]

                # print(header)
                ionsList = re.findall(
                    "calculating FatBand for Element: (.*) Orbital.*", self.lobsterout
                )
                proj_ions = re.findall(
                    "calculating FatBand for Element: (.*) Orbital\(s\):\s*.*",
                    self.lobsterout,
                )

                proj_orbitals.append((proj_ions[ion_index], data[iline].split(";")[2]))
                # print(proj_orbitals)
                ion_index += 1

                ndos = int(header[2])
                iline += 1
                tmp_dos = [
                    [float(x) for x in y.split()] for y in data[iline : iline + ndos]
                ]
                tmp_dos = np.array(tmp_dos)

                projected_dos.append(tmp_dos)

                iline += ndos

            final_projected = []

            for i_ion in range(len(projected_dos)):
                tmp_dos = np.zeros(shape=[len(projected_dos[i_ion][:, 0]), 10, 2])
                for ilabel, label in enumerate(proj_orbitals[i_ion][1].split(), 1):
                    if is_spin_polarized == False:
                        tmp_dos[:, 0, 0] = projected_dos[i_ion][:, 0]
                        if label.find("s") == True:
                            tmp_dos[:, 1, 0] += projected_dos[i_ion][:, ilabel]
                        elif label.find("p_y") == True:
                            tmp_dos[:, 2, 0] += projected_dos[i_ion][:, ilabel]
                        elif label.find("p_z") == True:
                            tmp_dos[:, 3, 0] += projected_dos[i_ion][:, ilabel]
                        elif label.find("p_x") == True:
                            tmp_dos[:, 4, 0] += projected_dos[i_ion][:, ilabel]
                        elif label.find("d_xy") == True:
                            tmp_dos[:, 5, 0] += projected_dos[i_ion][:, ilabel]
                        elif label.find("d_yz") == True:
                            tmp_dos[:, 6, 0] += projected_dos[i_ion][:, ilabel]
                        elif label.find("d_z^2") == True:
                            tmp_dos[:, 7, 0] += projected_dos[i_ion][:, ilabel]
                        elif label.find("d_xz") == True:
                            tmp_dos[:, 8, 0] += projected_dos[i_ion][:, ilabel]
                        elif label.find("d_x^2-y^2") == True:
                            tmp_dos[:, 9, 0] += projected_dos[i_ion][:, ilabel]
                    else:
                        tmp_dos[:, 0, 0] = projected_dos[i_ion][:, 0]
                        tmp_dos[:, 0, 1] = projected_dos[i_ion][:, 0]
                        if label.find("s") == True:
                            tmp_dos[:, 1, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 1, 1] += projected_dos[i_ion][:, 2 * ilabel]
                        elif label.find("p_y") == True:
                            tmp_dos[:, 2, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 2, 1] += projected_dos[i_ion][:, 2 * ilabel]
                        elif label.find("p_z") == True:
                            tmp_dos[:, 3, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 3, 1] += projected_dos[i_ion][:, 2 * ilabel]
                        elif label.find("p_x") == True:
                            tmp_dos[:, 4, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 4, 1] += projected_dos[i_ion][:, 2 * ilabel]
                        elif label.find("d_xy") == True:
                            tmp_dos[:, 5, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 5, 1] += projected_dos[i_ion][:, 2 * ilabel]
                        elif label.find("d_yz") == True:
                            tmp_dos[:, 6, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 6, 1] += projected_dos[i_ion][:, 2 * ilabel]
                        elif label.find("d_z^2") == True:
                            tmp_dos[:, 7, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 7, 1] += projected_dos[i_ion][:, 2 * ilabel]
                        elif label.find("d_xz") == True:
                            tmp_dos[:, 8, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 8, 1] += projected_dos[i_ion][:, 2 * ilabel]
                        elif label.find("d_x^2-y^2") == True:
                            tmp_dos[:, 9, 0] += projected_dos[i_ion][:, 2 * ilabel - 1]
                            tmp_dos[:, 9, 1] += projected_dos[i_ion][:, 2 * ilabel]
                final_projected.append(tmp_dos)
                final_labels_index = {
                    "energies": None,
                    "s": 0,
                    "p_y": 1,
                    "p_z": 2,
                    "p_x": 3,
                    "d_xy": 4,
                    "d_yz": 5,
                    "d_z^2": 6,
                    "d_xz": 7,
                    "d_x^2-y^2": 8,
                }
                final_labels = list(final_labels_index.keys())
            return {
                "total": total_dos,
                "projected": final_projected,
                "projected_labels_info": final_labels,
                "ions": ionsList,
            }

        else:

            return {"total": total_dos}

    def _readFileNames(self):
        """Helper method to parse filenames form the lobster outfile

        Returns
        -------
        None
            None
        """
        self.filepaths = []
        self.ionsList = re.findall(
            "calculating FatBand for Element: (.*) Orbital.*", self.lobsterout
        )
        proj_orbitals = re.findall(
            "calculating FatBand for Element: (.*) Orbital\(s\):\s*(.*)",
            self.lobsterout,
        )
        for proj in proj_orbitals:
            orbitals = proj[1].split()
            for orbital in orbitals:
                filepath = self.dirpath / f"FATBAND_{proj[0]}_{orbital}.lobster"
                self.filepaths.append(filepath)
        return None

    def _readFatBands(self):
        """Helpermethod to parse the fatbands

        Returns
        -------
        None
            None
        """
        with open(self.filepaths[0], "r") as rf:
            projFile = rf.read()

        ##########################################################################################
        # kpoints
        ##########################################################################################
        raw_kpoints = re.findall(
            "# K-Point \d+ :\s*([-\.\\d]*)\s*([-\.\\d]*)\s*([-\.\\d]*)", projFile
        )
        self.kpointsCount = len(raw_kpoints)
        self.kpoints = np.zeros(shape=(self.kpointsCount, 3))
        for ik in range(len(raw_kpoints)):
            for coord in range(3):
                self.kpoints[ik][coord] = raw_kpoints[ik][coord]

        self.bandsCount = int(re.findall("NBANDS (\d*)", projFile)[0])
        self.orbitalCount = 10
        self.spd = np.zeros(
            shape=(
                self.kpointsCount,
                self.bandsCount,
                self.nspin,
                self.ionsCount + 1,
                len(self.orbitals) + 2,
            )
        )

        self.bands = np.zeros(shape=(self.kpointsCount, self.bandsCount, self.nspin))
        for file in range(len(self.filepaths)):
            with open(self.filepaths[file], "r") as rf:
                projFile = rf.read()

            fatbands_info = re.findall("#\sFATBAND\sfor(.*)", projFile)[0].split()
            current_ion = fatbands_info[0]
            current_orbital = fatbands_info[1][1:]

            iion = 0
            iorbital = 0
            for i in range(len(self.ionsList)):
                if self.ionsList[i] == current_ion:
                    iion = i
            for i in range(len(self.orbitals)):
                if self.orbitals[i] == current_orbital:
                    iorbital = i + 1

            fatbands = re.split("# K-Point", projFile)[1:]
            for ik, fatband in enumerate(fatbands[:]):

                for iband, band in enumerate(fatband.split("\n")[1:-1]):
                    if self.nspin == 2:
                        if iband < self.bandsCount:
                            self.bands[ik, iband, 0] = float(band.split()[1])
                            self.bands[ik, iband, 1] = float(
                                fatband.split("\n")[1:][
                                    iband + self.bandsCount
                                ].split()[1]
                            )

                            self.spd[ik, iband, 0, iion, iorbital] = float(
                                band.split()[2]
                            )
                            self.spd[ik, iband, 1, iion, iorbital] = float(
                                fatband.split("\n")[1:-1][
                                    iband + self.bandsCount
                                ].split()[2]
                            )

                    else:
                        self.bands[ik, iband, 0] = float(band.split()[1])
                        self.spd[ik, iband, 0, iion, iorbital] = float(band.split()[2])
            self.spd[:, :, :, :, -1] = np.sum(self.spd[:, :, :, :, 1:-1], axis=4)
            self.spd[:, :, :, -1, :] = np.sum(self.spd[:, :, :, 0:-1, :], axis=3)
            self.spd[:, :, :, -1, 0] = 0

        return None

    def _spd2projected(self, spd, nprinciples=1):
        """A helper method to conver spd format to projected format
        to be inputed in to pyprocar.core.ElectronicBandStructure

        Parameters
        ----------
        spd : np.ndarray
            The spd array
        nprinciples : int, optional
            The principle quantum number, by default 1

        Returns
        -------
        np.ndarray
            The projected array
        """
        # This function is for VASP
        # non-pol and colinear
        # spd is formed as (nkpoints,nbands, nspin, natom+1, norbital+2)
        # natom+1 > last column is total
        # norbital+2 > 1st column is the number of atom last is total
        # non-colinear
        # spd is formed as (nkpoints,nbands, nspin +1 , natom+1, norbital+2)
        # natom+1 > last column is total
        # norbital+2 > 1st column is the number of atom last is total
        # nspin +1 > last column is total
        if spd is None:
            return None
        natoms = spd.shape[3] - 1
        nkpoints = spd.shape[0]

        nbands = spd.shape[1]
        nspins = spd.shape[2]

        norbitals = spd.shape[4] - 2
        # if spd.shape[2] == 4:
        #     nspins = 3
        # else:
        #     nspins = spd.shape[2]
        # if nspins == 2:
        #     nbands = int(spd.shape[1] / 2)
        # else:
        #     nbands = spd.shape[1]
        projected = np.zeros(
            shape=(nkpoints, nbands, natoms, nprinciples, norbitals, nspins),
            dtype=spd.dtype,
        )
        temp_spd = spd.copy()
        # (nkpoints,nbands, nspin, natom, norbital)
        temp_spd = np.swapaxes(temp_spd, 2, 4)
        # (nkpoints,nbands, norbital , natom , nspin)
        temp_spd = np.swapaxes(temp_spd, 2, 3)
        # (nkpoints,nbands, natom, norbital, nspin)
        # projected[ikpoint][iband][iatom][iprincipal][iorbital][ispin]
        # if nspins == 3:
        #     projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :-1]
        # elif nspins == 2:
        #     projected[:, :, :, 0, :, 0] = temp_spd[:, :nbands, :-1, 1:-1, 0]
        #     projected[:, :, :, 0, :, 1] = temp_spd[:, nbands:, :-1, 1:-1, 0]
        # else:
        projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]
        return projected

    def _createKPath(self):
        """Helper method to format thwe kpath information into
         the pyprocar.core.KPath

        Returns
        -------
        None
            None
        """
        self.special_kpoints = np.zeros(shape=(len(self.kticks) - 1, 2, 3))
        self.modified_knames = []
        for itick in range(len(self.kticks)):
            if itick != len(self.kticks) - 1:
                self.special_kpoints[itick, 0, :] = self.kpoints[self.kticks[itick]]
                self.special_kpoints[itick, 1, :] = self.kpoints[self.kticks[itick + 1]]
                self.modified_knames.append(
                    [self.knames[itick], self.knames[itick + 1]]
                )

        has_time_reversal = True

        self.kpath = KPath(
            knames=self.modified_knames,
            special_kpoints=self.special_kpoints,
            kticks=self.kticks,
            ngrids=self.ngrids,
            has_time_reversal=has_time_reversal,
        )

        return None

    def _getSpecialKpoints(self):
        """Helper method to get the special kpoints for a given code

        Returns
        -------
        None
            None
        """

        if self.code == "qe":
            numK = int(re.findall("K_POINTS.*\n([0-9]*)", self.scfIn)[0])

            raw_kpoints = re.findall(
                "K_POINTS.*\n\s*[0-9]*.*\n" + numK * "(.*)\n",
                self.scfIn,
            )[0]

            raw_high_symmetry = []
            self.knames = []
            self.kticks = []
            tickCountIndex = 0
            for x in raw_kpoints:
                if len(x.split()) == 5:

                    raw_high_symmetry.append(
                        (float(x.split()[0]), float(x.split()[1]), float(x.split()[2]))
                    )
                    self.knames.append("%s" % x.split()[4].replace("!", ""))
                    self.kticks.append(tickCountIndex)
                if float(x.split()[3]) == 0:
                    tickCountIndex += 1
            self.high_symmetry_points = np.array(raw_high_symmetry)
            self.nhigh_sym = len(self.knames)

            self.ngrids = []
            tick_Count = 1
            for ihs in range(self.nhigh_sym):
                # self.kticks.append(tick_Count - 1)
                self.ngrids.append(int(float(raw_kpoints[ihs].split()[3])))
                tick_Count += int(float(raw_kpoints[ihs].split()[3]))
        else:
            self.knames = []
            self.kticks = []
            self.high_symmetry_points = []
            self.nhigh_sym = []
            self.ngrids = []

        return None

    def parse_structure(
        self,
        outcar_filepath: str = None,
        poscar_filepath: str = None,
        procar_filepath: str = None,
        reciprocal_lattice: np.ndarray = None,
        kpoints: str = None,
        interpolation_factor: int = 1,
        fermi: float = None,
    ):
        """A method to parse the electronic sturcutue information

        Parameters
        ----------
        outcar_filepath : str, optional
            The OUTCAR filename, by default None
        poscar_filepath : str, optional
            The POSCAR filename, by default None
        procar_filepath : str, optional
            The POSCAR filename, by default None
        reciprocal_lattice : np.ndarray, optional
            The reciprocal lattice matrix, by default None
        kpoints : str, optional
            The KPOINTS file name, by default None
        interpolation_factor : int, optional
            The interpolation factor, by default 1
        fermi : float, optional
            The fermi energy, by default None

        Returns
        -------
        None
            None
        """

        if self.code == "vasp":
            if outcar_filepath is not None:
                outcar = vasp.Outcar(outcar_filepath)
                if fermi is None:
                    fermi = outcar.efermi
                reciprocal_lattice = outcar.reciprocal_lattice
            if poscar is not None:
                poscar = vasp.Poscar(poscar)
                structure = poscar.structure
                if reciprocal_lattice is None:
                    reciprocal_lattice = poscar.structure.reciprocal_lattice

            if kpoints is not None:
                kpoints = vasp.Kpoints(kpoints)
                kpath = kpoints.kpath

            procar = vasp.Procar(
                procar,
                structure,
                reciprocal_lattice,
                kpath,
                fermi,
                interpolation_factor=interpolation_factor,
            )
            ebs = procar.ebs

        elif self.code == "qe":
            scfin_filepath = self.dirpath / self.scfin_filepath.name
            with open(scfin_filepath, "r") as rf:
                self.scfIn = rf.read()

            if self.dirpath is None:
                self.dirpath = Path("bands")

            parser = qe.QEParser(
                scfIn_filename="scf.in",
                dirpath=str(
                    self.dirpath
                ),  # Convert to string for QEParser compatibility
                dos_interpolation_factor=None,
            )
            self.prefix = parser.prefix
            self.non_colinear = parser.non_colinear
            self.spinCalc = parser.spinCalc
            self.spin_orbit = parser.spin_orbit
            self.nspin = parser.nspin
            self.bandsCount = parser.bandsCount

            self.atomic_positions = parser.atomic_positions
            self.direct_lattice = parser.direct_lattice
            self.species_list = parser.nspecies
            self.species_list = parser.species_list
            self.ionsCount = parser.ionsCount
            self.ions = parser.ions
            self.alat = parser.alat
            self.composition = parser.composition
            self.efermi = parser.efermi
            self.reciprocal_lattice = parser.reciprocal_lattice

        return None
