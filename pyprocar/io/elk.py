__author__ = "Pedram Tavadze"
__maintainer__ = "Pedram Tavadze"
__email__ = "petavazohi@mix.wvu.edu"
__date__ = "March 29, 2023"

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy.typing as npt
import numpy as np


from ..core.dos import DensityOfStates
from ..core.ebs import ElectronicBandStructure
from ..core.kpath import KPath
from ..core.structure import Structure
from .. import utils
HARTREE_TO_EV = 27.211386245988
FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"
float_re = re.compile(FLOAT)


def parse_dos_block(dos_block: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the DOS block from elk output file.

    Parameters
    ----------
    dos_block : str
        input string containing the DOS block

    Returns
    -------
    Tuple[np.array, np.array]
        (energies, dos)
    """
    X = np.array([x.split() for x in dos_block.splitlines()]).astype(float)
    if len(X) == 0:
        return None, None
    return X[:, 0], X[:, 1]


class ElkParser:

    def __init__(self, dir_path: Union[str, Path], kdirect: bool = True):
        # elk specific input parameters
        self.dir_path = Path(dir_path)
        self.elkin_filepath = self.dir_path / "elk.in"
        self.filepaths = []
        self.elkin = None
        self.nbands = None

        self.nspin = None
        self.kticks = None

        self.kdirect = kdirect

        self.kpoints = None
        self.bands = None

        self.spd = None
        self.cspd = None

        # spin polarized parameters
        self.spinpol = None

        self.is_bands_calculation = False

        self.orbital_names = [
            "Y00",
            "Y1-1",
            "Y10",
            "Y11",
            "Y2-2",
            "Y2-1",
            "Y20",
            "Y21",
            "Y22",
            "Y3-3",
            "Y3-2",
            "Y3-1",
            "Y30",
            "Y3-1",
            "Y3-2",
            "Y3-3",
        ]
        self.orbital_name_short = ["s", "p", "d", "f"]
        self.norbital = None  # number of orbitals

        # number of spin components (blocks of data), 1: non-magnetic non
        # polarized, 2: spin polarized collinear, 4: non-collinear
        # spin.
        # NOTE: before calling to `self._readOrbital` the case '4'
        # is marked as '1'

        # rf = open(self.elkin_filepath, "r")
        # self.elkin = rf.read()
        # rf.close()

        # self._read_lattice_out()
        self._read_elkin()
        self._read_fermi()

        if self.is_bands_calculation:
            self._read_kpoints_info()
            self._read_bands()
        self._read_dos()

        has_time_reversal = True

        # Checks if filepaths exists, if not it is not a bands calculation
        if self.is_bands_calculation:
            self.kpath = KPath(
                knames=self.knames,
                special_kpoints=self.special_kpoints,
                kticks=self.kticks,
                ngrids=self.ngrids,
                has_time_reversal=has_time_reversal,
            )

            self.ebs = ElectronicBandStructure(
                kpoints=self.kpoints,
                bands=self.bands,
                projected=self._spd2projected(self.spd),
                efermi=self.fermi,
                kpath=self.kpath,
                projected_phase=None,
                labels=self.orbital_names[:-1],
                reciprocal_lattice=self.reclat,
            )

        return

    @property
    def spd_orb(self):
        # indices: ikpt, iband, ispin, iion, iorb
        # remove indices and total from iorb.
        return self.spd[:, :, :, 1:-1]

    def _read_kpoints_info(self):

        self.nkpoints = int(re.findall(
            r"plot1d\n\s*[0-9]*\s*([0-9]*)", self.elkin)[0])

        raw_ticks = re.findall(
            r"plot1d\n\s*([0-9]*)\s*([0-9]*)", self.elkin)[0]
        self.nhigh_sym = int(raw_ticks[0])
        n_segments = self.nhigh_sym - 1
        grid_points = int(raw_ticks[1]) / n_segments
        self.ngrids = [grid_points for x in range(self.nhigh_sym)]

        raw_ticks = re.findall(
            r"plot1d\n\s*[0-9]*\s*[0-9]*.*\n" +
            self.nhigh_sym * ".*:(.*)\n", self.elkin
        )[0]
        if len(raw_ticks) != self.nhigh_sym:
            knames = [str(x) for x in range(self.nhigh_sym)]
        else:
            knames = [
                "$%s$" % (x.replace(",", "").replace(
                    "vlvp1d", "").replace(" ", ""))
                for x in re.findall(
                    r"plot1d\n\s*[0-9]*\s*[0-9]*.*\n" +
                    self.nhigh_sym * ".*:(.*)\n",
                    self.elkin,
                )[0]
            ]
        self.high_symmetry_points = np.zeros(shape=(self.nhigh_sym, 3))
        raw_high_symmetry_points = re.findall(
            r"plot1d.*\n.*\n\s* " + self.nhigh_sym * "(.*)\n*", self.elkin
        )[0]
        for i, raw_kpoint in enumerate(raw_high_symmetry_points):
            kpoint = [float(kpoint) for kpoint in raw_kpoint.split()[:3]]
            self.high_symmetry_points[i, :] = kpoint

        self.special_kpoints = np.zeros(shape=(self.nhigh_sym - 1, 2, 3))
        self.knames = []
        for i in range(self.nhigh_sym - 1):
            self.special_kpoints[i, 0, :] = self.high_symmetry_points[i, :]
            self.special_kpoints[i, 1, :] = self.high_symmetry_points[i + 1, :]

            self.knames.append([knames[i], knames[i + 1]])
        return None

    def _read_elkin(self):
        """
        Reads and parses elk.in
        """

        path = self.dir_path / "elk.in"
        file_content = path.read_text()
        pattern_task = re.compile(
            r"(?m)^[ \t]*tasks[ \t]*\n"       # “tasks” line
            r"((?:[ \t]*\d+[ \t]*\n)+)"       # one or more lines of digits
        )
        match = pattern_task.search(file_content)
        if not match:
            raise ValueError("No 'tasks' block found in elk.in")
        self.tasks = [int(n) for n in match.group(1).split()]

        if 20 in self.tasks:
            self.is_bands_calculation = True
            self.filepaths.append(self.dir_path / "BANDS.OUT")
        if 21 in self.tasks or 22 in self.tasks:
            self.is_bands_calculation = True
            ispc = 1
            for spc in self.composition:
                for iatom in range(self.composition[spc]):
                    self.filepaths.append(
                        self.dir_path /
                        f"BAND_S{ispc:02d}_A{iatom + 1:04d}.OUT"
                    )
                ispc += 1

        self.spinpol = re.findall(r"spinpol\s*([.a-zA-Z]*)", file_content)
        if len(self.spinpol) != 0:
            self.spinpol = utils.strings.bool_fortran(self.spinpol[0])
        else:
            self.spinpol = False
        if self.spinpol:
            self.nspin = 2
        else:
            self.nspin = 1

        if (self.dir_path/'GEOMETRY.OUT').exists():
            lattice = self._read_geometry_out()

    def _read_bands(self):
        """
        if the task is any of 20,21,22 it parses the BANDS*.OUT files
        and prepares the spd, bands and kpoints for bandsplot
        """
        if np.any([x in self.tasks for x in [20, 21, 22]]):

            rf = open(self.filepaths[0], "r")
            lines = rf.readlines()
            rf.close()

            raw_nbands = int(len(lines) / (self.nkpoints + 1))

            rf = open(self.dir_path / "BANDLINES.OUT", "r")
            bandLines = rf.readlines()
            rf.close()

            tick_pos = []
            # using strings for a better comparision and avoiding rounding by python
            for iline in range(0, len(bandLines), 3):
                tick_pos.append(bandLines[iline].split()[0])
            x_points = []
            for iline in range(self.nkpoints):
                x_points.append(lines[iline].split()[0])

            self.kpoints = np.zeros(shape=(self.nkpoints, 3))
            x_points = np.array(x_points)
            tick_pos = np.array(tick_pos)

            self.kticks = []
            for ihs in range(1, self.nhigh_sym):
                start = np.where(x_points == tick_pos[ihs - 1])[0][0]
                end = np.where(x_points == tick_pos[ihs])[0][0] + 1
                self.kpoints[start:end][:] = np.linspace(
                    self.high_symmetry_points[ihs - 1],
                    self.high_symmetry_points[ihs],
                    end - start,
                )

                self.kticks.append(start)
            self.kticks.append(self.nkpoints - 1)

            self.ngrids = np.diff(np.array(self.kticks))
            self.ngrids[-1] += 1

            if not self.kdirect:
                self.kpoints = np.dot(self.kpoints, self.reclat)

            rf = open(self.filepaths[0], "r")
            lines = rf.readlines()
            rf.close()

            iline = 0
            raw_bands = np.zeros(shape=(self.nkpoints, raw_nbands))
            for iband in range(raw_nbands):
                for ikpoint in range(self.nkpoints):
                    raw_bands[ikpoint, iband] = float(lines[iline].split()[1])
                    iline += 1
                if ikpoint == self.nkpoints - 1:
                    iline += 1
            raw_bands *= HARTREE_TO_EV

            if self.nspin == 1:
                self.nbands = raw_nbands
            elif self.nspin == 2:
                self.nbands = raw_nbands // 2

            self.bands = np.zeros(
                shape=(self.nkpoints, self.nbands, self.nspin))
            if self.nspin == 1:
                self.bands[:, :, 0] = raw_bands
            elif self.nspin == 2:
                self.bands[:, :, 0] = raw_bands[:, : self.nbands]
                self.bands[:, :, 1] = raw_bands[:, self.nbands:]
            self.bands += self.fermi

            self.norbital = 16
            self.spd = np.zeros(
                shape=(
                    self.nkpoints,
                    self.nbands,
                    self.nspin,
                    self.natom + 1,
                    self.norbital + 2,
                )
            )
            idx_bands_out = None
            for ifile in range(len(self.filepaths)):
                if self.filepaths[ifile] == self.dir_path / "BANDS.OUT":
                    idx_bands_out = ifile
            if idx_bands_out != None:
                del self.filepaths[idx_bands_out]

            for ifile in range(self.natom):
                rf = open(self.filepaths[ifile], "r")
                lines = rf.readlines()
                rf.close()
                iline = 0

                for iband in range(self.nbands):
                    for ikpoint in range(self.nkpoints):
                        temp = np.array([float(x)
                                        for x in lines[iline].split()])
                        self.spd[ikpoint, iband, 0, ifile, 0] = ifile + 1
                        self.spd[ikpoint, iband, 0, ifile, 1:-1] = temp[2:]
                        iline += 1
                    if ikpoint == self.nkpoints - 1:
                        iline += 1
            # self.spd[:,:,:,-1,:] = self.spd.sum(axis=3)
            self.spd[:, :, :, :, -
                     1] = np.sum(self.spd[:, :, :, :, 1:-1], axis=4)
            self.spd[:, :, :, -1, :] = self.spd.sum(axis=3)
            self.spd[:, :, 0, -1, 0] = 0

            if self.nspin == 2:
                # spin up block for spin = 1
                self.spd[:, : self.nbands // 2, 1, :, :] = self.spd[
                    :, : self.nbands // 2, 0, :, :
                ]
                # spin down block for spin = 1
                self.spd[:, self.nbands // 2:, 1, :, :] = (
                    -1 * self.spd[:, self.nbands // 2:, 0, :, :]
                )

                # manipulating spd array for spin polarized calculations.
                # The shape is (nkpoints,2*nbands,2,natoms,norbitals)
                # The third dimension is for spin.
                # When this is zero, the bands*2 (all spin up and down bands) have positive projections.
                # When this is one, the the first half of bands (spin up) will have positive projections
                # and the second half (spin down) will have negative projections. This is to adhere to
                # the convention used in PyProcar to obtain spin density and spin magnetization.

                # spin up and spin down block for spin = 0
                # spd2[:, :, 0, :, :] = self.spd[:, :, 0, :, :]

    def _read_dos(self):
        """Read the DOS from elk output files. TDOS.OUT and PDOS_*.OUT

        Parameters
        ----------
        path : Union[str, Path]
            Path to the elk calculation.

        Returns
        -------
        DesityOfStates
            Returns a DensityOfStates object from pyprocar.core.dos
        """
        if not os.path.exists(self.dir_path / "TDOS.OUT"):

            return None
        tdos = []
        pdos = {}
        n_atoms = 0
        for i_file in self.dir_path.iterdir():
            if i_file.name == "TDOS.OUT":
                with open(i_file, "r") as f:
                    data = f.read()
                blocks = re.split(r"\n\s*\n", data)
                for i, block in enumerate(blocks):
                    energies, dos_total = parse_dos_block(block)
                    if dos_total is not None:
                        tdos.append(dos_total)
                        edos = energies

            if "PDOS" in i_file.stem:
                spc, atm = i_file.stem.split("_")[1:]
                if spc not in pdos:
                    pdos[spc] = {}
                if atm not in pdos[spc]:
                    pdos[spc][atm] = {}
                with open(i_file, "r") as f:
                    data = f.read()
                # parse all the blocks. Should be 16 * self.nspins.
                # Remove last element as it is empty string
                blocks = re.split(r"\n\s*\n", data)[:-1]
                n_atoms += 1
                for i, block in enumerate(blocks):
                    _, dos_projected = parse_dos_block(block)
                    if dos_projected is not None:
                        pdos[spc][atm][i] = dos_projected

        energies = edos * HARTREE_TO_EV
        n_spins = len(tdos)

        dos_total = np.zeros((n_spins, len(energies)))
        for i_spin in range(n_spins):
            dos_total[i_spin, :] = tdos[i_spin]

        if n_spins == 2:
            dos_total[1, :] = -1 * dos_total[1, :]

        n_orbitals = 16
        n_principals = 1
        dos_projected = np.zeros(
            (n_atoms, n_principals, n_orbitals, n_spins, len(energies))
        )
        iatom = -1
        for i_spc in range(1, len(pdos) + 1):
            for i_atom_for_specie in range(1, len(pdos[f"S{i_spc:02d}"]) + 1):
                # print(i_atom)
                iatom += 1
                for i_orbital in range(n_orbitals):
                    for i_spin in range(n_spins):
                        dos_projected[iatom, 0, i_orbital, i_spin, :] = pdos[
                            f"S{i_spc:02d}"
                        ][f"A{i_atom_for_specie:04d}"][i_orbital + i_spin * n_orbitals]

        self.dos = DensityOfStates(
            energies, dos_total, self.fermi, dos_projected)
        return self.dos

    def _read_fermi(self):
        """
        Returns the fermi energy read from FERMI.OUT
        """
        with open(self.dir_path / "EFERMI.OUT", "r") as rf:
            self.fermi = float(rf.readline().split()[0]) * HARTREE_TO_EV
        return self.fermi

    def _read_geometry_out(self):
        """sumary_line

        Keyword arguments:
        argument -- description
        Return: return_description
        """

        path_geometry_out = self.dir_path/'GEOMETRY.OUT'
        content_geometry_out = path_geometry_out.read_text()

        # Read lattice
        pattern_matrix = re.compile(
            r"avec[\s\S]*?\n"
            rf"((?:[ \t]*{FLOAT}\s+){{8}}"
            rf"{FLOAT}\s*\n)"
        )

        match = pattern_matrix.search(content_geometry_out)
        matrix_block_str = match.group(1)
        lattice = np.fromstring(matrix_block_str, sep=" ").reshape(3, 3)

        # read atoms and coords
        pattern_nspc = re.search(
            r"(?m)^atoms\s*\r?\n\s*(\d+)",
            content_geometry_out,
            re.IGNORECASE
        )

        nspecies = int(pattern_nspc.group(1)) if pattern_nspc else None

        pattern_spc = re.compile(
            r"(?mi)^\s*'([A-Za-z]+\.in)'[\s\S]*?^\s*(\d+).*\s")

        atoms: List[str] = []
        fractional_coords: Union[List[List[float]],
                                 npt.NDArray[np.float64]] = []
        for m in pattern_spc.finditer(content_geometry_out):
            atom_count = int(m.group(2))
            atom_symbols = [m.group(1).replace('.in', '')]*atom_count
            atoms += atom_symbols
            start = m.end()
            tail = content_geometry_out[start:].splitlines()

            pos_lines = tail[:atom_count]
            for line in pos_lines:
                fractional_coords += [[float(x)
                                       for x in float_re.findall(line)]]
        fractional_coords = np.array(fractional_coords)
        if fractional_coords.shape[1] == 6:
            external_mag_field = fractional_coords[:, 3:]
            fractional_coords = fractional_coords[:, :3]
        assert (len(set(atoms)) ==
                nspecies), "Number of species in GEOMETRY.OUT does not match the parsed species."
        self.structure = Structure(
            atoms=atoms,
            lattice=lattice,
            fractional_coordinates=fractional_coords,
        )
        return self.structure

    def _spd2projected(self, spd, nprinciples=1):
        """
        Helpermethod to project the spd array to the projected array
        which will be fed into pyprocar.coreElectronicBandStructure object

        Parameters
        ----------
        spd : np.ndarray
            The spd array from the earlier parse. This has a structure simlar to the PROCAR output in vasp
            Has the shape [n_kpoints,n_band,n_spins,n-orbital,n_atoms]
        nprinciples : int, optional
            The prinicipal quantum numbers, by default 1

        Returns
        -------
        np.ndarray
            The projected array. Has the shape [n_kpoints,n_band,n_atom,n_principal,n-orbital,n_spin]
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
        #     # Used if self.spins==3
        #     projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]
        #     # Used if self.spins == 4
        #     # projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, 1:]
        if nspins == 2:
            projected[:, :, :, 0, :, 0] = temp_spd[:, :, :-1, 1:-1, 0]
            projected[:, :, :, 0, :, 1] = temp_spd[:, :, :-1, 1:-1, 1]
        else:
            projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]
        return projected
