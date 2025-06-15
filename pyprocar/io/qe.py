__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import copy
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure
from pyprocar.utils.units import AU_TO_ANG, HARTREE_TO_EV

logger = logging.getLogger(__name__)


class QEParser:
    """The class is used to parse Quantum Expresso files.
    The most important objects that comes from this parser are the .ebs and .dos

    Parameters
    ----------
    dirpath : str, optional
        Directory path to where calculation took place, by default ""
    scf_in_filepath : str, optional
        The scf filename, by default "scf.in"
    bands_in_filepath : str, optional
        The bands filename in the case of a band structure calculation, by default "bands.in"
    pdos_in_filepath : str, optional
        The pdos filename in the case of a density ofstates calculation, by default "pdos.in"
    kpdos_in_filepath : str, optional
        The kpdos filename, by default "kpdos.in"
    atomic_proj_xml : str, optional
        The atomic projection xml name. This is located in the where the outdir is and in the {prefix}.save directory, by default "atomic_proj.xml"
    """

    def __init__(
        self,
        dirpath: Union[str, Path] = ".",
        scf_in_filepath: Union[str, Path] = "scf.in",
        bands_in_filepath: Union[str, Path] = "bands.in",
        pdos_in_filepath: Union[str, Path] = "pdos.in",
        kpdos_in_filepath: Union[str, Path] = "kpdos.in",
        atomic_proj_xml_filepath: Union[str, Path] = "atomic_proj.xml",
    ):
        self.dirpath = Path(dirpath).resolve()

        # Handles the pathing to the files
        (
            prefix,
            xml_root,
            atomic_proj_xml_filepath,
            pdos_in_filepath,
            bands_in_filepath,
            pdos_out_filepath,
            projwfc_out_filepath,
            self.scf_out_filepath,
        ) = self._initialize_filenames(
            scf_in_filepath=scf_in_filepath,
            bands_in_filepath=bands_in_filepath,
            pdos_in_filepath=pdos_in_filepath,
        )
        # Parsing structual and calculation type information
        self._parse_efermi(main_xml_root=xml_root)
        self._parse_magnetization(main_xml_root=xml_root)
        self._parse_structure(main_xml_root=xml_root)
        self._parse_band_structure_tag(main_xml_root=xml_root)
        self._parse_symmetries(main_xml_root=xml_root)

        # Parsing projections spd array and spd phase arrays
        if atomic_proj_xml_filepath.exists():
            self._parse_wfc_mapping(projwfc_out_filepath=projwfc_out_filepath)
            self._parse_atomic_projections(
                atomic_proj_xml_filepath=atomic_proj_xml_filepath
            )

        # Parsing density of states files
        if pdos_in_filepath.exists():
            self.dos = self._parse_pdos(
                pdos_in_filepath=pdos_in_filepath, dirpath=self.dirpath
            )
        else:
            self.dos = None
        # Parsing information related to the bandstructure calculations kpath and klabels
        self.kticks = None
        self.knames = None
        self.kpath = None
        if (
            xml_root.findall(".//input/control_variables/calculation")[0].text
            == "bands"
        ):
            self.isBandsCalc = True
            with open(bands_in_filepath, "r") as f:
                self.bandsIn = f.read()
            self._get_kpoint_labels()

        self.ebs = ElectronicBandStructure(
            kpoints=self.kpoints,
            n_kx=self.nkx,
            n_ky=self.nky,
            n_kz=self.nkz,
            bands=self.bands,
            projected=self._spd2projected(self.spd),
            efermi=self.efermi,
            kpath=self.kpath,
            projected_phase=self._spd2projected(self.spd_phase),
            labels=self.orbital_names[:-1],
            reciprocal_lattice=self.reciprocal_lattice,
        )

        return None

    def kpoints_cart(self):
        """Returns the kpoints in cartesian coordinates

        Returns
        -------
        np.ndarray
            Kpoints in cartesian coordinates
        """
        # cart_kpoints self.kpoints = self.kpoints*(2*np.pi /self.alat)
        # Converting back to crystal basis
        cart_kpoints = self.kpoints.dot(self.reciprocal_lattice)

        return cart_kpoints

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

        symbols = [x.strip() for x in self.ions]
        structures = []

        st = Structure(
            atoms=symbols,
            lattice=self.direct_lattice,
            fractional_coordinates=self.atomic_positions,
            rotations=self.rotations,
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

    def _parse_pdos(self, pdos_in_filepath, dirpath):
        """Helper method to parse the pdos files

        Parameters
        ----------
        pdos_in_filepath : str
            The pdos.in filename
        dirpath : str
            The directory path where the calculation took place.

        Returns
        -------
        pyprocar.core.DensityOfStates
            The density of states object for the calculation
        """

        with open(pdos_in_filepath, "r") as f:
            pdos_in = f.read()

        self.pdos_prefix = re.findall("filpdos\s*=\s*'(.*)'", pdos_in)[0]
        self.proj_prefix = re.findall("filproj\s*=\s*'(.*)'", pdos_in)[0]

        # Parsing total density of states
        energies, total_dos = self._parse_dos_total(
            dos_total_filename=f"{dirpath}{os.sep}{self.pdos_prefix}.pdos_tot"
        )
        self.n_energies = len(energies)
        # Finding all the density of states projections files
        wfc_filenames = self._parse_available_wfc_filenames()
        projected_dos = self._parse_dos_projections(
            wfc_filenames=wfc_filenames, n_energy=len(energies)
        )

        # print(projected_labels)
        dos = DensityOfStates(
            energies=energies,
            total=total_dos,
            efermi=self.efermi,
            projected=projected_dos,
            interpolation_factor=1,
        )
        return dos

    def _parse_dos_total(self, dos_total_filename):
        """Helper method to parse the dos total file

        Parameters
        ----------
        dos_total_filename : str
            The dos total filename

        Returns
        -------
        Tupole
            Returns a tuple with energies and the total dos arrays
        """
        with open(dos_total_filename) as f:
            tmp_text = f.readlines()
            header = tmp_text[0]
            dos_text = "".join(tmp_text[1:])

        # Strip ending spaces away. Avoind empty string at the end
        raw_dos_blocks_by_energy = dos_text.rstrip().split("\n")

        n_energies = len(raw_dos_blocks_by_energy)
        energies = np.zeros(shape=(n_energies))
        # total_dos =  np.zeros(shape=(n_energies, self.n_spin))
        total_dos = np.zeros(shape=(self.n_spin, n_energies))
        for ienergy, raw_dos_block_by_energy in enumerate(raw_dos_blocks_by_energy):
            energies[ienergy] = float(raw_dos_block_by_energy.split()[0])

            # Covers colinear spin-polarized. This is because these is a difference in energies
            if self.n_spin == 2:
                total_dos[:, ienergy] = [
                    float(val)
                    for val in raw_dos_block_by_energy.split()[-self.n_spin :]
                ]
            # Covers colinear non-spin-polarized and non-colinear. This is because the energies are the same
            else:
                total_dos[0, ienergy] = float(raw_dos_block_by_energy.split()[2])

        energies -= self.efermi
        return energies, total_dos

    def _parse_dos_projections(self, wfc_filenames, n_energy):
        """Parse the dos projection files using efficient array operations."""
        n_principal_number = 1

        projected_dos_array = np.zeros(
            (self.n_atoms, n_principal_number, self.n_orbitals, self.n_spin, n_energy)
        )

        for filename in wfc_filenames:
            self._validate_file(filename)

            # Some pdos files are not non-colinear.
            # In the version of the qe code where we get the spin projections.
            # Non-colinear pdos files should have the word 'j' in the filename.
            if not self._is_non_colinear_file(filename) and self.is_non_colinear:
                continue

            # Determine the parsing method based on calculation type
            if self.is_non_colinear:
                self._parse_non_colinear(filename, projected_dos_array)
            else:
                self._parse_colinear(filename, projected_dos_array)

        return projected_dos_array

    def _is_non_colinear_file(self, filename):
        """Determine if the file corresponds to non-colinear calculations."""
        file_tag = filename.split("_")[-1]
        return "j" in file_tag

    def _validate_file(self, filename):
        if not os.path.exists(filename):
            raise ValueError(f"ERROR: pdos file not found: {filename}")

    def _read_file_content(self, filename):
        with open(filename) as f:
            # Check if spin component projections are present
            lines = f.readlines()[1:]
            spin_projections_present = "lsigma" in lines[1]

            if spin_projections_present:
                pdos = self._parse_spin_components_lines(lines)
            elif self.is_non_colinear:
                pdos = self._parse_noncolinear_pdos(lines)
            else:
                pdos = self._parse_colinear_pdos(lines)
        return pdos

    def _parse_spin_components_lines(self, lines):

        n_orbitals = len(lines[0].split()) - 2
        pdos = np.zeros(shape=(self.n_energies, n_orbitals, self.n_spin))
        for i_energy in range(self.n_energies):
            iblock_start = i_energy * 6
            total_line = lines[iblock_start].split()[2:]
            x_line = lines[iblock_start + 2].split()[1:]
            y_line = lines[iblock_start + 3].split()[1:]
            z_line = lines[iblock_start + 4].split()[1:]
            pdos[i_energy, :, 0] = [float(tot) for tot in total_line]
            pdos[i_energy, :, 1] = [float(x) for x in x_line]
            pdos[i_energy, :, 2] = [float(y) for y in y_line]
            pdos[i_energy, :, 3] = [float(z) for z in z_line]

        # Move energy to the last axis
        pdos = np.moveaxis(pdos, 0, -1)
        return pdos

    def _parse_colinear_pdos(self, lines):
        # The energy column is first, the sum totals by spin channel
        n_orbitals = len(lines[0].split()) - self.n_spin - 1
        n_orbitals //= self.n_spin
        pdos = np.zeros(shape=(self.n_energies, n_orbitals, self.n_spin))
        col_start = 1 + self.n_spin
        for i_energy in range(self.n_energies):

            # Skip energies and sum over projections
            values = lines[i_energy].split()[col_start:]
            if self.n_spin == 1:
                pdos[i_energy, :, 0] = values
            else:
                pdos[i_energy, :, 0] = values[::2]
                pdos[i_energy, :, 1] = values[1::2]

        # Move energy to the last axis
        pdos = np.moveaxis(pdos, 0, -1)
        return pdos

    def _parse_noncolinear_pdos(self, lines):
        # The energy column is first, the sum totals by spin channel
        n_orbitals = len(lines[0].split()) - 1 - 1
        n_orbitals //= 1
        pdos = np.zeros(shape=(self.n_energies, n_orbitals, 4))
        col_start = 1 + 1
        for i_energy in range(self.n_energies):
            # Skip energies and sum over projections
            values = lines[i_energy].split()[col_start:]
            pdos[i_energy, :, 0] = values
        # Move energy to the last axis
        pdos = np.moveaxis(pdos, 0, -1)
        return pdos

    def _extract_file_info(self, filename):
        atom_num = int(re.findall(r"#(\d*)", filename)[0]) - 1
        wfc_name = re.findall(r"atm#\d*\(([a-zA-Z0-9]*)\)", filename)[0]
        orbital_name = filename.split("(")[-1][0]

        total_angular_momentum = None
        if self.is_non_colinear:
            tmp_str = filename.split("_")[-1]
            total_angular_momentum = float(tmp_str.strip("j").strip(")"))
        return atom_num, wfc_name, orbital_name, total_angular_momentum

    def _parse_non_colinear(self, filename, projected_dos_array):

        pdos_data = self._read_file_content(filename)
        atom_num, wfc_name, orbital_name, total_angular_momentum = (
            self._extract_file_info(filename)
        )

        orbital_nums, _ = self._get_orbital_info_non_colinear(
            orbital_name, total_angular_momentum
        )
        if not orbital_nums:
            raise ValueError(f"ERROR: orbital_nums is empty for {filename}")

        projected_dos_array[
            atom_num, 0, orbital_nums, :, : self.n_energies
        ] += pdos_data

    def _parse_colinear(self, filename, projected_dos_array):

        pdos_data = self._read_file_content(filename)
        atom_num, wfc_name, orbital_name, total_angular_momentum = (
            self._extract_file_info(filename)
        )
        orbital_nums, _ = self._get_orbital_info_colinear(orbital_name)

        if not orbital_nums:
            raise ValueError(f"ERROR: orbital_nums is empty for {filename}")
        projected_dos_array[
            atom_num, 0, orbital_nums, :, : self.n_energies
        ] += pdos_data

    def _get_orbital_info_non_colinear(self, l_orbital_name, tot_ang_mom):
        orbital_info = {
            "s": {0.5: ([0, 1], np.linspace(-0.5, 0.5, 2))},
            "p": {
                0.5: ([2, 3], np.linspace(-0.5, 0.5, 2)),
                1.5: ([4, 5, 6, 7], np.linspace(-1.5, 1.5, 4)),
            },
            "d": {
                0.5: ([2, 3], np.linspace(-0.5, 0.5, 2)),
                1.5: ([8, 9, 10, 11], np.linspace(-1.5, 1.5, 4)),
                2.5: ([12, 13, 14, 15, 16, 17], np.linspace(-2.5, 2.5, 6)),
            },
        }
        return orbital_info.get(l_orbital_name, {}).get(tot_ang_mom, ([], []))

    def _get_orbital_info_colinear(self, orbital_name):
        orbital_info = {
            "s": ([0], np.linspace(0, 0, 1)),
            "p": ([1, 2, 3], np.linspace(-1, 1, 3)),
            "d": ([4, 5, 6, 7, 8], np.linspace(-2, 2, 5)),
        }
        return orbital_info.get(orbital_name, ([], []))

    def _get_kpoint_labels(self):
        """
        This method will parse the bands.in file to get the kpath information.
        """

        # Parsing klabels
        self.ngrids = []
        kmethod = re.findall("K_POINTS[\s\{]*([a-z_]*)[\s\{]*", self.bandsIn)[0]
        self.discontinuities = []
        if kmethod == "crystal":
            numK = int(re.findall("K_POINTS.*\n([0-9]*)", self.bandsIn)[0])

            raw_khigh_sym = re.findall(
                "K_POINTS.*\n\s*[0-9]*.*\n" + numK * "(.*)\n*", self.bandsIn
            )[0]

            tickCountIndex = 0
            self.knames = []
            self.kticks = []

            for x in raw_khigh_sym:
                if len(x.split()) == 5:
                    self.knames.append("%s" % x.split()[4].replace("!", ""))
                    self.kticks.append(tickCountIndex)

                tickCountIndex += 1

            self.nhigh_sym = len(self.knames)

        elif kmethod == "crystal_b":
            self.nhigh_sym = int(re.findall("K_POINTS.*\n([0-9]*)", self.bandsIn)[0])

            raw_khigh_sym = re.findall(
                "K_POINTS.*\n.*\n" + self.nhigh_sym * "(.*)\n*",
                self.bandsIn,
            )[0]

            self.kticks = []
            self.high_symmetry_points = np.zeros(shape=(self.nhigh_sym, 3))
            tick_Count = 1
            for ihs in range(self.nhigh_sym):

                # In QE cyrstal_b mode, the user is able to specify grid on last high symmetry point.
                # QE just uses 1 for the last high symmetry point.
                grid_current = int(raw_khigh_sym[ihs].split()[3])
                if ihs < self.nhigh_sym - 2:
                    self.ngrids.append(grid_current)

                # Incrementing grid by 1 for seocnd to last high symmetry point
                elif ihs == self.nhigh_sym - 2:
                    self.ngrids.append(grid_current + 1)

                # I have no idea why I skip the last high symmetry point. I think it had to do with disconinuous points.
                # Need to code test case for this. Otherwise leave it as is.
                # elif ihs == self.nhigh_sym - 1:
                #     continue
                self.kticks.append(tick_Count - 1)
                tick_Count += grid_current

            raw_ticks = re.findall(
                "K_POINTS.*\n\s*[0-9]*\s*[0-9]*.*\n" + self.nhigh_sym * ".*!(.*)\n*",
                self.bandsIn,
            )[0]

            if len(raw_ticks) != self.nhigh_sym:
                self.knames = [str(x) for x in range(self.nhigh_sym)]

            else:
                self.knames = [
                    "%s" % (x.replace(",", "").replace("vlvp1d", "").replace(" ", ""))
                    for x in raw_ticks
                ]

        # Formating to conform with Kpath class
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

    def _initialize_filenames(
        self,
        scf_in_filepath: Union[str, Path],
        bands_in_filepath: Union[str, Path],
        pdos_in_filepath: Union[str, Path],
    ):
        """This helper method handles pathing to the to locate files

        Parameters
        ----------
        dirpath : str
            The directory path where the calculation is
        scf_in_filepath : str
            The input scf filename
        bands_in_filepath : str
            The input bands filename
        pdos_in_filepath : str
            The input pdos filename

        Returns
        -------
        Tuple
            Returns a tuple of important pathing information.
            Mainly, the directory path is prepended to the filenames.
        """
        scf_filename = Path(scf_in_filepath).name
        scf_in_filepath = Path(self.dirpath) / scf_filename
        scf_out_filepath = Path(self.dirpath) / f"{scf_filename.split('.')[0]}.out"

        bands_filename = Path(bands_in_filepath).name
        bands_in_filepath = Path(self.dirpath) / bands_filename
        bands_out_filepath = Path(self.dirpath) / f"{bands_filename.split('.')[0]}.out"

        pdos_filename = Path(pdos_in_filepath).name
        pdos_in_filepath = Path(self.dirpath) / pdos_filename
        pdos_out_filepath = Path(self.dirpath) / f"pdos.out"
        kpdos_out_filepath = Path(self.dirpath) / f"kpdos.out"

        if kpdos_out_filepath.exists():
            projwfc_out_filepath = kpdos_out_filepath
        elif pdos_out_filepath.exists():
            projwfc_out_filepath = pdos_out_filepath
        else:
            projwfc_out_filepath = None

        with open(scf_in_filepath, "r") as f:
            scf_in = f.read()

        outdir = re.findall("outdir\s*=\s*'\S*?(.*)'", scf_in)[0]
        prefix = re.findall("prefix\s*=\s*'(.*)'", scf_in)[0]
        xml_filename = prefix + ".xml"

        atomic_proj_xml_filepath = (
            Path(self.dirpath) / outdir / f"{prefix}.save" / "atomic_proj.xml"
        )
        if not atomic_proj_xml_filepath.exists():
            atomic_proj_xml_filepath = Path(self.dirpath) / "atomic_proj.xml"

        output_xml_filepath = Path(self.dirpath) / outdir / xml_filename
        logger.info(f"output_xml: {output_xml_filepath}")
        if not output_xml_filepath.exists():
            output_xml_filepath = Path(self.dirpath) / xml_filename

        tree = ET.parse(output_xml_filepath)

        xml_root = tree.getroot()
        prefix = xml_root.findall(".//input/control_variables/prefix")[0].text

        pdos_in_filepath = Path(self.dirpath) / pdos_in_filepath
        bands_in_filepath = Path(self.dirpath) / bands_in_filepath

        if pdos_out_filepath.exists():
            pdos_out_filepath = pdos_out_filepath

        return (
            prefix,
            xml_root,
            atomic_proj_xml_filepath,
            pdos_in_filepath,
            bands_in_filepath,
            pdos_out_filepath,
            projwfc_out_filepath,
            scf_out_filepath,
        )

    def _parse_available_wfc_filenames(self):
        """Helper method to parse the projection filename from the pdos.out file

        Parameters
        ----------
        dirpath : str
            The directory name where the calculation is.

        Returns
        -------
        List
            Returns a list of projection file names
        """

        wfc_filenames = []
        tmp_wfc_filenames = []
        atms_wfc_num = []

        # Parsing projection filnames for identification information
        for filepath in Path(self.dirpath).iterdir():
            if filepath.suffix in [
                ".pdos",
                ".pdos_tot",
                ".lowdin",
                ".projwfc_down",
                ".projwfc_up",
                ".xml",
            ]:
                continue

            if filepath.stem != self.pdos_prefix:
                continue

            filename = str(filepath.name)
            tmp_wfc_filenames.append(filename)

            atm_num = int(re.findall("_atm#([0-9]*)\(.*", filename)[0])
            wfc_num = int(re.findall("_wfc#([0-9]*)\(.*", filename)[0])
            wfc = re.findall("_wfc#[0-9]*\(([_A-Za-z0-9.]*)\).*", filename)[0]
            atm = re.findall("_atm#[0-9]*\(([A-Za-z]*[0-9]*)\).*", filename)[0]

            atms_wfc_num.append((atm_num, atm, wfc_num, wfc))

        # sort density of states projections files by atom number
        sorted_file_num = sorted(atms_wfc_num, key=lambda a: a[0])
        for index in sorted_file_num:
            wfc_filenames.append(
                f"{self.dirpath}{os.sep}{self.pdos_prefix}.pdos_atm#{index[0]}({index[1]})_wfc#{index[2]}({index[3]})"
            )

        return wfc_filenames

    def _parse_wfc_mapping(self, projwfc_out_filepath):
        """Helper method which creates a mapping between wfc number and the orbtial and atom numbers

        Parameters
        ----------
        projwfc_out_filepath : str
            The projwfc out filepath

        Returns
        -------
        None
            None
        """
        with open(projwfc_out_filepath) as f:
            pdos_out = f.read()
        raw_wfc = re.findall(
            "(?<=read\sfrom\spseudopotential\sfiles).*\n\n([\S\s]*?)\n\n(?=\sk\s=)",
            pdos_out,
        )[0]
        wfc_list = raw_wfc.split("\n")

        self.wfc_mapping = {}
        # print(self.orbitals)
        for i, wfc in enumerate(wfc_list):

            iwfc = int(re.findall("(?<=state\s#)\s*(\d*)", wfc)[0])
            iatm = int(re.findall("(?<=atom)\s*(\d*)", wfc)[0])
            l_orbital_type_index = int(re.findall("(?<=l=)\s*(\d*)", wfc)[0])

            if self.is_non_colinear:
                j_orbital_type_index = float(re.findall("(?<=j=)\s*([-\d.]*)", wfc)[0])
                m_orbital_type_index = float(
                    re.findall("(?<=m_j=)\s*([-\d.]*)", wfc)[0]
                )
                tmp_orb_dict = {
                    "l": self._convert_lorbnum_to_letter(lorbnum=l_orbital_type_index),
                    "j": j_orbital_type_index,
                    "m": m_orbital_type_index,
                }
                # print(self._convert_lorbnum_to_letter(lorbnum=l_orbital_type_index))
            else:
                m_orbital_type_index = int(re.findall("(?<=m=)\s*(\d*)", wfc)[0])
                tmp_orb_dict = {"l": l_orbital_type_index, "m": m_orbital_type_index}

            iorb = 0

            for iorbital, orb in enumerate(self.orbitals):
                if tmp_orb_dict == orb:
                    iorb = iorbital

            self.wfc_mapping.update({f"wfc_{iwfc}": {"orbital": iorb, "atom": iatm}})

        return None

    def _parse_atomic_projections(self, atomic_proj_xml_filepath):
        """A Helper method to parse the atomic projection xml file

        Parameters
        ----------
        atomic_proj_xml_filepath : str
            The atomic_proj.xml filename

        Returns
        -------
        None
            None
        """
        atmProj_tree = ET.parse(atomic_proj_xml_filepath)
        atm_proj_root = atmProj_tree.getroot()

        root_header = atm_proj_root.findall(".//HEADER")[0]

        nbnd = int(root_header.get("NUMBER_OF_BANDS"))
        nk = int(root_header.get("NUMBER_OF_K-POINTS"))
        nwfc = int(root_header.get("NUMBER_OF_ATOMIC_WFC"))
        norb = self.n_orbitals
        natm = self.n_atoms
        nspin_channels = int(root_header.get("NUMBER_OF_SPIN_COMPONENTS"))
        nspin_projections = self.n_spin
        # The indices are to match the format of the from PROCAR format.
        # In it there is an extra 2 columns for orbitals for the ion index and the total
        # Also there is and extra row for the totals
        self.spd = np.zeros(
            shape=(
                self.n_k,
                self.n_band,
                self.n_spin,
                self.n_atoms + 1,
                self.n_orbitals + 2,
            )
        )

        self.spd_phase = np.zeros(
            shape=(self.spd.shape),
            dtype=np.complex_,
        )

        bands = self._parse_bands_tag(atm_proj_root, nk, nbnd, nspin_channels)
        kpoints = self._parse_kpoints_tag(atm_proj_root, nk, nspin_channels)
        projs, projs_phase = self._parse_projections_tag(
            atm_proj_root, nk, nbnd, natm, norb, nspin_channels, nspin_projections
        )

        # maping the projections to the spd array. The spd array is the output of the PROCAR file
        self.spd[:, :, :, :-1, 1:-1] += projs[:, :, :, :, :]
        self.spd_phase[:, :, :, :-1, 1:-1] += projs_phase[:, :, :, :, :]

        # Adding atom index. This is a vasp output thing
        for ions in range(self.ionsCount):
            self.spd[:, :, :, ions, 0] = ions + 1

        # The following fills the totals for the spd array. Again this is a vasp output thing.
        self.spd[:, :, :, :, -1] = np.sum(self.spd[:, :, :, :, 1:-1], axis=4)
        self.spd[:, :, :, -1, :] = np.sum(self.spd[:, :, :, :-1, :], axis=3)
        self.spd[:, :, :, -1, 0] = 0

        return None

    def _parse_bands_tag(self, atm_proj_root, nk, nbnd, nspins):

        bands = np.zeros(shape=(nk, nbnd, nspins))
        results = atm_proj_root.findall(".//EIGENSTATES/E")
        # For spin-polarized calculations, there are two spin channels.
        # They add them by first adding the spin up and then the spin down
        # I break this down with the folloiwng indexing
        spin_reuslts = [results[i * nk : (i + 1) * nk] for i in range(nspins)]
        for ispin, spin_result in enumerate(spin_reuslts):
            for ik, result in enumerate(spin_result):
                bands_per_kpoint = result.text.split()
                bands[ik, :, ispin] = bands_per_kpoint

        return bands

    def _parse_kpoints_tag(self, atm_proj_root, nk, nspins):
        kpoints = np.zeros(shape=(nk, 3))
        kpoint_tags = atm_proj_root.findall(".//EIGENSTATES/K-POINT")
        # For spin-polarized calculations, there are two spin channels.
        # They add them by first adding the spin up and then the spin down
        # I break this down with the folloiwng indexing
        spin_reuslts = [kpoint_tags[i * nk : (i + 1) * nk] for i in range(nspins)]
        for ispin, spin_result in enumerate(spin_reuslts):
            for ik, kpoint_tag in enumerate(spin_result):
                kpoint = kpoint_tag.text.split()
                kpoints[ik, :] = kpoint
        return kpoints

    def _parse_projections_tag(
        self, atm_proj_root, nk, nbnd, natm, norb, nspin_channels, nspin_projections
    ):

        projs = np.zeros(shape=(nk, nbnd, nspin_projections, natm, norb))
        projs_phase = np.zeros(
            shape=(nk, nbnd, nspin_projections, natm, norb), dtype=np.complex_
        )
        proj_tags = atm_proj_root.findall(".//EIGENSTATES/PROJS")
        # For spin-polarized calculations, there are two spin channels.
        # They add them by first adding the spin up and then the spin down
        # I break this down with the folloiwng indexing
        spin_reuslts = [proj_tags[i * nk : (i + 1) * nk] for i in range(nspin_channels)]
        for ispin, spin_result in enumerate(spin_reuslts):
            for ik, proj_tag in enumerate(spin_result):
                atm_wfs_tags = proj_tag.findall("ATOMIC_WFC")
                for atm_wfs_tag in atm_wfs_tags:
                    iwfc = int(atm_wfs_tag.get("index"))
                    iorb = self.wfc_mapping[f"wfc_{iwfc}"]["orbital"]
                    iatm = self.wfc_mapping[f"wfc_{iwfc}"]["atom"] - 1
                    band_projections = atm_wfs_tag.text.strip().split("\n")
                    for iband, band_projection in enumerate(band_projections):
                        real = float(band_projection.split()[0])
                        imag = float(band_projection.split()[1])
                        comp = complex(real, imag)
                        comp_squared = np.absolute(comp) ** 2

                        projs_phase[ik, iband, ispin, iatm, iorb] = complex(real, imag)
                        projs[ik, iband, ispin, iatm, iorb] = comp_squared

                atm_sigma_wfs_tags = proj_tag.findall("ATOMIC_SIGMA_PHI")
                if atm_sigma_wfs_tags:
                    spin_x_projections = [
                        atm_sigma_wfs_tag
                        for atm_sigma_wfs_tag in atm_sigma_wfs_tags
                        if atm_sigma_wfs_tag.get("ipol") == "1"
                    ]
                    spin_y_projections = [
                        atm_sigma_wfs_tag
                        for atm_sigma_wfs_tag in atm_sigma_wfs_tags
                        if atm_sigma_wfs_tag.get("ipol") == "2"
                    ]
                    spin_z_projections = [
                        atm_sigma_wfs_tag
                        for atm_sigma_wfs_tag in atm_sigma_wfs_tags
                        if atm_sigma_wfs_tag.get("ipol") == "3"
                    ]
                    spin_projections = [
                        spin_x_projections,
                        spin_y_projections,
                        spin_z_projections,
                    ]
                    for i_spin_component, spin_projection_tags in enumerate(
                        spin_projections
                    ):
                        for spin_projection_tag in spin_projection_tags:
                            iwfc = int(spin_projection_tag.get("index"))
                            iorb = self.wfc_mapping[f"wfc_{iwfc}"]["orbital"]
                            iatm = self.wfc_mapping[f"wfc_{iwfc}"]["atom"] - 1
                            band_projections = spin_projection_tag.text.strip().split(
                                "\n"
                            )
                            for iband, band_projection in enumerate(band_projections):
                                real = float(band_projection.split()[0])
                                imag = float(band_projection.split()[1])
                                comp = complex(real, imag)
                                comp_squared = np.absolute(comp) ** 2

                                # Move spin index by 1 to match the order in the spd array
                                # First index should be total,
                                # second, third, and fourth should be x,y,z, respoectively
                                projs_phase[
                                    ik, iband, i_spin_component + 1, iatm, iorb
                                ] = complex(real, imag)
                                projs[ik, iband, i_spin_component + 1, iatm, iorb] = (
                                    real
                                )

        return projs, projs_phase

    def _parse_structure(self, main_xml_root):
        """A helper method to parse the structure tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """

        self.nspecies = len(main_xml_root.findall(".//output/atomic_species")[0])
        self.composition = {
            species.attrib["name"]: 0
            for species in main_xml_root.findall(".//output/atomic_species")[0]
        }
        self.species_list = list(self.composition.keys())
        self.ionsCount = int(
            main_xml_root.findall(".//output/atomic_structure")[0].attrib["nat"]
        )
        self.alat = float(
            main_xml_root.findall(".//output/atomic_structure")[0].attrib["alat"]
        )
        self.alat = self.alat * AU_TO_ANG

        self.ions = []
        for ion in main_xml_root.findall(".//output/atomic_structure/atomic_positions")[
            0
        ]:
            self.ions.append(ion.attrib["name"][:2])
            self.composition[ion.attrib["name"]] += 1

        self.n_atoms = len(self.ions)

        self.atomic_positions = np.array(
            [
                ion.text.split()
                for ion in main_xml_root.findall(
                    ".//output/atomic_structure/atomic_positions"
                )[0]
            ],
            dtype=float,
        )
        # in a.u
        self.direct_lattice = np.array(
            [
                acell.text.split()
                for acell in main_xml_root.findall(".//output/atomic_structure/cell")[0]
            ],
            dtype=float,
        )

        # in a.u
        self.reciprocal_lattice = (2 * np.pi / self.alat) * np.array(
            [
                acell.text.split()
                for acell in main_xml_root.findall(
                    ".//output/basis_set/reciprocal_lattice"
                )[0]
            ],
            dtype=float,
        )

        # Convert to angstrom
        self.direct_lattice = self.direct_lattice * AU_TO_ANG

        return None

    def _parse_symmetries(self, main_xml_root):
        """A helper method to parse the symmetries tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """
        self.nsym = int(main_xml_root.findall(".//output/symmetries/nsym")[0].text)
        self.nrot = int(main_xml_root.findall(".//output/symmetries/nrot")[0].text)
        self.spg = int(
            main_xml_root.findall(".//output/symmetries/space_group")[0].text
        )
        self.nsymmetry = len(main_xml_root.findall(".//output/symmetries/symmetry"))

        self.rotations = np.zeros(shape=(self.nsymmetry, 3, 3))

        for isymmetry, symmetry_operation in enumerate(
            main_xml_root.findall(".//output/symmetries/symmetry")
        ):

            symmetry_matrix = (
                np.array(
                    symmetry_operation.findall(".//rotation")[0].text.split(),
                    dtype=float,
                )
                .reshape(3, 3)
                .T
            )

            self.rotations[isymmetry, :, :] = symmetry_matrix
        return None

    def _parse_magnetization(self, main_xml_root):
        """A helper method to parse the magnetization tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """
        is_non_colinear = str2bool(
            main_xml_root.findall(".//output/magnetization/noncolin")[0].text
        )
        is_spin_calc = str2bool(
            main_xml_root.findall(".//output/magnetization/lsda")[0].text
        )
        is_spin_orbit_calc = str2bool(
            main_xml_root.findall(".//output/magnetization/spinorbit")[0].text
        )

        # The calcuulation is non-colinear
        if is_non_colinear:
            n_spin = 4

            orbitals = [
                {"l": "s", "j": 0.5, "m": -0.5},
                {"l": "s", "j": 0.5, "m": 0.5},
                {"l": "p", "j": 0.5, "m": -0.5},
                {"l": "p", "j": 0.5, "m": 0.5},
                {"l": "p", "j": 1.5, "m": -1.5},
                {"l": "p", "j": 1.5, "m": -0.5},
                {"l": "p", "j": 1.5, "m": -0.5},
                {"l": "p", "j": 1.5, "m": 1.5},
                {"l": "d", "j": 1.5, "m": -1.5},
                {"l": "d", "j": 1.5, "m": -0.5},
                {"l": "d", "j": 1.5, "m": -0.5},
                {"l": "d", "j": 1.5, "m": 1.5},
                {"l": "d", "j": 2.5, "m": -2.5},
                {"l": "d", "j": 2.5, "m": -1.5},
                {"l": "d", "j": 2.5, "m": -0.5},
                {"l": "d", "j": 2.5, "m": 0.5},
                {"l": "d", "j": 2.5, "m": 1.5},
                {"l": "d", "j": 2.5, "m": 2.5},
            ]
            orbitalNames = []
            for orbital in orbitals:
                tmp_name = ""
                for key, value in orbital.items():
                    # print(key,value)
                    if key != "l":
                        tmp_name = tmp_name + key + str(value)
                    else:
                        tmp_name = tmp_name + str(value) + "_"
                orbitalNames.append(tmp_name)

        # The calcuulation is colinear
        else:
            # colinear spin or non spin polarized
            if is_spin_calc:
                n_spin = 2
            else:
                n_spin = 1
            orbitals = [
                {"l": 0, "m": 1},
                {"l": 1, "m": 3},
                {"l": 1, "m": 1},
                {"l": 1, "m": 2},
                {"l": 2, "m": 5},
                {"l": 2, "m": 3},
                {"l": 2, "m": 1},
                {"l": 2, "m": 2},
                {"l": 2, "m": 4},
            ]
            orbitalNames = [
                "s",
                "py",
                "pz",
                "px",
                "dxy",
                "dyz",
                "dz2",
                "dxz",
                "dx2",
                "tot",
            ]

        self.is_non_colinear = is_non_colinear
        self.is_spin_calc = is_spin_calc
        self.is_spin_orbit_calc = is_spin_orbit_calc
        self.n_spin = n_spin
        self.orbitals = orbitals
        self.n_orbitals = len(orbitals)
        self.orbital_names = orbitalNames
        return None

    def _parse_band_structure_tag(self, main_xml_root):
        """A helper method to parse the band_structure tag of the main xml file

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """

        self.nkx = None
        self.nky = None
        self.nkz = None
        self.nk1 = None
        self.nk2 = None
        self.nk3 = None
        monkhorst_tag = main_xml_root.findall(
            ".//output/band_structure/starting_k_points"
        )[0][0]
        if "monkhorst_pack" in monkhorst_tag.tag:
            self.nkx = float(monkhorst_tag.attrib["nk1"])
            self.nky = float(monkhorst_tag.attrib["nk2"])
            self.nkz = float(monkhorst_tag.attrib["nk3"])

            self.nk1 = float(monkhorst_tag.attrib["k1"])
            self.nk2 = float(monkhorst_tag.attrib["k2"])
            self.nk3 = float(monkhorst_tag.attrib["k3"])

        self.nks = int(main_xml_root.findall(".//output/band_structure/nks")[0].text)
        self.atm_wfc = int(
            main_xml_root.findall(".//output/band_structure/num_of_atomic_wfc")[0].text
        )

        self.nelec = float(
            main_xml_root.findall(".//output/band_structure/nelec")[0].text
        )
        if self.n_spin == 2:

            self.n_band = int(
                main_xml_root.findall(".//output/band_structure/nbnd_up")[0].text
            )
            self.nbnd_up = int(
                main_xml_root.findall(".//output/band_structure/nbnd_up")[0].text
            )
            self.nbnd_down = int(
                main_xml_root.findall(".//output/band_structure/nbnd_dw")[0].text
            )

            self.bands = np.zeros(shape=(self.nks, self.n_band, 2))
            self.kpoints = np.zeros(shape=(self.nks, 3))
            self.weights = np.zeros(shape=(self.nks))
            self.occupations = np.zeros(shape=(self.nks, self.n_band, 2))

            band_structure_element = main_xml_root.findall(".//output/band_structure")[
                0
            ]

            for ikpoint, kpoint_element in enumerate(
                main_xml_root.findall(".//output/band_structure/ks_energies")
            ):

                self.kpoints[ikpoint, :] = np.array(
                    kpoint_element.findall(".//k_point")[0].text.split(), dtype=float
                )
                self.weights[ikpoint] = np.array(
                    kpoint_element.findall(".//k_point")[0].attrib["weight"],
                    dtype=float,
                )

                self.bands[ikpoint, :, 0] = (
                    HARTREE_TO_EV
                    * np.array(
                        kpoint_element.findall(".//eigenvalues")[0].text.split(),
                        dtype=float,
                    )[: self.nbnd_up]
                )

                self.occupations[ikpoint, :, 0] = np.array(
                    kpoint_element.findall(".//occupations")[0].text.split(),
                    dtype=float,
                )[: self.nbnd_up]

                self.bands[ikpoint, :, 1] = (
                    HARTREE_TO_EV
                    * np.array(
                        kpoint_element.findall(".//eigenvalues")[0].text.split(),
                        dtype=float,
                    )[self.nbnd_down :]
                )
                self.occupations[ikpoint, :, 1] = np.array(
                    kpoint_element.findall(".//occupations")[0].text.split(),
                    dtype=float,
                )[self.nbnd_down :]

        # For non-spin-polarized and non colinear
        else:
            self.n_band = int(
                main_xml_root.findall(".//output/band_structure/nbnd")[0].text
            )
            self.bands = np.zeros(shape=(self.nks, self.n_band, 1))
            self.kpoints = np.zeros(shape=(self.nks, 3))
            self.weights = np.zeros(shape=(self.nks))
            self.occupations = np.zeros(shape=(self.nks, self.n_band))
            for ikpoint, kpoint_element in enumerate(
                main_xml_root.findall(".//output/band_structure/ks_energies")
            ):
                self.kpoints[ikpoint, :] = np.array(
                    kpoint_element.findall(".//k_point")[0].text.split(), dtype=float
                )
                self.weights[ikpoint] = np.array(
                    kpoint_element.findall(".//k_point")[0].attrib["weight"],
                    dtype=float,
                )
                self.bands[ikpoint, :, 0] = HARTREE_TO_EV * np.array(
                    kpoint_element.findall(".//eigenvalues")[0].text.split(),
                    dtype=float,
                )

                self.occupations[ikpoint, :] = np.array(
                    kpoint_element.findall(".//occupations")[0].text.split(),
                    dtype=float,
                )
        # Multiply in 2pi/alat
        self.kpoints = self.kpoints * (2 * np.pi / self.alat)
        # Converting back to crystal basis
        self.kpoints = np.around(
            self.kpoints.dot(np.linalg.inv(self.reciprocal_lattice)), decimals=8
        )
        self.n_k = len(self.kpoints)

        self.kpointsCount = len(self.kpoints)
        self.bandsCount = self.n_band

        return None

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

    def _parse_efermi(self, main_xml_root):
        """A helper method to parse the band_structure tag of the main xml file for the fermi energy

        Parameters
        ----------
        main_xml_root : xml.etree.ElementTree.Element
            The main xml Element

        Returns
        -------
        None
            None
        """
        # self.efermi =  float(main_xml_root.findall(".//output/band_structure/fermi_energy")[0].text) * HARTREE_TO_EV

        with open(self.scf_out_filepath, "r") as f:
            scf_out = f.read()
            self.efermi = float(
                re.findall("the Fermi energy is\s*([-\d.]*)", scf_out)[0]
            )
        return None

    def _convert_lorbnum_to_letter(self, lorbnum):
        """A helper method to convert the lorb number to the letter format

        Parameters
        ----------
        lorbnum : int
            The number of the l orbital

        Returns
        -------
        str
            The l orbital name
        """
        lorb_mapping = {0: "s", 1: "p", 2: "d", 3: "f"}
        return lorb_mapping[lorbnum]


def str2bool(v):
    """Converts a string of a boolean to an actual boolean

    Parameters
    ----------
    v : str
        The string of the boolean value

    Returns
    -------
    boolean
        The boolean value
    """
    return v.lower() in ("true")
    return v.lower() in ("true")

    return v.lower() in ("true")
