__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import copy
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure

HARTREE_TO_EV = 27.211386245988  # eV/Hartree


class SiestaParser:
    def __init__(self, fdf_filepath: Union[str, Path]):
        """The class is used to parse information in a siesta calculation

        Parameters
        ----------
        fdf_file : str
            The .fdf file that has the inputs for the Siesta calculation
        """

        self.dirname = Path(fdf_filepath).parent

        # Parse some initial information
        # This contains kpath info, prefix for files, and structure info
        self._parse_fdf(fdf_filepath=fdf_filepath)

        # parses the bands file. This will initiate the bands array
        bands_filepath = self.dirname / f"{self.prefix}.bands"
        self._parse_bands(bands_filepath=bands_filepath)

        # self._parse_struct_out(struct_out_file=f"{self.prefix}{os.sep}STRUCT_OUT")

    def _parse_fdf(self, fdf_filepath: Union[str, Path]):
        """A helper method to parse the infromation inside the fdf file

        Parameters
        ----------
        fdf_filepath : str
            The .fdf file that has the inputs for the Siesta calculation

        Returns
        -------
        None
            None
        """
        with open(fdf_filepath) as f:
            fdf_text = f.read()

        self.prefix = re.findall("SystemLabel ([0-9A-Za-z]*)", fdf_text)[0]

        self._parse_direct_lattice(fdf_text=fdf_text)
        self._parse_atomic_positions(fdf_text=fdf_text)
        self._create_structure()

        is_bands_calc = len(re.findall("%block (BandLines)", fdf_text)) == 1
        if is_bands_calc:
            self._parse_kpath(fdf_text=fdf_text)

        is_dos_calc = (
            len(re.findall("%block (ProjectedDensityOfStates)", fdf_text)) == 1
        )
        if is_dos_calc:
            self._parse_dos_info(fdf_text=fdf_text)

        return None

    def _parse_kpath(self, fdf_text):
        """
        A helper method to parse the kpath information

        Parameters
        ----------
        fdf_text : str
            The .fdf file text that has the inputs for the Siesta calculation

        Returns
        -------
        None
            None
        """
        raw_kpath = (
            re.findall(
                "(?<=%block BandLines).*\n([\s\S]*?)(?=%endblock BandLines)", fdf_text
            )[0]
            .rstrip()
            .split("\n")
        )

        k_names = []
        special_kpoints = []
        kticks = []
        ngrids = []
        for i, raw_k_point in enumerate(raw_kpath):
            k_name = raw_k_point.split()[-1]
            special_kpoint = raw_k_point.split()[1:-1]
            special_kpoint = [float(coord) for coord in special_kpoint]
            k_tick_points = int(raw_k_point.split()[0])

            special_kpoints.append(special_kpoint)
            k_names.append(k_name)

            ngrids.append(k_tick_points)
            if i == 0:
                kticks.append(0)
            else:
                current_k_tick_point = kticks[i - 1] + k_tick_points
                kticks.append(current_k_tick_point)

        special_kpoint = np.array(special_kpoint)

        self.k_names = [raw_k_point.split()[-1] for raw_k_point in raw_kpath]
        self.kticks = kticks
        self.ngrids = ngrids

        self.special_kpoints = np.zeros(shape=(len(self.kticks) - 1, 2, 3))
        self.modified_knames = []
        for i, special_kpoint in enumerate(special_kpoints):

            if i != len(special_kpoints) - 1:
                self.special_kpoints[i, 0, :] = special_kpoints[i]
                self.special_kpoints[i, 1, :] = special_kpoints[i + 1]

                self.modified_knames.append([k_names[i], k_names[i + 1]])

        print(self.special_kpoints)
        has_time_reversal = True
        self.kpath = KPath(
            knames=self.modified_knames,
            special_kpoints=self.special_kpoints,
            kticks=self.kticks,
            ngrids=self.ngrids,
            has_time_reversal=has_time_reversal,
        )

        return None

    def _parse_direct_lattice(self, fdf_text):
        """
        A helper method to parse the direct lattice information

        Parameters
        ----------
        fdf_text : str
            The .fdf file text that has the inputs for the Siesta calculation

        Returns
        -------
        None
            None
        """
        raw_lattice = (
            re.findall(
                "(?<=%block [Ll]atticeVectors).*\n([\s\S]*?)(?=%endblock [Ll]atticeVectors)",
                fdf_text,
            )[0]
            .rstrip()
            .split("\n")
        )

        direct_lattice = np.zeros(shape=(3, 3))
        for i, raw_vec in enumerate(raw_lattice):
            for j, coord in enumerate(raw_vec.split()):
                direct_lattice[i, j] = float(coord)
        self.direct_lattice = direct_lattice
        return None

    def _parse_atomic_positions(self, fdf_text):
        """
        A helper method to parse the atomic positions information

        Parameters
        ----------
        fdf_text : str
            The .fdf file text that has the inputs for the Siesta calculation

        Returns
        -------
        None
            None
        """

        raw_atom_positions = (
            re.findall(
                "(?<=%block atomiccoordinatesandatomicspecies).*\n([\s\S]*?)(?=%endblock atomiccoordinatesandatomicspecies)",
                fdf_text,
            )[0]
            .rstrip()
            .split("\n")
        )
        raw_species_labels = (
            re.findall(
                "(?<=%block ChemicalSpeciesLabel).*\n([\s\S]*?)(?=%endblock ChemicalSpeciesLabel)",
                fdf_text,
            )[0]
            .rstrip()
            .split("\n")
        )
        atomic_coords_format = re.findall(
            "AtomicCoordinatesFormat\s([A-Za-z]*)", fdf_text
        )[0]
        lattice_constant = float(
            re.findall("LatticeConstant\s([0-9.]*\s)", fdf_text)[0]
        )

        species_list = []
        index_species_mapping = {}
        for raw_species_label in raw_species_labels:
            specie_label = raw_species_label.split()[2]
            specie_index = raw_species_label.split()[0]
            species_list.append(specie_label)
            index_species_mapping.update({specie_index: specie_label})

        n_atoms = len(raw_atom_positions)
        atomic_positions = np.zeros(shape=(n_atoms, 3))
        atom_list = []
        for i, raw_atom_position in enumerate(raw_atom_positions):
            raw_atom_position_list = raw_atom_position.split()
            specie_index = raw_atom_position_list[3]
            atom_list.append(index_species_mapping[specie_index])
            for j, raw_atom_coord in enumerate(raw_atom_position_list[:3]):
                atomic_positions[i, j] = float(raw_atom_coord)

        self.atom_list = atom_list
        self.atomic_positions = atomic_positions
        self.species_list = species_list
        self.index_species_mapping = index_species_mapping
        self.atomic_coords_format = atomic_coords_format
        self.lattice_constant = lattice_constant
        return None

    def _create_structure(self):
        """
        A helper method to create a pyprocar.core.Structure

        Returns
        -------
        None
            None
        """

        # Depends on atomic coords format
        if self.atomic_coords_format == "Fractional":
            structure = Structure(
                atoms=self.atom_list,
                lattice=self.direct_lattice,
                fractional_coordinates=self.atomic_positions,
            )
        else:
            structure = Structure(
                atoms=self.atom_list,
                lattice=self.direct_lattice,
                cartesian_coordinates=self.atomic_positions,
            )

        self.structure = structure

        return None

    def _parse_dos_info(self, fdf_text):
        """
        A helper method to parse the density of states information

        Parameters
        ----------
        fdf_text : str
            The .fdf file text that has the inputs for the Siesta calculation

        Returns
        -------
        None
            None
        """
        raw_pdos_info = (
            re.findall(
                "(?<=%block ProjectedDensityOfStates).*\n([\s\S]*?)(?=%endblock ProjectedDensityOfStates)",
                fdf_text,
            )[0]
            .rstrip()
            .split("\n")
        )
        raw_pdos_kmesh = (
            re.findall(
                "(?<=%block PDOS\.kgrid_Monkhorst_Pack).*\n([\s\S]*?)(?=%endblock PDOS\.kgrid_Monkhorst_Pack)",
                fdf_text,
            )[0]
            .rstrip()
            .split("\n")
        )

        print(raw_pdos_info)

        print(raw_pdos_kmesh)

        return None

    def _parse_bands(self, bands_filepath: Union[str, Path]):
        """
        A helper method to parse the density of states information

        Parameters
        ----------
        bands_filepath : str
            The .BANDS file that has the band structure output information

        Returns
        -------
        None
            None
        """
        with open(bands_filepath) as f:

            bands_text = f.readlines()

            bands_info = bands_text[3]
            raw_bands = "".join(bands_text[4:])

        n_bands = int(bands_info.split()[0])
        n_band_spins = int(bands_info.split()[1])
        n_kpoints = int(bands_info.split()[2])

        bands = np.zeros(shape=(n_kpoints, n_bands, n_band_spins))

        raw_bands_list = raw_bands.split()
        counter = 0
        kdists = []
        for ik in range(n_kpoints):
            # Skipping kdistance value
            kdists.append(float(raw_bands_list[counter]))
            counter += 1
            for ispin in range(n_band_spins):
                for iband in range(n_bands):
                    bands[ik, iband, ispin] = float(raw_bands_list[counter])
                    # Procced to next value inlist
                    counter += 1

        self.bands = bands
        return None
