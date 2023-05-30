__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import re
import copy
import os 
import math
import xml.etree.ElementTree as ET

import numpy as np
from pyprocar.core import DensityOfStates, Structure, ElectronicBandStructure, KPath
from warnings import warn

HARTREE_TO_EV = 27.211386245988  #eV/Hartree
class SiestaParser():
    def __init__(self,
                    fdf_file:str = None,
                    proj_file:str = "atom_proj.projs",
                    efermi:float = None,
                    out_file:str = None
        ):
        """The class is used to parse information in a siesta calculation

        Parameters
        ----------
        fdf_file : str
            The .fdf file that has the inputs for the Siesta calculation
        """

        # Parse some initial information
        # This contains kpath info, prefix for files, and structure info
        self._parse_fdf(fdf_file=fdf_file)
        self._parse_structure()
        if efermi is None:
            self._parse_out(out_file=out_file)

        # parses the bands file. This will initiate the bands array
        self.dirname = os.path.dirname(self.fdf_file) or '.'
        self._parse_bands(bands_file=f'{self.dirname}{os.sep}{self.prefix}.bands')


        # self._parse_struct_out(struct_out_file=f"{self.prefix}{os.sep}STRUCT_OUT")
        
        if os.path.exists(proj_file):
            self._parse_projections(proj_file)

        self.ebs = ElectronicBandStructure(
            kpoints=self.kpoints,
            bands=self.bands,
            projected=self._spd2projected(self.spd),
            efermi=self.efermi,
            kpath=self.kpath,
            reciprocal_lattice=self.reciprocal_lattice
        )


    def _parse_fdf(self,fdf_file):
        """A helper method to parse the infromation inside the fdf file

        Parameters
        ----------
        fdf_file : str
            The .fdf file that has the inputs for the Siesta calculation

        Returns
        -------
        None
            None
        """
        if fdf_file is None:
            try:
                self.fdf_file = [f for f in os.listdir() if '.fdf' in f][0]
            except IndexError:
                raise FileNotFoundError("The .fdf file could not be found.")
        else:
            self.fdf_file = fdf_file

        with open(self.fdf_file, 'r') as f:
            fdf_text = f.read()

        self.prefix = re.findall(r'SystemLabel\s*(.*)' , fdf_text)[0]

        #self._parse_direct_lattice(fdf_text=fdf_text)
        #self._parse_atomic_positions(fdf_text=fdf_text)
        #self._create_structure()

        is_bands_calc = len(re.findall("%block (BandLines)", fdf_text)) == 1
        if is_bands_calc:
            self._parse_kpath(fdf_text=fdf_text)

        is_dos_calc = len(re.findall("%block (ProjectedDensityOfStates)", fdf_text)) == 1
        if is_dos_calc:
            self._parse_dos_info(fdf_text=fdf_text)

        return None

    def _parse_out(self, out_file):

        if out_file is None:
            try:
                self.out_file = [f for f in os.listdir() if '.out' in f][0]
            except IndexError:
                warn(
                "WARNING: No output file was provided or could be found. Setting efermi = 0"
                )
                self.efermi = 0.0
                return None
        else:
            self.out_file = out_file

        with open(self.out_file, 'r') as f:
            out_text = f.read()

        try:
            efermi = re.findall(r'Fermi =\s*(.*)', out_text)[0]
        except IndexError:
            warn(
                "WARNING: Failed to read the output file. Setting efermi = 0"
            )
            efermi = 0.0
        self.efermi = float(efermi)


    def _parse_kpath(self,fdf_text):
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
        raw_kpath = re.findall("(?<=%block BandLines).*\n([\s\S]*?)(?=%endblock BandLines)", fdf_text)[0].rstrip().split('\n')

        k_names=[]
        special_kpoints=[]
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
            if i==0:
                kticks.append(0)
            else:
                current_k_tick_point = kticks[i-1]+k_tick_points
                kticks.append(current_k_tick_point )
                
        special_kpoint = np.array(special_kpoint)

        self.k_names = [raw_k_point.split()[-1] for raw_k_point in raw_kpath] 
        self.kticks = kticks
        self.ngrids = ngrids

        self.special_kpoints = np.zeros(shape = (len(self.kticks) -1 ,2,3) )
        self.modified_knames = []
        for i, special_kpoint in enumerate(special_kpoints):
            
            if i != len(special_kpoints)-1:
                self.special_kpoints[i,0,:] = special_kpoints[i]
                self.special_kpoints[i,1,:] = special_kpoints[i+1]

                self.modified_knames.append([k_names[i], k_names[i+1] ])


        print(self.special_kpoints)
        has_time_reversal = True
        self.kpath = KPath(
                    knames=self.modified_knames,
                    special_kpoints=self.special_kpoints,
                    kticks = self.kticks,
                    ngrids=self.ngrids,
                    has_time_reversal=has_time_reversal,
                )

        return None

    def _parse_direct_lattice(self,fdf_text):
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
        raw_lattice = re.findall("(?<=%block [Ll]atticeVectors).*\n([\s\S]*?)(?=%endblock [Ll]atticeVectors)", fdf_text)[0].rstrip().split('\n')

        direct_lattice = np.zeros(shape=(3,3))
        for i, raw_vec in enumerate(raw_lattice):
            for j, coord in enumerate(raw_vec.split()):
                direct_lattice[i,j] = float(coord)
        self.direct_lattice=direct_lattice
        self.reciprocal_lattice = 2*np.pi* np.linalg.inv( direct_lattice ).T
        return None

    def _parse_atomic_positions(self,fdf_text):
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

        raw_atom_positions = re.findall("(?<=%block atomiccoordinatesandatomicspecies).*\n([\s\S]*?)(?=%endblock atomiccoordinatesandatomicspecies)", fdf_text)[0].rstrip().split('\n')
        raw_species_labels = re.findall("(?<=%block ChemicalSpeciesLabel).*\n([\s\S]*?)(?=%endblock ChemicalSpeciesLabel)", fdf_text)[0].rstrip().split('\n')
        atomic_coords_format = re.findall("AtomicCoordinatesFormat\s([A-Za-z]*)",fdf_text)[0]
        lattice_constant = float(re.findall("LatticeConstant\s([0-9.]*\s)",fdf_text)[0])

        species_list=[]
        index_species_mapping = {}
        for raw_species_label in raw_species_labels:
            specie_label = raw_species_label.split()[2]
            specie_index = raw_species_label.split()[0]
            species_list.append(specie_label)
            index_species_mapping.update({specie_index:specie_label})

        n_atoms = len(raw_atom_positions)
        atomic_positions=np.zeros(shape = (n_atoms,3) )
        atom_list = []
        for i,raw_atom_position in enumerate(raw_atom_positions):
            raw_atom_position_list= raw_atom_position.split()
            specie_index = raw_atom_position_list[3]
            atom_list.append(index_species_mapping[specie_index])
            for j,raw_atom_coord in enumerate(raw_atom_position_list[:3]):
                atomic_positions[i,j] = float(raw_atom_coord)

        self.atom_list = atom_list
        self.atomic_positions = atomic_positions
        self.species_list=species_list
        self.index_species_mapping=index_species_mapping
        self.atomic_coords_format=atomic_coords_format
        self.lattice_constant=lattice_constant
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
        if self.atomic_coords_format == 'Fractional':
            structure = Structure(atoms=self.atom_list, lattice = self.direct_lattice, fractional_coordinates =self.atomic_positions )
        else:
            structure = Structure(atoms=self.atom_list, lattice = self.direct_lattice, cartesian_coordinates =self.atomic_positions )

        self.structure = structure

        return None

    def _parse_structure(self):

        from ase.data import atomic_numbers
        atomic_symbols = {key: value for value, key in atomic_numbers.items()}

        print(self.prefix)
        filename = f'{self.prefix}.STRUCT_OUT'

        cell = np.loadtxt(filename, max_rows=3)
        self.reciprocal_lattice = 2*np.pi* np.linalg.inv( cell ).T

        raw_data = np.loadtxt(filename, skiprows=4).reshape((-1, 5))
        numbers = raw_data[:, 1]
        positions = raw_data[:, (2, 3, 4)]
        atom_list = [atomic_symbols[x] for x in numbers]

        self.structure = Structure(
            atoms=atom_list,
            lattice=cell,
            fractional_coordinates=positions
        )

    def _parse_dos_info(self,fdf_text):
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
        raw_pdos_info = re.findall("(?<=%block ProjectedDensityOfStates).*\n([\s\S]*?)(?=%endblock ProjectedDensityOfStates)", fdf_text)[0].rstrip().split('\n')
        raw_pdos_kmesh = re.findall("(?<=%block PDOS\.kgrid_Monkhorst_Pack).*\n([\s\S]*?)(?=%endblock PDOS\.kgrid_Monkhorst_Pack)", fdf_text)[0].rstrip().split('\n')

        print(raw_pdos_info)

        print(raw_pdos_kmesh)

        return None

    def _parse_bands(self,bands_file):
        """
        A helper method to parse the density of states information

        Parameters
        ----------
        bands_file : str
            The .BANDS file that has the band structure output information

        Returns
        -------
        None
            None
        """
        with open(bands_file) as f:

            bands_text = f.readlines()

            bands_info = bands_text[3]
            raw_bands = "".join(bands_text[4:])

        n_bands = int(bands_info.split()[0])
        n_band_spins = int(bands_info.split()[1])
        n_kpoints = int(bands_info.split()[2])

        bands = np.zeros(shape = (n_kpoints,n_bands,n_band_spins))
 
        raw_bands_list = raw_bands.split()
        counter=0
        kdists = []
        for ik in range(n_kpoints):
            # Skipping kdistance value
            kdists.append(float(raw_bands_list[counter]))
            counter +=1
            for ispin in range(n_band_spins):
                for iband in range(n_bands):
                    bands[ik,iband,ispin] = float(raw_bands_list[counter])
                    # Procced to next value inlist
                    counter +=1
        
        self.bands=bands
        return None

    def _parse_projections(self, proj_file):

        with open(proj_file, 'r') as f:
            file_str = f.read().replace('*', '')

        nkpoints, nbands, nspin, natoms, norbitals = [int(x) for x in file_str.split('\n')[1].split()]

        kpoints = re.findall(r"point:(.+)", file_str)
        kpoints = [x.split() for x in kpoints]
        kpoints = np.array(kpoints, dtype=float)
        self.kpoints = kpoints[:, (1, 2, 3)]

        spd = re.findall(r"s      py.+([-.\d\seto]+)", file_str)
        spd = [x.split() for x in spd]
        spd = np.array(spd, dtype=float)
        spd = spd.reshape((nkpoints, nbands, nspin, natoms, norbitals+2))
        tot = np.sum(spd, axis=-2).reshape((nkpoints, nbands, nspin, 1, norbitals+2))

        self.spd = np.concatenate((spd, tot), axis=3)

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
        norbitals = spd.shape[4] - 2
        if spd.shape[2] == 4:
            nspins = 3
        else:
            nspins = spd.shape[2]
        if nspins == 2:
            nbands = int(spd.shape[1] / 2)
        else:
            nbands = spd.shape[1]
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
        if nspins == 3:
            projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :-1]
        elif nspins == 2:
            projected[:, :, :, 0, :, 0] = temp_spd[:, :nbands, :-1, 1:-1, 0]
            projected[:, :, :, 0, :, 1] = temp_spd[:, nbands:, :-1, 1:-1, 0]
        else:
            projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]

        return projected
